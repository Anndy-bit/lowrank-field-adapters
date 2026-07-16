"""
S3Block — implementation-correct S³ transformer layer.

Instead of reimplementing attention/MLP (which dropped RoPE, causal masking and
GQA in the previous version), we WRAP the original HuggingFace decoder layer:

  * its 7 Linear projections (q,k,v,o, gate,up,down) are swapped for SVMOLinear
    (frozen W + learnable spectral delta), so weights stay correct and full;
  * the attention/MLP submodules are called as-is, so RoPE / causal mask / GQA /
    RMSNorm come straight from HF and are correct by construction;
  * STB (residual, zero at init) preconditions the block input;
  * two NMF flows (zero at init) deform post-attention and post-MLP states.

At initialisation every adapter is identity, so an S3-wrapped model reproduces
the frozen base model exactly (verified by tests/test_s3_parity.py).
"""

from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.adapters.svmo_linear import SVMOLinear
from src.adapters.nmf import NMFFlow


def load_svd_factors(svd_dir, layer_idx: int, proj_name: str):
    """Load precomputed (U_k, S_k, Vt_k) for one projection, or None if absent.

    Matches the on-disk layout produced by src/utils/randomized_svd.py:
        {svd_dir}/layer_{idx}.{proj}/{U_k,S_k,Vt_k}.safetensors
    """
    if svd_dir is None:
        return None
    layer_dir = Path(svd_dir) / f"layer_{layer_idx}.{proj_name}"
    if not layer_dir.exists():
        return None
    try:
        from safetensors.torch import load_file
        U = load_file(str(layer_dir / "U_k.safetensors"))["U_k"]
        S = load_file(str(layer_dir / "S_k.safetensors"))["S_k"]
        Vt = load_file(str(layer_dir / "Vt_k.safetensors"))["Vt_k"]
        return U, S, Vt
    except Exception:
        return None


class STBResidual(nn.Module):
    """Spectral Transport Bridge as a residual, identity-at-init preconditioner.

        h_out = h + U_k · c_φ(U_k^T h, log σ),   c_φ zero at init.

    Unlike the previous STB (which replaced h by U_k U_k^T h, destroying the
    orthogonal complement even at init), this only ever *adds* a correction that
    lives in span(U_k) and is zero at the start of training.
    """

    def __init__(self, U_k: torch.Tensor, k: int, head_dim: Optional[int] = None, beta: float = 0.5):
        super().__init__()
        self.k = k
        self.beta = beta
        self.head_dim = head_dim or max(k // 4, 8)
        self.register_buffer("U_k", U_k.detach().clone())   # [d, k]
        self.W_q = nn.Linear(k, self.head_dim, bias=False)
        self.W_k = nn.Linear(k, self.head_dim, bias=False)
        self.W_v = nn.Linear(k, self.head_dim, bias=False)
        self.W_o = nn.Linear(self.head_dim, k, bias=False)
        nn.init.normal_(self.W_q.weight, std=0.02)
        nn.init.normal_(self.W_k.weight, std=0.02)
        nn.init.normal_(self.W_v.weight, std=0.02)
        nn.init.zeros_(self.W_o.weight)   # zero-init output -> identity at start
        self.num_params = sum(p.numel() for p in self.parameters())

    def forward(self, h: torch.Tensor, sigma_log: torch.Tensor) -> torch.Tensor:
        U_k = self.U_k.to(h.dtype)
        s = F.linear(h, U_k.t())                       # U_k^T h -> [*, k]
        q = self.W_q(s)
        k = self.W_k(sigma_log.to(h.dtype).expand(s.shape[:-1] + (self.k,)))
        v = self.W_v(s)
        score = (q * k).sum(dim=-1, keepdim=True) * (self.head_dim ** -0.5)
        delta_k = self.W_o(torch.sigmoid(score) * v)   # [*, k], zero at init
        delta = F.linear(delta_k, U_k)                 # -> [*, d]
        return h + self.beta * delta


class S3Block(nn.Module):
    """S³-adapted wrapper around one HuggingFace decoder layer (in place)."""

    def __init__(
        self,
        hf_layer: nn.Module,
        svmo_k: int = 128,
        svmo_hidden: int = 32,
        svmo_alpha: float = 0.3,
        nmf_bottleneck: int = 8,
        nmf_T: float = 1.0,
        nmf_N: int = 4,
        nmf_solver: str = "rk4",
        stb_beta: float = 0.5,
        enable_svmo: bool = True,
        enable_stb: bool = True,
        enable_nmf: bool = True,
        use_randomized_svd: bool = False,
        svd_dir=None,
        layer_idx: int = 0,
        sbs_p: float = 1.0,
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.sbs_p = sbs_p  # SBS (Thm 10): prob. of applying STB per forward; 1.0 = always
        self.self_attn = hf_layer.self_attn
        self.mlp = hf_layer.mlp
        self.input_layernorm = hf_layer.input_layernorm
        self.post_attention_layernorm = hf_layer.post_attention_layernorm
        self.dim = self.self_attn.q_proj.weight.shape[1]

        self.enable_svmo = enable_svmo
        self.enable_stb = enable_stb
        self.enable_nmf = enable_nmf

        def wrap(linear, name):
            fac = load_svd_factors(svd_dir, layer_idx, name)
            return SVMOLinear(linear, svmo_k, svmo_hidden, svmo_alpha,
                              use_randomized=use_randomized_svd, svd_factors=fac)

        if enable_svmo:
            self.self_attn.q_proj = wrap(self.self_attn.q_proj, "q_proj")
            self.self_attn.k_proj = wrap(self.self_attn.k_proj, "k_proj")
            self.self_attn.v_proj = wrap(self.self_attn.v_proj, "v_proj")
            self.self_attn.o_proj = wrap(self.self_attn.o_proj, "o_proj")
            self.mlp.gate_proj = wrap(self.mlp.gate_proj, "gate_proj")
            self.mlp.up_proj = wrap(self.mlp.up_proj, "up_proj")
            self.mlp.down_proj = wrap(self.mlp.down_proj, "down_proj")

        if enable_stb and enable_svmo:
            self.stb = STBResidual(self.self_attn.q_proj.U_k, self.self_attn.q_proj.k,
                                   beta=stb_beta)
        else:
            self.stb = None

        if enable_nmf:
            self.nmf_attn = NMFFlow(self.dim, nmf_bottleneck, nmf_T, nmf_N, nmf_solver)
            self.nmf_mlp = NMFFlow(self.dim, nmf_bottleneck, nmf_T, nmf_N, nmf_solver)
        else:
            self.nmf_attn = self.nmf_mlp = None

    def _sigma_log(self) -> torch.Tensor:
        m = self.self_attn.q_proj.modulated_sigma()
        return torch.log(m + 1e-8)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_values=None,
        use_cache: bool = False,
        cache_position=None,
        position_embeddings=None,
        nmf_N_steps: Optional[int] = None,
        stb_apply: Optional[bool] = None,
        **kwargs,
    ):
        h = hidden_states
        # SBS (Thm 10): apply STB with prob sbs_p during training; always at eval.
        # `stb_apply` lets the streaming trainer replay the SAME decision in its
        # manual backward recompute (otherwise forward/backward would diverge).
        if self.stb is not None:
            if stb_apply is None:
                stb_apply = (not self.training) or self.sbs_p >= 1.0 or (torch.rand(()) < self.sbs_p)
            if stb_apply:
                h = self.stb(h, self._sigma_log())

        residual = h
        x = self.input_layernorm(h)
        attn_out = self.self_attn(
            hidden_states=x,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            cache_position=cache_position,
            **kwargs,
        )
        attn_out = attn_out[0] if isinstance(attn_out, tuple) else attn_out
        h = residual + attn_out
        if self.nmf_attn is not None:
            h = self.nmf_attn(h, N_steps=nmf_N_steps)

        residual = h
        x = self.post_attention_layernorm(h)
        h = residual + self.mlp(x)
        if self.nmf_mlp is not None:
            h = self.nmf_mlp(h, N_steps=nmf_N_steps)

        return h

    def trainable_parameters(self):
        for p in self.parameters():
            if p.requires_grad:
                yield p


def wrap_model_with_s3(model, freeze_base: bool = True, **block_kwargs) -> nn.ModuleList:
    """Replace every decoder layer in a HF causal-LM with an S3Block, in place.

    Returns the new ModuleList of S3Blocks. The rest of `model` (embeddings,
    rotary_emb, final norm, lm_head) is reused unchanged, so `model(input_ids)`
    runs the correct forward with S³ adapters inserted.
    """
    if freeze_base:
        for p in model.parameters():
            p.requires_grad = False

    layers = model.model.layers
    new_layers = nn.ModuleList()
    for i in range(len(layers)):
        block = S3Block(layers[i], layer_idx=i, **block_kwargs)
        layers[i] = block
        new_layers.append(block)
    # ensure only adapter params require grad
    for blk in new_layers:
        for name, p in blk.named_parameters():
            p.requires_grad = (
                "modulation" in name or name.startswith("stb.")
                or ".stb." in name or "nmf_attn" in name or "nmf_mlp" in name
                or "field" in name or name.startswith("W_")
            )
    return new_layers
