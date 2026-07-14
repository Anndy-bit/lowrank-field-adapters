"""
S³ Hybrid Transformer Layer

Complete S³ (Spectral-Spatial-Smooth) adaptation of one transformer layer.

Architecture:
    Input h_in
      ↓
    [STB] spectral preconditioning with U_k, Σ_log
      ↓
    [Attention] with SVMO on W_Q, W_K, W_V, W_O
      ↓
    h_attn_res = h_attn + h_in (residual)
      ↓
    [NMF_1] post-attention manifold flow (ODE)
      ↓
    [RMSNorm] (frozen)
      ↓
    [MLP] with SVMO on W_up, W_gate, W_down
      ↓
    h_mlp_res = h_mlp + h_post_attn (residual)
      ↓
    [NMF_2] post-MLP manifold flow (ODE)
      ↓
    Output h_out

VRAM per layer: ~250MB (pretrained weights) + ~16MB (adapter overhead).
Training: one layer at a time via CPU↔GPU swap.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any, Tuple
import math

from .svmo import SVMOAdapter, create_svmo_from_linear
from .nmf import NMFFlow, create_nmf_pair
from .stb import STBBridge, create_stb_bridge_from_svmo
from .base import count_total_params, vram_estimate_all


class S3TransformerLayer(nn.Module):
    """One transformer layer fully adapted with S³.

    All pretrained weights frozen. SVMO on 7 projection matrices.
    STB at layer input. NMF flows after attention and MLP blocks.

    This layer is designed for LAYER-BY-LAYER training: weights are
    loaded to GPU one layer at a time, processed, and unloaded.

    Args:
        pretrained_layer: the original HuggingFace transformer block
        svmo_k: SVD rank for SVMO
        svmo_hidden: hidden dim of modulation MLPs
        svmo_alpha: maximum modulation amplitude
        nmf_bottleneck: bottleneck dim for NMF velocity field
        nmf_T: NMF flow duration
        nmf_N: NMF RK4 steps
        nmf_solver: 'rk4' or 'euler'
        stb_head_dim: cross-attention head dim
        stb_beta: coupling strength
    """

    def __init__(
        self,
        pretrained_layer: nn.Module,
        svmo_k: int = 128,
        svmo_hidden: int = 32,
        svmo_alpha: float = 0.3,
        nmf_bottleneck: int = 8,
        nmf_T: float = 1.0,
        nmf_N: int = 4,
        nmf_solver: str = "rk4",
        stb_head_dim: Optional[int] = None,
        stb_beta: float = 0.5,
        enable_sbs: bool = True,
        sbs_p: float = 0.3,
        svd_dir: Optional[str] = None,
        layer_idx: int = 0,
        svd_dtype: torch.dtype = torch.float32,
    ):
        super().__init__()

        # Store all hyperparams for _build_from_pretrained
        self.svmo_k = svmo_k
        self.svmo_hidden = svmo_hidden
        self.svmo_alpha = svmo_alpha
        self.nmf_bottleneck = nmf_bottleneck
        self.nmf_T = nmf_T
        self.nmf_N = nmf_N
        self.nmf_solver = nmf_solver
        self.stb_head_dim = stb_head_dim
        self.stb_beta = stb_beta
        self.enable_sbs = enable_sbs
        self.sbs_p = sbs_p
        self.svd_dir = svd_dir
        self.layer_idx = layer_idx
        self.svd_dtype = svd_dtype

        self._build_from_pretrained(pretrained_layer)
        dim = self.dim

        attn = pretrained_layer.self_attn
        mlp = pretrained_layer.mlp

        def _svmo(proj, name):
            return create_svmo_from_linear(
                proj, svmo_k, svmo_hidden, svmo_alpha,
                svd_dir=svd_dir, layer_name=f"layer_{layer_idx}.{name}",
                dtype=svd_dtype,
            )

        self.svmo_q = _svmo(attn.q_proj, "q_proj")
        self.svmo_k_proj = _svmo(attn.k_proj, "k_proj")
        self.svmo_v = _svmo(attn.v_proj, "v_proj")
        self.svmo_o = _svmo(attn.o_proj, "o_proj")

        self.svmo_up = _svmo(mlp.up_proj, "up_proj")
        self.svmo_gate = _svmo(mlp.gate_proj, "gate_proj")
        self.svmo_down = _svmo(mlp.down_proj, "down_proj")

        self.stb = STBBridge(
            self.svmo_q.U_k.detach().clone(), svmo_k, stb_head_dim, stb_beta
        )
        self.enable_sbs = enable_sbs
        self.sbs_p = sbs_p

        self.nmf_attn, self.nmf_mlp = create_nmf_pair(
            dim, nmf_bottleneck, nmf_T, nmf_N, nmf_solver
        )

        self._count_params()

    def _build_from_pretrained(self, pretrained_layer):
        attn = pretrained_layer.self_attn
        if hasattr(attn, "q_proj"):
            self.dim = attn.q_proj.in_features
            self.num_heads = getattr(attn, "num_heads", 28)
            self.head_dim = getattr(attn, "head_dim", self.dim // self.num_heads)
        elif hasattr(attn, "q_proj") and hasattr(attn.q_proj, "weight"):
            self.dim = attn.q_proj.weight.shape[1]
            self.num_heads = getattr(attn, "num_heads", 28)
            self.head_dim = self.dim // self.num_heads
        else:
            raise ValueError(
                "Cannot infer dimensions from pretrained layer"
            )

        for name in [
            "input_layernorm", "post_attention_layernorm",
            "input_norm", "post_attn_norm",
        ]:
            if hasattr(pretrained_layer, name):
                norm = getattr(pretrained_layer, name)
                self.register_buffer(f"norm_{name}", norm.weight.data.clone().detach())

    def _count_params(self):
        self.trainable_params = sum(
            m.trainable_param_count()
            for m in [
                self.svmo_q, self.svmo_k_proj, self.svmo_v, self.svmo_o,
                self.svmo_up, self.svmo_gate, self.svmo_down,
                self.stb, self.nmf_attn, self.nmf_mlp,
            ]
        )
        self.frozen_params = sum(
            m.frozen_param_count()
            for m in [
                self.svmo_q, self.svmo_k_proj, self.svmo_v, self.svmo_o,
                self.svmo_up, self.svmo_gate, self.svmo_down,
                self.stb,
            ]
        )

    @torch.no_grad()
    def _update_sigma_logs(self):
        """Sync Σ_log buffer on STB with current SVMO modulation state."""
        S_mod = self.svmo_q._modulate()
        stb_sigma_log = torch.log(S_mod + 1e-8)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[Any] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        is_causal: bool = False,
        nmf_N_steps: Optional[int] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, ...]:
        residual = hidden_states

        # Normalize to exactly 3D [B, S, D]
        ndim = hidden_states.dim()
        if ndim == 2:
            # [S, D] → [1, S, D]
            hidden_states = hidden_states.unsqueeze(0)
            residual = hidden_states
        elif ndim == 1:
            # [D] → [1, 1, D]
            hidden_states = hidden_states.unsqueeze(0).unsqueeze(0)
            residual = hidden_states
        elif ndim == 4:
            # [B, 1, S, D] — the 2nd dim (index 1) is a spurious singleton from checkpoint storage.
            # Remove it via squeeze(1) → [B, S, D].
            # If shape is [B, X, S, D] with X!=1, flatten dims 1..-2 as extra batch: [B*X, S, D].
            B, X, S, D = hidden_states.shape
            if X == 1:
                hidden_states = hidden_states.squeeze(1)  # → [B, S, D]
                residual = hidden_states
            else:
                hidden_states = hidden_states.reshape(B * X, S, D)
                residual = hidden_states
        elif ndim > 4:
            # [B, X, Y, ..., S, D] — flatten dims 1..-2 as extra batch dim
            B = hidden_states.shape[0]
            hidden_states = hidden_states.reshape(B, -1, hidden_states.shape[-1])
            residual = hidden_states

        B, S, D = hidden_states.shape
        hidden_states_flat = hidden_states.reshape(-1, D)
        sigma_log = torch.log(self.svmo_q._modulate() + 1e-8)

        if not self.enable_sbs or torch.rand(1).item() < self.sbs_p:
            hidden_states = self.stb(hidden_states, sigma_log)

        normed = self._apply_norm(hidden_states, "input")

        # Ensure normed is exactly 3D [B, S, D]
        if normed.dim() == 2:
            normed = normed.unsqueeze(0)  # [B, D] → [1, B, D]
        elif normed.dim() == 1:
            normed = normed.unsqueeze(0).unsqueeze(0)  # [D] → [1, 1, D]
        elif normed.dim() == 4:
            # [B, 1, S, D] → squeeze out the second dim
            normed = normed.squeeze(1)  # → [B, S, D]
        elif normed.dim() > 4:
            normed = normed.flatten(0, -3)  # → [B*..., S, D]

        B, S, D = normed.shape
        normed_flat = normed.reshape(B * S, D)

        q_out = self.svmo_q(normed_flat)
        k_out = self.svmo_k_proj(normed_flat)
        v_out = self.svmo_v(normed_flat)

        q_dim = q_out.shape[-1]
        k_dim = k_out.shape[-1]
        v_dim = v_out.shape[-1]

        num_q_heads = q_dim // self.head_dim
        num_kv_heads = k_dim // self.head_dim

        q = q_out.reshape(B, S, num_q_heads, self.head_dim).transpose(1, 2)
        k = k_out.reshape(B, S, num_kv_heads, self.head_dim).transpose(1, 2)
        v = v_out.reshape(B, S, num_kv_heads, self.head_dim).transpose(1, 2)

        if num_kv_heads < num_q_heads:
            repeat_factor = num_q_heads // num_kv_heads
            k = k.repeat_interleave(repeat_factor, dim=1)
            v = v.repeat_interleave(repeat_factor, dim=1)

        scale = 1.0 / math.sqrt(self.head_dim)
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * scale

        if is_causal:
            causal_mask = torch.triu(
                torch.ones(S, S, device=hidden_states.device, dtype=q.dtype),
                diagonal=1,
            )
            causal_mask = causal_mask.masked_fill(
                causal_mask.bool(), float("-inf")
            )
            causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)
            attn_weights = attn_weights + causal_mask

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).reshape(B, S, D)
        attn_output = self.svmo_o(attn_output)

        hidden_states = residual + attn_output
        hidden_states = self.nmf_attn(hidden_states, N_steps=nmf_N_steps)

        normed = self._apply_norm(hidden_states, "post_attn")
        normed_flat = normed.reshape(-1, D)

        up = self.svmo_up(normed_flat)
        gate = self.svmo_gate(normed_flat)
        mlp_hidden = F.silu(gate) * up
        mlp_output = self.svmo_down(mlp_hidden).reshape(B, S, D)

        hidden_states = hidden_states + mlp_output
        hidden_states = self.nmf_mlp(hidden_states, N_steps=nmf_N_steps)

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (attn_weights,)
        if use_cache:
            outputs += (None,)

        return outputs

    def _apply_norm(
        self, hidden_states: torch.Tensor, norm_type: str
    ) -> torch.Tensor:
        # Guarantee 3D at entry
        if hidden_states.dim() == 2:
            hidden_states = hidden_states.unsqueeze(1)  # [B, D] → [B, 1, D]
        elif hidden_states.dim() == 1:
            hidden_states = hidden_states.unsqueeze(0).unsqueeze(0)  # [D] → [1, 1, D]

        attr_map = {
            "input": ["norm_input_layernorm", "norm_input_norm"],
            "post_attn": ["norm_post_attention_layernorm", "norm_post_attn_norm"],
        }
        candidates = attr_map[norm_type]
        for name in candidates:
            if hasattr(self, name):
                weight = getattr(self, name)
                if weight.ndim > 0:
                    variance = hidden_states.pow(2).mean(-1, keepdim=True)
                    hidden_states = hidden_states * torch.rsqrt(variance + 1e-6)
                    return weight * hidden_states
                else:
                    return F.layer_norm(
                        hidden_states, (hidden_states.shape[-1],),
                        weight=None, bias=None, eps=1e-6
                    )
        return hidden_states

    def summary(self) -> Dict[str, Any]:
        return {
            "trainable_params": self.trainable_params,
            "frozen_params": self.frozen_params,
            "svmo_k": self.svmo_q.k,
            "svmo_hidden": self.svmo_q.modulation.hidden_dim,
            "svmo_alpha": self.svmo_q.alpha,
            "nmf_bottleneck": self.nmf_attn.field.bottleneck_dim,
            "nmf_T": self.nmf_attn.T,
            "nmf_N": self.nmf_attn.N_steps,
            "stb_beta": self.stb.beta,
            "total_vram_mb": vram_estimate_all(
                [
                    self.svmo_q, self.svmo_k_proj, self.svmo_v, self.svmo_o,
                    self.svmo_up, self.svmo_gate, self.svmo_down,
                    self.stb, self.nmf_attn, self.nmf_mlp,
                ]
            ),
        }


def create_s3_layer_from_hf(
    hf_layer: nn.Module,
    svmo_k: int = 128,
    svmo_hidden: int = 32,
    svmo_alpha: float = 0.3,
    nmf_bottleneck: int = 8,
    nmf_T: float = 1.0,
    nmf_N: int = 4,
    stb_beta: float = 0.5,
    enable_sbs: bool = True,
    sbs_p: float = 0.3,
    svd_dir: Optional[str] = None,
    layer_idx: int = 0,
) -> S3TransformerLayer:
    """Create an S³ layer from a HuggingFace transformer block."""
    return S3TransformerLayer(
        hf_layer,
        svmo_k=svmo_k,
        svmo_hidden=svmo_hidden,
        svmo_alpha=svmo_alpha,
        nmf_bottleneck=nmf_bottleneck,
        nmf_T=nmf_T,
        nmf_N=nmf_N,
        stb_beta=stb_beta,
        enable_sbs=enable_sbs,
        sbs_p=sbs_p,
        svd_dir=svd_dir,
        layer_idx=layer_idx,
    )