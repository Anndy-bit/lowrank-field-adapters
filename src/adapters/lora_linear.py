"""
LoRA baseline — the reference every PEFT paper is measured against.

Implemented with the SAME contract as SVMOLinear so it drops into the same
streaming trainer and FrozenStreamer (which resolves `.base` to find the frozen
weight):

    y = W·x  +  (alpha/r) · B(A(x))          A ~ N(0, 0.02), B = 0

B is zero-initialised, so Δ = 0 at init and the wrapped model reproduces the
frozen base exactly — same identity-init property as S³, which makes the
base-vs-after comparison fair.

Params per matrix: r·(d_in + d_out). For Qwen2.5-7B (28 layers × 7 projections)
the totals are ≈ 2.52M·r  →  r=1: 2.52M · r=2: 5.05M · r=8: 20.2M.
(S³ full is 3.90M, so r=1 and r=2 bracket it for a params-matched comparison.)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    def __init__(self, linear: nn.Linear, r: int = 8, alpha: float = 16.0):
        super().__init__()
        d_out, d_in = linear.weight.shape
        self.d_in, self.d_out, self.r = d_in, d_out, r
        self.scaling = alpha / r

        for p in linear.parameters():
            p.requires_grad = False
        self.base = linear                      # frozen; streamer targets `.base`

        self.lora_A = nn.Parameter(torch.empty(r, d_in))
        self.lora_B = nn.Parameter(torch.zeros(d_out, r))
        nn.init.normal_(self.lora_A, std=0.02)  # B stays zero -> identity at init

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = self.base(x)
        z = F.linear(x, self.lora_A.to(x.dtype))          # [*, r]
        delta = F.linear(z, self.lora_B.to(x.dtype))      # [*, d_out]
        return base + delta * self.scaling

    def trainable_param_count(self) -> int:
        return self.lora_A.numel() + self.lora_B.numel()

    def extra_repr(self) -> str:
        return f"d_in={self.d_in}, d_out={self.d_out}, r={self.r}"


_PROJ = [("self_attn", "q_proj"), ("self_attn", "k_proj"), ("self_attn", "v_proj"),
         ("self_attn", "o_proj"), ("mlp", "gate_proj"), ("mlp", "up_proj"), ("mlp", "down_proj")]


def apply_lora_to_blocks(blocks, r: int = 8, alpha: float = 16.0) -> int:
    """Swap the 7 projections of each block for LoRALinear. Returns trainable count.

    `blocks` are S3Blocks built with every S³ operator disabled — i.e. plain
    decoder wiring — so the only adaptation is LoRA. Same forward path, same
    streamer, same trainer: an apples-to-apples baseline.
    """
    n = 0
    for blk in blocks:
        for sub, proj in _PROJ:
            parent = getattr(blk, sub)
            lin = getattr(parent, proj)
            lin = getattr(lin, "base", lin)     # unwrap if already wrapped
            lora = LoRALinear(lin, r=r, alpha=alpha)
            setattr(parent, proj, lora)
            n += lora.trainable_param_count()
        for name, p in blk.named_parameters():
            p.requires_grad = ("lora_A" in name) or ("lora_B" in name)
    return n
