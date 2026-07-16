"""
SVMOLinear — corrected, residual Singular Value Modulation for a frozen Linear.

This is the implementation-correct form of SVMO (formalismo.md §1.1):

    W_SVMO · x = W · x + Δ_SVMO · x,   Δ_SVMO = U_k (m_θ(Σ_k) - Σ_k) V_k^T

Crucially, the FULL frozen weight W is kept and applied every forward. The
rank-k SVD factors only parameterise the *delta*. This fixes the previous
implementation, which replaced W by its rank-k truncation W_k (destroying the
pretrained model, especially for the wide MLP matrices) and never restored the
tail (W - W_k).

Properties preserved from the theory:
- U_k, V_k, Σ_k frozen; only θ of g_θ trains.
- m_θ(σ) = σ (1 + α tanh(g_θ(log(σ+ε)))) ⇒ Δσ = σ·α·tanh(g_θ) = 0 at init.
- Identity init: g_θ(·) ≈ 0 ⇒ Δ_SVMO = 0 ⇒ forward == frozen Linear exactly.
- O(H²) trainable params per matrix, independent of d and k.

Drop-in replacement for nn.Linear inside a HuggingFace attention/MLP block.
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class ModulationMLP(nn.Module):
    """Tiny scalar MLP g_θ: R -> R, 2 hidden layers of width H, GELU.

    Last layer initialised near zero so g_θ(x) ≈ 0 at start (identity init).
    """

    def __init__(self, hidden_dim: int = 32):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.fc1 = nn.Linear(1, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
        nn.init.normal_(self.fc1.weight, std=0.02)
        nn.init.zeros_(self.fc1.bias)
        nn.init.normal_(self.fc2.weight, std=0.02)
        nn.init.zeros_(self.fc2.bias)
        nn.init.zeros_(self.fc3.weight)
        nn.init.zeros_(self.fc3.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.gelu(self.fc1(x))
        x = F.gelu(self.fc2(x))
        return self.fc3(x)


def _svd_topk(weight: torch.Tensor, k: int, use_randomized: bool):
    d_out, d_in = weight.shape
    k = min(k, min(d_out, d_in))
    W = weight.detach().float().cpu()
    if use_randomized and min(d_out, d_in) > 512:
        from src.adapters.svmo import randomized_svd
        U, S, Vt = randomized_svd(W, k)
        return U[:, :k].contiguous(), S[:k].contiguous(), Vt[:k, :].contiguous()
    U, S, Vt = torch.linalg.svd(W, full_matrices=False)
    return U[:, :k].contiguous(), S[:k].contiguous(), Vt[:k, :].contiguous()


class SVMOLinear(nn.Module):
    """Frozen Linear + learnable spectral (singular-value) modulation delta.

    Args:
        linear: the frozen nn.Linear to wrap (weight & bias are copied, frozen).
        k: SVD rank for the modulation delta.
        hidden_dim: width H of g_θ.
        alpha: max fractional modulation amplitude.
        use_randomized: use randomized SVD for large matrices.
        svd_factors: optional (U_k, S_k, Vt_k) precomputed offline.
    """

    def __init__(
        self,
        linear: nn.Linear,
        k: int = 128,
        hidden_dim: int = 32,
        alpha: float = 0.3,
        use_randomized: bool = False,
        svd_factors: Optional[tuple] = None,
        eps: float = 1e-8,
    ):
        super().__init__()
        d_out, d_in = linear.weight.shape
        self.d_out, self.d_in = d_out, d_in
        self.alpha = alpha
        self.eps = eps

        # Delegate the base (frozen) forward to the original module. Keeps ONE
        # copy of W and works whether it is fp16, 8-bit/4-bit quantized, or
        # accelerate-offloaded — we never materialise/duplicate the weight.
        for p in linear.parameters():
            p.requires_grad = False
        self.base = linear

        if svd_factors is not None:
            U_k, S_k, Vt_k = svd_factors
            k = min(k, S_k.shape[0])
            U_k, S_k, Vt_k = U_k[:, :k], S_k[:k], Vt_k[:k, :]
        else:
            U_k, S_k, Vt_k = _svd_topk(linear.weight.data, k, use_randomized)
        self.k = S_k.shape[0]

        dtype = linear.weight.dtype if linear.weight.dtype.is_floating_point else torch.float16
        self.register_buffer("U_k", U_k.to(dtype).contiguous())      # [d_out, k]
        self.register_buffer("S_k", S_k.to(dtype).contiguous())      # [k]
        self.register_buffer("Vt_k", Vt_k.to(dtype).contiguous())    # [k, d_in]
        self.register_buffer("S_k_log", torch.log(S_k.float() + eps).to(dtype))

        self.modulation = ModulationMLP(hidden_dim)
        self.num_params = sum(p.numel() for p in self.modulation.parameters())

    def delta_sigma(self) -> torch.Tensor:
        """Δσ = m_θ(σ) - σ = σ · α · tanh(g_θ(log σ)).  Zero at init."""
        x = self.S_k_log.unsqueeze(-1).to(next(self.modulation.parameters()).dtype)
        g = self.modulation(x).squeeze(-1)
        return self.S_k.to(g.dtype) * self.alpha * torch.tanh(g)

    def modulated_sigma(self) -> torch.Tensor:
        """m_θ(σ) = σ + Δσ (used by STB for the current spectral state)."""
        return self.S_k.to(self.delta_sigma().dtype) + self.delta_sigma()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = self.base(x)                          # frozen W·x (+bias), any dtype/backend
        dsig = self.delta_sigma().to(x.dtype)
        z = F.linear(x, self.Vt_k.to(x.dtype))       # x V_k^T -> [*, k]
        z = z * dsig
        delta = F.linear(z, self.U_k.to(x.dtype))    # z U_k^T -> [*, d_out]
        return base + delta

    def trainable_param_count(self) -> int:
        return self.num_params

    def extra_repr(self) -> str:
        return f"d_in={self.d_in}, d_out={self.d_out}, k={self.k}, alpha={self.alpha}"
