"""
Singular Value Modulation Operator (SVMO)

Pointwise nonlinear modulation of singular values of frozen weight matrices.
W = U Σ V^T (frozen U,V) → W_adapted = U · m_θ(Σ) · V^T

Key properties:
- U and V completely frozen, only θ of m_θ receives gradients
- m_θ(σ) = σ · (1 + α · tanh(g_θ(log(σ + ε))))
- Bounded modulation: m_θ(σ) ∈ [σ(1-α), σ(1+α)]
- Identity initialization: m_θ(σ) ≈ σ at start
- O(H²) trainable params per matrix, independent of model dimension
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math


class ModulationMLP(nn.Module):
    """Tiny MLP g_θ: scalar → scalar with bounded output via tanh.

    Architecture: input → GELU → H → GELU → H → output (scalar)
    Total params: ~H² + 3H + 1
    """

    def __init__(self, hidden_dim: int = 32):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.fc1 = nn.Linear(1, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
        self._init_near_zero()

    def _init_near_zero(self):
        for m in [self.fc1, self.fc2]:
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            nn.init.zeros_(m.bias)
        nn.init.normal_(self.fc3.weight, mean=0.0, std=1e-5)
        nn.init.zeros_(self.fc3.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.gelu(self.fc1(x))
        x = F.gelu(self.fc2(x))
        x = self.fc3(x)
        return x


class SVMOAdapter(nn.Module):
    """SVMO applied to one weight matrix (e.g. W_Q, W_K, W_V, W_O, etc.).

    Stores frozen SVD factors U_k, Σ_k, V_k and a small learnable modulation
    MLP m_θ. Replaces a standard nn.Linear forward pass.

    Args:
        weight: frozen weight matrix W ∈ R^(d_out × d_in)
        k: number of singular components to keep (SVD rank)
        hidden_dim: hidden dimension of modulation MLP g_θ
        alpha: maximum fractional modulation amplitude ∈ (0, 1]
        bias: whether original layer had bias (pass through unchanged)
    """

    def __init__(
        self,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
        k: int = 128,
        hidden_dim: int = 32,
        alpha: float = 0.3,
    ):
        super().__init__()
        d_out, d_in = weight.shape
        r = min(d_out, d_in)
        k = min(k, r)

        self.d_out = d_out
        self.d_in = d_in
        self.k = k
        self.alpha = alpha

        U_k, S_k, Vt_k = self._compute_svd(weight, k)
        self.register_buffer("U_k", U_k)
        self.register_buffer("S_k", S_k)
        self.register_buffer("Vt_k", Vt_k)
        self.S_k_log = torch.log(S_k + 1e-8)

        self.modulation = ModulationMLP(hidden_dim)
        self.num_params = sum(p.numel() for p in self.modulation.parameters())

        self.has_bias = bias is not None
        if self.has_bias:
            self.register_buffer("bias", bias.data.clone())
        else:
            self.bias = None

    def _compute_svd(self, weight: torch.Tensor, k: int):
        weight_cpu = weight.detach().float().cpu()
        U_full, S_full, Vt_full = torch.linalg.svd(weight_cpu, full_matrices=False)
        U_k = U_full[:, :k].clone()
        S_k = S_full[:k].clone()
        Vt_k = Vt_full[:k, :].clone()
        return U_k, S_k, Vt_k

    def _modulate(self) -> torch.Tensor:
        """Apply m_θ to Σ_k: returns S_mod ∈ R^k."""
        g_out = self.modulation(self.S_k_log.unsqueeze(-1)).squeeze(-1)
        factor = 1.0 + self.alpha * torch.tanh(g_out)
        return self.S_k * factor

    def _reconstruct_weight(self) -> torch.Tensor:
        """W_adapted = U_k · diag(m_θ(S_k_log)) · Vt_k"""
        S_mod = self._modulate()
        W = (self.U_k * S_mod.unsqueeze(0)) @ self.Vt_k
        return W

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply adapted linear transformation.

        Efficient two-step forward using U_k, V_k decomposition:
        z = x · V_k^T  (right singular projection)
        z_mod = z ⊙ m_θ(Σ_k) (pointwise modulation)
        y = z_mod · U_k^T (left singular projection)

        Memory: stores z ∈ R^(B×k) instead of full weight.

        Args:
            x: input (batch_size, d_in) or (*, d_in)

        Returns:
            output: (batch_size, d_out) or (*, d_out)
        """
        S_mod = self._modulate()
        orig_shape = x.shape
        x_flat = x.reshape(-1, self.d_in)
        z = x_flat @ self.Vt_k.t()
        z_mod = z * S_mod.unsqueeze(0)
        y = z_mod @ self.U_k.t()
        y = y.reshape(orig_shape[:-1] + (self.d_out,))
        if self.has_bias:
            y = y + self.bias
        return y

    def forward_full(self, x: torch.Tensor) -> torch.Tensor:
        """Forward using reconstructed weight (for verification). Same output."""
        W = self._reconstruct_weight()
        return F.linear(x, W, self.bias)

    @torch.no_grad()
    def effective_rank(self) -> float:
        S_mod = self._modulate()
        S_mod_norm = S_mod / S_mod.sum()
        entropy = -(S_mod_norm * torch.log(S_mod_norm + 1e-8)).sum()
        max_entropy = math.log(self.k)
        return torch.exp(entropy).item()

    @torch.no_grad()
    def modulation_profile(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (original singular values, modulated singular values)."""
        S_mod = self._modulate()
        return self.S_k.clone(), S_mod.clone()

    def trainable_param_count(self) -> int:
        return self.num_params

    def frozen_param_count(self) -> int:
        return self.U_k.numel() + self.S_k.numel() + self.Vt_k.numel() + (
            self.bias.numel() if self.has_bias else 0
        )

    def vram_estimate_mb(self, dtype_size: int = 2) -> float:
        bytes_total = (
            self.U_k.numel() * dtype_size
            + self.S_k.numel() * dtype_size
            + self.Vt_k.numel() * dtype_size
            + self.num_params * dtype_size * 2
        )
        return bytes_total / (1024 * 1024)

    def extra_repr(self) -> str:
        return (
            f"d_in={self.d_in}, d_out={self.d_out}, k={self.k}, "
            f"alpha={self.alpha}, trainable={self.num_params}, "
            f"VRAM={self.vram_estimate_mb():.1f}MB"
        )


def randomized_svd(
    weight: torch.Tensor,
    k: int,
    n_oversamples: int = 10,
    n_iter: int = 2,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Randomized SVD (Halko, Martinsson, Tropp 2011).

    For large matrices (d > 1024), randomized SVD is ~50× faster than
    torch.linalg.svd. Only computes top-k components.

    Args:
        weight: (d_out, d_in) matrix
        k: target rank
        n_oversamples: extra random samples for accuracy (typically 5-10)
        n_iter: power iterations for accuracy (typically 1-2)

    Returns:
        U_k: (d_out, k), S_k: (k,), Vt_k: (k, d_in)
    """
    d_out, d_in = weight.shape
    k_target = min(k + n_oversamples, min(d_out, d_in))

    weight_cpu = weight.detach().float().cpu()

    omega = torch.randn(d_in, k_target, device=weight_cpu.device)
    Y = weight_cpu @ omega

    for _ in range(n_iter):
        Q, _ = torch.linalg.qr(Y)
        Y = weight_cpu.t() @ Q
        Q, _ = torch.linalg.qr(Y)
        Y = weight_cpu @ Q

    Q, _ = torch.linalg.qr(Y)
    B = Q.t() @ weight_cpu
    Ub, Sb, Vtb = torch.linalg.svd(B, full_matrices=False)

    U_k = (Q @ Ub)[:, :k]
    S_k = Sb[:k]
    Vt_k = Vtb[:k, :]
    return U_k, S_k, Vt_k


def create_svmo_from_linear(
    layer: nn.Linear,
    k: int = 128,
    hidden_dim: int = 32,
    alpha: float = 0.3,
    use_randomized_svd: bool = False,
) -> SVMOAdapter:
    """Convenience: create SVMOAdapter from an existing nn.Linear.

    Args:
        layer: nn.Linear with frozen weights
        k: SVD rank
        hidden_dim: hidden dim of modulation MLP
        alpha: modulation amplitude
        use_randomized_svd: if True, use randomized SVD (faster for d>1024)

    Returns:
        SVMOAdapter configured with layer's weights and bias.
    """
    weight = layer.weight.data.detach().float()
    bias = layer.bias.data.detach().float() if layer.bias is not None else None

    if use_randomized_svd:
        svmo = SVMOAdapter.__new__(SVMOAdapter)
        nn.Module.__init__(svmo)
        d_out, d_in = weight.shape
        r = min(d_out, d_in)
        k_eff = min(k, r)
        svmo.d_out = d_out
        svmo.d_in = d_in
        svmo.k = k_eff
        svmo.alpha = alpha
        U_k, S_k, Vt_k = randomized_svd(weight, k_eff)
        svmo.register_buffer("U_k", U_k)
        svmo.register_buffer("S_k", S_k)
        svmo.register_buffer("Vt_k", Vt_k)
        svmo.S_k_log = torch.log(S_k + 1e-8)
        svmo.modulation = ModulationMLP(hidden_dim)
        svmo.num_params = sum(p.numel() for p in svmo.modulation.parameters())
        svmo.has_bias = bias is not None
        if svmo.has_bias:
            svmo.register_buffer("bias", bias.clone())
        else:
            svmo.bias = None
        return svmo
    return SVMOAdapter(weight, bias, k, hidden_dim, alpha)