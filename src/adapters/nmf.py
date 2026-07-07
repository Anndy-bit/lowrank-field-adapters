"""
Neural Manifold Flow (NMF)

Continuous deformation of latent representations via Neural ODE.
dh(t)/dt = f_θ(h(t), t), h(0) = h → h_adapted = h(T)

Key properties:
- First application of Neural ODEs to LLM parameter-efficient fine-tuning
- Bottleneck MLP velocity field f_θ: d → d_bottleneck → d
- Fixed-step RK4 integrator (N ∈ {2, 4, 8} steps)
- Identity initialization: f_θ ≈ 0 → h(T) ≈ h(0)
- O(d · d_bottleneck) params per instance (~65K for d=4096, d_bottleneck=8)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Callable
import math


class VelocityField(nn.Module):
    """Bottleneck MLP velocity field: f_θ(h, t) = W_out · tanh(W_in · [h; t]).

    Parameters: 2 · d · d_bottleneck
    For d=4096, d_bottleneck=8: ~65K params.
    """

    def __init__(self, dim: int, bottleneck_dim: int = 8):
        super().__init__()
        self.dim = dim
        self.bottleneck_dim = bottleneck_dim

        self.W_in = nn.Linear(dim + 1, bottleneck_dim, bias=False)
        self.W_out = nn.Linear(bottleneck_dim, dim, bias=False)
        self._init_near_zero()

    def _init_near_zero(self):
        nn.init.normal_(self.W_in.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.W_out.weight, mean=0.0, std=1e-5)

    def forward(self, h: torch.Tensor, t: float) -> torch.Tensor:
        t_tensor = torch.full(
            (h.shape[0], 1),
            t,
            device=h.device,
            dtype=h.dtype,
        )
        h_t = torch.cat([h, t_tensor], dim=-1)
        x = torch.tanh(self.W_in(h_t))
        x = self.W_out(x)
        return x


class NMFFlow(nn.Module):
    """NMF applied at one point in the transformer (post-attention or post-MLP).

    Solves dh/dt = f_θ(h, t) from t=0 to t=T using fixed-step RK4.

    Args:
        dim: latent dimension d (e.g. 4096)
        bottleneck_dim: hidden dim of velocity field
        T: total integration time
        N_steps: number of RK4 integration steps
        solver: 'rk4' (default) or 'euler'
    """

    def __init__(
        self,
        dim: int,
        bottleneck_dim: int = 8,
        T: float = 1.0,
        N_steps: int = 4,
        solver: str = "rk4",
    ):
        super().__init__()
        self.dim = dim
        self.T = T
        self.N_steps = N_steps
        self.solver = solver
        self.dt = T / N_steps
        self.field = VelocityField(dim, bottleneck_dim)
        self.num_params = sum(p.numel() for p in self.field.parameters())

    def _rk4_step(
        self, h: torch.Tensor, t: float, dt: float
    ) -> torch.Tensor:
        k1 = self.field(h, t)
        k2 = self.field(h + dt / 2 * k1, t + dt / 2)
        k3 = self.field(h + dt / 2 * k2, t + dt / 2)
        k4 = self.field(h + dt * k3, t + dt)
        return h + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

    def _euler_step(
        self, h: torch.Tensor, t: float, dt: float
    ) -> torch.Tensor:
        return h + dt * self.field(h, t)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        if self.solver == "rk4":
            step_fn = self._rk4_step
        elif self.solver == "euler":
            step_fn = self._euler_step
        else:
            raise ValueError(f"Unknown solver: {self.solver}")

        h_current = h
        t_current = 0.0
        for _ in range(self.N_steps):
            h_current = step_fn(h_current, t_current, self.dt)
            t_current += self.dt
        return h_current

    def deformation(self, h: torch.Tensor) -> torch.Tensor:
        """Return Δh = h_adapted - h_original."""
        h_adapted = self.forward(h)
        return h_adapted - h

    @torch.no_grad()
    def deformation_norm(self, h: torch.Tensor) -> float:
        """Average L2 norm of deformation."""
        return self.deformation(h).norm(dim=-1).mean().item()

    def trainable_param_count(self) -> int:
        return self.num_params

    def vram_estimate_mb(self, dtype_size: int = 2) -> float:
        field_bytes = self.num_params * dtype_size * 2
        ode_bytes = self.N_steps * self.dim * dtype_size
        bytes_total = field_bytes + ode_bytes
        return bytes_total / (1024 * 1024)

    def extra_repr(self) -> str:
        return (
            f"dim={self.dim}, bottleneck={self.field.bottleneck_dim}, "
            f"T={self.T}, N={self.N_steps}, solver={self.solver}, "
            f"params={self.num_params}"
        )


class NMFAdjoint(nn.Module):
    """NMF with adjoint sensitivity method for backward pass.

    Uses O(1) memory regardless of N_steps (no intermediate state storage).
    More computationally intensive per backward step.

    Args:
        Same as NMFFlow.
    """

    def __init__(
        self,
        dim: int,
        bottleneck_dim: int = 8,
        T: float = 1.0,
        N_steps: int = 4,
    ):
        super().__init__()
        self.dim = dim
        self.T = T
        self.N_steps = N_steps
        self.dt = T / N_steps
        self.field = VelocityField(dim, bottleneck_dim)
        self.num_params = sum(p.numel() for p in self.field.parameters())

    def _rk4_step_no_grad(self, h: torch.Tensor, t: float) -> torch.Tensor:
        """RK4 step without building autograd graph (used for adjoint replay)."""
        with torch.no_grad():
            k1 = self.field(h, t)
            k2 = self.field(h + self.dt / 2 * k1, t + self.dt / 2)
            k3 = self.field(h + self.dt / 2 * k2, t + self.dt / 2)
            k4 = self.field(h + self.dt * k3, t + self.dt)
            return h + self.dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return NMFFlow.forward.__wrapped__(self, h)

    def extra_repr(self) -> str:
        return f"dim={self.dim}, adjoint=True, T={self.T}, N={self.N_steps}"


def create_nmf_pair(
    dim: int,
    bottleneck_dim: int = 8,
    T: float = 1.0,
    N_steps: int = 4,
    solver: str = "rk4",
) -> tuple[NMFFlow, NMFFlow]:
    """Create the two NMF instances per transformer layer (post-attn + post-MLP)."""
    return (
        NMFFlow(dim, bottleneck_dim, T, N_steps, solver),
        NMFFlow(dim, bottleneck_dim, T, N_steps, solver),
    )