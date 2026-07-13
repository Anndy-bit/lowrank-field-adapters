"""
Spectral Transport Bridge (STB)

Bidirectional coupling between SVMO (spectral weight modulation) and NMF
(latent manifold flow). Transports latent representations through the
spectral basis U_k of frozen weight matrices.

Operation: h → U_k^T·h = s (spectral signature)
          s̃ = c_φ(s, Σ_k) (cross-attention coupling)
          h_STB = U_k · s̃ (back to latent space)

Key properties:
- Cross-attention between spectral signature and current singular values
- Creates information channel between weight-space and representation-space adaptations
- 3·k·(k/4) ≈ 12K params per instance (for k=128)
- Applied at input to each transformer layer (spectral preconditioning)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class SpectralCoupling(nn.Module):
    """Cross-attention coupling function: c_φ(s, σ_log).

    s ∈ R^k (spectral signature of latent representation h)
    σ_log ∈ R^k (log of current modulated singular values from SVMO)

    Architecture:
    Q = W_q · s  (query: what does the representation look like spectrally?)
    K = W_k · σ_log  (key: what is the current singular value structure?)
    V = W_v · s  (value: representation content to be transformed)
    attention = softmax(Q·K^T / √k_h)
    output = s + β · (attention · V)

    Args:
        k: spectral dimension (equal to SVMO's k)
        head_dim: dimension of Q,K,V projections (k/4 by default)
        beta: coupling strength ∈ [0, 1]
    """

    def __init__(
        self,
        k: int,
        head_dim: Optional[int] = None,
        beta: float = 0.5,
    ):
        super().__init__()
        self.k = k
        self.head_dim = head_dim or max(k // 4, 8)
        self.beta = beta
        self.scale = self.head_dim ** -0.5

        self.W_q = nn.Linear(k, self.head_dim, bias=False)
        self.W_k = nn.Linear(k, self.head_dim, bias=False)
        self.W_v = nn.Linear(k, self.head_dim, bias=False)
        self.W_o = nn.Linear(self.head_dim, k, bias=False)

        self._init_weights()
        self.num_params = sum(p.numel() for p in self.parameters())

    def _init_weights(self):
        for m in [self.W_q, self.W_k]:
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.W_v.weight, mean=0.0, std=1e-5)
        nn.init.normal_(self.W_o.weight, mean=0.0, std=1e-5)

    def forward(
        self,
        s: torch.Tensor,
        sigma_log: torch.Tensor,
    ) -> torch.Tensor:
        # s: [B, ..., k] or [..., k]
        # sigma_log: [k] - broadcast to match s shape
        Q = self.W_q(s)
        # Expand sigma_log to match s's batch/seq dimensions
        sigma_log_expanded = sigma_log.expand(s.shape[:-1] + (self.k,))
        K = self.W_k(sigma_log_expanded)
        V = self.W_v(s)

        attn = (Q @ K.transpose(-2, -1)) * self.scale
        attn_weights = F.softmax(attn, dim=-1)
        delta = self.W_o(attn_weights @ V)

        return s + self.beta * delta


class STBBridge(nn.Module):
    """Full STB operator applied at input to a transformer layer.

    h → U_k^T·h (spectral projection) → c_φ(s, Σ_log) → U_k·s̃ → h_STB

    Args:
        U_k: frozen left singular vectors from SVMO (d_out × k)
        k: spectral dimension
        head_dim: cross-attention head dim
        beta: coupling strength
    """

    def __init__(
        self,
        U_k: torch.Tensor,
        k: int,
        head_dim: Optional[int] = None,
        beta: float = 0.5,
    ):
        super().__init__()
        self.k = k
        self.beta = beta
        self.register_buffer("U_k", U_k)
        self.coupling = SpectralCoupling(k, head_dim, beta)
        self.num_params = self.coupling.num_params

    def forward(
        self,
        h: torch.Tensor,
        sigma_log: torch.Tensor,
    ) -> torch.Tensor:
        # Ensure U_k matches input dtype
        U_k = self.U_k.to(h.dtype)
        s = h @ U_k
        s_tilde = self.coupling(s, sigma_log)
        h_stb = s_tilde @ U_k.t()
        return h_stb

    def spectral_signature(self, h: torch.Tensor) -> torch.Tensor:
        """Return s = U_k^T·h for analysis/debugging."""
        with torch.no_grad():
            return h @ self.U_k

    @torch.no_grad()
    def attention_weights(
        self, h: torch.Tensor, sigma_log: torch.Tensor
    ) -> torch.Tensor:
        s = h @ self.U_k
        Q = self.coupling.W_q(s)
        K = self.coupling.W_k(sigma_log).unsqueeze(0)
        attn = (Q @ K.transpose(-2, -1)) * self.coupling.scale
        return F.softmax(attn, dim=-1).squeeze(-1)

    def trainable_param_count(self) -> int:
        return self.num_params

    def frozen_param_count(self) -> int:
        return self.U_k.numel()

    def vram_estimate_mb(self, dtype_size: int = 2) -> float:
        bytes_total = (
            self.U_k.numel() * dtype_size
            + self.num_params * dtype_size * 2
        )
        return bytes_total / (1024 * 1024)

    def extra_repr(self) -> str:
        return (
            f"k={self.k}, head_dim={self.coupling.head_dim}, "
            f"beta={self.beta}, params={self.num_params}"
        )

    @torch.no_grad()
    def set_k(self, new_k: int):
        """Dynamically change spectral dimension k (for DRA).
        
        Truncates U_k and updates coupling layer dimensions.
        Only supports reducing k.
        """
        new_k = min(new_k, self.U_k.shape[1])
        if new_k == self.k:
            return
        if new_k > self.k:
            raise ValueError(f"Cannot increase k from {self.k} to {new_k} without full SVD")
        
        self.k = new_k
        self.U_k = self.U_k[:, :new_k].contiguous()
        
        # Update coupling layer dimensions
        old_head_dim = self.coupling.head_dim
        self.coupling.head_dim = max(new_k // 4, 8)
        self.coupling.scale = self.coupling.head_dim ** -0.5
        
        # Recreate linear layers with new dimensions
        device = self.U_k.device
        dtype = self.U_k.dtype
        self.coupling.W_q = nn.Linear(new_k, self.coupling.head_dim, bias=False).to(device, dtype)
        self.coupling.W_k = nn.Linear(new_k, self.coupling.head_dim, bias=False).to(device, dtype)
        self.coupling.W_v = nn.Linear(new_k, self.coupling.head_dim, bias=False).to(device, dtype)
        self.coupling.W_o = nn.Linear(self.coupling.head_dim, new_k, bias=False).to(device, dtype)
        
        self.coupling._init_weights()
        self.coupling.num_params = sum(p.numel() for p in self.coupling.parameters())
        self.num_params = self.coupling.num_params
        
        print(f"  [DRA] STB rank adapted: k={self.k} -> {new_k}")


def create_stb_bridge_from_svmo(
    svmo_adapter,
    head_dim: Optional[int] = None,
    beta: float = 0.5,
) -> STBBridge:
    """Create STBBridge using U_k and k from an existing SVMOAdapter.

    The STB uses the FIRST U_k from the SVMO adapters in the layer
    (typically W_Q's U_k, since it's the first projection the input sees).

    Args:
        svmo_adapter: SVMOAdapter instance (provides U_k, k)
        head_dim: cross-attention head dimension
        beta: coupling strength

    Returns:
        STBBridge configured for the layer
    """
    k = svmo_adapter.k
    U_k = svmo_adapter.U_k.detach().clone()
    return STBBridge(U_k, k, head_dim, beta)