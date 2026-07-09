"""Hardware profiles for the S³ Frugal Trainer.

Defines three predefined tiers (LOW / MEDIUM / HIGH) covering the typical
GPU classes targeted by the S³ method, plus an `HardwareProfile` dataclass
that bundles a fully populated `FrugalConfig` with the extra knobs that the
trainer needs to adapt its execution path to the chosen GPU class.

S3-OPT flags are automatically configured per tier:
- LOW (GTX 1050 2GB): ALL optimizations enabled — VRAM is the bottleneck
- MEDIUM (RTX 3060-5090 Ti 8-16GB): ALL optimizations — balanced VRAM/speed
- HIGH (RTX 4090 32GB+): Core speedup only — VRAM is ample

Usage:
    from src.training.hardware_profiles import HardwareTier, get_profile
    profile = get_profile(HardwareTier.MEDIUM)
    trainer = FrugalTrainer(..., config=profile.config)
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict

import torch

from .frugal_trainer import FrugalConfig


class HardwareTier(Enum):
    """Three GPU classes targeted by the S³ method."""

    LOW = "low"        # 2-4 GB VRAM: GTX 1050, 1050 Ti, MX550
    MEDIUM = "medium"  # 8-16 GB VRAM: RTX 3060, 4060, 4070, 5090 Ti
    HIGH = "high"      # 32+ GB VRAM: RTX 4090, A6000, A100, H100


@dataclass
class HardwareProfile:
    """Container exposing a ready-to-use `FrugalConfig` plus the
    hardware-specific knobs that the trainer branches on."""

    tier: HardwareTier
    config: FrugalConfig
    layer_swap: bool
    token_by_token: bool
    vram_budget_mb: int
    description: str


def _low_config() -> FrugalConfig:
    """GTX 1050 / 2GB — ALL optimizations enabled for VRAM survival.

    VRAM budget: 2048 MB (layer_swap=True, token_by_token=True mandatory)
    All S3-OPT optimizations active because every byte matters.
    """
    return FrugalConfig(
        micro_batch_size=1,
        gradient_accumulation_steps=128,
        learning_rate=1e-3,
        weight_decay=0.01,
        betas=(0.9, 0.999),
        max_epochs=3,
        warmup_ratio=0.05,
        max_grad_norm=1.0,
        log_interval=10,
        use_amp=True,
        amp_dtype=torch.float16,
        layer_swap=True,
        token_by_token=True,
        vram_budget_mb=2048,
        # S3-OPT: ALL enabled for MAX VRAM savings on 2GB
        use_smv=True,          # PCIe vector delta-modulation (~32x bandwidth reduction)
        use_emp=True,          # MLP μ prediction (skip MLP forward when H predictable)
        use_ter=True,          # ODE N routing (1/2/4 steps vs always 4)
        use_fdgd=True,         # Fourier gradient filtering (stability)
        use_gns=True,          # Skip backward when grad < ε·max (skip ~20-30% layers)
        use_sma=True,          # Spectral checkpoint compression (4096→128 = 16x)
        use_tows=True,         # Token warmstart (reduces ODE init overhead)
        use_dra=True,          # Dynamic rank k (128→64 when s(t) low)
        use_mso=True,          # Shortcut detection (linear bypass when κ < κ_thresh)
        use_hfisc=True,        # Optimal init (√(2/d) for GELU)
        use_sgc=True,          # Spectral gradient compression (backward pass)
        use_nfr=True,          # NMF flow recycling (reuse intermediate states)
        use_sbs=True,          # STB bypass sampling (p_stb=0.3)
        sbs_p_stb=0.3,
    )


def _medium_config() -> FrugalConfig:
    """RTX 3060-5090 Ti / 8-16GB — ALL optimizations, balanced VRAM/speed.

    VRAM budget: 12288 MB (layer_swap=False, token_by_token=False)
    All optimizations active — more VRAM means we can do more per pass.
    """
    return FrugalConfig(
        micro_batch_size=8,
        gradient_accumulation_steps=16,
        learning_rate=1e-3,
        weight_decay=0.01,
        betas=(0.9, 0.999),
        max_epochs=3,
        warmup_ratio=0.05,
        max_grad_norm=1.0,
        log_interval=10,
        use_amp=True,
        amp_dtype=torch.bfloat16,
        layer_swap=False,
        token_by_token=False,
        vram_budget_mb=12288,
        # S3-OPT: ALL enabled — VRAM allows full compute mode
        use_smv=True,          # Bandwidth reduction still useful
        use_emp=True,          # Skip MLP when H predictable
        use_ter=True,          # ODE routing (1/2/4 steps)
        use_fdgd=True,         # Gradient stability filtering
        use_gns=True,          # Skip low-gradient backward passes
        use_sma=True,          # Compression for activation storage
        use_tows=True,         # Token warmstart
        use_dra=True,          # Rank adaptation
        use_mso=True,          # Shortcut detection
        use_hfisc=True,        # Optimal init
        use_sgc=True,          # Gradient compression
        use_nfr=True,          # Flow recycling
        use_sbs=True,          # STB bypass
        sbs_p_stb=0.3,
    )


def _high_config() -> FrugalConfig:
    """RTX 4090 / A100 / 32GB+ — compute speedup focus, VRAM is ample.

    VRAM budget: 40960 MB (layer_swap=False, token_by_token=False)
    Prioritize speedup over memory savings.
    """
    return FrugalConfig(
        micro_batch_size=32,
        gradient_accumulation_steps=4,
        learning_rate=1e-3,
        weight_decay=0.01,
        betas=(0.9, 0.999),
        max_epochs=3,
        warmup_ratio=0.05,
        max_grad_norm=1.0,
        log_interval=10,
        use_amp=True,
        amp_dtype=torch.bfloat16,
        layer_swap=False,
        token_by_token=False,
        vram_budget_mb=40960,
        # S3-OPT: compute speedup only — disable memory optimizations
        use_smv=False,         # No PCIe transfer bottleneck
        use_emp=True,          # Skip MLP when predictable (speedup)
        use_ter=True,          # ODE routing (1/2/4 steps)
        use_fdgd=False,        # No need for gradient filtering
        use_gns=False,         # Process all layers — VRAM is ample
        use_sma=False,         # No need to compress activations
        use_tows=True,         # Token warmstart (speedup)
        use_dra=False,         # Keep full rank (k=128)
        use_mso=True,          # Shortcut detection (speedup)
        use_hfisc=True,        # Optimal init (convergence)
        use_sgc=False,         # No gradient compression needed
        use_nfr=False,         # Full NMF flow (no recycling needed)
        use_sbs=True,          # STB bypass (speedup)
        sbs_p_stb=0.3,
    )


_PROFILES: Dict[HardwareTier, HardwareProfile] = {
    HardwareTier.LOW: HardwareProfile(
        tier=HardwareTier.LOW,
        config=_low_config(),
        layer_swap=True,
        token_by_token=True,
        vram_budget_mb=2048,
        description="2-4 GB VRAM: GTX 1050, 1050 Ti, MX550",
    ),
    HardwareTier.MEDIUM: HardwareProfile(
        tier=HardwareTier.MEDIUM,
        config=_medium_config(),
        layer_swap=False,
        token_by_token=False,
        vram_budget_mb=12288,
        description="8-16 GB VRAM: RTX 3060, 4060, 4070",
    ),
    HardwareTier.HIGH: HardwareProfile(
        tier=HardwareTier.HIGH,
        config=_high_config(),
        layer_swap=False,
        token_by_token=False,
        vram_budget_mb=40960,
        description="32+ GB VRAM: RTX 4090, A6000, A100, H100",
    ),
}


def get_profile(tier: HardwareTier) -> HardwareProfile:
    """Return the `HardwareProfile` preset associated with `tier`."""
    return _PROFILES[tier]
