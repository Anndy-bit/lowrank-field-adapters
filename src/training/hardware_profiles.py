"""Hardware profiles for the S³ Frugal Trainer.

Defines three predefined tiers (LOW / MEDIUM / HIGH) covering the typical
GPU classes targeted by the S³ method, plus an `HardwareProfile` dataclass
that bundles a fully populated `FrugalConfig` with the extra knobs that the
trainer needs to adapt its execution path to the chosen GPU class.

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
    MEDIUM = "medium"  # 8-16 GB VRAM: RTX 3060, 4060, 4070
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
    )


def _medium_config() -> FrugalConfig:
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
    )


def _high_config() -> FrugalConfig:
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
