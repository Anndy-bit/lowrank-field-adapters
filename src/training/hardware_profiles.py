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
    """Four GPU classes targeted by the S³ method."""

    LOW = "low"          # 2-4 GB VRAM: GTX 1050, 1050 Ti, MX550
    MEDIUM = "medium"    # 8-16 GB VRAM: RTX 3060, 4060, 4070, 5090 Ti
    HIGH = "high"        # 32+ GB VRAM: RTX 4090, A6000, A100, H100
    EXTREME = "extreme"  # Multi-GPU / 70B+ models: cuantización 4-bit + FSDP


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
        # FASE 1: Memory management for 2GB VRAM / 10GB RAM
        svd_dir="./svd_factors/",
        svd_storage="mmap",              # Lazy load SVD per layer from disk
        base_model_offload=True,         # Accelerate offload to CPU + disk
        base_model_offload_dir="/tmp/s3_offload",
        dataset_streaming=True,          # 0 RAM for data
        cpu_ram_budget_gb=10.0,          # Hard limit CPU RAM
        model_quantization="none",
        # S3-OPT: ALL enabled for MAX VRAM savings on 2GB
        use_smv=True,
        use_emp=True,
        use_ter=True,
        use_fdgd=True,
        use_gns=True,
        use_sma=True,
        use_tows=True,
        use_dra=True,
        use_mso=True,
        use_hfisc=True,
        use_sgc=True,
        use_nfr=True,
        use_sbs=True,
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
        # FASE 1: Memory management for 8-16GB VRAM
        svd_dir="./svd_factors/",
        svd_storage="mmap",
        base_model_offload=False,        # Full model fits in VRAM
        base_model_offload_dir="/tmp/s3_offload",
        dataset_streaming=False,         # RAM OK for data
        cpu_ram_budget_gb=24.0,
        model_quantization="none",
        # S3-OPT: ALL enabled — VRAM allows full compute mode
        use_smv=True,
        use_emp=True,
        use_ter=True,
        use_fdgd=True,
        use_gns=True,
        use_sma=True,
        use_tows=True,
        use_dra=True,
        use_mso=True,
        use_hfisc=True,
        use_sgc=True,
        use_nfr=True,
        use_sbs=True,
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
        # FASE 1: Memory management for 32GB+ VRAM
        svd_dir="./svd_factors/",
        svd_storage="ram",               # Keep all SVD in RAM (fastest)
        base_model_offload=False,
        base_model_offload_dir="/tmp/s3_offload",
        dataset_streaming=False,
        cpu_ram_budget_gb=64.0,
        model_quantization="none",
        # S3-OPT: compute speedup only — disable memory optimizations
        use_smv=False,
        use_emp=True,
        use_ter=True,
        use_fdgd=False,
        use_gns=False,
        use_sma=False,
        use_tows=True,
        use_dra=False,
        use_mso=True,
        use_hfisc=True,
        use_sgc=False,
        use_nfr=False,
        use_sbs=True,
        sbs_p_stb=0.3,
    )


def _extreme_config() -> FrugalConfig:
    """70B+ models — cuantización 4-bit NF4 + offload + gradient checkpointing.

    VRAM budget: configurable (típicamente 8-24GB para 70B en 4-bit).
    Modelo base en NF4 (~35GB), SVD factors en mmap, todas las optimizaciones S3-OPT.
    Compatible con multi-GPU FSDP si hay varias GPUs disponibles.
    """
    return FrugalConfig(
        micro_batch_size=1,
        gradient_accumulation_steps=64,
        learning_rate=5e-4,
        weight_decay=0.01,
        betas=(0.9, 0.999),
        max_epochs=3,
        warmup_ratio=0.03,
        max_grad_norm=1.0,
        log_interval=5,
        use_amp=True,
        amp_dtype=torch.bfloat16,
        layer_swap=True,
        token_by_token=True,
        vram_budget_mb=8192,
        svd_dir="./svd_factors/",
        svd_storage="mmap",
        base_model_offload=True,
        base_model_offload_dir="/tmp/s3_offload_70b",
        dataset_streaming=True,
        cpu_ram_budget_gb=32.0,
        model_quantization="4bit",
        quant_compute_dtype=torch.bfloat16,
        quant_double_quant=True,
        quant_quant_type="nf4",
        fsdp_enabled=False,
        fsdp_world_size=1,
        cpu_offload_params=True,
        cpu_offload_optims=True,
        use_gradient_checkpointing=True,
        use_smv=True,
        use_emp=True,
        use_ter=True,
        use_fdgd=True,
        use_gns=True,
        use_sma=True,
        use_tows=True,
        use_dra=True,
        use_mso=True,
        use_hfisc=True,
        use_sgc=True,
        use_nfr=True,
        use_sbs=True,
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
    HardwareTier.EXTREME: HardwareProfile(
        tier=HardwareTier.EXTREME,
        config=_extreme_config(),
        layer_swap=True,
        token_by_token=True,
        vram_budget_mb=8192,
        description="70B+ models: 4-bit NF4 quantization + CPU offload + FSDP",
    ),
}


def get_profile(tier: HardwareTier) -> HardwareProfile:
    """Return the `HardwareProfile` preset associated with `tier`."""
    return _PROFILES[tier]
