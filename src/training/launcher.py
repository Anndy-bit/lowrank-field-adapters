"""Interactive / CLI launcher for the S³ Frugal Trainer.

Presents the user with a hardware-tier selector and a HuggingFace model menu,
then resolves the chosen `HardwareProfile` plus the dataset path. The actual
model + dataset loading and `FrugalTrainer` instantiation happens in
`experiments/run_training.py`; this module only resolves the configuration.
"""

import argparse
import sys
from typing import List, Optional, Tuple

from .hardware_profiles import HardwareProfile, HardwareTier, get_profile


BANNER = r"""
========================================================================
   ____   ____   ___  ____   ____   ___   ____   ___   ____   __  __
  / __/  / __/  / _/ / __/  / __/  / _/  / __/  / _/  / __/  / / / /
 / /    / /    / /  / /    / /    / /   / /    / /   / /    / /_/ /
/__/   /__/   /__/  /__/   /__/   /__/  /__/   /__/  /__/    \____/

        S^3 Frugal Trainer -- Hardware Selector
========================================================================
"""

# Numbered model presets offered by the interactive menu.
MODEL_PRESETS: List[Tuple[str, str]] = [
    ("Qwen2.5-7B-Instruct", "Qwen/Qwen2.5-7B-Instruct"),
    ("Mistral-7B-v0.3",     "mistralai/Mistral-7B-v0.3"),
    ("Llama-3-8B",          "meta-llama/Meta-Llama-3-8B"),
]

DEFAULT_DATASET = "data/alpaca_52k.jsonl"

TIER_MENU: List[Tuple[str, HardwareTier, str]] = [
    ("1", HardwareTier.LOW,    "Low tier   (2-4 GB VRAM:  GTX 1050, 1050 Ti, MX550)"),
    ("2", HardwareTier.MEDIUM, "Medium tier (8-16 GB VRAM: RTX 3060, 4060, 4070)"),
    ("3", HardwareTier.HIGH,    "High tier   (32+ GB VRAM:  RTX 4090, A6000, A100, H100)"),
]


def _print_banner() -> None:
    print(BANNER)


def _prompt_tier() -> HardwareTier:
    """Loop on stdin until a valid tier is selected."""
    while True:
        print("\nSelect your hardware tier:")
        for key, tier, desc in TIER_MENU:
            print(f"  {key}. {desc}")
        choice = input("> ").strip()
        for key, tier, _ in TIER_MENU:
            if choice == key or choice.lower() == tier.value:
                print(f"  -> Selected tier: {tier.value}")
                return tier
        print("  ! Invalid choice. Please enter 1, 2 or 3.")


def _prompt_model() -> str:
    """Loop on stdin until a valid HuggingFace model id is selected."""
    # Lazily probe whether `transformers` is available; we use it only to
    # hint the user, never as a hard requirement for selecting a model id.
    transformers_available = False
    try:
        import transformers  # noqa: F401
        transformers_available = True
    except Exception:
        transformers_available = False

    while True:
        print("\nSelect a HuggingFace model to fine-tune:")
        for idx, (label, _) in enumerate(MODEL_PRESETS, start=1):
            print(f"  {idx}. {label}")
        print(f"  {len(MODEL_PRESETS) + 1}. Custom (type any HuggingFace model id)")
        if not transformers_available:
            print("  (note: `transformers` library not detected; "
                  "presets will still resolve to a HF id.)")
        choice = input("> ").strip()

        if choice.isdigit():
            n = int(choice)
            if 1 <= n <= len(MODEL_PRESETS):
                model_id = MODEL_PRESETS[n - 1][1]
                print(f"  -> Selected model: {model_id}")
                return model_id
            elif n == len(MODEL_PRESETS) + 1:
                custom = input("  Enter HuggingFace model id: ").strip()
                if custom:
                    print(f"  -> Selected model: {custom}")
                    return custom
        print("  ! Invalid choice. Try again.")


def _prompt_dataset() -> str:
    """Read the dataset path with the default already prefilled."""
    print(f"\nDataset path (press Enter for default: {DEFAULT_DATASET}):")
    choice = input("> ").strip()
    if not choice:
        choice = DEFAULT_DATASET
    print(f"  -> Dataset: {choice}")
    return choice


def _print_summary(
    profile: HardwareProfile, model_name: str, dataset_path: str
) -> None:
    cfg = profile.config
    sep = "=" * 70
    print("\n" + sep)
    print("Resolved S^3 configuration")
    print(sep)
    print(f"  Hardware tier          : {profile.tier.value}")
    print(f"  Description            : {profile.description}")
    print(f"  VRAM budget            : {profile.vram_budget_mb} MB")
    print(f"  layer_swap             : {profile.layer_swap}")
    print(f"  token_by_token         : {profile.token_by_token}")
    print("-" * 70)
    print(f"  Model (HuggingFace id) : {model_name}")
    print(f"  Dataset path           : {dataset_path}")
    print("-" * 70)
    print("FrugalConfig:")
    print(f"  micro_batch_size       : {cfg.micro_batch_size}")
    print(f"  gradient_accumulation  : {cfg.gradient_accumulation_steps}")
    print(f"  learning_rate          : {cfg.learning_rate}")
    print(f"  weight_decay           : {cfg.weight_decay}")
    print(f"  betas                  : {cfg.betas}")
    print(f"  max_epochs             : {cfg.max_epochs}")
    print(f"  warmup_ratio           : {cfg.warmup_ratio}")
    print(f"  max_grad_norm          : {cfg.max_grad_norm}")
    print(f"  log_interval           : {cfg.log_interval}")
    print(f"  use_amp                : {cfg.use_amp}")
    print(f"  amp_dtype              : {cfg.amp_dtype}")
    print(sep + "\n")


def interactive_menu() -> Tuple[HardwareProfile, str, str]:
    """Interactive menu. Returns (profile, model_name, dataset_path)."""
    _print_banner()
    tier = _prompt_tier()
    profile = get_profile(tier)
    model_name = _prompt_model()
    dataset_path = _prompt_dataset()
    _print_summary(profile, model_name, dataset_path)
    return profile, model_name, dataset_path


def launch_from_args(argv: Optional[List[str]] = None) -> Tuple[HardwareProfile, str, str]:
    """CLI alternative to `interactive_menu`.

    Parses `--tier {low,medium,high}`, `--model HF_ID`, `--dataset PATH`.
    Falls back to the interactive menu if any required argument is missing.
    """
    parser = argparse.ArgumentParser(
        description="S^3 Frugal Trainer launcher",
    )
    parser.add_argument(
        "--tier", choices=("low", "medium", "high"), default=None,
        help="Hardware tier: low (2-4 GB), medium (8-16 GB), high (32+ GB).",
    )
    parser.add_argument(
        "--model", default=None,
        help="HuggingFace model id to fine-tune (e.g. Qwen/Qwen2.5-7B-Instruct).",
    )
    parser.add_argument(
        "--dataset", default=None,
        help=f"Dataset path (default: {DEFAULT_DATASET}).",
    )
    args = parser.parse_args(argv)

    missing = (
        args.tier is None
        or args.model is None
        or args.dataset is None
    )
    if missing:
        return interactive_menu()

    tier_map = {
        "low": HardwareTier.LOW,
        "medium": HardwareTier.MEDIUM,
        "high": HardwareTier.HIGH,
    }
    profile = get_profile(tier_map[args.tier])
    _print_summary(profile, args.model, args.dataset)
    return profile, args.model, args.dataset


if __name__ == "__main__":  # pragma: no cover
    launch_from_args()
