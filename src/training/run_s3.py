"""Thin entry-point for the S^3 Frugal Trainer.

This script presents the interactive menu to the user and prints the
resolved configuration. The actual model + dataset loading, S^3 layer
construction and `FrugalTrainer` instantiation happen in
`experiments/run_training.py`; here we only resolve and display the
configuration so the user can sanity-check it before kicking off a full run.

Usage:
    python src/training/run_s3.py
    python src/training/run_s3.py --tier medium --model Qwen/Qwen2.5-7B-Instruct
"""

import sys
import os

if __name__ == "__main__":
    _repo_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    if _repo_root not in sys.path:
        sys.path.insert(0, _repo_root)

from src.training.hardware_profiles import HardwareTier, get_profile
from src.training.launcher import interactive_menu, launch_from_args


def main(argv=None) -> int:
    if argv is None:
        argv = sys.argv[1:]

    if argv:
        profile, model_name, dataset_path = launch_from_args(argv)
    else:
        profile, model_name, dataset_path = interactive_menu()

    cfg = profile.config
    sep = "=" * 60
    print(sep)
    print("S^3 Configuration — Resolved")
    print(sep)
    print(f"  tier          : {profile.tier.value}")
    print(f"  model         : {model_name}")
    print(f"  dataset       : {dataset_path}")
    print(f"  layer_swap    : {profile.layer_swap}")
    print(f"  token_by_token: {profile.token_by_token}")
    print(f"  micro_batch   : {cfg.micro_batch_size}")
    print(f"  grad_accum    : {cfg.gradient_accumulation_steps}")
    print(f"  amp_dtype     : {cfg.amp_dtype}")
    print(f"  VRAM budget   : {profile.vram_budget_mb} MB")
    print(sep)
    print(
        "(stub) The HuggingFace model + dataset loading, S^3 layer "
        "construction and FrugalTrainer instantiation live in "
        "experiments/run_training.py. Exiting cleanly."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
