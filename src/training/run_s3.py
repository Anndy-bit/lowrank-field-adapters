"""Thin entry-point for the S^3 Frugal Trainer.

This script presents the interactive menu to the user and prints the
resolved configuration. The actual model + dataset loading, S^3 layer
construction and `FrugalTrainer` instantiation happen in
`experiments/run_training.py` (which the user already has under
`experiments/`); here we only resolve and display the configuration so the
user can sanity-check it before kicking off a full run.

Usage:
    python -m src.training.run_s3
    python src/training/run_s3.py
"""

import sys

from .launcher import interactive_menu


def main() -> int:
    profile, model_name, dataset_path = interactive_menu()
    print("Selected: "
          f"profile={profile.tier.value}, "
          f"model={model_name}, "
          f"dataset={dataset_path}")
    print(
        "(stub) The actual HuggingFace model + dataset loading, S^3 layer "
        "construction and FrugalTrainer instantiation live in "
        "experiments/run_training.py. Exiting cleanly."
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
