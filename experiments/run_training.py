"""Experiments runner for the S³ Frugal Trainer — used by run_s3.py (interactive menu).

This module is the primary entry point when running via the interactive menu
(run_s3.py). It uses the HardwareProfile from the menu to configure everything,
including all S3-OPT optimizations.

Usage (internal, called by run_s3.py):
    from experiments.run_training import run_training
    run_training(profile, model_name, dataset_path)
"""

import torch
import os
import sys
import time
from typing import Optional, Callable

from src.training.hardware_profiles import HardwareProfile
from src.training.training_pipeline import build_s3_model, load_dataset
from src.training.training_monitor import TrainingMonitor


def run_training(
    profile: HardwareProfile,
    model_name: str,
    dataset_path: str,
    max_epochs: int = 3,
    seed: int = 42,
    checkpoint_dir: Optional[str] = None,
    eval_only: bool = False,
    checkpoint_path: Optional[str] = None,
    ablation: Optional[dict] = None,
    log_fn: Optional[Callable] = None,
    monitor: Optional[TrainingMonitor] = None,
    device: str = "cuda:0",
):
    """Run S³ training using the hardware profile configuration.

    This is the main training function called by run_s3.py (interactive menu).
    All S3-OPT flags come from the HardwareProfile — no YAML needed.

    Args:
        profile: HardwareProfile with tier-specific FrugalConfig + S3-OPT flags
        model_name: HuggingFace model id (e.g. "Qwen/Qwen2.5-7B-Instruct")
        dataset_path: path to dataset or HuggingFace dataset id
        max_epochs: number of training epochs (default: 3)
        seed: random seed (default: 42)
        checkpoint_dir: directory to save checkpoints
        eval_only: if True, skip training and only run benchmarks
        checkpoint_path: path to checkpoint for eval_only mode
        ablation: optional dict with svmo/stb/nmf bools
        log_fn: optional callback for logging
        monitor: optional TrainingMonitor instance
        device: GPU device string
    """
    torch.manual_seed(seed)

    print(f"[S³] Loading model: {model_name}")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        low_cpu_mem_usage=True,
    )
    for param in base_model.parameters():
        param.requires_grad = False

    svd_dir = f"./svd_factors/{model_name.split('/')[-1].lower()}/"

    config = profile.config
    # Use profile's max_epochs unless explicitly overridden
    config = _with_max_epochs(config, max_epochs)

    print(f"[S³] Building S³ model (tier={profile.tier.value})...")
    print(f"[S³] S3-OPT active flags:")
    print(f"     SMV={config.use_smv} EMP={config.use_emp} TER={config.use_ter}")
    print(f"     FDGD={config.use_fdgd} GNS={config.use_gns} SMA={config.use_sma}")
    print(f"     TOWS={config.use_tows} DRA={config.use_dra} MSO={config.use_mso}")
    print(f"     HFISC={config.use_hfisc} SGC={config.use_sgc} NFR={config.use_nfr}")
    print(f"     SBS={config.use_sbs} (p_stb={config.sbs_p_stb})")

    trainer = build_s3_model(
        base_model=base_model,
        svd_dir=svd_dir,
        config=config,
        device=device,
        ablation=ablation,
    )

    if eval_only:
        if checkpoint_path:
            trainer.load_checkpoint(checkpoint_path)
        print("[S³] Running benchmarks (eval only)...")
        from src.benchmarks.run_benchmarks import run_all_benchmarks, save_benchmark_results, format_latex_table
        results = run_all_benchmarks(
            base_model.to(device), tokenizer, device,
            ["mmlu", "hellaswag", "arc"],
        )
        output_dir = "./results/"
        os.makedirs(output_dir, exist_ok=True)
        save_benchmark_results(results, os.path.join(output_dir, "benchmarks.json"))
        format_latex_table(results, os.path.join(output_dir, "benchmarks_table.tex"))
        return

    dataset_config = {
        "path": dataset_path,
        "split": "train",
        "format": "instruction_output",
        "instruction_key": "instruction",
        "output_key": "output",
        "max_seq_length": 512,
    }
    dataloader = load_dataset(dataset_config, tokenizer)

    output_dir = checkpoint_dir or "./checkpoints/"
    os.makedirs(output_dir, exist_ok=True)

    if monitor is None:
        monitor_dir = "./results/monitoring/"
        os.makedirs(monitor_dir, exist_ok=True)
        monitor = TrainingMonitor(
            device=device,
            log_dir=monitor_dir,
            sample_interval=2.0,
            print_interval=5,
            project_name=f"s3_{model_name.split('/')[-1].lower()}",
        )
        monitor.start()

    def logging_fn(stats):
        pass

    print(f"[S³] Starting training (tier={profile.tier.value}, epochs={max_epochs})...")
    t_start = time.time()
    trainer.train(dataloader, log_fn=log_fn or logging_fn, monitor=monitor)
    elapsed = time.time() - t_start
    print(f"[S³] Training complete in {elapsed:.1f}s ({elapsed/3600:.1f}h)")

    monitor.stop()

    from src.training.training_monitor import generate_plots, print_paper_table
    generate_plots(monitor_dir, monitor.run_id)
    print_paper_table(monitor_dir, monitor.run_id)

    ckpt_path = os.path.join(output_dir, "checkpoint.pt")
    trainer.save_checkpoint(ckpt_path)
    print(f"[S³] Checkpoint saved: {ckpt_path}")

    print("[S³] Running benchmarks...")
    from src.benchmarks.run_benchmarks import run_all_benchmarks, save_benchmark_results, format_latex_table
    results = run_all_benchmarks(
        base_model.to(device), tokenizer, device,
        ["mmlu", "hellaswag", "arc", "gsm8k"],
    )
    results_dir = "./results/"
    os.makedirs(results_dir, exist_ok=True)
    save_benchmark_results(results, os.path.join(results_dir, "benchmarks.json"))
    format_latex_table(results, os.path.join(results_dir, "benchmarks_table.tex"))

    print("[S³] Pipeline complete.")


def _with_max_epochs(config: HardwareProfile, max_epochs: int):
    """Create a copy of config with updated max_epochs."""
    cfg_dict = {
        'micro_batch_size': config.micro_batch_size,
        'gradient_accumulation_steps': config.gradient_accumulation_steps,
        'learning_rate': config.learning_rate,
        'weight_decay': config.weight_decay,
        'betas': config.betas,
        'max_epochs': max_epochs,
        'warmup_ratio': config.warmup_ratio,
        'max_grad_norm': config.max_grad_norm,
        'log_interval': config.log_interval,
        'use_amp': config.use_amp,
        'amp_dtype': config.amp_dtype,
        'layer_swap': config.layer_swap,
        'token_by_token': config.token_by_token,
        'vram_budget_mb': config.vram_budget_mb,
        'use_smv': config.use_smv,
        'use_emp': config.use_emp,
        'use_ter': config.use_ter,
        'use_fdgd': config.use_fdgd,
        'use_gns': config.use_gns,
        'use_sma': config.use_sma,
        'use_tows': config.use_tows,
        'use_dra': config.use_dra,
        'use_mso': config.use_mso,
        'use_hfisc': config.use_hfisc,
        'use_sgc': config.use_sgc,
        'use_nfr': config.use_nfr,
        'use_sbs': config.use_sbs,
        'sbs_p_stb': config.sbs_p_stb,
    }
    from src.training.frugal_trainer import FrugalConfig
    return FrugalConfig(**cfg_dict)


if __name__ == "__main__":
    from src.training.launcher import interactive_menu, launch_from_args
    from src.training.hardware_profiles import HardwareTier, get_profile

    profile, model_name, dataset_path = launch_from_args()
    run_training(profile, model_name, dataset_path)