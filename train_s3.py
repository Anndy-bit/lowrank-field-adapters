#!/usr/bin/env python3
"""
S³ Training Pipeline — End-to-end fine-tuning on ≤2GB VRAM.

Orchestrates the complete workflow:
  1. Load pretrained model + tokenizer
  2. Load pre-computed SVD factors
  3. Build S³ layers (SVMO × 7 + STB + NMF × 2 per transformer block)
  4. Frugal layer-by-layer training on Alpaca/OpenOrca/FLAN
  5. Save adapter checkpoint
  6. Run benchmarks

Usage:
    # Step 1: Pre-compute SVD (run once per model)
    python src/utils/randomized_svd.py --model Qwen/Qwen2.5-7B-Instruct --k 128

    # Step 2: Train S³
    python train_s3.py --config experiments/configs/s3_standard.yaml

    # Step 3: Evaluate
    python train_s3.py --config experiments/configs/s3_standard.yaml --eval_only \
        --checkpoint ./checkpoints/s3_qwen7b_alpaca/checkpoint.pt
"""

import torch
import argparse
import yaml
import os
import sys
import json
import time
from typing import Dict, Optional, Any

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.adapters.hybrid import S3TransformerLayer, create_s3_layer_from_hf
from src.adapters.svmo import SVMOAdapter, create_svmo_from_linear
from src.adapters.nmf import NMFFlow
from src.adapters.stb import STBBridge
from src.training.frugal_trainer import FrugalTrainer, FrugalConfig
from src.benchmarks.run_benchmarks import run_all_benchmarks, save_benchmark_results, format_latex_table


def load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def build_s3_model(
    base_model,
    svd_dir: str,
    config: Dict[str, Any],
    device: str = "cuda:0",
    ablation: Optional[Dict[str, bool]] = None,
) -> FrugalTrainer:
    """Build S³ adapted model from pretrained base + SVD factors + config.

    Args:
        base_model: HuggingFace CausalLM model on CPU
        svd_dir: directory with pre-computed SVD factors
        config: S³ config dict
        device: GPU device
        ablation: optional dict with keys svmo/stb/nmf (bool) for ablation

    Returns:
        FrugalTrainer ready for training
    """
    s3_cfg = config["s3"]
    if ablation is None:
        ablation = {"svmo": True, "stb": True, "nmf": True}

    svd_on = s3_cfg.copy() if ablation["svmo"] else {**s3_cfg, "svmo_k": 0}
    
    s3_layers = torch.nn.ModuleList()
    gpu_device = torch.device(device)
    cpu_device = torch.device("cpu")

    transformer_blocks = getattr(base_model.model, "layers", None)
    if transformer_blocks is None:
        for attr in ["decoder", "transformer", "encoder"]:
            inner = getattr(base_model.model, attr, None)
            if inner and hasattr(inner, "layers"):
                transformer_blocks = inner.layers
                break

    if transformer_blocks is None:
        raise ValueError("Cannot find transformer layers in model structure")

    n_layers = len(transformer_blocks)
    print(f"[S3] Found {n_layers} transformer layers")

    for i, block in enumerate(transformer_blocks):
        block = block.to(cpu_device)
        for param in block.parameters():
            param.requires_grad = False

        try:
            s3_layer = create_s3_layer_from_hf(
                block,
                svmo_k=s3_cfg["svmo_k"] if ablation["svmo"] else 0,
                svmo_hidden=s3_cfg["svmo_hidden"],
                svmo_alpha=s3_cfg["svmo_alpha"],
                nmf_bottleneck=s3_cfg["nmf_bottleneck"],
                nmf_T=s3_cfg["nmf_T"],
                nmf_N=s3_cfg["nmf_N"],
                stb_beta=s3_cfg["stb_beta"],
            )
        except Exception as e:
            print(f"[S3] Skipping layer {i}: {e}")
            continue

        if not ablation["svmo"]:
            s3_layer.svmo_q = None
        if not ablation["stb"]:
            s3_layer.stb = None
        if not ablation["nmf"]:
            s3_layer.nmf_attn = None
            s3_layer.nmf_mlp = None

        s3_layers.append(s3_layer)

        trainable = sum(p.numel() for p in s3_layer.parameters() if p.requires_grad)
        print(f"[S3] Layer {i}: {trainable:,} trainable params")

    total_trainable = sum(
        sum(p.numel() for p in layer.parameters() if p.requires_grad)
        for layer in s3_layers
    )
    total_params = sum(
        sum(p.numel() for p in layer.parameters())
        for layer in s3_layers
    )
    print(f"[S3] Total: {total_trainable:,} trainable / {total_params:,} all params")

    embedding = base_model.model.embed_tokens
    lm_head = base_model.lm_head
    embedding.weight.requires_grad = False
    if hasattr(lm_head, "weight"):
        lm_head.weight.requires_grad = False

    train_cfg = config["training"]

    trainer = FrugalTrainer(
        s3_layers=s3_layers,
        embedding=embedding,
        lm_head=lm_head,
        config=FrugalConfig(
            micro_batch_size=train_cfg["micro_batch_size"],
            gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
            learning_rate=train_cfg["learning_rate"],
            weight_decay=train_cfg.get("weight_decay", 0.01),
            betas=tuple(train_cfg.get("betas", [0.9, 0.999])),
            max_epochs=train_cfg["max_epochs"],
            warmup_ratio=train_cfg["warmup_ratio"],
            max_grad_norm=train_cfg["max_grad_norm"],
            use_amp=train_cfg.get("use_amp", True),
        ),
        device=device,
    )
    return trainer


def load_dataset(config: Dict[str, Any], tokenizer):
    """Load and tokenize dataset according to config."""
    from datasets import load_dataset
    from torch.utils.data import DataLoader

    data_cfg = config["data"]
    dataset = load_dataset(data_cfg["path"], split=data_cfg.get("split", "train"), trust_remote_code=True)

    fmt = data_cfg.get("preprocessing", {}).get("format", "instruction_output")
    inst_key = data_cfg.get("preprocessing", {}).get("instruction_key", "instruction")
    out_key = data_cfg.get("preprocessing", {}).get("output_key", "output")

    def tokenize_fn(example):
        instruction = example[inst_key]
        output = example.get(out_key, "")
        text = f"### Instruction:\n{instruction}\n\n### Response:\n{output}"
        tokens = tokenizer.encode(
            text,
            truncation=True,
            max_length=config["model"].get("max_seq_length", 512),
        )
        return {"input_ids": tokens}

    tokenized = dataset.map(tokenize_fn, remove_columns=dataset.column_names)
    tokenized.set_format(type="torch", columns=["input_ids"])

    sorted_dataset = sorted(tokenized, key=lambda x: len(x["input_ids"]))

    def collate(batch):
        max_len = max(len(x["input_ids"]) for x in batch)
        padded = torch.zeros(len(batch), max_len, dtype=torch.long)
        for i, x in enumerate(batch):
            padded[i, :len(x["input_ids"])] = x["input_ids"]
        return {"input_ids": padded}

    dataloader = DataLoader(
        sorted_dataset,
        batch_size=1,
        shuffle=True,
        collate_fn=collate,
    )

    print(f"[Data] Loaded {len(dataset)} examples")
    return dataloader


def main():
    parser = argparse.ArgumentParser(description="S³ Training Pipeline")
    parser.add_argument("--config", type=str, required=True, help="YAML config path")
    parser.add_argument("--eval_only", action="store_true", help="Skip training, only evaluate")
    parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint path for eval")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="cuda:0", help="GPU device")
    parser.add_argument("--ablation", type=str, default=None,
                        help="Ablation: s3_full, s3_no_stb, s3_no_nmf, s3_no_svmo, svmo_only, nmf_only, stb_only")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    config = load_config(args.config)

    if not args.device.startswith("cuda"):
        print("[WARN] Training on CPU only — will be very slow")
        config["training"]["use_amp"] = False

    print(f"[S³] Loading model: {config['model']['name']}")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(config["model"]["name"])
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    base_model = AutoModelForCausalLM.from_pretrained(
        config["model"]["name"],
        torch_dtype=getattr(torch, config["model"]["dtype"]),
        device_map="cpu",
        low_cpu_mem_usage=True,
    )
    for param in base_model.parameters():
        param.requires_grad = False

    ablation = None
    if args.ablation:
        from yaml import safe_load
        with open("experiments/configs/ablations.yaml") as f:
            ablations_cfg = safe_load(f)
        ablation_entry = ablations_cfg["ablations"].get(args.ablation, {})
        ablation = {
            "svmo": ablation_entry.get("svmo", True),
            "stb": ablation_entry.get("stb", True),
            "nmf": ablation_entry.get("nmf", True),
        }
        print(f"[S³] Ablation mode: {args.ablation} → svmo={ablation['svmo']}, stb={ablation['stb']}, nmf={ablation['nmf']}")

    svd_dir = config["model"].get("svd_dir", f"./svd_factors/{config['model']['name'].split('/')[-1].lower()}/")

    trainer = build_s3_model(base_model, svd_dir, config, args.device, ablation)

    if args.eval_only:
        if args.checkpoint:
            trainer.load_checkpoint(args.checkpoint)

        print("[S³] Running benchmarks (eval only)...")
        results = run_all_benchmarks(
            base_model.to(args.device), tokenizer, args.device,
            config.get("benchmarks", ["mmlu", "hellaswag", "arc"]),
        )
        output_dir = config["output"].get("results_dir", "./results/")
        os.makedirs(output_dir, exist_ok=True)
        save_benchmark_results(results, os.path.join(output_dir, "benchmarks.json"))
        format_latex_table(results, os.path.join(output_dir, "benchmarks_table.tex"))
        return

    dataloader = load_dataset(config, tokenizer)

    output_cfg = config["output"]
    os.makedirs(output_cfg["checkpoint_dir"], exist_ok=True)
    os.makedirs(output_cfg["results_dir"], exist_ok=True)

    def log_fn(stats):
        pass

    print("[S³] Starting frugal training...")
    t_start = time.time()
    trainer.train(dataloader, log_fn=log_fn)
    elapsed = time.time() - t_start
    print(f"[S³] Training complete in {elapsed:.1f}s ({elapsed/3600:.1f}h)")

    ckpt_path = os.path.join(output_cfg["checkpoint_dir"], "checkpoint.pt")
    trainer.save_checkpoint(ckpt_path)

    print("[S³] Running benchmarks on trained model...")
    results = run_all_benchmarks(
        base_model.to(args.device), tokenizer, args.device,
        config.get("benchmarks", ["mmlu", "hellaswag", "arc"]),
    )
    save_benchmark_results(results, os.path.join(output_cfg["results_dir"], "benchmarks.json"))
    format_latex_table(results, os.path.join(output_cfg["results_dir"], "benchmarks_table.tex"))

    print("[S³] Pipeline complete.")


if __name__ == "__main__":
    main()