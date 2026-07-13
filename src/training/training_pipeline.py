"""Shared training pipeline for S³ — used by both train_s3.py and run_training.py.

This module contains the core model-building and dataset-loading logic that
both CLI entry points (make train / python train_s3.py) and the interactive
menu (run_s3.py) go through. The difference is:
- CLI (make train): FrugalConfig from YAML + hardware profile defaults
- Menu (run_s3.py): FrugalConfig from hardware profile only

Both ultimately call build_s3_model() and load_dataset().
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Optional, Any, Tuple

from src.adapters.hybrid import S3TransformerLayer, create_s3_layer_from_hf
from src.training.frugal_trainer import FrugalTrainer, FrugalConfig
from src.training.hardware_profiles import HardwareProfile


def merge_frugal_config(
    base: FrugalConfig,
    training_overrides: Optional[Dict[str, Any]] = None,
    s3_opt_overrides: Optional[Dict[str, Any]] = None,
) -> FrugalConfig:
    """Create a new FrugalConfig by merging base with overrides.

    S3-OPT flags default to the base config values. Only overrides
    explicitly passed in s3_opt_overrides will change them.
    This ensures the hardware profile's S3-OPT defaults are preserved
    unless explicitly overridden (e.g. for ablations).

    Args:
        base: FrugalConfig to use as base (typically from hardware profile)
        training_overrides: dict of FrugalConfig fields to override from YAML
        s3_opt_overrides: dict of S3-OPT flags to override (e.g. for ablations)
    """
    # Start from base
    cfg_dict = {
        'micro_batch_size': base.micro_batch_size,
        'gradient_accumulation_steps': base.gradient_accumulation_steps,
        'learning_rate': base.learning_rate,
        'weight_decay': base.weight_decay,
        'betas': base.betas,
        'max_epochs': base.max_epochs,
        'warmup_ratio': base.warmup_ratio,
        'max_grad_norm': base.max_grad_norm,
        'log_interval': base.log_interval,
        'use_amp': base.use_amp,
        'amp_dtype': base.amp_dtype,
        'layer_swap': base.layer_swap,
        'token_by_token': base.token_by_token,
        'vram_budget_mb': base.vram_budget_mb,
        # S3-OPT flags from base
        'use_smv': base.use_smv,
        'use_emp': base.use_emp,
        'use_ter': base.use_ter,
        'use_fdgd': base.use_fdgd,
        'use_gns': base.use_gns,
        'use_sma': base.use_sma,
        'use_tows': base.use_tows,
        'use_dra': base.use_dra,
        'use_mso': base.use_mso,
        'use_hfisc': base.use_hfisc,
        'use_sgc': base.use_sgc,
        'use_nfr': base.use_nfr,
        'use_sbs': base.use_sbs,
        'sbs_p_stb': base.sbs_p_stb,
    }

    # Apply training overrides (from YAML)
    if training_overrides:
        for key, value in training_overrides.items():
            if hasattr(base, key):
                cfg_dict[key] = value

    # Apply S3-OPT overrides (from YAML s3_opt section)
    if s3_opt_overrides:
        for key, value in s3_opt_overrides.items():
            if key in cfg_dict:
                cfg_dict[key] = value

    return FrugalConfig(**cfg_dict)


def build_s3_model(
    base_model,
    svd_dir: str,
    config: FrugalConfig,
    device: str = "cuda:0",
    ablation: Optional[Dict[str, bool]] = None,
) -> FrugalTrainer:
    """Build S³ adapted model from pretrained base + SVD factors + config.

    Args:
        base_model: HuggingFace CausalLM model on CPU
        svd_dir: directory with pre-computed SVD factors
        config: FrugalConfig (from hardware profile + optional YAML overrides)
        device: GPU device
        ablation: optional dict with keys svmo/stb/nmf (bool) for ablation

    Returns:
        FrugalTrainer ready for training
    """
    s3_cfg = {
        "svmo_k": 128,       # Default; override via config if needed
        "svmo_hidden": 32,
        "svmo_alpha": 0.3,
        "nmf_bottleneck": 8,
        "nmf_T": 1.0,
        "nmf_N": 4,
        "stb_beta": 0.5,
    }
    if ablation is None:
        ablation = {"svmo": True, "stb": True, "nmf": True}

    s3_layers = torch.nn.ModuleList()
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
                svd_dir=svd_dir,
                layer_idx=i,
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

    trainer = FrugalTrainer(
        s3_layers=s3_layers,
        embedding=embedding,
        lm_head=lm_head,
        config=config,
        device=device,
    )
    return trainer


def load_dataset(config: Dict[str, Any], tokenizer) -> DataLoader:
    """Load and tokenize dataset according to config dict.

    Supports both YAML config dict format and simple dict format.
    If config.dataset_streaming is True, uses streaming mode (0 RAM for data).
    """
    from datasets import load_dataset

    # Handle different config formats
    if "data" in config:
        data_cfg = config["data"]
        path = data_cfg["path"]
        split = data_cfg.get("split", "train")
        fmt = data_cfg.get("preprocessing", {}).get("format", "instruction_output")
        inst_key = data_cfg.get("preprocessing", {}).get("instruction_key", "instruction")
        out_key = data_cfg.get("preprocessing", {}).get("output_key", "output")
        max_seq = config.get("model", {}).get("max_seq_length", 512)
    else:
        path = config.get("path", "tatsu-lab/alpaca")
        split = config.get("split", "train")
        fmt = config.get("format", "instruction_output")
        inst_key = config.get("instruction_key", "instruction")
        out_key = config.get("output_key", "output")
        max_seq = config.get("max_seq_length", 512)

    streaming = config.get("dataset_streaming", False)
    dataset = load_dataset(path, split=split, trust_remote_code=True, streaming=streaming)

    def tokenize_fn(example):
        instruction = example[inst_key]
        output = example.get(out_key, "")
        text = f"### Instruction:\n{instruction}\n\n### Response:\n{output}"
        tokens = tokenizer.encode(text, truncation=True, max_length=max_seq)
        return {"input_ids": tokens}

    if streaming:
        # Streaming (IterableDataset): use with_format instead of set_format
        tokenized = dataset.map(tokenize_fn, remove_columns=dataset.column_names)
        tokenized = tokenized.with_format("torch")
        sorted_dataset = tokenized  # Can't sort streaming
        shuffle = False
    else:
        # Regular dataset: can sort by length
        tokenized = dataset.map(tokenize_fn, remove_columns=dataset.column_names)
        tokenized.set_format(type="torch", columns=["input_ids"])
        sorted_dataset = sorted(tokenized, key=lambda x: len(x["input_ids"]))
        shuffle = True

    def collate(batch):
        max_len = max(len(x["input_ids"]) for x in batch)
        padded = torch.zeros(len(batch), max_len, dtype=torch.long)
        for i, x in enumerate(batch):
            padded[i, :len(x["input_ids"])] = x["input_ids"]
        return {"input_ids": padded}

    dataloader = DataLoader(
        sorted_dataset,
        batch_size=1,
        shuffle=shuffle,
        collate_fn=collate,
    )

    print(f"[Data] Loaded {len(dataset) if not streaming else 'streaming'} examples")
    return dataloader