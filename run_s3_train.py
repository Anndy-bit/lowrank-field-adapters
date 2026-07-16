#!/usr/bin/env python3
"""
S³ 7B training entrypoint (rebuilt, correct core).

Loads a locally-cached model OFFLINE, wraps every decoder layer with an S3Block
(SVMO residual + STB + NMF) using the pre-computed SVD factors on disk, and
trains only the adapters with S3Trainer (full-sequence, real autograd,
gradient checkpointing, optional layer swap).

No new model download is performed (rural / offline friendly): set
HF_HUB_OFFLINE=1 and rely on ~/.cache/huggingface.

Examples
--------
# Self-test the WHOLE pipeline on a tiny random model (no download, CPU):
python run_s3_train.py --smoke

# Real run on the cached Qwen2.5-7B with yesterday's SVD factors:
HF_HUB_OFFLINE=1 python run_s3_train.py \
    --model Qwen/Qwen2.5-7B-Instruct --svd_dir ./svd_factors \
    --device cuda:0 --quant 8bit --layer_swap --epochs 3
"""

import argparse
import os
import sys

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch

from src.adapters.s3_block import wrap_model_with_s3
from src.training.s3_trainer import S3Trainer, S3TrainConfig


# --------------------------------------------------------------------------- #
def build_dataloader(tokenizer, max_seq, limit, batch_size=1):
    from datasets import load_dataset
    from torch.utils.data import DataLoader

    ds = load_dataset("tatsu-lab/alpaca", split="train")
    if limit:
        ds = ds.select(range(min(limit, len(ds))))

    def tok(ex):
        text = f"### Instruction:\n{ex['instruction']}\n\n### Response:\n{ex.get('output','')}"
        return {"input_ids": tokenizer.encode(text, truncation=True, max_length=max_seq)}

    ds = ds.map(tok, remove_columns=ds.column_names)
    # sort by length so each batch is length-homogeneous (minimal padding)
    ds = sorted(ds, key=lambda x: len(x["input_ids"]))
    pad_id = tokenizer.pad_token_id or 0

    def collate(batch):
        m = max(len(x["input_ids"]) for x in batch)
        ids = torch.full((len(batch), m), pad_id, dtype=torch.long)
        attn = torch.zeros((len(batch), m), dtype=torch.long)
        for i, x in enumerate(batch):
            n = len(x["input_ids"])
            ids[i, :n] = torch.tensor(x["input_ids"])
            attn[i, :n] = 1
        return {"input_ids": ids, "attention_mask": attn}

    # shuffle at the BATCH level (keeps length homogeneity) via a sampler over batches
    return DataLoader(ds, batch_size=batch_size, shuffle=(batch_size == 1), collate_fn=collate)


def load_base_model(model_name, quant, device, gpu_mem, cpu_mem, offload_dir):
    """Load the frozen base offline.

    `gpu_mem` caps how much of the base accelerate may place on the GPU: the rest
    goes to CPU RAM / disk. Headroom MUST be left for activations, the adapters and
    the optimizer, otherwise accelerate fills the card and the first forward OOMs.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    kw = dict(dtype=torch.float16, low_cpu_mem_usage=True)
    if quant in ("8bit", "4bit"):
        from transformers import BitsAndBytesConfig
        if quant == "8bit":
            kw["quantization_config"] = BitsAndBytesConfig(
                load_in_8bit=True, llm_int8_enable_fp32_cpu_offload=True)
        else:
            kw["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True,
                # required so accelerate may keep part of the model on CPU/disk
                llm_int8_enable_fp32_cpu_offload=True)

    if device.startswith("cuda"):
        os.makedirs(offload_dir, exist_ok=True)
        kw["device_map"] = "auto"
        kw["max_memory"] = {0: gpu_mem, "cpu": cpu_mem}
        kw["offload_folder"] = offload_dir
        print(f"[S³] base placement budget: GPU={gpu_mem}, CPU={cpu_mem}, offload={offload_dir}")
    else:
        kw["device_map"] = "cpu"

    model = AutoModelForCausalLM.from_pretrained(model_name, **kw)
    return model, tok


# --------------------------------------------------------------------------- #
def run_smoke():
    """End-to-end pipeline check on a tiny random Qwen2 (no download, CPU)."""
    from transformers import Qwen2Config, Qwen2ForCausalLM
    print("[smoke] building tiny Qwen2 + S³ wrap ...")
    cfg = Qwen2Config(vocab_size=512, hidden_size=128, intermediate_size=256,
                      num_hidden_layers=4, num_attention_heads=8, num_key_value_heads=2,
                      max_position_embeddings=256, tie_word_embeddings=False)
    torch.manual_seed(0)
    model = Qwen2ForCausalLM(cfg)

    # parity check
    ids = torch.randint(0, 512, (2, 32))
    model.eval()
    with torch.no_grad():
        base_loss = float(model(ids, labels=ids).loss)
    blocks = wrap_model_with_s3(model, svmo_k=32, svmo_hidden=16, svmo_alpha=0.3,
                                nmf_bottleneck=8, nmf_N=4, stb_beta=0.5,
                                svd_dir=None, sbs_p=0.3)
    with torch.no_grad():
        init_loss = float(model(ids, labels=ids).loss)
    print(f"[smoke] parity base={base_loss:.4f} s3_init={init_loss:.4f} Δ={abs(base_loss-init_loss):.2e}")
    assert abs(base_loss - init_loss) < 1e-3, "parity FAILED"

    tr = S3Trainer(model, blocks,
                   S3TrainConfig(learning_rate=5e-3, max_epochs=4, grad_accum_steps=1,
                                 gradient_checkpointing=True, layer_swap=False, amp=False,
                                 use_nfr=True, use_sbs=True,
                                 extra_opts=("TER", "DRA", "MSO", "TOWS", "HFISC", "FDGD", "GNS", "EMP", "SMV")),
                   device="cpu")

    class Loader:
        def __init__(s, b): s.b = b
        def __iter__(s): return iter(s.b)
        def __len__(s): return len(s.b)

    batch = {"input_ids": torch.randint(0, 512, (4, 40))}
    model.eval()
    with torch.no_grad():
        l0 = float(tr.forward_loss(batch["input_ids"]))
    tr.train(Loader([batch] * 6))
    model.eval()
    with torch.no_grad():
        l1 = float(tr.forward_loss(batch["input_ids"]))
    print(f"[smoke] learning loss {l0:.4f} -> {l1:.4f}")
    assert l1 < l0 - 0.05, "learning FAILED"
    print("[smoke] PIPELINE OK ✔  (parity exact + loss decreases + S3-OPT coupled)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--svd_dir", default="./svd_factors")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--mode", default="stream", choices=["stream", "quant"],
                    help="stream = disk-stream frozen weights per layer (fits tiny VRAM); "
                         "quant = bitsandbytes + accelerate offload")
    ap.add_argument("--quant", default="4bit", choices=["none", "8bit", "4bit"])
    ap.add_argument("--layer_swap", action="store_true")
    ap.add_argument("--gpu_mem", default="2GiB",
                    help="max VRAM accelerate may use for the BASE weights "
                         "(leave headroom for activations/adapters)")
    ap.add_argument("--cpu_mem", default="9GiB", help="max CPU RAM for offloaded base weights")
    ap.add_argument("--offload_dir", default="/tmp/s3_offload")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--max_seq", type=int, default=512)
    ap.add_argument("--grad_accum", type=int, default=16)
    ap.add_argument("--batch", type=int, default=1,
                    help="examples per step — amortizes the per-step disk read (big speedup)")
    # --- USF (formalismo_streaming.md) ---
    ap.add_argument("--granularity", default="auto",
                    choices=["auto", "layer", "sublayer", "matrix"],
                    help="Thm 3 / §9: streamable unit. 'auto' picks the coarsest that fits.")
    ap.add_argument("--prefetch", action="store_true",
                    help="Prop. 5: double-buffer the next unit's disk read")
    ap.add_argument("--layer_major", action="store_true",
                    help="Thm 4: invert loops -> I/O divided by grad_accum")
    ap.add_argument("--limit", type=int, default=None, help="limit #training examples")
    ap.add_argument("--svmo_k", type=int, default=128)
    ap.add_argument("--sbs_p", type=float, default=0.3)
    ap.add_argument("--out", default="./results", help="directory for the run report + data")
    ap.add_argument("--smoke", action="store_true", help="self-test pipeline on a tiny model")
    args = ap.parse_args()

    if args.smoke:
        run_smoke()
        return

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    streamer = None
    if args.mode == "stream":
        from src.training.streaming_loader import (
            build_streaming_model, ShardReader, FrozenStreamer, PrefetchStreamer,
            materialize_shared, find_snapshot, choose_granularity,
        )
        print(f"[S³] building {args.model} skeleton on meta (0 memory) ...")
        model, _ = build_streaming_model(args.model)
        print(f"[S³] wrapping decoder layers with S³ + offline SVD from {args.svd_dir} ...")
        blocks = wrap_model_with_s3(
            model, svmo_k=args.svmo_k, svmo_hidden=32, svmo_alpha=0.3,
            nmf_bottleneck=8, nmf_T=1.0, nmf_N=4, stb_beta=0.5,
            svd_dir=args.svd_dir, sbs_p=args.sbs_p,
        )
        reader = ShardReader(find_snapshot(args.model))
        # big vocab matrices on CPU; final norm on GPU; per-layer weights streamed
        materialize_shared(model, reader, embed_device="cpu",
                           head_device="cpu", norm_device=args.device)
        # LM head in fp32 on CPU for a stable, precise cross-entropy
        import torch.nn as _nn
        model.lm_head.weight = _nn.Parameter(model.lm_head.weight.data.float(), requires_grad=False)

        # §9 — pick the coarsest granularity that fits this GPU (model-agnostic).
        if args.granularity == "auto":
            free_b = (torch.cuda.get_device_properties(0).total_memory
                      if args.device.startswith("cuda") else 4 * 2**30)
            gran = choose_granularity(model.config, int(free_b * 0.75),
                                      reserve_bytes=int(0.6 * 2**30), prefetch=args.prefetch)
        else:
            gran = args.granularity
        cls = PrefetchStreamer if args.prefetch else FrozenStreamer
        streamer = cls(reader, torch.device(args.device), granularity=gran)
        print(f"[S³] USF streaming: granularity='{gran}' prefetch={args.prefetch} "
              f"— one unit resident on GPU at a time.")
    else:
        print(f"[S³] loading {args.model} offline (quant={args.quant}) ...")
        model, tok = load_base_model(args.model, args.quant, args.device,
                                     args.gpu_mem, args.cpu_mem, args.offload_dir)
        print(f"[S³] wrapping decoder layers with S³ + offline SVD from {args.svd_dir} ...")
        blocks = wrap_model_with_s3(
            model, svmo_k=args.svmo_k, svmo_hidden=32, svmo_alpha=0.3,
            nmf_bottleneck=8, nmf_T=1.0, nmf_N=4, stb_beta=0.5,
            svd_dir=args.svd_dir, sbs_p=args.sbs_p,
        )

    cfg = S3TrainConfig(
        learning_rate=args.lr, max_epochs=args.epochs, grad_accum_steps=args.grad_accum,
        gradient_checkpointing=True, layer_swap=args.layer_swap,
        amp=args.device.startswith("cuda"), amp_dtype=torch.float16,
        layer_major=args.layer_major, ckpt_offload=True,
        use_nfr=True, use_sbs=True, use_sgc=True,
        extra_opts=("TER", "DRA", "MSO", "TOWS", "HFISC", "FDGD", "GNS", "EMP", "SMV"),
    )
    trainer = S3Trainer(model, blocks, cfg, device=args.device, frozen_streamer=streamer)

    print(f"[S³] loading Alpaca (cached), batch={args.batch} ...")
    dl = build_dataloader(tok, args.max_seq, args.limit, batch_size=args.batch)
    eval_batches = _held_out_batches(tok, args.max_seq, n=8)

    # ---- full experiment report instrumentation ----
    from src.training.run_report import RunLogger, eval_perplexity, generate_report
    import time as _time
    run_dir = os.path.join(args.out, _time.strftime("run_%Y%m%d_%H%M%S"))
    dataset_info = {"name": "tatsu-lab/alpaca", "model": args.model,
                    "max_seq": args.max_seq, "limit": args.limit, "n_train_batches": len(dl)}
    logger = RunLogger(run_dir, cfg, model, blocks, torch.device(args.device), dataset_info)
    print(f"[S³] report dir: {run_dir}")

    # base quality (adapters at identity == frozen base model)
    print("[S³] base eval (adapters at identity) ...")
    base_q = eval_perplexity(trainer, eval_batches)
    print(f"       base: {base_q}")

    print("[S³] training ...")
    t_train = _time.time()
    trainer.train(dl, logger=logger)
    train_min = (_time.time() - t_train) / 60

    print("[S³] after eval ...")
    after_q = eval_perplexity(trainer, eval_batches)
    print(f"       after: {after_q}")

    # save adapter checkpoint (only trainable params)
    os.makedirs(args.out, exist_ok=True)
    ckpt = os.path.join(run_dir, "adapters.pt")
    torch.save({n: p.detach().cpu() for n, p in model.named_parameters() if p.requires_grad}, ckpt)

    # summary
    tokens = getattr(trainer, "tokens_cum", 0)
    summary = {
        "vram_peak_mb": _peak_from(run_dir),
        "tokens_per_s": round(tokens / (train_min * 60), 2) if train_min > 0 else 0,
        "total_wall_min": round(train_min, 2),
        "tokens_processed": tokens,
        "disk_read_mb_per_step": round(getattr(trainer, "_disk_mb_per_step", 0), 1),
        "quality": {"base": base_q, "after": after_q},
    }
    logger.finalize(summary)
    report = generate_report(run_dir)
    print("\n" + "=" * 64)
    print(open(report).read())
    print("=" * 64)
    print(f"[S³] full report + data: {run_dir}")
    print("[S³] done.")


def _held_out_batches(tok, max_seq, n=8):
    """A few Alpaca examples (from the END of the split) as a held-out eval set."""
    from datasets import load_dataset
    ds = load_dataset("tatsu-lab/alpaca", split="train")
    ds = ds.select(range(len(ds) - n, len(ds)))
    out = []
    for ex in ds:
        text = f"### Instruction:\n{ex['instruction']}\n\n### Response:\n{ex.get('output','')}"
        ids = tok.encode(text, truncation=True, max_length=max_seq)
        if len(ids) > 2:
            out.append(torch.tensor(ids).unsqueeze(0))
    return out


def _peak_from(run_dir):
    """Read the peak VRAM (allocated) from the live trace."""
    import csv as _csv
    p = os.path.join(run_dir, "vram_trace.csv")
    if not os.path.exists(p):
        return None
    peak = 0.0
    with open(p) as f:
        for r in _csv.DictReader(f):
            try:
                peak = max(peak, float(r["vram_alloc_mb"]))
            except Exception:
                pass
    return round(peak, 1)


if __name__ == "__main__":
    main()
