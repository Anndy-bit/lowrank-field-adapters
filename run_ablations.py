#!/usr/bin/env python3
"""
S³ ablation runner (RQ4) — one command, one comparison table.

Trains several operator configurations on the SAME data + seed and emits a
side-by-side comparison (quality delta, trainable params, VRAM, throughput,
per-operator gradient magnitude). Each variant also gets its own full report.

Configs (structurally valid — STB needs SVMO's U_k, so −SVMO implies −STB):
  full       SVMO + STB + NMF
  no_stb     SVMO + NMF          (does the bridge help?)
  no_nmf     SVMO + STB          (does the flow help?)
  svmo_only  SVMO                (is spectral modulation enough?)
  nmf_only   NMF                 (is the flow enough alone?)

Examples
--------
python run_ablations.py --smoke                    # CPU tiny-model self-test
make ablations LIMIT=100                            # real 7B, 100 examples each
"""

import argparse
import csv
import os
import time

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import torch

from src.adapters.s3_block import wrap_model_with_s3
from src.training.s3_trainer import S3Trainer, S3TrainConfig
from src.training.run_report import RunLogger, eval_perplexity, generate_report

# Suite 1 — operator ablation (which operator matters). Done previously.
OPERATOR_SUITE = {
    "full":      dict(enable_svmo=True,  enable_stb=True,  enable_nmf=True),
    "no_stb":    dict(enable_svmo=True,  enable_stb=False, enable_nmf=True),
    "no_nmf":    dict(enable_svmo=True,  enable_stb=True,  enable_nmf=False),
    "svmo_only": dict(enable_svmo=True,  enable_stb=False, enable_nmf=False),
    "nmf_only":  dict(enable_svmo=False, enable_stb=False, enable_nmf=True),
}

# Suite 2 — PARAMS-MATCHED (does NMF win by design or by size?).
# NMF params = 2*d*bottleneck*2flows*28layers ; SVMO params ~= H^2 per matrix.
#   nmf_b1  ~0.40M  |  nmf_b2 ~0.80M  |  svmo_h64 ~0.82M  (matches nmf_b2)
# The decisive pair: nmf_b2 (0.80M, NMF) vs svmo_h64 (0.82M, SVMO) at equal budget.
CAPACITY_SUITE = {
    "nmf_b1":      dict(enable_svmo=False, enable_stb=False, enable_nmf=True,  nmf_bottleneck=1),
    "nmf_b2":      dict(enable_svmo=False, enable_stb=False, enable_nmf=True,  nmf_bottleneck=2),
    "svmo_h64":    dict(enable_svmo=True,  enable_stb=False, enable_nmf=False, svmo_hidden=64),
    "full_nmf_b1": dict(enable_svmo=True,  enable_stb=True,  enable_nmf=True,  nmf_bottleneck=1),
}

# Suite 3 — BASELINES: S³ vs LoRA at matched parameter budget, same data/hardware.
# LoRA params ~= 2.52M * r  |  S³ full = 3.90M  ->  r=1 (2.52M) and r=2 (5.05M) bracket it.
# r=8 (20.2M) is the literature-standard LoRA, 5x S³'s budget.
BASELINE_SUITE = {
    "s3_full":  dict(enable_svmo=True, enable_stb=True, enable_nmf=True),          # 3.90M
    "lora_r1":  dict(lora_r=1),                                                     # 2.52M
    "lora_r2":  dict(lora_r=2),                                                     # 5.05M
    "lora_r8":  dict(lora_r=8),                                                     # 20.2M (standard)
}

SUITES = {"operators": OPERATOR_SUITE, "capacity": CAPACITY_SUITE, "baselines": BASELINE_SUITE}
_DEFAULTS = dict(svmo_hidden=32, svmo_alpha=0.3, nmf_bottleneck=8, nmf_T=1.0, nmf_N=4, stb_beta=0.5)


def _tiny_model():
    from transformers import Qwen2Config, Qwen2ForCausalLM
    cfg = Qwen2Config(vocab_size=512, hidden_size=64, intermediate_size=128,
                      num_hidden_layers=3, num_attention_heads=8, num_key_value_heads=2,
                      max_position_embeddings=128, tie_word_embeddings=False)
    return Qwen2ForCausalLM(cfg)


def _build_variant(args, vcfg):
    """Return (model, blocks, streamer). Fresh model per variant.
    vcfg carries enable_* flags plus optional per-variant overrides
    (nmf_bottleneck, svmo_hidden, ...) merged over the defaults."""
    vcfg = dict(vcfg)
    lora_r = vcfg.pop("lora_r", None)
    if lora_r is not None:
        # LoRA baseline: plain decoder wiring (all S³ operators off) + LoRA on the 7 projections
        vcfg.update(enable_svmo=False, enable_stb=False, enable_nmf=False)
    kw = {**_DEFAULTS, **vcfg}

    if args.smoke:
        torch.manual_seed(args.seed)
        model = _tiny_model()
        kw = {**kw, "svmo_hidden": vcfg.get("svmo_hidden", 16)}
        blocks = wrap_model_with_s3(model, svmo_k=16, sbs_p=args.sbs_p, svd_dir=None, **kw)
        if lora_r is not None:
            from src.adapters.lora_linear import apply_lora_to_blocks
            apply_lora_to_blocks(blocks, r=lora_r)
        return model, blocks, None

    import torch.nn as nn
    from src.training.streaming_loader import (
        build_streaming_model, ShardReader, FrozenStreamer, materialize_shared, find_snapshot)
    torch.manual_seed(args.seed)
    model, _ = build_streaming_model(args.model)
    blocks = wrap_model_with_s3(model, svmo_k=args.svmo_k, svd_dir=args.svd_dir,
                                sbs_p=args.sbs_p, **kw)
    if lora_r is not None:
        from src.adapters.lora_linear import apply_lora_to_blocks
        n = apply_lora_to_blocks(blocks, r=lora_r)
        print(f"[lora] r={lora_r} -> {n:,} trainable params")
    reader = ShardReader(find_snapshot(args.model))
    materialize_shared(model, reader, "cpu", "cpu", args.device)
    model.lm_head.weight = nn.Parameter(model.lm_head.weight.data.float(), requires_grad=False)
    return model, blocks, FrozenStreamer(reader, torch.device(args.device))


def _pack(token_lists, batch, pad_id):
    """Sort by length, group into padded batches of `batch` with attention masks."""
    token_lists = sorted(token_lists, key=len)
    out = []
    for i in range(0, len(token_lists), batch):
        grp = token_lists[i:i + batch]
        m = max(len(x) for x in grp)
        ids = torch.full((len(grp), m), pad_id, dtype=torch.long)
        attn = torch.zeros((len(grp), m), dtype=torch.long)
        for j, x in enumerate(grp):
            ids[j, :len(x)] = torch.tensor(x)
            attn[j, :len(x)] = 1
        out.append({"input_ids": ids, "attention_mask": attn})
    return out


def _batches(args, tok):
    """Fixed training + eval batches shared across all variants (fair comparison)."""
    if args.smoke:
        g = torch.Generator().manual_seed(0)
        toks = [torch.randint(0, 512, (24,), generator=g).tolist() for _ in range(args.limit)]
        train = _pack(toks, args.batch, 0)
        ev = [torch.randint(0, 512, (1, 20), generator=g) for _ in range(4)]
        return train, ev
    from datasets import load_dataset
    ds = load_dataset("tatsu-lab/alpaca", split="train")
    tr = ds.select(range(min(args.limit, len(ds))))
    ev = ds.select(range(len(ds) - args.eval_n, len(ds)))   # held-out tail, never trained on

    def enc(ex):
        t = f"### Instruction:\n{ex['instruction']}\n\n### Response:\n{ex.get('output','')}"
        return tok.encode(t, truncation=True, max_length=args.max_seq)
    pad_id = tok.pad_token_id or 0
    toks = [ids for ids in (enc(ex) for ex in tr) if len(ids) > 2]
    train = _pack(toks, args.batch, pad_id)
    # eval is batched too — otherwise 100+ examples cost an eternity under streaming
    evtoks = [ids for ids in (enc(ex) for ex in ev) if len(ids) > 2]
    evb = _pack(evtoks, args.batch, pad_id)
    return train, evb


class _Loader:
    def __init__(self, b): self.b = b
    def __iter__(self): return iter(self.b)
    def __len__(self): return len(self.b)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--svd_dir", default="./svd_factors")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--suite", default="operators", choices=list(SUITES),
                    help="operators = which operator matters; capacity = params-matched (NMF design vs size)")
    ap.add_argument("--ablations", default="",
                    help="comma list to restrict variants; empty = whole suite")
    ap.add_argument("--limit", type=int, default=150)
    ap.add_argument("--eval_n", type=int, default=100,
                    help="held-out examples for base/after perplexity (batched)")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--max_seq", type=int, default=256)
    ap.add_argument("--grad_accum", type=int, default=2)
    ap.add_argument("--batch", type=int, default=4, help="examples per step (batch=4 ~2.3x faster)")
    ap.add_argument("--svmo_k", type=int, default=128)
    ap.add_argument("--sbs_p", type=float, default=1.0,
                    help="STB always applied in ablations (clean 'does STB help' signal)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="./results")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    if args.smoke:
        args.device = "cpu"
    suite = SUITES[args.suite]
    if args.ablations.strip():
        names = [a.strip() for a in args.ablations.split(",") if a.strip() in suite]
    else:
        names = list(suite)
    out_root = os.path.join(args.out, time.strftime(f"ablations_{args.suite}_%Y%m%d_%H%M%S"))
    os.makedirs(out_root, exist_ok=True)

    tok = None
    if not args.smoke:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(args.model)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token

    train_b, eval_b = _batches(args, tok)
    n_ex = sum(b["input_ids"].shape[0] for b in train_b)
    sec_per_ex = 27 if args.device.startswith("cuda") else 0.05  # measured @ batch=4, S=256
    eta_h = len(names) * n_ex * sec_per_ex / 3600
    print(f"\n[ablations] {len(names)} variants x {n_ex} examples (batch={args.batch}) "
          f"~ ETA {eta_h:.1f}h total (~{eta_h/len(names):.1f}h/variant)")

    rows = []
    for name in names:
        print("\n" + "#" * 64 + f"\n# ABLATION [{args.suite}]: {name}\n" + "#" * 64)
        flags = suite[name]
        model, blocks, streamer = _build_variant(args, flags)
        cfg = S3TrainConfig(learning_rate=args.lr, max_epochs=args.epochs,
                            grad_accum_steps=args.grad_accum, gradient_checkpointing=True,
                            amp=args.device.startswith("cuda") and not args.smoke,
                            use_nfr=flags.get("enable_nmf", False),
                            use_sbs=flags.get("enable_stb", False), log_interval=10)
        trainer = S3Trainer(model, blocks, cfg, device=args.device, frozen_streamer=streamer)

        run_dir = os.path.join(out_root, name)
        logger = RunLogger(run_dir, cfg, model, blocks, torch.device(args.device),
                           {"name": "tatsu-lab/alpaca", "model": args.model, "ablation": name})
        base = eval_perplexity(trainer, eval_b)
        t0 = time.time()
        trainer.train(_Loader(train_b), logger=logger)
        wall = (time.time() - t0) / 60
        after = eval_perplexity(trainer, eval_b)
        tokens = getattr(trainer, "tokens_cum", 0)
        summary = {"quality": {"base": base, "after": after},
                   "total_wall_min": round(wall, 2),
                   "tokens_per_s": round(tokens / (wall * 60), 2) if wall > 0 else 0,
                   "vram_peak_mb": _peak(run_dir)}
        logger.finalize(summary)
        generate_report(run_dir)

        n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
        gm = _grad_means(run_dir)
        fl = _loss_stats(run_dir)
        rows.append({
            "ablation": name, "trainable": n_train,
            "base_ppl": base["perplexity"], "after_ppl": after["perplexity"],
            "delta_ppl": round(after["perplexity"] - base["perplexity"], 4),
            "final_loss": fl.get("final"), "min_loss": fl.get("min"),
            "vram_peak_mb": summary["vram_peak_mb"], "tok_s": summary["tokens_per_s"],
            "wall_min": summary["total_wall_min"],
            "grad_svmo": gm.get("grad_svmo"), "grad_nmf": gm.get("grad_nmf"),
            "grad_stb": gm.get("grad_stb"), "grad_lora": gm.get("grad_lora"),
        })
        # trained adapter weights for this variant (reproducibility)
        torch.save({n: p.detach().cpu() for n, p in model.named_parameters() if p.requires_grad},
                   os.path.join(run_dir, "adapters.pt"))
        print(f"[{name}] ppl {base['perplexity']:.3f} -> {after['perplexity']:.3f} "
              f"(Δ {rows[-1]['delta_ppl']:+.3f})  params={n_train:,}  VRAM={summary['vram_peak_mb']}MB")
        del model, blocks, trainer, streamer
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()

    _write_comparison(out_root, rows)
    print(f"\n[ablations] comparison table + per-variant reports: {out_root}")


def _peak(run_dir):
    p = os.path.join(run_dir, "vram_trace.csv")
    if not os.path.exists(p):
        return None
    peak = 0.0
    for r in csv.DictReader(open(p)):
        try:
            peak = max(peak, float(r["vram_alloc_mb"]))
        except Exception:
            pass
    return round(peak, 1)


def _loss_stats(run_dir):
    p = os.path.join(run_dir, "steps.csv")
    L = []
    if os.path.exists(p):
        for r in csv.DictReader(open(p)):
            try:
                L.append(float(r["loss"]))
            except (ValueError, KeyError):
                pass
    return {"final": round(L[-1], 4), "min": round(min(L), 4)} if L else {}


def _grad_means(run_dir):
    import statistics
    p = os.path.join(run_dir, "steps.csv")
    cols = {"grad_svmo": [], "grad_nmf": [], "grad_stb": [], "grad_lora": []}
    if os.path.exists(p):
        for r in csv.DictReader(open(p)):
            for k in cols:
                try:
                    cols[k].append(float(r[k]))
                except (ValueError, KeyError):
                    pass
    return {k: round(statistics.mean(v), 5) if v else None for k, v in cols.items()}


def _write_comparison(out_root, rows):
    csv_path = os.path.join(out_root, "comparison.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    md = ["# S³ Ablations (RQ4)\n", "Same data + seed across all variants. Lower Δppl = better.\n",
          "| ablation | trainable | base ppl | after ppl | Δ ppl | final loss | min loss | VRAM MB | tok/s | wall min | grad svmo | grad nmf | grad stb | grad lora |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        md.append("| {ablation} | {trainable:,} | {base_ppl} | {after_ppl} | {delta_ppl:+} | "
                  "{final_loss} | {min_loss} | {vram_peak_mb} | {tok_s} | {wall_min} | "
                  "{grad_svmo} | {grad_nmf} | {grad_stb} | {grad_lora} |".format(**r))
    # contribution of each operator vs full
    full = next((r for r in rows if r["ablation"] == "full"), None)
    if full:
        md.append("\n## Operator contribution (Δppl vs `full`, more negative = operator helps)\n")
        for r in rows:
            if r["ablation"] != "full":
                diff = round(r["delta_ppl"] - full["delta_ppl"], 4)
                md.append(f"- removing/keeping-only **{r['ablation']}**: {diff:+} ppl vs full")
    with open(os.path.join(out_root, "comparison.md"), "w") as f:
        f.write("\n".join(md))
    print("\n" + "\n".join(md))


if __name__ == "__main__":
    main()
