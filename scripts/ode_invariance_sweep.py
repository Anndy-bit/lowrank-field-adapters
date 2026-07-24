"""Barrido de invariancia del solver ODE de NMF (N=1,2,4,8,16), sin reentrenar.

Test falsable (docs/estado_benchmarks.md #4): si NMF aprendio un flujo continuo
de verdad, la salida (y por tanto la perplejidad) debe ser ~invariante al numero
de pasos del solver RK4 una vez que N es "suficiente". Si oscila fuerte, el
modelo esta explotando el paso de discretizacion concreto usado en entrenamiento,
no un flujo real.

Carga el checkpoint ya entrenado de s3_full y solo hace forward (eval_loss),
sobreescribiendo N_steps/dt en cada NMFFlow antes de cada medicion.
"""
import sys, os, json, math, torch
sys.path.insert(0, "/home/gula/Documentos/Proyectos Github/lowrank-field-adapters")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from transformers import AutoTokenizer
from datasets import load_dataset
from src.adapters.s3_block import wrap_model_with_s3
from src.training.s3_trainer import S3Trainer, S3TrainConfig
from src.training.streaming_loader import (
    build_streaming_model, ShardReader, FrozenStreamer, materialize_shared, find_snapshot,
)
from src.training.run_report import eval_perplexity

MODEL = "Qwen/Qwen2.5-7B-Instruct"
DEV = "cuda"
CKPT = "results/ablations_baselines_20260716_024551/s3_full/adapters.pt"
N_LIST = [1, 2, 4, 8, 16]

tok = AutoTokenizer.from_pretrained(MODEL)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

model, _ = build_streaming_model(MODEL)
blocks = wrap_model_with_s3(
    model, svmo_k=128, svd_dir="./svd_factors/",
    svmo_hidden=32, svmo_alpha=0.3,
    nmf_bottleneck=8, nmf_T=1.0, nmf_N=4, stb_beta=0.5,
    enable_svmo=True, enable_stb=True, enable_nmf=True,
)
sd = torch.load(CKPT, map_location="cpu", weights_only=False)
missing, unexpected = model.load_state_dict(sd, strict=False)
print(f"[ode-sweep] checkpoint {CKPT}: unexpected={len(unexpected)}", flush=True)

import torch.nn as nn
reader = ShardReader(find_snapshot(MODEL))
materialize_shared(model, reader, "cpu", "cpu", DEV)
model.lm_head.weight = nn.Parameter(model.lm_head.weight.data.float(), requires_grad=False)
streamer = FrozenStreamer(reader, torch.device(DEV))

cfg = S3TrainConfig(gradient_checkpointing=False, amp=True, amp_dtype=torch.float16, use_nfr=False)
trainer = S3Trainer(model, blocks, cfg, device=DEV, frozen_streamer=streamer)

# Held-out eval set, misma construccion que run_ablations.py (cola de alpaca, nunca entrenada)
ds = load_dataset("tatsu-lab/alpaca", split="train")
ev = ds.select(range(len(ds) - 100, len(ds)))


def enc(ex):
    t = f"### Instruction:\n{ex['instruction']}\n\n### Response:\n{ex.get('output', '')}"
    return tok.encode(t, truncation=True, max_length=256)


pad_id = tok.pad_token_id or 0
evtoks = sorted([ids for ids in (enc(ex) for ex in ev) if len(ids) > 2], key=len)
batch = 4
evb = []
for i in range(0, len(evtoks), batch):
    grp = evtoks[i:i + batch]
    m = max(len(x) for x in grp)
    ids = torch.full((len(grp), m), pad_id, dtype=torch.long)
    attn = torch.zeros((len(grp), m), dtype=torch.long)
    for j, x in enumerate(grp):
        ids[j, :len(x)] = torch.tensor(x)
        attn[j, :len(x)] = 1
    evb.append({"input_ids": ids, "attention_mask": attn})
print(f"[ode-sweep] {len(evb)} batches de eval (batch={batch})", flush=True)

results = {}
for N in N_LIST:
    for blk in blocks:
        for nmf in (getattr(blk, "nmf_attn", None), getattr(blk, "nmf_mlp", None)):
            if nmf is not None:
                nmf.N_steps = N
                nmf.dt = nmf.T / N
    r = eval_perplexity(trainer, evb)
    results[N] = r
    print(f"[ode-sweep] N={N:2d}  eval_loss={r['eval_loss']:.5f}  ppl={r['perplexity']:.4f}", flush=True)

ppls = [results[N]["perplexity"] for N in N_LIST]
spread = max(ppls) - min(ppls)
mean = sum(ppls) / len(ppls)
results["_summary"] = {
    "ppl_min": min(ppls), "ppl_max": max(ppls), "ppl_spread": round(spread, 4),
    "ppl_spread_pct_of_mean": round(100 * spread / mean, 3),
}
print(f"[ode-sweep] spread={spread:.4f} ppl ({100*spread/mean:.3f}% de la media)", flush=True)

os.makedirs("results", exist_ok=True)
with open("results/ode_invariance_sweep.json", "w") as f:
    json.dump(results, f, indent=2)
print("[ode-sweep] guardado en results/ode_invariance_sweep.json", flush=True)
