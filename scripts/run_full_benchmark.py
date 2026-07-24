"""Corrida completa: 7 tareas x {base, LoRA r8, S3 full} via USFLM (layer-major).

Usa el harness real de lm_eval (simple_evaluate) para que precision/agregacion
sean las estandar de la literatura -- no reinventamos el calculo de accuracy.

Orden deliberado: base y LoRA (sin NMF, deberian ser rapidos) primero; S3 (con
el ODE de NMF, medido a ~7 tok/s) al final, porque es el que tarda ~18-19h.
Resultados se guardan incrementalmente: si algo se cae a medio camino, lo ya
corrido no se pierde.
"""
import sys, os, json, time, torch
sys.path.insert(0, "/home/gula/Documentos/Proyectos Github/lowrank-field-adapters")

# Pure engineering, zero effect on results: every score is a fixed function of
# its own tokens (verified: Thm 4 chunking-invariance test gave delta=0), so
# letting cuDNN autotune kernels for the repeated micro-batch shapes only
# changes wall time, never a number in the output. Portable -- no path/disk
# assumption, works on any CUDA GPU.
torch.backends.cudnn.benchmark = True

from transformers import AutoTokenizer
from lm_eval import simple_evaluate
from src.adapters.s3_block import wrap_model_with_s3
from src.adapters.lora_linear import apply_lora_to_blocks
from src.training.s3_trainer import S3Trainer, S3TrainConfig
from src.training.streaming_loader import (
    build_streaming_model, ShardReader, FrozenStreamer,
    materialize_shared, find_snapshot, choose_granularity,
)
from src.eval.usf_lm import USFLM

MODEL = "Qwen/Qwen2.5-7B-Instruct"
DEV = "cuda"
TASKS = ["boolq", "piqa", "hellaswag", "winogrande",
         "arc_easy", "arc_challenge", "openbookqa"]
LIMIT = 500
CKPT_DIR = "results/ablations_baselines_20260716_024551"
OUT = "results/full_benchmark_7tasks.json"

CONFIGS = [
    ("base",   dict(enable_svmo=False, enable_stb=False, enable_nmf=False, svd_dir=None),
     None, None),
    ("lora_r8", dict(enable_svmo=False, enable_stb=False, enable_nmf=False, svd_dir=None),
     "lora", f"{CKPT_DIR}/lora_r8/adapters.pt"),
    ("s3_full", dict(svmo_k=128, svmo_hidden=32, svmo_alpha=0.3,
                      nmf_bottleneck=8, nmf_T=1.0, nmf_N=4, stb_beta=0.5,
                      svd_dir="./svd_factors/"),
     None, f"{CKPT_DIR}/s3_full/adapters.pt"),
]

tok = AutoTokenizer.from_pretrained(MODEL)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token


def build_trainer(wrap_kwargs, lora_r, ckpt_path):
    model, _cfg = build_streaming_model(MODEL)
    blocks = wrap_model_with_s3(model, sbs_p=1.0, **wrap_kwargs)
    if lora_r is not None:
        n = apply_lora_to_blocks(blocks, r=8)
        print(f"[bench] LoRA r=8 -> {n:,} trainable params", flush=True)
    if ckpt_path is not None:
        sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        missing, unexpected = model.load_state_dict(sd, strict=False)
        print(f"[bench] checkpoint {ckpt_path}: unexpected={len(unexpected)}", flush=True)

    reader = ShardReader(find_snapshot(MODEL))
    materialize_shared(model, reader, embed_device="cpu", head_device="cpu", norm_device=DEV)
    import torch.nn as nn
    model.lm_head.weight = nn.Parameter(model.lm_head.weight.data.half(), requires_grad=False)

    free_b = torch.cuda.get_device_properties(0).total_memory
    gran = choose_granularity(model.config, int(free_b * 0.75),
                              reserve_bytes=int(0.6 * 2**30), prefetch=False)
    streamer = FrozenStreamer(reader, torch.device(DEV), granularity=gran)
    tcfg = S3TrainConfig(gradient_checkpointing=True, amp=True, amp_dtype=torch.float16,
                         layer_major=True, ckpt_offload=True,
                         use_nfr=True, use_sbs=True, use_sgc=True)
    return S3Trainer(model, blocks, tcfg, device=DEV, frozen_streamer=streamer)


def load_existing():
    if os.path.exists(OUT):
        with open(OUT) as f:
            return json.load(f)
    return {}


def save(all_results):
    with open(OUT, "w") as f:
        json.dump(all_results, f, indent=2, default=str)


all_results = load_existing()
print(f"[bench] configs ya completados: {list(all_results.keys())}", flush=True)

for name, wrap_kwargs, lora_r, ckpt in CONFIGS:
    existing = all_results.get(name, {})
    done_tasks = set(existing.get("results", {}).keys())
    pending = [t for t in TASKS if t not in done_tasks]

    if not pending:
        print(f"[bench] {name}: ya en {OUT}, salto", flush=True)
        continue

    print(f"\n{'#'*70}\n# {name}\n{'#'*70}", flush=True)
    if done_tasks:
        print(f"[bench] {name}: reanudando, ya hechas: {sorted(done_tasks)}, "
              f"faltan: {pending}", flush=True)

    trainer = build_trainer(wrap_kwargs, lora_r, ckpt)
    # chunk_requests=200 (not 100000): s3_full OOM'd on its very first task with the
    # full-task chunk size -- SVMO+STB+NMF all resident leaves little headroom, and
    # bundling ~500-2000 requests into one forward_layer_major call held too many
    # micro-batches' activations at once. S3 is compute-bound (Sec. cost model), so
    # more frequent disk reads cost little; this trades a negligible amount of I/O
    # amortization for real OOM margin.
    lm = USFLM(trainer, tok, chunk_requests=200, batch_size=16, max_length=2048)

    # Un simple_evaluate por tarea (no la lista completa de TASKS) para que un
    # crash a mitad de config solo pierda la tarea en curso, no las ya corridas:
    # guardamos en OUT despues de cada tarea individual.
    acc_results = dict(existing.get("results", {}))
    acc_wall_s = existing.get("wall_hours", 0.0) * 3600
    for task in pending:
        t0 = time.time()
        res = simple_evaluate(
            model=lm, tasks=[task], limit=LIMIT,
            log_samples=False, verbosity="WARNING",
            bootstrap_iters=1000,
        )
        acc_wall_s += time.time() - t0
        acc_results.update(res["results"])

        all_results[name] = {
            "results": acc_results,
            "wall_hours": acc_wall_s / 3600,
            "vram_peak_mb": torch.cuda.max_memory_allocated() / 2**20,
        }
        save(all_results)
        print(f"[bench] {name}/{task} guardado en {OUT} "
              f"({acc_wall_s/3600:.2f} h acumuladas)", flush=True)

    print(f"[bench] {name} terminado en {acc_wall_s/3600:.2f} h", flush=True)

    del trainer, lm
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

print("\n[bench] TODO TERMINADO", flush=True)
for name, r in all_results.items():
    print(f"\n=== {name} ({r['wall_hours']:.2f} h) ===")
    for task, metrics in r["results"].items():
        acc = metrics.get("acc,none", metrics.get("acc_norm,none"))
        print(f"  {task:16s} acc={acc}")
