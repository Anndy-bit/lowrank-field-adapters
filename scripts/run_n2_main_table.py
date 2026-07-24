"""n=2 para la tabla principal (S3 vs LoRA, 7 tareas): reentrena lora_r8 y
s3_full con seed=43 y vuelve a correr el benchmark de 7 tareas sobre esos
checkpoints nuevos. Checkpointed en dos niveles (paso de entrenamiento +
por-tarea en la evaluacion), igual que scripts/run_full_benchmark.py y
scripts/run_seed_sweep.py -- si se cae a mitad de camino, no repite lo ya
hecho.
"""
import glob, json, os, subprocess, sys, time

ROOT = "/home/gula/Documentos/Proyectos Github/lowrank-field-adapters"
os.chdir(ROOT)
sys.path.insert(0, ROOT)

STATE_FILE = "results/n2_main_table_state.json"


def load_state():
    if os.path.exists(STATE_FILE):
        return json.load(open(STATE_FILE))
    return {}


def save_state(s):
    with open(STATE_FILE, "w") as f:
        json.dump(s, f, indent=2)


state = load_state()

# Paso 1: reentrenar baselines a seed=43 (lora_r1, lora_r2, lora_r8, s3_full)
if "train_seed43" not in state:
    print("=== Paso 1: reentrenando baselines seed=43 ===", flush=True)
    env = dict(os.environ)
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    r = subprocess.run(["./.venv/bin/python", "-u", "run_ablations.py",
                         "--suite", "baselines", "--seed", "43"], env=env)
    if r.returncode != 0:
        print(f"[n2] Paso 1 FALLO (returncode={r.returncode})", flush=True)
        sys.exit(1)
    dirs = sorted(glob.glob("results/ablations_baselines_*"))
    ckpt_dir = dirs[-1]
    state["train_seed43"] = {"ckpt_dir": ckpt_dir}
    save_state(state)
    print(f"=== Paso 1 OK, checkpoints en {ckpt_dir} ===", flush=True)
else:
    ckpt_dir = state["train_seed43"]["ckpt_dir"]
    print(f"=== Paso 1 ya hecho ({ckpt_dir}), salto ===", flush=True)

# Paso 2: evaluar 7 tareas para lora_r8 y s3_full (seed43), incremental por tarea
import torch
import torch.nn as nn
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
OUT = "results/full_benchmark_7tasks_seed43.json"

CONFIGS = [
    ("lora_r8", dict(enable_svmo=False, enable_stb=False, enable_nmf=False),
     "lora", f"{ckpt_dir}/lora_r8/adapters.pt"),
    ("s3_full", dict(svmo_k=128, svmo_hidden=32, svmo_alpha=0.3,
                      nmf_bottleneck=8, nmf_T=1.0, nmf_N=4, stb_beta=0.5,
                      svd_dir="./svd_factors/"),
     None, f"{ckpt_dir}/s3_full/adapters.pt"),
]

tok = AutoTokenizer.from_pretrained(MODEL)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token


def build_trainer(wrap_kwargs, lora_r, ckpt_path):
    model, _cfg = build_streaming_model(MODEL)
    blocks = wrap_model_with_s3(model, sbs_p=1.0, **wrap_kwargs)
    if lora_r is not None:
        n = apply_lora_to_blocks(blocks, r=8)
        print(f"[n2] LoRA r=8 -> {n:,} trainable params", flush=True)
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"[n2] checkpoint {ckpt_path}: unexpected={len(unexpected)}", flush=True)

    reader = ShardReader(find_snapshot(MODEL))
    materialize_shared(model, reader, embed_device="cpu", head_device="cpu", norm_device=DEV)
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
print(f"[n2] configs ya completados: {list(all_results.keys())}", flush=True)

for name, wrap_kwargs, lora_r, ckpt in CONFIGS:
    existing = all_results.get(name, {})
    done_tasks = set(existing.get("results", {}).keys())
    pending = [t for t in TASKS if t not in done_tasks]

    if not pending:
        print(f"[n2] {name}: ya en {OUT}, salto", flush=True)
        continue

    print(f"\n{'#' * 70}\n# {name} (seed43)\n{'#' * 70}", flush=True)
    if done_tasks:
        print(f"[n2] {name}: reanudando, ya hechas: {sorted(done_tasks)}, faltan: {pending}", flush=True)

    trainer = build_trainer(wrap_kwargs, lora_r, ckpt)
    lm = USFLM(trainer, tok, chunk_requests=200, batch_size=16, max_length=2048)

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
        print(f"[n2] {name}/{task} guardado en {OUT} ({acc_wall_s/3600:.2f} h acumuladas)", flush=True)

    print(f"[n2] {name} terminado en {acc_wall_s/3600:.2f} h", flush=True)

    del trainer, lm
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

print("\n[n2] TODO TERMINADO", flush=True)
for name, r in all_results.items():
    print(f"\n=== {name} seed43 ({r['wall_hours']:.2f} h) ===")
    for task, metrics in r["results"].items():
        acc = metrics.get("acc,none", metrics.get("acc_norm,none"))
        print(f"  {task:16s} acc={acc}")
