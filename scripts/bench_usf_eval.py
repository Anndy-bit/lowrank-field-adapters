"""Mide USFLM sobre una tarea real, variando cuantas peticiones comparten una
lectura del modelo.

Dos cosas a la vez:
  1. Thm 4 (correccion): trocear distinto NO puede mover las puntuaciones.
  2. Thm 4 (coste): amortizar la lectura sobre mas peticiones debe acelerar.

Si (1) falla, el wrapper esta roto. Si (2) no aparece, mi modelo de coste esta
roto y los benchmarks no caben en una noche.
"""
import sys, time, torch
sys.path.insert(0, "/home/gula/Documentos/Proyectos Github/lowrank-field-adapters")

from transformers import AutoTokenizer
from lm_eval.tasks import TaskManager, get_task_dict
from src.adapters.s3_block import wrap_model_with_s3
from src.training.s3_trainer import S3Trainer, S3TrainConfig
from src.training.streaming_loader import (
    build_streaming_model, ShardReader, FrozenStreamer,
    materialize_shared, find_snapshot, choose_granularity,
)
from src.eval.usf_lm import USFLM

MODEL = "Qwen/Qwen2.5-7B-Instruct"
DEV = "cuda"
TASK = "openbookqa"
LIMIT = 100                      # 100 preguntas x 4 opciones = 400 peticiones

tok = AutoTokenizer.from_pretrained(MODEL)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

CKPT = "results/ablations_baselines_20260716_024551/s3_full/adapters.pt"

print("[bench] montando modelo streaming (svmo_k=128, checkpoint real) ...", flush=True)
model, cfg = build_streaming_model(MODEL)
blocks = wrap_model_with_s3(model, svmo_k=128, svmo_hidden=32, svmo_alpha=0.3,
                            nmf_bottleneck=8, nmf_T=1.0, nmf_N=4, stb_beta=0.5,
                            svd_dir="./svd_factors/", sbs_p=1.0)
sd = torch.load(CKPT, map_location="cpu", weights_only=False)
missing, unexpected = model.load_state_dict(sd, strict=False)
print(f"[bench] checkpoint cargado: unexpected={len(unexpected)}", flush=True)
reader = ShardReader(find_snapshot(MODEL))
materialize_shared(model, reader, embed_device="cpu", head_device="cpu", norm_device=DEV)
import torch.nn as nn
# fp16 en la cabeza: el wrapper solo proyecta las posiciones de la continuacion,
# asi que no necesita el fp32 que eval_loss usa para una CE sobre todo el batch.
model.lm_head.weight = nn.Parameter(model.lm_head.weight.data.half(), requires_grad=False)

free_b = torch.cuda.get_device_properties(0).total_memory
gran = choose_granularity(model.config, int(free_b * 0.75),
                          reserve_bytes=int(0.6 * 2**30), prefetch=False)
streamer = FrozenStreamer(reader, torch.device(DEV), granularity=gran)
tcfg = S3TrainConfig(gradient_checkpointing=True, amp=True, amp_dtype=torch.float16,
                     layer_major=True, ckpt_offload=True,
                     use_nfr=True, use_sbs=True, use_sgc=True)
trainer = S3Trainer(model, blocks, tcfg, device=DEV, frozen_streamer=streamer)

print(f"[bench] cargando {TASK} (limit={LIMIT}) ...", flush=True)
task = get_task_dict([TASK], TaskManager())[TASK]
task.build_all_requests(limit=LIMIT, rank=0, world_size=1)
reqs = [r for r in task.instances if len(r.args) >= 2]
n_tok = sum(len(tok.encode(r.args[0], add_special_tokens=False)) +
            len(tok.encode(r.args[1], add_special_tokens=False)) for r in reqs)
print(f"[bench] {len(reqs)} peticiones, {n_tok:,} tokens\n", flush=True)

print("=" * 72)
print(f"{'chunk':>7}{'lecturas':>10}{'seg':>9}{'tok/s':>9}{'suite 7 tareas':>17}")
print("-" * 72)

results = {}
for chunk in [80, 400]:      # ambos multiplos de batch_size=8: micro-batches
                              # identicos en ambos casos, sin artefacto de borde
    lm = USFLM(trainer, tok, chunk_requests=chunk, batch_size=8)
    torch.cuda.synchronize(); t0 = time.time()
    out = lm.loglikelihood(reqs)
    torch.cuda.synchronize(); dt = time.time() - t0
    reads = (len(reqs) + chunk - 1) // chunk
    tps = n_tok / dt
    print(f"{chunk:>7}{reads:>10}{dt:>9.1f}{tps:>9.1f}{460263/tps/3600:>15.1f} h", flush=True)
    results[chunk] = out

print("=" * 72)
a, b = results[80], results[400]
worst = max(abs(x[0] - y[0]) for x, y in zip(a, b))
print(f"Thm 4 (correccion): max|Delta logprob| entre trocear 5x vs 1x (mismo batch_size) = {worst:.3e}")
print(f"VRAM pico: {torch.cuda.max_memory_allocated()/2**20:.0f} MB")
