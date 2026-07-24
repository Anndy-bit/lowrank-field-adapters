"""Orquesta corridas de run_ablations.py para llegar a n>=3 semillas por vía
(significancia estadistica, docs/estado_benchmarks.md #5) y para medir los
dos puntos de ablation que faltaban (docs/mejoras_futuras.md #4).

Resumible: si un (suite, seed, filtro) ya corrio (existe su directorio de
salida con comparison.md), lo salta. Si un job falla, se detiene la
secuencia sin perder los jobs ya completados (quedan sus resultados en disco).
"""
import json, os, subprocess, sys, time

ROOT = "/home/gula/Documentos/Proyectos Github/lowrank-field-adapters"
os.chdir(ROOT)

# (suite, seed, filtro de --ablations o None = toda la suite)
JOBS = [
    ("capacity", 42, "nmf_b3,no_nmf_h64"),  # puntos nuevos, misma semilla que los ya medidos
    ("operators", 43, None),
    ("capacity", 43, None),
    ("baselines", 43, None),
    ("operators", 44, None),
    ("capacity", 44, None),
    ("baselines", 44, None),
]

STATE_FILE = "results/seed_sweep_state.json"


def load_state():
    if os.path.exists(STATE_FILE):
        return json.load(open(STATE_FILE))
    return {"done": {}}


def save_state(state):
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


state = load_state()

for suite, seed, filt in JOBS:
    key = f"{suite}_seed{seed}_{filt or 'all'}"
    if key in state["done"]:
        print(f"[seed-sweep] {key}: ya en {STATE_FILE}, salto", flush=True)
        continue

    print(f"\n{'=' * 70}\n[seed-sweep] LANZANDO {key}\n{'=' * 70}", flush=True)
    cmd = [
        "./.venv/bin/python", "-u", "run_ablations.py",
        "--suite", suite, "--seed", str(seed),
    ]
    if filt:
        cmd += ["--ablations", filt]

    env = dict(os.environ)
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    t0 = time.time()
    r = subprocess.run(cmd, env=env)
    dt = (time.time() - t0) / 60

    if r.returncode != 0:
        print(f"[seed-sweep] {key} FALLO (returncode={r.returncode}) tras {dt:.1f} min "
              f"-- deteniendo la secuencia, lo ya hecho queda guardado", flush=True)
        sys.exit(1)

    print(f"[seed-sweep] {key} OK en {dt:.1f} min", flush=True)
    state["done"][key] = {"wall_min": round(dt, 1), "ts": time.strftime("%Y-%m-%dT%H:%M:%S")}
    save_state(state)

print("\n[seed-sweep] TODO TERMINADO", flush=True)
