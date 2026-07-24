"""Cuenta los tokens reales que exige cada benchmark. CPU puro, sin GPU.

El techo de cómputo del hardware es fijo: Phi=0.7 TFLOP/s medidos, y un forward
sobre 7.62B params cuesta 2N FLOPs/token. Todo el presupuesto de tiempo es, por
tanto, (tokens / techo). Contar los tokens es contar las horas.
"""
import os, sys



from transformers import AutoTokenizer
from lm_eval.tasks import TaskManager, get_task_dict

MODEL = "Qwen/Qwen2.5-7B-Instruct"
N_PARAMS = 7.62e9
PHI = 0.7e12                       # FLOP/s efectivos medidos en la GTX 1050
TOK_PER_S = PHI / (2 * N_PARAMS)   # forward = 2N FLOPs/token

TASKS = ["boolq", "piqa", "social_iqa", "hellaswag",
         "winogrande", "arc_easy", "arc_challenge", "openbookqa"]
LIMIT = int(sys.argv[1]) if len(sys.argv) > 1 else 500

tok = AutoTokenizer.from_pretrained(MODEL)
tm = TaskManager()

print(f"Techo de cómputo: {TOK_PER_S:.1f} tokens/s (Phi=0.7 TFLOP/s, 2N FLOPs/token)")
print(f"Límite por tarea: {LIMIT} preguntas\n")
print(f"{'tarea':<16}{'reqs':>7}{'tokens':>12}{'ctx único':>12}{'horas':>8}{'h·caché':>9}")
print("-" * 64)

tot_tokens = tot_shared = tot_reqs = 0
for name in TASKS:
    try:
        td = get_task_dict([name], tm)
        task = td[name] if name in td else list(td.values())[0]
        task.build_all_requests(limit=LIMIT, rank=0, world_size=1)
        reqs = task.instances

        n_tok = 0
        ctx_seen = {}
        for r in reqs:
            if len(r.args) < 2:
                continue
            ctx, cont = r.args[0], r.args[1]
            c_ids = tok(ctx, add_special_tokens=False)["input_ids"]
            k_ids = tok(cont, add_special_tokens=False)["input_ids"]
            n_tok += len(c_ids) + len(k_ids)
            # Con caché de contexto: el ctx se paga UNA vez por grupo de opciones.
            if ctx not in ctx_seen:
                ctx_seen[ctx] = len(c_ids)
            ctx_seen[ctx] = ctx_seen[ctx]  # ya contado
            n_shared_add = len(k_ids)
            ctx_seen.setdefault("__conts__", 0)
            ctx_seen["__conts__"] += n_shared_add

        shared = sum(v for k, v in ctx_seen.items() if k != "__conts__") + ctx_seen.get("__conts__", 0)
        h = n_tok / TOK_PER_S / 3600
        h_s = shared / TOK_PER_S / 3600
        print(f"{name:<16}{len(reqs):>7}{n_tok:>12,}{shared:>12,}{h:>8.2f}{h_s:>9.2f}")
        tot_tokens += n_tok; tot_shared += shared; tot_reqs += len(reqs)
    except Exception as e:
        print(f"{name:<16}  ERROR: {type(e).__name__}: {str(e)[:60]}")

print("-" * 64)
print(f"{'TOTAL':<16}{tot_reqs:>7}{tot_tokens:>12,}{tot_shared:>12,}"
      f"{tot_tokens/TOK_PER_S/3600:>8.2f}{tot_shared/TOK_PER_S/3600:>9.2f}")
print(f"\nPor adapter: {tot_tokens/TOK_PER_S/3600:.1f} h sin caché de contexto, "
      f"{tot_shared/TOK_PER_S/3600:.1f} h con caché.")
