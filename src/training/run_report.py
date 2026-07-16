"""
Full experiment-report instrumentation for S³ training.

Captures — during and after a run — everything a paper needs (grounded in
docs/formalismo.md §5): provenance, training dynamics, memory (RQ1), throughput,
per-component and per-layer gradient norms (Thm 2 / GNS), spectral modulation
profiles (Thm 1), and before/after quality (perplexity). Emits raw CSV/JSON so
any plot can be regenerated, PNG figures (matplotlib, optional), and a
human-readable report.md printed at the end of `make train`.

Nothing here is load-bearing for correctness — it only observes.
"""

import csv
import json
import os
import platform
import subprocess
import threading
import time
from datetime import datetime, timezone

import torch


# --------------------------------------------------------------------------- #
# Provenance
# --------------------------------------------------------------------------- #
def _pkg_version(name):
    try:
        return __import__(name).__version__
    except Exception:
        return "n/a"


def _git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL,
            cwd=os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        ).decode().strip()
    except Exception:
        return "n/a"


def collect_provenance(config, model, blocks, device, dataset_info):
    import psutil
    gpu = {}
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        gpu = {"name": p.name, "total_vram_gb": round(p.total_memory / 1024**3, 2),
                "driver_cuda": torch.version.cuda}

    trainable = [p for p in model.parameters() if p.requires_grad]
    def count(pred):
        return sum(p.numel() for n, p in model.named_parameters()
                   if p.requires_grad and pred(n))
    comp = {
        "svmo": count(lambda n: "modulation" in n),
        "nmf": count(lambda n: "nmf" in n or "field" in n),
        "stb": count(lambda n: "stb" in n or ".W_" in n),
        "lora": count(lambda n: "lora_" in n),
    }
    total = sum(p.numel() for p in model.parameters())
    n_train = sum(p.numel() for p in trainable)

    cfg = {k: (str(v) if isinstance(v, torch.dtype) else v)
           for k, v in vars(config).items()}
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "hardware": {
            "gpu": gpu,
            "cpu_count": os.cpu_count(),
            "ram_total_gb": round(psutil.virtual_memory().total / 1024**3, 1),
            "platform": platform.platform(),
        },
        "versions": {lib: _pkg_version(lib) for lib in
                     ["torch", "transformers", "accelerate", "safetensors",
                      "bitsandbytes", "datasets", "numpy"]},
        "python": platform.python_version(),
        "train_config": cfg,
        "params": {
            "total": total, "trainable": n_train,
            "trainable_pct": round(100 * n_train / total, 4),
            "by_component": comp,
        },
        "dataset": dataset_info,
    }


# --------------------------------------------------------------------------- #
# Live VRAM / GPU poller (RQ1 trace)
# --------------------------------------------------------------------------- #
class VramPoller(threading.Thread):
    def __init__(self, path, device, interval=0.1):
        super().__init__(daemon=True)
        self.path, self.device, self.interval = path, device, interval
        self._stop = threading.Event()
        self.t0 = time.time()

    def _smi(self):
        try:
            out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=memory.used,utilization.gpu,temperature.gpu",
                 "--format=csv,noheader,nounits"], stderr=subprocess.DEVNULL).decode().strip()
            used, util, temp = [x.strip() for x in out.split(",")]
            return used, util, temp
        except Exception:
            return "", "", ""

    def run(self):
        import psutil
        with open(self.path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["t_s", "vram_alloc_mb", "vram_reserved_mb",
                        "smi_used_mb", "gpu_util_pct", "gpu_temp_c", "cpu_ram_mb"])
            while not self._stop.is_set():
                t = round(time.time() - self.t0, 3)
                if self.device.type == "cuda":
                    alloc = torch.cuda.memory_allocated(self.device) / 1024**2
                    resv = torch.cuda.memory_reserved(self.device) / 1024**2
                else:
                    alloc = resv = 0
                used, util, temp = self._smi()
                cpu_ram = psutil.Process().memory_info().rss / 1024**2
                w.writerow([t, round(alloc, 1), round(resv, 1), used, util, temp, round(cpu_ram, 1)])
                f.flush()
                self._stop.wait(self.interval)

    def stop(self):
        self._stop.set()


# --------------------------------------------------------------------------- #
# Per-step logger
# --------------------------------------------------------------------------- #
STEP_FIELDS = [
    "wall_s", "epoch", "opt_step", "batch_idx", "seq_len", "tokens_cum",
    "loss", "perplexity", "lr", "grad_total", "grad_svmo", "grad_nmf", "grad_stb", "grad_lora",
    "scaler_scale", "skipped", "fwd_ms", "bwd_ms", "step_ms",
    "vram_alloc_mb", "vram_peak_mb", "cpu_ram_mb", "disk_read_mb",
]


class RunLogger:
    def __init__(self, run_dir, config, model, blocks, device, dataset_info):
        self.dir = run_dir
        os.makedirs(run_dir, exist_ok=True)
        os.makedirs(os.path.join(run_dir, "figures"), exist_ok=True)
        self.device = device
        self.t0 = time.time()

        self.provenance = collect_provenance(config, model, blocks, device, dataset_info)
        with open(os.path.join(run_dir, "provenance.json"), "w") as f:
            json.dump(self.provenance, f, indent=2)

        # grad groups (component + per layer) for Thm-2 / GNS analysis
        self.groups = {"svmo": [], "nmf": [], "stb": [], "lora": []}
        self.layer_params = {}
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if "lora_" in name:                       # LoRA baseline
                self.groups["lora"].append(p)
            elif "modulation" in name:
                self.groups["svmo"].append(p)
            elif "nmf" in name or "field" in name:
                self.groups["nmf"].append(p)
            elif "stb" in name or ".W_" in name:
                self.groups["stb"].append(p)
        for i, blk in enumerate(blocks):
            self.layer_params[i] = [p for p in blk.parameters() if p.requires_grad]
        self.blocks = blocks

        self._steps_f = open(os.path.join(run_dir, "steps.csv"), "w", newline="")
        self._steps = csv.DictWriter(self._steps_f, fieldnames=STEP_FIELDS)
        self._steps.writeheader()
        self._layerg_f = open(os.path.join(run_dir, "layer_grads.csv"), "w", newline="")
        self._layerg = csv.writer(self._layerg_f)
        self._layerg.writerow(["opt_step", "layer", "grad_norm"])

        self.poller = VramPoller(os.path.join(run_dir, "vram_trace.csv"), device)
        self.poller.start()
        self.tokens_cum = 0

    @staticmethod
    def _gnorm(params):
        s = 0.0
        for p in params:
            if p.grad is not None:
                s += float(p.grad.detach().float().pow(2).sum())
        return s ** 0.5

    def grad_components(self):
        return {k: self._gnorm(v) for k, v in self.groups.items()}

    def log_layer_grads(self, opt_step):
        for i, params in self.layer_params.items():
            self._layerg.writerow([opt_step, i, round(self._gnorm(params), 6)])
        self._layerg_f.flush()

    def log_step(self, row):
        full = {k: row.get(k, "") for k in STEP_FIELDS}
        self._steps.writerow(full)
        self._steps_f.flush()

    def snapshot_modulation(self):
        """m_θ(σ)/σ per singular value per projection (Thm 1 interpretability)."""
        path = os.path.join(self.dir, "modulation.csv")
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["layer", "proj", "idx", "sigma", "m_sigma", "ratio"])
            for i, blk in enumerate(self.blocks):
                attn = getattr(blk, "self_attn", None)
                mods = []
                if attn is not None:
                    for pn in ("q_proj", "k_proj", "v_proj", "o_proj"):
                        m = getattr(attn, pn, None)
                        if hasattr(m, "modulated_sigma"):
                            mods.append((pn, m))
                for pn, m in mods:
                    with torch.no_grad():
                        sig = m.S_k.float().cpu()
                        msig = m.modulated_sigma().float().cpu()
                    for j in range(0, len(sig), max(1, len(sig) // 32)):  # subsample
                        s = float(sig[j])
                        w.writerow([i, pn, j, round(s, 6), round(float(msig[j]), 6),
                                    round(float(msig[j]) / s if s else 1.0, 6)])

    def finalize(self, summary):
        self.snapshot_modulation()
        with open(os.path.join(self.dir, "summary.json"), "w") as f:
            json.dump(summary, f, indent=2)
        self.poller.stop()
        self._steps_f.close()
        self._layerg_f.close()


# --------------------------------------------------------------------------- #
# Quality eval (offline: perplexity on held-out batches)
# --------------------------------------------------------------------------- #
@torch.no_grad()
def eval_perplexity(trainer, eval_batches):
    trainer.model.eval()
    losses = []
    for ids in eval_batches:
        losses.append(float(trainer.eval_loss(ids)))
    trainer.model.train()
    import math
    mean = sum(losses) / max(len(losses), 1)
    return {"eval_loss": round(mean, 5), "perplexity": round(math.exp(mean), 4),
            "n_batches": len(losses)}


# --------------------------------------------------------------------------- #
# Report generation (plots + report.md)
# --------------------------------------------------------------------------- #
def _read_csv(path):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return list(csv.DictReader(f))


def _to_float(rows, key):
    out = []
    for r in rows:
        try:
            out.append(float(r[key]))
        except (ValueError, KeyError, TypeError):
            out.append(float("nan"))
    return out


def generate_report(run_dir):
    prov = json.load(open(os.path.join(run_dir, "provenance.json")))
    summary = json.load(open(os.path.join(run_dir, "summary.json"))) \
        if os.path.exists(os.path.join(run_dir, "summary.json")) else {}
    steps = _read_csv(os.path.join(run_dir, "steps.csv"))
    vram = _read_csv(os.path.join(run_dir, "vram_trace.csv"))
    layerg = _read_csv(os.path.join(run_dir, "layer_grads.csv"))
    mod = _read_csv(os.path.join(run_dir, "modulation.csv"))
    figdir = os.path.join(run_dir, "figures")

    figs = _make_figures(steps, vram, layerg, mod, figdir)
    _write_markdown(run_dir, prov, summary, steps, vram, figs)
    return os.path.join(run_dir, "report.md")


def _make_figures(steps, vram, layerg, mod, figdir):
    figs = {}
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return figs

    def save(name):
        p = os.path.join(figdir, name)
        plt.tight_layout(); plt.savefig(p, dpi=110); plt.close()
        figs[name] = p

    if steps:
        x = _to_float(steps, "opt_step")
        # loss + perplexity
        fig, ax1 = plt.subplots(figsize=(7, 4))
        ax1.plot(x, _to_float(steps, "loss"), color="tab:blue", label="loss")
        ax1.set_xlabel("optimizer step"); ax1.set_ylabel("loss", color="tab:blue")
        ax2 = ax1.twinx()
        ax2.plot(x, _to_float(steps, "perplexity"), color="tab:orange", alpha=0.6, label="perplexity")
        ax2.set_ylabel("perplexity", color="tab:orange")
        plt.title("Training loss / perplexity"); save("loss_curve.png")

        # learning rate
        plt.figure(figsize=(7, 3)); plt.plot(x, _to_float(steps, "lr"))
        plt.xlabel("optimizer step"); plt.ylabel("lr"); plt.title("LR schedule"); save("lr.png")

        # gradient norms per component
        plt.figure(figsize=(7, 4))
        for k, c in [("grad_total", "black"), ("grad_svmo", "tab:blue"),
                     ("grad_nmf", "tab:green"), ("grad_stb", "tab:red")]:
            plt.plot(x, _to_float(steps, k), label=k.replace("grad_", ""), color=c, alpha=0.8)
        plt.legend(); plt.xlabel("optimizer step"); plt.ylabel("grad norm")
        plt.title("Gradient norm — total & per operator (Thm 2)"); save("grad_norms.png")

        # step time breakdown
        plt.figure(figsize=(7, 3.5))
        plt.plot(x, _to_float(steps, "fwd_ms"), label="forward")
        plt.plot(x, _to_float(steps, "bwd_ms"), label="backward")
        plt.legend(); plt.xlabel("optimizer step"); plt.ylabel("ms")
        plt.title("Per-step time (streaming fwd/bwd)"); save("step_time.png")

    if vram:
        t = _to_float(vram, "t_s")
        plt.figure(figsize=(8, 4))
        plt.plot(t, _to_float(vram, "vram_alloc_mb"), label="allocated")
        plt.plot(t, _to_float(vram, "vram_reserved_mb"), label="reserved", alpha=0.6)
        smi = _to_float(vram, "smi_used_mb")
        if any(v == v for v in smi):
            plt.plot(t, smi, label="nvidia-smi used", alpha=0.5)
        plt.legend(); plt.xlabel("time (s)"); plt.ylabel("VRAM (MB)")
        plt.title("VRAM trace (RQ1)"); save("vram_trace.png")

    if layerg:
        last = max(_to_float(layerg, "opt_step"))
        rows = [r for r in layerg if float(r["opt_step"]) == last]
        if rows:
            plt.figure(figsize=(8, 3.5))
            plt.bar([int(r["layer"]) for r in rows], [float(r["grad_norm"]) for r in rows])
            plt.xlabel("layer"); plt.ylabel("grad norm"); plt.title("Per-layer grad norm (GNS)")
            save("layer_grads.png")

    if mod:
        plt.figure(figsize=(7, 4))
        sig = _to_float(mod, "sigma"); ratio = _to_float(mod, "ratio")
        plt.scatter(sig, ratio, s=6, alpha=0.4)
        plt.axhline(1.0, color="gray", ls="--", lw=0.8)
        plt.xlabel("singular value σ"); plt.ylabel("m_θ(σ)/σ")
        plt.title("Spectral modulation profile (Thm 1)"); save("modulation.png")

    return figs


def _write_markdown(run_dir, prov, summary, steps, vram, figs):
    import numpy as np
    L = []
    L.append("# S³ Training Report\n")
    L.append(f"_generated {datetime.now(timezone.utc).isoformat()}_\n")

    # headline
    hw = prov["hardware"]["gpu"].get("name", "?")
    pk = summary.get("vram_peak_mb", "?")
    L.append("## Headline\n")
    L.append(f"- **Model**: {prov['dataset'].get('model','?')}")
    L.append(f"- **GPU**: {hw} ({prov['hardware']['gpu'].get('total_vram_gb','?')} GB)")
    L.append(f"- **Trainable params**: {prov['params']['trainable']:,} "
             f"({prov['params']['trainable_pct']}% of {prov['params']['total']:,})")
    L.append(f"- **VRAM peak**: {pk} MB")
    L.append(f"- **Throughput**: {summary.get('tokens_per_s','?')} tok/s")
    L.append(f"- **Total wall-clock**: {summary.get('total_wall_min','?')} min\n")

    # quality
    q = summary.get("quality", {})
    if q:
        L.append("## Quality (RQ2 gate — did it learn?)\n")
        L.append("| metric | base (init) | after | Δ |")
        L.append("|---|---|---|---|")
        b, a = q.get("base", {}), q.get("after", {})
        for m in ("eval_loss", "perplexity"):
            bv, av = b.get(m), a.get(m)
            d = round(av - bv, 4) if (isinstance(av, (int, float)) and isinstance(bv, (int, float))) else "?"
            L.append(f"| {m} | {bv} | {av} | {d} |")
        L.append("")

    # training dynamics
    if steps:
        loss = np.array(_to_float(steps, "loss"))
        L.append("## Training dynamics\n")
        L.append(f"- optimizer steps: {len(steps)}")
        L.append(f"- loss: first {loss[0]:.4f} → last {loss[-1]:.4f} (min {np.nanmin(loss):.4f})")
        sk = sum(1 for r in steps if str(r.get('skipped')).lower() in ('true', '1'))
        L.append(f"- fp16 steps skipped by GradScaler (overflow warmup): {sk}")
        for comp in ("grad_svmo", "grad_nmf", "grad_stb"):
            v = np.array(_to_float(steps, comp))
            L.append(f"- mean {comp}: {np.nanmean(v):.4g}")
        L.append("")

    # memory
    if vram:
        alloc = np.array(_to_float(vram, "vram_alloc_mb"))
        cpu = np.array(_to_float(vram, "cpu_ram_mb"))
        L.append("## Memory (RQ1)\n")
        L.append(f"- VRAM allocated: peak {np.nanmax(alloc):.0f} MB, "
                 f"median {np.nanmedian(alloc):.0f} MB")
        L.append(f"- CPU RAM: peak {np.nanmax(cpu):.0f} MB")
        L.append(f"- samples: {len(vram)} @ ~100ms\n")

    # provenance
    L.append("## Provenance & config\n")
    L.append("```json")
    L.append(json.dumps({"versions": prov["versions"], "hardware": prov["hardware"],
                         "params": prov["params"], "git": prov["git_commit"]}, indent=2))
    L.append("```\n")

    # figures
    if figs:
        L.append("## Figures\n")
        for name in figs:
            L.append(f"### {name}")
            L.append(f"![{name}](figures/{name})\n")

    # data files
    L.append("## Raw data (regenerate any plot)\n")
    for fn in ("steps.csv", "vram_trace.csv", "layer_grads.csv", "modulation.csv",
               "provenance.json", "summary.json"):
        if os.path.exists(os.path.join(run_dir, fn)):
            L.append(f"- `{fn}`")
    L.append("")

    # what's still needed for the paper
    L.append("## Still needed for the paper (not auto-collected here)\n")
    L.append("- Downstream benchmarks: MMLU / HellaSwag / ARC / GSM8K / AlpacaEval "
             "(need datasets cached; run `make evaluate`).")
    L.append("- Baselines: LoRA r=8/r=64, QLoRA, full-FT (same seeds/eval).")
    L.append("- Ablations: −STB / −NMF / −SVMO and single-operator runs.")
    L.append("- n≥3 seeds → mean ± std / 95% CI.")

    with open(os.path.join(run_dir, "report.md"), "w") as f:
        f.write("\n".join(L))
