#!/usr/bin/env python3
"""Regenerate the paper's data figures as vector PDFs, straight from the run CSVs.

Every figure in main.tex/main_es.tex that carries measured data is produced here, so
the paper's plots are traceable to the run that made them and can be rebuilt from
scratch. The schedule diagram is TikZ and lives in the .tex itself.

    python3 paper/usf/make_figures.py

Both language variants are written on every run: labels in English under figures/,
in Spanish under figures/es/, so main_es.tex is not a Spanish paper with English
axes. Titles are deliberately omitted: IEEE puts that text in the caption.
"""

import os
import sys

# Axis text, per language. Numbers and units stay identical across variants; only
# the words change, so the two papers cannot drift apart on the data.
STR = {
    "en": {
        "time_h": "time (h)", "device_mem": "device memory (MB)",
        "capacity": "card capacity 4034 MB", "smi": "nvidia-smi",
        "reserved": "reserved", "allocated": "allocated",
        "opt_step": "optimizer step", "train_loss": "training loss",
        "per_step": "per optimizer step", "mean9": "9-step mean",
        "ppl_before": "held-out ppl 11.96", "ppl_after": "held-out ppl 5.50",
    },
    "es": {
        "time_h": "tiempo (h)", "device_mem": "memoria del dispositivo (MB)",
        "capacity": "capacidad de la tarjeta 4034 MB", "smi": "nvidia-smi",
        "reserved": "reservada", "allocated": "asignada",
        "opt_step": "paso de optimizador", "train_loss": "pérdida de entrenamiento",
        "per_step": "por paso", "mean9": "media de 9 pasos",
        "ppl_before": "ppl reservada 11,96", "ppl_after": "ppl reservada 5,50",
    },
}

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
RUN = os.path.join(ROOT, "results", "run_20260714_225107")
OUT = os.path.join(HERE, "figures")

# IEEE single column is 3.5in wide; keep type at ~8pt so it matches the body text
# after LaTeX includes the PDF at \columnwidth.
COL_W = 3.5
matplotlib.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 8,
    "legend.fontsize": 7,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "axes.linewidth": 0.6,
    "lines.linewidth": 0.8,
    "grid.linewidth": 0.4,
    "figure.dpi": 150,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})

CARD_MB = 4034.0  # CUDA-visible capacity of the GTX 1050 (3.94 GiB)


def fig_vram_trace(lang, out):
    """The device-memory trace. The sawtooth *is* the Load/Free cycle."""
    T = STR[lang]
    df = pd.read_csv(os.path.join(RUN, "vram_trace.csv"))
    h = df["t_s"] / 3600.0

    fig, ax = plt.subplots(figsize=(COL_W, 2.05))
    ax.plot(h, df["smi_used_mb"], color="#1b7837", label=T["smi"])
    ax.plot(h, df["vram_reserved_mb"], color="#e08214", label=T["reserved"])
    ax.plot(h, df["vram_alloc_mb"], color="#2166ac", label=T["allocated"], alpha=0.85)
    ax.axhline(CARD_MB, color="#b2182b", ls="--", lw=0.9)
    ax.text(h.max() * 0.99, CARD_MB - 70, T["capacity"],
            ha="right", va="top", fontsize=6.5, color="#b2182b")

    ax.set_xlabel(T["time_h"])
    ax.set_ylabel(T["device_mem"])
    ax.set_ylim(0, CARD_MB * 1.09)
    ax.set_xlim(0, h.max())
    # Legend above the axes: the plot area is full and the capacity line would
    # otherwise strike through the labels.
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, frameon=False,
              handlelength=1.5, columnspacing=1.4, borderaxespad=0.15)
    ax.grid(alpha=0.25, ls=":")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    p = os.path.join(out, "vram_trace.pdf")
    fig.savefig(p)
    plt.close(fig)
    print(f"  [{lang}] vram_trace -> {os.path.relpath(p, ROOT)}  "
          f"({len(df):,} samples, peak alloc {df['vram_alloc_mb'].max():.0f} MB, "
          f"peak smi {df['smi_used_mb'].max():.0f} MB)")


def fig_loss_curve(lang, out):
    """The learning trajectory. x is optimizer steps, NOT micro-batches.

    We plot training loss only. Logged perplexity is exp(loss) to within float32
    rounding (max relative deviation 2.4e-5), so a second perplexity axis would
    replot the same curve. The held-out evaluations are the independent numbers,
    so those are what we annotate.
    """
    T = STR[lang]
    df = pd.read_csv(os.path.join(RUN, "steps.csv"))
    # One row per micro-batch; the optimizer steps once per grad-accum group.
    per_step = df.groupby("opt_step").agg(loss=("loss", "mean")).reset_index()
    x, y = per_step["opt_step"], per_step["loss"]

    fig, ax = plt.subplots(figsize=(COL_W, 2.05))
    ax.plot(x, y, color="#6baed6", lw=0.6, alpha=0.9, label=T["per_step"])
    ax.plot(x, y.rolling(9, center=True, min_periods=1).mean(),
            color="#08306b", lw=1.3, label=T["mean9"])

    # The held-out numbers: measured before and after, on 100 unseen examples.
    ax.annotate(T["ppl_before"], xy=(0, y.iloc[0]), xytext=(9, 4.55),
                fontsize=6.5, color="#b2182b",
                arrowprops=dict(arrowstyle="->", color="#b2182b", lw=0.6))
    ax.annotate(T["ppl_after"], xy=(x.iloc[-1], y.iloc[-1]), xytext=(84, 2.75),
                fontsize=6.5, color="#b2182b",
                arrowprops=dict(arrowstyle="->", color="#b2182b", lw=0.6))

    ax.set_xlabel(T["opt_step"])
    ax.set_ylabel(T["train_loss"])
    ax.set_xlim(0, x.max())
    ax.set_ylim(0.9, 5.3)
    ax.legend(loc="upper right", frameon=False, handlelength=1.5)
    ax.grid(alpha=0.25, ls=":")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    p = os.path.join(out, "loss_curve.pdf")
    fig.savefig(p)
    plt.close(fig)
    print(f"  [{lang}] loss_curve -> {os.path.relpath(p, ROOT)}  "
          f"({len(per_step)} optimizer steps from {len(df)} micro-batches, "
          f"loss {y.iloc[0]:.2f} -> {y.iloc[-1]:.2f})")


def main():
    if not os.path.isdir(RUN):
        sys.exit(f"run directory not found: {RUN}")
    print(f"source: {os.path.relpath(RUN, ROOT)}")
    for lang in ("en", "es"):
        out = OUT if lang == "en" else os.path.join(OUT, "es")
        os.makedirs(out, exist_ok=True)
        fig_vram_trace(lang, out)
        fig_loss_curve(lang, out)
    print("done.")


if __name__ == "__main__":
    main()
