# S³ Training Report

_generated 2026-07-15T08:15:02.405504+00:00_

## Headline

- **Model**: Qwen/Qwen2.5-7B-Instruct
- **GPU**: NVIDIA GeForce GTX 1050 (3.94 GB)
- **Trainable params**: 3,896,452 (0.0511% of 7,619,512,964)
- **VRAM peak**: 2232.8 MB
- **Throughput**: 4.85 tok/s
- **Total wall-clock**: 255.84 min

## Quality (RQ2 gate — did it learn?)

| metric | base (init) | after | Δ |
|---|---|---|---|
| eval_loss | 2.48164 | 1.7054 | -0.7762 |
| perplexity | 11.9609 | 5.5036 | -6.4573 |

## Training dynamics

- optimizer steps: 275
- loss: first 5.0325 → last 1.5311 (min 0.8188)
- fp16 steps skipped by GradScaler (overflow warmup): 0
- mean grad_svmo: 0.06999
- mean grad_nmf: 0.6025
- mean grad_stb: 0.1015

## Memory (RQ1)

- VRAM allocated: peak 2233 MB, median 930 MB
- CPU RAM: peak 13613 MB
- samples: 114766 @ ~100ms

## Provenance & config

```json
{
  "versions": {
    "torch": "2.5.1+cu121",
    "transformers": "5.13.0",
    "accelerate": "1.14.0",
    "safetensors": "0.8.0",
    "bitsandbytes": "0.49.2",
    "datasets": "5.0.0",
    "numpy": "2.5.0"
  },
  "hardware": {
    "gpu": {
      "name": "NVIDIA GeForce GTX 1050",
      "total_vram_gb": 3.94,
      "driver_cuda": "12.1"
    },
    "cpu_count": 8,
    "ram_total_gb": 15.5,
    "platform": "Linux-6.18.7-76061807-generic-x86_64-with-glibc2.39"
  },
  "params": {
    "total": 7619512964,
    "trainable": 3896452,
    "trainable_pct": 0.0511,
    "by_component": {
      "svmo": 225988,
      "nmf": 3211712,
      "stb": 3670464
    }
  },
  "git": "f970a614a37c3c27a390323cfedce21ec51d5264"
}
```

## Figures

### loss_curve.png
![loss_curve.png](figures/loss_curve.png)

### lr.png
![lr.png](figures/lr.png)

### grad_norms.png
![grad_norms.png](figures/grad_norms.png)

### step_time.png
![step_time.png](figures/step_time.png)

### vram_trace.png
![vram_trace.png](figures/vram_trace.png)

### layer_grads.png
![layer_grads.png](figures/layer_grads.png)

### modulation.png
![modulation.png](figures/modulation.png)

## Raw data (regenerate any plot)

- `steps.csv`
- `vram_trace.csv`
- `layer_grads.csv`
- `modulation.csv`
- `provenance.json`
- `summary.json`

## Still needed for the paper (not auto-collected here)

- Downstream benchmarks: MMLU / HellaSwag / ARC / GSM8K / AlpacaEval (need datasets cached; run `make evaluate`).
- Baselines: LoRA r=8/r=64, QLoRA, full-FT (same seeds/eval).
- Ablations: −STB / −NMF / −SVMO and single-operator runs.
- n≥3 seeds → mean ± std / 95% CI.