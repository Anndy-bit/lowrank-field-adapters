# S³ Training Report

_generated 2026-07-15T23:44:42.336231+00:00_

## Headline

- **Model**: Qwen/Qwen2.5-7B-Instruct
- **GPU**: NVIDIA GeForce GTX 1050 (3.94 GB)
- **Trainable params**: 684,740 (0.009% of 7,616,301,252)
- **VRAM peak**: 1939.4 MB
- **Throughput**: 5.25 tok/s
- **Total wall-clock**: 41.44 min

## Quality (RQ2 gate — did it learn?)

| metric | base (init) | after | Δ |
|---|---|---|---|
| eval_loss | 2.48118 | 1.82887 | -0.6523 |
| perplexity | 11.9554 | 6.2269 | -5.7285 |

## Training dynamics

- optimizer steps: 50
- loss: first 3.4443 → last 1.3215 (min 1.0361)
- fp16 steps skipped by GradScaler (overflow warmup): 0
- mean grad_svmo: 0.2603
- mean grad_nmf: 0
- mean grad_stb: 0.3902

## Memory (RQ1)

- VRAM allocated: peak 1939 MB, median 878 MB
- CPU RAM: peak 13895 MB
- samples: 21289 @ ~100ms

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
    "total": 7616301252,
    "trainable": 684740,
    "trainable_pct": 0.009,
    "by_component": {
      "svmo": 225988,
      "nmf": 0,
      "stb": 458752
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