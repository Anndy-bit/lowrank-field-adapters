# S³ Training Report

_generated 2026-07-19T13:15:32.824972+00:00_

## Headline

- **Model**: Qwen/Qwen2.5-7B-Instruct
- **GPU**: NVIDIA GeForce GTX 1050 (3.94 GB)
- **Trainable params**: 802,928 (0.0105% of 7,616,419,440)
- **VRAM peak**: 1359.8 MB
- **Throughput**: 6.31 tok/s
- **Total wall-clock**: 26.86 min

## Quality (RQ2 gate — did it learn?)

| metric | base (init) | after | Δ |
|---|---|---|---|
| eval_loss | 2.25347 | 1.69447 | -0.559 |
| perplexity | 9.5207 | 5.4438 | -4.0769 |

## Training dynamics

- optimizer steps: 38
- loss: first 3.1574 → last 1.1599 (min 1.0430)
- fp16 steps skipped by GradScaler (overflow warmup): 0
- mean grad_svmo: 0
- mean grad_nmf: 0.7098
- mean grad_stb: 0

## Memory (RQ1)

- VRAM allocated: peak 1360 MB, median 212 MB
- CPU RAM: peak 13899 MB
- samples: 21220 @ ~100ms

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
    "total": 7616419440,
    "trainable": 802928,
    "trainable_pct": 0.0105,
    "by_component": {
      "svmo": 0,
      "nmf": 802928,
      "stb": 802928,
      "lora": 0
    }
  },
  "git": "d1be1682af63e6e461496f01cb1f4c24c32568fb"
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