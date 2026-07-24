# S³ Training Report

_generated 2026-07-16T04:57:40.506436+00:00_

## Headline

- **Model**: Qwen/Qwen2.5-7B-Instruct
- **GPU**: NVIDIA GeForce GTX 1050 (3.94 GB)
- **Trainable params**: 1,086,204 (0.0143% of 7,616,702,716)
- **VRAM peak**: 2114.9 MB
- **Throughput**: 5.1 tok/s
- **Total wall-clock**: 33.21 min

## Quality (RQ2 gate — did it learn?)

| metric | base (init) | after | Δ |
|---|---|---|---|
| eval_loss | 2.48128 | 1.73567 | -0.7456 |
| perplexity | 11.9566 | 5.6728 | -6.2838 |

## Training dynamics

- optimizer steps: 38
- loss: first 3.1571 → last 1.1363 (min 0.9528)
- fp16 steps skipped by GradScaler (overflow warmup): 0
- mean grad_svmo: 0.2056
- mean grad_nmf: 0.6192
- mean grad_stb: 0.3041

## Memory (RQ1)

- VRAM allocated: peak 2115 MB, median 881 MB
- CPU RAM: peak 13878 MB
- samples: 17302 @ ~100ms

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
    "total": 7616702716,
    "trainable": 1086204,
    "trainable_pct": 0.0143,
    "by_component": {
      "svmo": 225988,
      "nmf": 401464,
      "stb": 860216
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