# Perplexity Evaluation

## Overview

Perplexity is the primary metric to answer the fundamental question: **"Did the model learn?"**

It measures how well the model predicts the next token in a text sequence. Lower perplexity means better prediction accuracy.

$$PPL = \exp\left(\frac{1}{N}\sum_{i=1}^{N} -\log P(x_i | x_{<i})\right)$$

## Implementation

The perplexity evaluation is implemented in `src/benchmarks/run_benchmarks.py`:

```python
def evaluate_perplexity(
    model, tokenizer, device,
    dataset_name: str = "wikitext",
    dataset_config: str = "wikitext-2-raw-v1",
    split: str = "test",
    limit: Optional[int] = None,
) -> BenchmarkResult:
```

### Features

- **Dataset**: Uses `wikitext-2-raw-v1` by default (standard benchmark dataset)
- **Fallback**: Falls back to `wikitext` if primary dataset unavailable
- **Error handling**: Returns `perplexity=inf` if dataset cannot be loaded (does not crash)
- **Token-level**: Computes cross-entropy loss at the token level for precise measurement

## Usage

### Via train_s3.py (automatic after training)

```bash
python train_s3.py --config experiments/configs/s3_standard.yaml --device cuda:0
```

Perplexity is included in the default benchmark list and runs automatically after training completes.

### Via train_s3.py (eval-only mode)

```bash
python train_s3.py --eval_only --checkpoint ./checkpoints/s3_qwen7b_alpaca/checkpoint.pt --device cuda:0
```

### Via run_benchmarks.py (standalone)

```bash
python src/benchmarks/run_benchmarks.py \
    --model_path Qwen/Qwen2.5-7B-Instruct \
    --adapter_checkpoint ./checkpoints/s3_qwen7b_alpaca/checkpoint.pt \
    --benchmarks perplexity \
    --output ./results/
```

### Combined with other benchmarks

```bash
python src/benchmarks/run_benchmarks.py \
    --model_path Qwen/Qwen2.5-7B-Instruct \
    --adapter_checkpoint ./checkpoints/s3_qwen7b_alpaca/checkpoint.pt \
    --benchmarks perplexity,mmlu,hellaswag,arc,gsm8k \
    --output ./results/benchmarks.json
```

## Interpreting Results

| Scenario | Perplexity Value | Interpretation |
|----------|------------------|----------------|
| Base model (before fine-tuning) | ~10-15 | Baseline performance |
| After S³ fine-tuning | Lower than base | **Model learned** |
| After S³ fine-tuning | Same or higher | **Problem: model did not learn** |
| LoRA baseline | ~8-12 (varies) | Compare to validate S³ competitiveness |

### Example Output

```
[Benchmark] Running perplexity...
  → perplexity=12.3456 (12345 tokens, 45.2s)
```

### Benchmark JSON Output

```json
{
  "Perplexity": {
    "metric": "perplexity",
    "score": 12.3456,
    "num_examples": 12345,
    "runtime_seconds": 45.2
  }
}
```

## Why Perplexity?

Perplexity is the **fastest way to verify learning**:

1. **Computationally cheap** - No generation required, just forward passes
2. **Interpretable** - Direct measure of prediction quality
3. **Sensitive** - Captures improvements immediately
4. **Standard** - Widely used in language model evaluation

Benchmarks like MMLU and HellaSwag answer "does the model know facts?" but take much longer. Perplexity answers "did the training help?" in minutes.

## Integration with Training Pipeline

The full evaluation pipeline runs in this order:

1. **Perplexity** (fastest, runs first)
2. **MMLU** (knowledge benchmark)
3. **HellaSwag** (commonsense reasoning)
4. **ARC-Challenge** (science questions)
5. **GSM8K** (math reasoning)

Results are saved to:
- `results/benchmarks.json` - Machine-readable format
- `results/benchmarks_table.tex` - LaTeX table for papers

## Configuration

Perplexity can be customized in `experiments/configs/s3_standard.yaml`:

```yaml
benchmarks:
  - perplexity      # Included by default
  - mmlu
  - hellaswag
  - arc
  - gsm8k
```

To disable perplexity:

```yaml
benchmarks:
  - mmlu
  - hellaswag
```

## Troubleshooting

### Dataset not loading

```
[Perplexity] Could not load wikitext/wikitext-2-raw-v1, trying alternative...
[Perplexity] Failed to load dataset. Using training data perplexity instead.
```

This is normal if the dataset cannot be downloaded. The evaluation continues without failing.

### High perplexity after training

If perplexity increases after fine-tuning:

1. Check learning rate (may be too high)
2. Verify adapter weights are being loaded correctly
3. Check that model is in training mode during fine-tuning
4. Verify SVD factors are being applied during forward pass

### Memory issues

If GPU memory is insufficient for perplexity evaluation, use:

```bash
python src/benchmarks/run_benchmarks.py \
    --benchmarks perplexity \
    --limit 100  # Limit to 100 examples
```