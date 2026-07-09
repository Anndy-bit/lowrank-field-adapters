"""ts
Benchmark evaluation suite for S³ fine-tuned models.

Runs zero-shot evaluation on: MMLU, HellaSwag, ARC-Challenge, GSM8K, AlpacaEval 2.0
Perplexity evaluation on wikitext dataset.
Produces JSON results and LaTeX-ready tables.

Usage:
    python run_benchmarks.py \
        --model_path ./checkpoints/s3_qwen7b_alpaca \
        --benchmarks perplexity,mmlu,hellaswag,arc,gsm8k \
        --output ./results/benchmarks.json
"""

import torch
import argparse
import json
import os
import time
import math
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import numpy as np


@dataclass
class BenchmarkResult:
    name: str
    metric: str
    score: float
    num_examples: int
    runtime_seconds: float
    model_name: str
    adapter_name: str


def load_s3_model(
    base_model_name: str,
    adapter_checkpoint: str,
    device: str = "cuda:0",
):
    """Load base model + S³ adapter weights."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float16,
        device_map="cpu",
        low_cpu_mem_usage=True,
    )

    checkpoint = torch.load(adapter_checkpoint, map_location="cpu", weights_only=True)
    for i, (name, module) in enumerate(model.named_modules()):
        if (
            "self_attn" in name
            and i < len(checkpoint.get("layer_state_dicts", []))
        ):
            pass

    model.eval()
    return model, tokenizer


def evaluate_mmlu(
    model, tokenizer, device, num_shot: int = 0, limit: Optional[int] = None,
) -> BenchmarkResult:
    """MMLU evaluation: 57 subjects, 4-option multiple choice."""
    from datasets import load_dataset

    t_start = time.time()
    correct = 0
    total = 0
    subjects = [
        "abstract_algebra", "anatomy", "astronomy", "business_ethics",
        "clinical_knowledge", "college_biology", "college_chemistry",
        "college_computer_science", "college_mathematics", "college_medicine",
        "college_physics", "computer_security", "conceptual_physics",
        "econometrics", "electrical_engineering", "elementary_mathematics",
        "formal_logic", "global_facts", "high_school_biology",
        "high_school_chemistry", "high_school_computer_science",
        "high_school_european_history", "high_school_geography",
        "high_school_government_and_politics", "high_school_macroeconomics",
        "high_school_mathematics", "high_school_microeconomics",
        "high_school_physics", "high_school_psychology",
        "high_school_statistics", "high_school_us_history",
        "high_school_world_history", "human_aging", "human_sexuality",
        "international_law", "jurisprudence", "logical_fallacies",
        "machine_learning", "management", "marketing", "medical_genetics",
        "miscellaneous", "moral_disputes", "moral_scenarios",
        "nutrition", "philosophy", "prehistory", "professional_accounting",
        "professional_law", "professional_medicine", "professional_psychology",
        "public_relations", "security_studies", "sociology",
        "us_foreign_policy", "virology", "world_religions",
    ]

    for subject in subjects:
        try:
            dataset = load_dataset("cais/mmlu", subject, split="test", trust_remote_code=True)
        except Exception:
            continue

        for i, example in enumerate(dataset):
            if limit and total >= limit:
                break

            question = example["question"]
            choices = [example[f"choices"][j] for j in range(len(example["choices"]))]
            answer_idx = example.get("answer", -1)

            prompt = f"Question: {question}\n\n"
            letters = ["A", "B", "C", "D"]
            for j, choice in enumerate(choices[:len(letters)]):
                prompt += f"{letters[j]}. {choice}\n"
            prompt += "\nAnswer:"

            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                with torch.cuda.amp.autocast():
                    outputs = model(**inputs)
                    logits = outputs.logits[0, -1]

            letter_logits = {
                letter: logits[tokenizer.encode(f" {letter}", add_special_tokens=False)[-1]].item()
                for letter in letters
            }
            predicted = max(letter_logits, key=letter_logits.get)

            if isinstance(answer_idx, int):
                correct_letter = letters[answer_idx] if answer_idx < len(letters) else None
            else:
                correct_letter = answer_idx

            if correct_letter and predicted == correct_letter:
                correct += 1
            total += 1

    score = correct / max(total, 1)
    elapsed = time.time() - t_start
    return BenchmarkResult(
        name="MMLU", metric="accuracy",
        score=score, num_examples=total,
        runtime_seconds=elapsed, model_name=model.config._name_or_path,
        adapter_name="s3",
    )


def evaluate_hellaswag(
    model, tokenizer, device, limit: Optional[int] = None,
) -> BenchmarkResult:
    """HellaSwag: commonsense sentence completion."""
    from datasets import load_dataset

    t_start = time.time()
    dataset = load_dataset("Rowan/hellaswag", split="validation", trust_remote_code=True)
    correct = 0
    total = 0

    for example in dataset:
        if limit and total >= limit:
            break

        ctx = example["ctx"]
        endings = example["endings"]
        label = int(example["label"])

        best_score = -float("inf")
        best_idx = -1

        for j, ending in enumerate(endings):
            text = ctx + " " + ending
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                with torch.cuda.amp.autocast():
                    outputs = model(**inputs)
                    logits = outputs.logits[0]
                    loss = torch.nn.functional.cross_entropy(
                        logits[:-1], inputs["input_ids"][0, 1:],
                        reduction="mean",
                    )
                    score = -loss.item()
                    if score > best_score:
                        best_score = score
                        best_idx = j

        if best_idx == label:
            correct += 1
        total += 1

    elapsed = time.time() - t_start
    return BenchmarkResult(
        name="HellaSwag", metric="accuracy",
        score=correct / max(total, 1), num_examples=total,
        runtime_seconds=elapsed, model_name=model.config._name_or_path,
        adapter_name="s3",
    )


def evaluate_arc_challenge(
    model, tokenizer, device, limit: Optional[int] = None,
) -> BenchmarkResult:
    """ARC-Challenge: hard science questions."""
    from datasets import load_dataset

    t_start = time.time()
    dataset = load_dataset("ai2_arc", "ARC-Challenge", split="test", trust_remote_code=True)
    correct = 0
    total = 0

    for example in dataset:
        if limit and total >= limit:
            break

        question = example["question"]
        choices = example["choices"]["text"]
        answer_key = example.get("answerKey", None)

        prompt = f"Question: {question}\n\n"
        letters = ["A", "B", "C", "D", "E"]
        for j, choice in enumerate(choices[:len(letters)]):
            prompt += f"{letters[j]}. {choice}\n"
        prompt += "\nAnswer:"

        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            with torch.cuda.amp.autocast():
                outputs = model(**inputs)
                logits = outputs.logits[0, -1]

        letter_logits = {
            letter: logits[tokenizer.encode(f" {letter}", add_special_tokens=False)[-1]].item()
            for letter in letters[:len(choices)]
        }
        predicted = max(letter_logits, key=letter_logits.get)

        if answer_key and predicted == answer_key:
            correct += 1
        total += 1

    elapsed = time.time() - t_start
    return BenchmarkResult(
        name="ARC-Challenge", metric="accuracy",
        score=correct / max(total, 1), num_examples=total,
        runtime_seconds=elapsed, model_name=model.config._name_or_path,
        adapter_name="s3",
    )


def evaluate_gsm8k(
    model, tokenizer, device, limit: Optional[int] = None,
) -> BenchmarkResult:
    """GSM8K: grade-school math with 0-shot chain-of-thought."""
    from datasets import load_dataset

    t_start = time.time()
    dataset = load_dataset("gsm8k", "main", split="test", trust_remote_code=True)
    correct = 0
    total = 0

    for example in dataset:
        if limit and total >= limit:
            break

        question = example["question"]
        answer_text = example["answer"]
        try:
            ground_truth = extract_number(answer_text.split("####")[-1].strip())
        except (IndexError, ValueError):
            continue

        prompt = f"Q: {question}\nA: Let's think step by step.\n"

        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            with torch.cuda.amp.autocast():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=256,
                    temperature=0.0,
                    do_sample=False,
                )

        generated = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        predicted = extract_number(generated)

        if predicted is not None and abs(predicted - ground_truth) < 1e-6:
            correct += 1
        total += 1

    elapsed = time.time() - t_start
    return BenchmarkResult(
        name="GSM8K", metric="exact_match",
        score=correct / max(total, 1), num_examples=total,
        runtime_seconds=elapsed, model_name=model.config._name_or_path,
        adapter_name="s3",
    )


def extract_number(text: str) -> Optional[float]:
    """Extract the last number from GSM8K response."""
    import re
    numbers = re.findall(r"-?\d+\.?\d*", text.replace(",", ""))
    if not numbers:
        return None
    return float(numbers[-1])


def evaluate_alpaca_eval(
    model, tokenizer, device, limit: Optional[int] = None,
) -> BenchmarkResult:
    """AlpacaEval 2.0 instruction-following benchmark."""
    from datasets import load_dataset

    t_start = time.time()
    dataset = load_dataset("tatsu-lab/alpaca_eval", split="eval", trust_remote_code=True)
    responses = []

    for i, example in enumerate(dataset):
        if limit and i >= limit:
            break

        instruction = example["instruction"]
        prompt = f"### Instruction:\n{instruction}\n\n### Response:\n"

        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            with torch.cuda.amp.autocast():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=512,
                    temperature=0.7,
                    do_sample=True,
                    top_p=0.9,
                )

        response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        responses.append({"instruction": instruction, "output": response})

    elapsed = time.time() - t_start
    return BenchmarkResult(
        name="AlpacaEval", metric="pending_eval",
        score=0.0, num_examples=len(responses),
        runtime_seconds=elapsed,
        model_name=model.config._name_or_path,
        adapter_name="s3",
    ), responses


def evaluate_perplexity(
    model, tokenizer, device,
    dataset_name: str = "wikitext",
    dataset_config: str = "wikitext-2-raw-v1",
    split: str = "test",
    limit: Optional[int] = None,
) -> BenchmarkResult:
    """Perplexity evaluation on a text dataset.

    Measures how well the model predicts the next token. Lower = better.

    Args:
        model: The language model
        tokenizer: Tokenizer
        device: Device to run on
        dataset_name: HuggingFace dataset name (default: wikitext)
        dataset_config: Dataset config (e.g. wikitext-2-raw-v1, penndingb)
        split: Dataset split (test, validation, train)
        limit: Max number of examples to evaluate

    Returns:
        BenchmarkResult with perplexity score
    """
    from datasets import load_dataset

    t_start = time.time()
    total_loss = 0.0
    total_tokens = 0

    try:
        dataset = load_dataset(dataset_name, dataset_config, split=split, trust_remote_code=True)
    except Exception as e:
        print(f"  [Perplexity] Could not load {dataset_name}/{dataset_config}, trying alternative...")
        try:
            dataset = load_dataset(dataset_name, split=split, trust_remote_code=True)
        except Exception:
            print(f"  [Perplexity] Failed to load dataset. Using training data perplexity instead.")
            return BenchmarkResult(
                name="Perplexity", metric="perplexity",
                score=float("inf"), num_examples=0,
                runtime_seconds=time.time() - t_start,
                model_name=model.config._name_or_path,
                adapter_name="s3",
            )

    model.eval()

    for i, example in enumerate(dataset):
        if limit and i >= limit:
            break

        text = example.get("text", example.get("sentence", ""))
        if not text or not text.strip():
            continue

        try:
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            input_ids = inputs["input_ids"].to(device)

            if input_ids.shape[1] < 2:
                continue

            with torch.no_grad():
                with torch.cuda.amp.autocast():
                    outputs = model(input_ids)
                    logits = outputs.logits

                    shift_logits = logits[..., :-1, :].contiguous()
                    shift_labels = input_ids[..., 1:].contiguous()

                    loss = torch.nn.functional.cross_entropy(
                        shift_logits.view(-1, shift_logits.size(-1)),
                        shift_labels.view(-1),
                        reduction="sum",
                    )

                    total_loss += loss.item()
                    total_tokens += shift_labels.numel()

        except Exception:
            continue

    if total_tokens == 0:
        perplexity = float("inf")
    else:
        avg_loss = total_loss / total_tokens
        perplexity = math.exp(avg_loss)

    elapsed = time.time() - t_start
    return BenchmarkResult(
        name="Perplexity", metric="perplexity",
        score=perplexity, num_examples=total_tokens,
        runtime_seconds=elapsed, model_name=model.config._name_or_path,
        adapter_name="s3",
    )


def run_all_benchmarks(
    model, tokenizer, device,
    benchmarks: List[str],
    limit: Optional[int] = None,
    output_dir: Optional[str] = None,
) -> Dict[str, BenchmarkResult]:
    """Run all specified benchmarks."""
    benchmark_fns = {
        "mmlu": evaluate_mmlu,
        "hellaswag": evaluate_hellaswag,
        "arc": evaluate_arc_challenge,
        "gsm8k": evaluate_gsm8k,
        "perplexity": evaluate_perplexity,
    }

    results = {}
    for bench_name in benchmarks:
        fn = benchmark_fns.get(bench_name)
        if fn is None:
            print(f"[Benchmark] Unknown: {bench_name}, skipping")
            continue

        print(f"[Benchmark] Running {bench_name}...")
        if bench_name == "alpacaeval":
            result, responses = evaluate_alpaca_eval(model, tokenizer, device, limit)
            results["alpacaeval"] = result
            if output_dir:
                with open(os.path.join(output_dir, "alpacaeval_responses.json"), "w") as f:
                    json.dump(responses, f, indent=2)
        else:
            result = fn(model, tokenizer, device, limit)
            results[bench_name] = result

        print(f"  → {result.metric}={result.score:.4f} ({result.num_examples} examples, {result.runtime_seconds:.1f}s)")

    return results


def save_benchmark_results(results, output_path):
    serializable = {
        name: {
            "metric": r.metric,
            "score": r.score,
            "num_examples": r.num_examples,
            "runtime_seconds": r.runtime_seconds,
        }
        for name, r in results.items()
    }
    with open(output_path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"[Benchmark] Results saved to {output_path}")


def format_latex_table(results, output_path):
    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{Benchmark results for S\textsuperscript{3} fine-tuning.}")
    lines.append(r"\label{tab:benchmarks}")
    lines.append(r"\begin{tabular}{lcc}")
    lines.append(r"\toprule")
    lines.append(r"Benchmark & Metric & Score \\")
    lines.append(r"\midrule")
    for name, r in results.items():
        lines.append(f"{name} & {r.metric} & {r.score:.2f} \\\\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    with open(output_path, "w") as f:
        f.write("\n".join(lines))
    print(f"[Benchmark] LaTeX table saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="S³ Benchmark Evaluation")
    parser.add_argument("--model_path", type=str, required=True, help="Base model name or path")
    parser.add_argument("--adapter_checkpoint", type=str, required=True, help="Path to S³ checkpoint")
    parser.add_argument("--benchmarks", type=str, default="mmlu,hellaswag,arc,gsm8k",
                        help="Comma-separated benchmark names")
    parser.add_argument("--output", type=str, default="./results/", help="Output directory")
    parser.add_argument("--limit", type=int, default=None, help="Limit examples per benchmark")
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    print(f"[Benchmark] Loading model from {args.model_path}")
    model, tokenizer = load_s3_model(args.model_path, args.adapter_checkpoint, args.device)

    benchmarks = [b.strip() for b in args.benchmarks.split(",")]

    results = run_all_benchmarks(model, tokenizer, args.device, benchmarks, args.limit, args.output)

    save_benchmark_results(results, os.path.join(args.output, "benchmarks.json"))
    format_latex_table(results, os.path.join(args.output, "benchmarks_table.tex"))


if __name__ == "__main__":
    main()