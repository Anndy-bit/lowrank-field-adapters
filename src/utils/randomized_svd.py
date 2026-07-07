"""
Offline randomized SVD computation for pretrained LLM weights.

Pre-computes U_k, Σ_k, V_k for all linear layers in a HuggingFace model.
Results saved to disk for use by SVMO during frugal training.

Usage:
    python randomized_svd.py --model Qwen/Qwen2.5-7B-Instruct --k 128 --output ./svd_factors/
"""

import torch
import argparse
import os
import sys
import json
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from typing import Dict, Tuple
from tqdm import tqdm


def randomized_svd(
    weight: torch.Tensor,
    k: int,
    n_oversamples: int = 10,
    n_iter: int = 2,
    device: str = "cpu",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Randomized SVD (Halko, Martinsson, Tropp 2011).

    For matrices with d > 1024, ~50× faster than torch.linalg.svd.

    Args:
        weight: (d_out, d_in) float32 tensor
        k: target rank
        n_oversamples: extra samples for accuracy
        n_iter: power iterations for accuracy
        device: computation device ('cpu' recommended for large matrices)

    Returns:
        U_k (d_out, k), S_k (k,), Vt_k (k, d_in) in float32
    """
    d_out, d_in = weight.shape
    k_target = min(k + n_oversamples, min(d_out, d_in))

    weight = weight.to(device=device, dtype=torch.float32)

    omega = torch.randn(d_in, k_target, device=device, dtype=torch.float32)
    Y = weight @ omega

    for _ in range(n_iter):
        Q, _ = torch.linalg.qr(Y)
        Y = weight.t() @ Q
        Q, _ = torch.linalg.qr(Y)
        Y = weight @ Q

    Q, _ = torch.linalg.qr(Y)
    B = Q.t() @ weight
    Ub, Sb, Vtb = torch.linalg.svd(B, full_matrices=False)

    U_k = (Q @ Ub)[:, :k].cpu()
    S_k = Sb[:k].cpu()
    Vt_k = Vtb[:k, :].cpu()

    del weight, omega, Y, Q, B, Ub, Sb, Vtb
    torch.cuda.empty_cache() if device.startswith("cuda") else None

    return U_k, S_k, Vt_k


def extract_linear_weights(
    model,
) -> Dict[str, torch.Tensor]:
    """Extract all nn.Linear weight matrices from a HuggingFace model.

    Returns: dict mapping layer path → weight tensor (float32, CPU).
    """
    weights = {}
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            w = module.weight.data.detach().clone().float().cpu()
            weights[name] = w
    return weights


def compute_all_svd(
    weights: Dict[str, torch.Tensor],
    k: int = 128,
    n_oversamples: int = 10,
    n_iter: int = 2,
    device: str = "cpu",
    show_progress: bool = True,
) -> Dict[str, Dict[str, torch.Tensor]]:
    """Compute randomized SVD for all extracted weight matrices.

    Returns: dict mapping layer_name → {"U_k": Tensor, "S_k": Tensor, "Vt_k": Tensor}
    """
    results = {}
    weight_items = list(weights.items())
    iterator = tqdm(weight_items, desc="SVD") if show_progress else weight_items

    total_params = 0
    t_start = time.time()

    for name, weight in iterator:
        d_out, d_in = weight.shape
        k_eff = min(k, min(d_out, d_in))
        if k_eff < 4:
            continue

        U_k, S_k, Vt_k = randomized_svd(
            weight, k_eff, n_oversamples, n_iter, device
        )

        results[name] = {
            "U_k": U_k,
            "S_k": S_k,
            "Vt_k": Vt_k,
        }

        total_params += U_k.numel() + S_k.numel() + Vt_k.numel()

    elapsed = time.time() - t_start
    print(f"\n[SVD] Computed {len(results)} matrices in {elapsed:.1f}s")
    print(f"[SVD] Total SVD factor params: {total_params:,} ({total_params * 4 / (1024**3):.2f} GB fp32)")

    return results


def save_svd_factors(
    svd_results: Dict[str, Dict[str, torch.Tensor]],
    output_dir: str,
    dtype: torch.dtype = torch.float16,
):
    """Save SVD factors to disk in fp16 format.

    Saves one file per layer for lazy loading during training.
    Also saves a metadata.json with paths and shapes.
    """
    os.makedirs(output_dir, exist_ok=True)

    metadata = {}
    total_bytes = 0

    for name, factors in svd_results.items():
        U_k = factors["U_k"].to(dtype)
        S_k = factors["S_k"].to(dtype)
        Vt_k = factors["Vt_k"].to(dtype)

        safe_name = name.replace(".", "_").replace("/", "_")
        path = os.path.join(output_dir, f"{safe_name}.pt")
        torch.save({"U_k": U_k, "S_k": S_k, "Vt_k": Vt_k}, path)

        total_bytes += os.path.getsize(path)
        metadata[name] = {
            "file": path,
            "shape_U": list(U_k.shape),
            "shape_Vt": list(Vt_k.shape),
            "k": U_k.shape[1],
            "size_bytes": os.path.getsize(path),
        }

    metadata_path = os.path.join(output_dir, "metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"[SVD] Saved {len(svd_results)} files to {output_dir}")
    print(f"[SVD] Total size: {total_bytes / (1024**3):.2f} GB {dtype}")


def load_svd_factors(
    output_dir: str,
    device: str = "cpu",
) -> Dict[str, Dict[str, torch.Tensor]]:
    """Load SVD factors for a specific layer during training."""
    metadata_path = os.path.join(output_dir, "metadata.json")
    if os.path.exists(metadata_path):
        with open(metadata_path) as f:
            metadata = json.load(f)
    else:
        metadata = {}
        for fname in os.listdir(output_dir):
            if fname.endswith(".pt"):
                metadata[fname[:-3]] = {"file": os.path.join(output_dir, fname)}

    results = {}
    for name, info in metadata.items():
        path = info.get("file", os.path.join(output_dir, f"{name}.pt"))
        data = torch.load(path, map_location=device, weights_only=True)
        results[name] = {
            "U_k": data["U_k"],
            "S_k": data["S_k"],
            "Vt_k": data["Vt_k"],
        }
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Pre-compute randomized SVD for LLM weights (S³ SVMO offline step)"
    )
    parser.add_argument(
        "--model", type=str, required=True,
        help="HuggingFace model name (e.g. Qwen/Qwen2.5-7B-Instruct)"
    )
    parser.add_argument(
        "--k", type=int, default=128,
        help="SVD rank (default 128)"
    )
    parser.add_argument(
        "--output", type=str, default="./svd_factors/",
        help="Output directory for SVD factors"
    )
    parser.add_argument(
        "--oversamples", type=int, default=10,
        help="Randomized SVD oversamples"
    )
    parser.add_argument(
        "--iterations", type=int, default=2,
        help="Randomized SVD power iterations"
    )
    parser.add_argument(
        "--device", type=str, default="cpu",
        help="Computation device (cpu recommended for large models)"
    )
    parser.add_argument(
        "--dtype", type=str, default="fp16",
        choices=["fp16", "fp32"],
        help="Storage dtype"
    )
    parser.add_argument(
        "--target-layers", type=str, nargs="*", default=None,
        help="Filter: only compute SVD for layers matching these substrings"
    )

    args = parser.parse_args()

    print(f"[SVD] Loading model: {args.model}")
    from utils.resilient_download import load_model_with_resume
    model, _, _ = load_model_with_resume(
        repo_id=args.model,
        device="cpu",
        torch_dtype=torch.float16,
        timeout=120,
        max_retries=10,
        retry_delay=5.0,
    )

    print(f"[SVD] Extracting weights...")
    all_weights = extract_linear_weights(model)
    del model
    torch.cuda.empty_cache()

    if args.target_layers:
        all_weights = {
            name: w for name, w in all_weights.items()
            if any(target in name for target in args.target_layers)
        }
        print(f"[SVD] Filtered to {len(all_weights)} layers: {args.target_layers}")

    print(f"[SVD] Computing randomized SVD for {len(all_weights)} matrices (k={args.k})...")
    results = compute_all_svd(
        all_weights,
        k=args.k,
        n_oversamples=args.oversamples,
        n_iter=args.iterations,
        device=args.device,
    )

    storage_dtype = torch.float16 if args.dtype == "fp16" else torch.float32
    save_svd_factors(results, args.output, dtype=storage_dtype)
    print("[SVD] Done.")


if __name__ == "__main__":
    main()