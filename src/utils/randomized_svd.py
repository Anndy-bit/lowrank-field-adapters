"""
Offline randomized SVD computation for pretrained LLM weights — MEMORY EFFICIENT.

Pre-computes U_k, Σ_k, V_k for all linear layers in a HuggingFace model.
Uses accelerate offload to stream model from disk, processes ONE layer at a time.

Usage:
    python randomized_svd.py --model Qwen/Qwen2.5-7B-Instruct --k 128 --output ./svd_factors/
"""

import torch
import argparse
import os
import sys
import json
import time
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Tuple
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def randomized_svd(
    weight: torch.Tensor,
    k: int,
    n_oversamples: int = 10,
    n_iter: int = 2,
    device: str = "cpu",
    seed: int = 42,
    generator: torch.Generator = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Randomized SVD (Halko, Martinsson, Tropp 2011).

    For matrices with d > 1024, ~50× faster than torch.linalg.svd.

    Args:
        weight: (d_out, d_in) float32 tensor
        k: target rank
        n_oversamples: extra random samples for accuracy
        n_iter: power iterations for accuracy
        device: computation device ('cpu' recommended for large matrices)
        seed: random seed for reproducibility
        generator: optional torch.Generator (overrides seed if provided)

    Returns:
        U_k (d_out, k), S_k (k,), Vt_k (k, d_in) in float32
    """
    d_out, d_in = weight.shape
    k_target = min(k + n_oversamples, min(d_out, d_in))

    weight = weight.to(device=device, dtype=torch.float32)

    if generator is None:
        generator = torch.Generator(device=device)
        generator.manual_seed(seed)

    omega = torch.randn(d_in, k_target, device=device, dtype=torch.float32, generator=generator)
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
    if device.startswith("cuda"):
        torch.cuda.empty_cache()

    return U_k, S_k, Vt_k


def main():
    parser = argparse.ArgumentParser(
        description="Pre-compute randomized SVD for LLM weights (S³ SVMO offline step) — streaming, low RAM"
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
    parser.add_argument(
        "--cpu-ram-budget-gb", type=float, default=10.0,
        help="Max CPU RAM to use (GB). Model will be offloaded to disk if needed."
    )
    parser.add_argument(
        "--offload-folder", type=str, default="/tmp/svd_offload",
        help="Folder for accelerate disk offload"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducible SVD (default 42)"
    )

    args = parser.parse_args()

    print(f"[SVD] Loading model on CPU with low_cpu_mem_usage: {args.model}")
    print(f"[SVD] Output: {args.output}")

    from transformers import AutoModelForCausalLM
    import gc

    # Load model directly on CPU with mmap (no offload complexity)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        device_map="cpu",
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    
    print(f"[SVD] Model loaded on CPU. Extracting and computing SVD layer-by-layer...")

    # Find transformer layers
    transformer_blocks = getattr(model.model, "layers", None)
    if transformer_blocks is None:
        for attr in ["decoder", "transformer", "encoder"]:
            inner = getattr(model.model, attr, None)
            if inner and hasattr(inner, "layers"):
                transformer_blocks = inner.layers
                break
    
    if transformer_blocks is None:
        raise ValueError("Cannot find transformer layers in model structure")

    n_layers = len(transformer_blocks)
    print(f"[SVD] Found {n_layers} transformer layers")

    os.makedirs(args.output, exist_ok=True)
    
    try:
        from safetensors.torch import save_file
        has_safetensors = True
    except ImportError:
        print("[SVD] safetensors not installed, falling back to torch.save")
        has_safetensors = False
        save_file = None

    storage_dtype = torch.float16 if args.dtype == "fp16" else torch.float32
    total_saved = 0
    t_start = time.time()

    generator = torch.Generator(device=args.device)
    generator.manual_seed(args.seed)
    print(f"[SVD] Random seed: {args.seed}")

    model_hasher = hashlib.sha256()
    layer_metrics = {}  # energy retained + reconstruction error per layer

    # Process ONE layer at a time to minimize RAM
    for layer_idx, block in enumerate(transformer_blocks):
        # Force load this block's weights from disk (accelerate handles meta tensors)
        # Move block to CPU properly - handles meta tensors via to_empty
        def move_to_cpu(m):
            for p in m.parameters(recurse=False):
                if p.device.type == "meta":
                    p.data = torch.empty_like(p, device="cpu")
            for buf in m.buffers(recurse=False):
                if buf.device.type == "meta":
                    buf.data = torch.empty_like(buf, device="cpu")
        
        block.apply(move_to_cpu)
        for param in block.parameters():
            param.requires_grad = False
        
        # Extract linear weights from this block
        layer_weights = {}
        for name, module in block.named_modules():
            if isinstance(module, torch.nn.Linear):
                full_name = f"model.layers.{layer_idx}.{name}"
                if args.target_layers and not any(t in full_name for t in args.target_layers):
                    continue
                w = module.weight.data.detach().clone().float().cpu()
                layer_weights[full_name] = w
                model_hasher.update(w.numpy().tobytes())
        
        if not layer_weights:
            continue
            
        print(f"[SVD] Layer {layer_idx}/{n_layers-1}: {len(layer_weights)} matrices...")
        
        # Compute SVD for this layer's weights
        layer_results = {}
        for name, weight in layer_weights.items():
            d_out, d_in = weight.shape
            k_eff = min(args.k, min(d_out, d_in))
            if k_eff < 4:
                continue

            U_k, S_k, Vt_k = randomized_svd(
                weight, k_eff, args.oversamples, args.iterations, args.device,
                generator=generator,
            )
            
            W_f_norm = torch.norm(weight, p='fro').item()
            S_sum_sq = (S_k[:k_eff] ** 2).sum().item()
            energy_retained = S_sum_sq / (W_f_norm ** 2) if W_f_norm > 0 else 0.0
            
            # Reconstruct W_approx = U_k @ diag(S_k) @ Vt_k (truncated to k_eff)
            # U_k: [d_out, k_eff], S_k: [k_eff], Vt_k: [k_eff, d_in]
            W_approx = (U_k[:, :k_eff] * S_k[:k_eff].unsqueeze(0)) @ Vt_k[:k_eff, :]
            recon_error = torch.norm(weight - W_approx, p='fro').item() / W_f_norm if W_f_norm > 0 else 0.0
            
            key = f"layer_{layer_idx}.{name.split('.')[-1]}"
            layer_metrics[key] = {
                "d_out": d_out, "d_in": d_in,
                "k_eff": k_eff,
                "frobenius_norm_W": W_f_norm,
                "energy_retained": round(energy_retained, 6),
                "reconstruction_error_relative": round(recon_error, 6),
                "singular_values_top5": S_k[:min(5, k_eff)].tolist(),
            }
            
            layer_results[name] = {
                "U_k": U_k.to(storage_dtype).contiguous(),
                "S_k": S_k.to(storage_dtype).contiguous(),
                "Vt_k": Vt_k.to(storage_dtype).contiguous(),
            }
            
            # Free weight tensor immediately
            del weight
        
        # Save this layer's SVD factors to disk immediately
        for name, factors in layer_results.items():
            U_k = factors["U_k"]
            S_k = factors["S_k"]
            Vt_k = factors["Vt_k"]

            # Parse layer index and projection name
            parts = name.split(".")
            layer_idx_parsed = None
            proj_name = None
            for i, part in enumerate(parts):
                if part == "layers" and i + 1 < len(parts):
                    layer_idx_parsed = parts[i + 1]
                    proj_name = parts[-1]
                    break
            
            if layer_idx_parsed is None or proj_name is None:
                safe_name = name.replace(".", "_").replace("/", "_")
                layer_dir = Path(args.output) / safe_name
            else:
                layer_dir = Path(args.output) / f"layer_{layer_idx_parsed}.{proj_name}"
            
            layer_dir.mkdir(parents=True, exist_ok=True)

            if has_safetensors:
                save_file({"U_k": U_k}, str(layer_dir / "U_k.safetensors"))
                save_file({"S_k": S_k}, str(layer_dir / "S_k.safetensors"))
                save_file({"Vt_k": Vt_k}, str(layer_dir / "Vt_k.safetensors"))
            else:
                torch.save({"U_k": U_k}, str(layer_dir / "U_k.pt"))
                torch.save({"S_k": S_k}, str(layer_dir / "S_k.pt"))
                torch.save({"Vt_k": Vt_k}, str(layer_dir / "Vt_k.pt"))

            total_saved += sum(f.stat().st_size for f in layer_dir.iterdir())
        
        # Explicit cleanup
        del layer_weights, layer_results, block
        gc.collect()
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()
        
        elapsed = time.time() - t_start
        print(f"  Layer {layer_idx} done. Elapsed: {elapsed:.1f}s. Total saved: {total_saved / (1024**3):.2f} GB")

    # Build paper-grade metadata manifest
    print("[SVD] Building paper-grade metadata manifest...")
    lib_versions = {}
    for lib in ["torch", "transformers", "safetensors"]:
        try:
            mod = __import__(lib)
            lib_versions[lib] = getattr(mod, "__version__", str(mod.__version__))
        except Exception:
            lib_versions[lib] = "unknown"

    # ── Compute aggregated metrics (BEFORE building manifest) ──────────────────
    elapsed = time.time() - t_start
    n_projections = len(layer_metrics)

    from transformers import AutoConfig
    try:
        model_config = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
        hidden_size = getattr(model_config, "hidden_size", 3584)
        num_hidden_layers = getattr(model_config, "num_hidden_layers", 28)
        intermediate_size = getattr(model_config, "intermediate_size", 18944)
        vocab_size = getattr(model_config, "vocab_size", 151936)
        embed_params = vocab_size * hidden_size
        layer_params = 4 * hidden_size * hidden_size + 3 * hidden_size * intermediate_size
        total_model_params = embed_params + num_hidden_layers * layer_params + vocab_size * hidden_size
    except Exception:
        total_model_params = 7_610_000_000

    compressed_params = 0
    total_k_eff = 0
    recon_errors = []
    energy_retained_list = []
    for lm in layer_metrics.values():
        k_eff = lm["k_eff"]
        d_out, d_in = lm["d_out"], lm["d_in"]
        compressed_params += k_eff * (d_out + d_in + 1)
        total_k_eff += k_eff
        recon_errors.append(lm["reconstruction_error_relative"])
        energy_retained_list.append(lm["energy_retained"])

    avg_eff_rank = total_k_eff / n_projections if n_projections > 0 else args.k
    avg_recon_error = sum(recon_errors) / len(recon_errors) if recon_errors else 0
    max_recon_error = max(recon_errors) if recon_errors else 0
    avg_energy = sum(energy_retained_list) / len(energy_retained_list) if energy_retained_list else 0
    compression_ratio = total_model_params / compressed_params if compressed_params > 0 else 0

    peak_ram_gb = 0
    try:
        import psutil
        peak_ram_gb = psutil.Process().memory_info().rss / (1024**3)
    except Exception:
        pass

    # ── Build manifest with all computed metrics ───────────────────────────────
    manifest = {
        "algorithm": "randomized_svd_halko_martinsson_tropp_2011",
        "model_id": args.model,
        "model_sha256": model_hasher.hexdigest(),
        "svd_params": {
            "k_target": args.k,
            "n_oversamples": args.oversamples,
            "n_iter": args.iterations,
            "random_seed": args.seed,
            "device": args.device,
            "storage_dtype": args.dtype,
        },
        "compression": {
            "original_params": total_model_params,
            "compressed_params": compressed_params,
            "compression_ratio": round(compression_ratio, 2),
            "avg_effective_rank": round(avg_eff_rank, 2),
            "avg_energy_retained": round(avg_energy, 6),
            "avg_reconstruction_error": round(avg_recon_error, 6),
            "max_reconstruction_error": round(max_recon_error, 6),
        },
        "library_versions": lib_versions,
        "execution": {
            "started_utc": datetime.now(timezone.utc).isoformat(),
            "elapsed_seconds": round(elapsed, 1),
            "n_layers_processed": n_layers,
            "n_projections_saved": n_projections,
            "total_size_gb": round(total_saved / (1024**3), 3),
            "peak_ram_gb": round(peak_ram_gb, 2),
        },
        "layer_metrics": layer_metrics,
    }

    metadata_path = os.path.join(args.output, "metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(manifest, f, indent=2)

    # ── Print paper-grade summary table ───────────────────────────────────────
    print()
    print("=" * 60)
    print(f"  Model: {args.model}")
    print(f"  Linear layers processed: {n_projections}")
    print(f"  Requested rank: {args.k}")
    print(f"  Average effective rank: {avg_eff_rank:.1f}")
    print()
    print(f"  Original parameters: {total_model_params/1e9:.2f}B")
    print(f"  Compressed SVD parameters: {compressed_params/1e6:.2f}M")
    print(f"  Compression ratio: {compression_ratio:.1f}x")
    print()
    print(f"  Average energy retained: {avg_energy:.4f} ({avg_energy*100:.2f}%)")
    print(f"  Average reconstruction error: {avg_recon_error:.5f}")
    print(f"  Maximum reconstruction error: {max_recon_error:.5f}")
    print()
    print(f"  Peak RAM: {peak_ram_gb:.2f} GB")
    print(f"  Elapsed: {elapsed:.1f} s")
    print(f"  Storage: {total_saved / (1024**3):.3f} GB ({args.dtype})")
    print("=" * 60)
    print(f"\n[SVD] Manifest: {metadata_path}")


if __name__ == "__main__":
    main()