"""Resilient model download using HuggingFace's file-by-file download.

Uses sequential file download (not parallel) for slow/unstable connections:
    - Downloads one file at a time
    - Each file resumes on interruption
    - Cleans up incomplete files before starting
    - Handles connection errors with retry
"""

import os
import time
import requests
from pathlib import Path
from typing import Optional, Tuple, List


HF_TOKEN = os.environ.get("HF_TOKEN", None)
HF_ENDPOINT = os.environ.get("HF_ENDPOINT", "https://huggingface.co")
HEADERS = {"User-Agent": "lowrank-field-adapters/1.0"}
if HF_TOKEN:
    HEADERS["authorization"] = f"Bearer {HF_TOKEN}"


def _clean_incomplete_files(model_cache: Path) -> int:
    """Remove .incomplete files from interrupted downloads."""
    cleaned = 0
    if model_cache.exists():
        incomplete_files = list(model_cache.rglob("*.incomplete"))
        for f in incomplete_files:
            try:
                f.unlink()
                cleaned += 1
            except Exception:
                pass
    return cleaned


def download_model_resilient(
    repo_id: str,
    timeout: int = 300,
    max_retries: int = 20,
    retry_delay: float = 10.0,
    force: bool = False,
) -> str:
    """Download a HuggingFace model file-by-file for unstable connections.

    Uses HF's hf_hub_download which:
    - Downloads one file at a time (stable for slow internet)
    - Automatically resumes interrupted files
    - Uses cache for already-downloaded files

    Args:
        repo_id: HuggingFace model repo
        timeout: Seconds per request (increased for slow connections)
        max_retries: Max retry attempts per file
        retry_delay: Seconds between retries
        force: Force re-download

    Returns:
        Local model directory path
    """
    from huggingface_hub import HfApi, hf_hub_download

    cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
    model_cache = Path(cache_dir) / f"models--{repo_id.replace('/', '--')}"

    # Clean incomplete files from previous interrupted downloads
    cleaned = _clean_incomplete_files(model_cache)
    if cleaned > 0:
        print(f"  Cleaned {cleaned} incomplete file(s) from previous run")

    print(f"  Downloading model: {repo_id}")
    print(f"  (sequential download for unstable connections, do NOT cancel)")

    try:
        api = HfApi()
        repo_files = api.list_repo_files(repo_id, token=HF_TOKEN)
        print(f"  Found {len(repo_files)} files to download")
    except Exception as e:
        print(f"  Error listing repo files: {e}")
        raise RuntimeError(f"Cannot access repo {repo_id}: {e}")

    for i, filename in enumerate(repo_files, 1):
        print(f"  [{i}/{len(repo_files)}] {filename}...", end=" ", flush=True)
        
        last_error = None
        for attempt in range(max_retries):
            try:
                hf_hub_download(
                    repo_id=repo_id,
                    filename=filename,
                    cache_dir=cache_dir,
                    token=HF_TOKEN,
                    force_download=force,
                )
                print("OK")
                break
                
            except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
                last_error = e
                print(f"connection error, retrying ({attempt+1}/{max_retries})...")
                time.sleep(retry_delay)
                
            except Exception as e:
                last_error = e
                err_str = str(e).lower()
                if "404" in err_str or "not found" in err_str:
                    print(f"NOT FOUND (skipping)")
                    break
                print(f"error: {e}, retrying ({attempt+1}/{max_retries})...")
                time.sleep(retry_delay)
        else:
            print(f"FAILED after {max_retries} attempts")
            raise RuntimeError(f"Failed to download {filename} after {max_retries} attempts: {last_error}")

    print(f"  Download complete: {model_cache}")
    return str(model_cache)


def load_model_with_resume(
    repo_id: str,
    device: str = "cpu",
    torch_dtype=None,
    timeout: int = 300,
    max_retries: int = 20,
    retry_delay: float = 10.0,
) -> Tuple:
    """Load a HuggingFace model with resilient, resumable downloads.

    Args:
        repo_id: HuggingFace model repo
        device: Device to load model on
        torch_dtype: Model dtype
        timeout: Seconds per request
        max_retries: Max retries per file
        retry_delay: Delay between retries

    Returns:
        (model, tokenizer, model_dir)
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_dir = download_model_resilient(
        repo_id,
        timeout=timeout,
        max_retries=max_retries,
        retry_delay=retry_delay,
    )

    print(f"  Loading tokenizer from: {model_dir}")
    tokenizer = None
    tokenizer_error = None
    for attempt in range(max_retries):
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
            break
        except Exception as e:
            tokenizer_error = e
            err_str = str(e).lower()
            if "tokenizer" in err_str or "sentencepiece" in err_str or "tiktoken" in err_str:
                print(f"  [{attempt+1}/{max_retries}] Tokenizer error (may be incomplete): {e}")
                print(f"  Re-downloading model to fix corrupted files...")
                try:
                    model_dir = download_model_resilient(
                        repo_id,
                        timeout=timeout,
                        max_retries=max_retries,
                        retry_delay=retry_delay,
                        force=True,
                    )
                    print(f"  Retrying tokenizer load from: {model_dir}")
                except Exception as dl_err:
                    print(f"  Re-download failed: {dl_err}, retrying original cache...")
                    time.sleep(retry_delay)
            else:
                print(f"  [{attempt+1}/{max_retries}] Error loading tokenizer: {e}, retrying in {retry_delay}s...")
                time.sleep(retry_delay)

    if tokenizer is None:
        raise RuntimeError(f"Failed to load tokenizer after {max_retries} attempts: {tokenizer_error}")

    print(f"  Loading model from: {model_dir}")
    model = None
    model_error = None
    for attempt in range(max_retries):
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_dir,
                torch_dtype=torch_dtype,
                device_map=device,
                trust_remote_code=True,
            )
            break
        except Exception as e:
            model_error = e
            print(f"  [{attempt+1}/{max_retries}] Error loading model: {e}")
            if attempt < max_retries - 1:
                print(f"  Retrying in {retry_delay}s...")
                time.sleep(retry_delay)

    if model is None:
        raise RuntimeError(f"Failed to load model after {max_retries} attempts: {model_error}")

    return model, tokenizer, model_dir