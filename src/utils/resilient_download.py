"""Resilient model download utilities with resume support.

Handles HuggingFace model downloads that can be interrupted
(1-minute internet drops, power outages, etc.).
"""

import os
import time
import hashlib
import requests
import threading
import torch
from pathlib import Path
from typing import Optional, List, Tuple


HF_TOKEN = os.environ.get("HF_TOKEN", None)
HF_ENDPOINT = os.environ.get("HF_ENDPOINT", "https://huggingface.co")
HEADERS = {"User-Agent": "lowrank-field-adapters/1.0"}
if HF_TOKEN:
    HEADERS["authorization"] = f"Bearer {HF_TOKEN}"


def _get_hf_file_url(repo_id: str, filename: str) -> str:
    return f"{HF_ENDPOINT}/api/models/{repo_id}/resolve/main/{filename}"


def _get_local_file_path(repo_id: str, filename: str, cache_dir: Optional[str] = None) -> Path:
    if cache_dir is None:
        cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
    return Path(cache_dir) / f"models--{repo_id.replace('/', '--')}" / "blobs" / filename


def _get_tmp_path(local_path: Path) -> Path:
    return local_path.with_suffix(".tmp_download")


def _get_etag_from_server(url: str, timeout: int = 30) -> Optional[str]:
    resp = requests.head(url, headers=HEADERS, timeout=timeout, allow_redirects=True)
    if resp.status_code == 200:
        return resp.headers.get("etag", "").strip('"')
    return None


def hf_download_file(
    repo_id: str,
    filename: str,
    cache_dir: Optional[str] = None,
    resume: bool = True,
    timeout: int = 120,
    max_retries: int = 10,
    retry_delay: float = 5.0,
    chunk_size: int = 10 * 1024 * 1024,
) -> Tuple[str, bool]:
    """Download a single file from HuggingFace Hub with resume support.

    Downloads start 10MB before the last downloaded byte, so connection
    drops lose at most ~10MB of download.

    Args:
        repo_id: HuggingFace model repo (e.g. 'Qwen/Qwen2.5-7B')
        filename: Name of file to download
        cache_dir: HuggingFace cache directory
        resume: If True, continue from where download left off
        timeout: Seconds per request attempt
        max_retries: Max retry attempts per file
        retry_delay: Seconds to wait before retry after failure
        chunk_size: Bytes to redownload before interruption (~10MB)

    Returns:
        (local_path_str, downloaded_fresh): Path to downloaded file and
        whether it was freshly downloaded (True) or from cache (False)
    """
    url = _get_hf_file_url(repo_id, filename)
    local_path = _get_local_file_path(repo_id, filename, cache_dir)
    tmp_path = _get_tmp_path(local_path)
    downloaded_fresh = False

    local_path.parent.mkdir(parents=True, exist_ok=True)

    existing_size = 0
    if local_path.exists():
        local_path.chmod(0o644)
        return (str(local_path), False)

    if resume and tmp_path.exists():
        existing_size = tmp_path.stat().st_size
        print(f"  Resuming {filename}: {existing_size / (1024*1024):.1f}MB already downloaded")

    headers = dict(HEADERS)
    start_byte = max(0, existing_size - chunk_size)

    for attempt in range(max_retries):
        try:
            if start_byte > 0:
                headers["Range"] = f"bytes={start_byte}-"

            resp = requests.get(url, headers=headers, timeout=timeout, stream=True)
            resp.raise_for_status()

            mode = "ab" if (start_byte > 0 and resume) else "wb"
            bytes_to_skip = 0

            if start_byte > 0 and resp.status_code in (200, 206):
                if resp.status_code == 206:
                    content_range = resp.headers.get("content-range", "")
                    if "/" in content_range:
                        total_size = content_range.split("/")[-1]
                        if total_size not in ("*", ""):
                            total_size = int(total_size)
                            if start_byte >= total_size:
                                print(f"  {filename} already complete, renaming")
                                tmp_path.rename(local_path)
                                return (str(local_path), False)
                bytes_to_skip = existing_size - start_byte

            with open(tmp_path, mode) as f:
                if bytes_to_skip > 0:
                    resp.raw.read(bytes_to_skip)
                for chunk in resp.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)

            tmp_path.rename(local_path)
            downloaded_fresh = True
            print(f"  {filename}: download complete ({local_path.stat().st_size / (1024*1024):.1f}MB)")
            return (str(local_path), True)

        except requests.exceptions.Timeout:
            print(f"  [{attempt+1}/{max_retries}] Timeout downloading {filename}, retrying in {retry_delay}s...")
        except requests.exceptions.ConnectionError as e:
            print(f"  [{attempt+1}/{max_retries}] Connection error for {filename}: {e}, retrying in {retry_delay}s...")
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 416:
                tmp_path.rename(local_path)
                return (str(local_path), True)
            print(f"  [{attempt+1}/{max_retries}] HTTP error for {filename}: {e}, retrying...")
        except (IOError, OSError) as e:
            print(f"  [{attempt+1}/{max_retries}] IO error for {filename}: {e}, retrying...")

        if attempt < max_retries - 1:
            time.sleep(retry_delay)

    raise RuntimeError(f"Failed to download {filename} after {max_retries} attempts")


def hf_download_model(
    repo_id: str,
    filenames: Optional[List[str]] = None,
    cache_dir: Optional[str] = None,
    timeout: int = 120,
    max_retries: int = 10,
    retry_delay: float = 5.0,
    progress_callback=None,
) -> str:
    """Download all (or specific) files of a HuggingFace model with resume support.

    Args:
        repo_id: HuggingFace model repo
        filenames: Specific files to download (None = discover from API)
        cache_dir: Cache directory
        timeout: Seconds per request
        max_retries: Max retries per file
        retry_delay: Delay between retries
        progress_callback: Called with (downloaded_count, total_count, filename)

    Returns:
        Local model directory path

    Raises:
        RuntimeError: If download fails after all retries
    """
    if filenames is None:
        import requests as req

        api_url = f"{HF_ENDPOINT}/api/models/{repo_id}"
        resp = req.get(api_url, headers=HEADERS, timeout=timeout)
        resp.raise_for_status()
        meta = resp.json()
        filenames = [f["rfilename"] for f in meta.get("siblings", [])]
        if not filenames:
            print(f"  Could not discover files from API, trying safetensors checkpoint...")
            filenames = ["model.safetensors", "pytorch_model.bin", "model.bin"]

    print(f"Downloading model: {repo_id} ({len(filenames)} files)")
    if cache_dir is None:
        cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
    model_dir = Path(cache_dir) / f"models--{repo_id.replace('/', '--')}"
    model_dir.mkdir(parents=True, exist_ok=True)

    for i, fname in enumerate(filenames):
        try:
            _, fresh = hf_download_file(
                repo_id, fname, cache_dir=cache_dir,
                resume=True, timeout=timeout, max_retries=max_retries,
                retry_delay=retry_delay
            )
            if progress_callback:
                progress_callback(i + 1, len(filenames), fname)
        except RuntimeError:
            raise

    return str(model_dir)


def check_model_cached(repo_id: str, filenames: List[str], cache_dir: Optional[str] = None) -> bool:
    """Check if all model files are present in cache."""
    if cache_dir is None:
        cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
    return all(
        _get_local_file_path(repo_id, f, cache_dir).exists()
        for f in filenames
    )


def _mock_model_download(repo_id: str, shapes: dict) -> str:
    import tempfile, json
    cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
    model_dir = Path(cache_dir) / f"models--{repo_id.replace('/', '--')}"
    config = {
        "model_type": "custom",
        "architectures": ["S3ForCausalLM"],
        "torch_dtype": "float16",
    }
    (model_dir / "configs.json").write_text(json.dumps(config, indent=2))
    meta = {
        "format": "gguf", "parameter_count": sum(v[0]*v[1] for v in shapes.values()),
        "quantization": "f16"
    }
    for name, (rows, cols) in shapes.items():
        safetensors_file = model_dir / "blobs" / f"{name}.safetensors"
        safetensors_file.parent.mkdir(parents=True, exist_ok=True)
        with open(safetensors_file, "wb") as f:
            import struct
            header = struct.pack("QQ", rows, cols)
            f.write(header)
    return str(model_dir)


def load_model_with_resume(
    repo_id: str,
    device: str = "cpu",
    torch_dtype=torch.float16,
    timeout: int = 120,
    max_retries: int = 10,
    retry_delay: float = 5.0,
    filenames: Optional[List[str]] = None,
) -> Tuple:
    """Load a HuggingFace model with resilient, resumable downloads.

    Downloads are resumable - if the internet drops for ~1 minute,
    the download continues from ~10MB before the interruption point.

    Args:
        repo_id: HuggingFace model repo id
        device: Device to load model on
        torch_dtype: Model dtype
        timeout: Seconds per request
        max_retries: Max retries per file
        retry_delay: Delay between retries
        filenames: Specific files (auto-discovered if None)

    Returns:
        (model, tokenizer, model_dir)
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
    model_dir = Path(cache_dir) / f"models--{repo_id.replace('/', '--')}"

    if filenames is None:
        api_url = f"{HF_ENDPOINT}/api/models/{repo_id}"
        resp = requests.get(api_url, headers=HEADERS, timeout=timeout)
        if resp.status_code == 200:
            meta = resp.json()
            filenames = [f["rfilename"] for f in meta.get("siblings", [])]
        if not filenames:
            filenames = ["model.safetensors", "config.json"]

    all_cached = check_model_cached(repo_id, filenames, cache_dir)

    if not all_cached:
        print(f"Model files not fully cached for {repo_id}, downloading with resume support...")
        model_dir_str = hf_download_model(
            repo_id, filenames, cache_dir=cache_dir,
            timeout=timeout, max_retries=max_retries,
            retry_delay=retry_delay,
        )
        model_dir = Path(model_dir_str)

    print(f"Loading model from: {model_dir}")
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        str(model_dir),
        torch_dtype=torch_dtype,
        device_map=device,
        trust_remote_code=True,
    )
    return model, tokenizer, str(model_dir)