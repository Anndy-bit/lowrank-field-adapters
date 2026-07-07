"""
Memory utilities for S³ frugal training.

VRAM monitoring, gradient accumulation helpers, CPU↔GPU transfer helpers.
"""

import torch
import time
import gc
from collections import deque
from typing import Optional, Dict, Any
from dataclasses import dataclass, field


@dataclass
class MemorySnapshot:
    allocated_mb: float = 0.0
    reserved_mb: float = 0.0
    max_allocated_mb: float = 0.0
    timestamp: float = 0.0


class VRAMMonitor:
    """Real-time GPU VRAM usage tracker.

    Usage:
        monitor = VRAMMonitor("cuda:0")
        monitor.start()
        # ... training ...
        peak = monitor.peak_mb
        monitor.stop()
    """

    def __init__(self, device: str = "cuda:0", poll_interval_ms: int = 100):
        self.device = torch.device(device) if device.startswith("cuda") else None
        self.poll_interval_ms = poll_interval_ms
        self.history: deque[MemorySnapshot] = deque(maxlen=10000)
        self._running = False
        self._peak_mb = 0.0

    @property
    def peak_mb(self) -> float:
        return self._peak_mb

    def snapshot(self) -> MemorySnapshot:
        if self.device is None:
            return MemorySnapshot()
        return MemorySnapshot(
            allocated_mb=torch.cuda.memory_allocated(self.device) / (1024 * 1024),
            reserved_mb=torch.cuda.memory_reserved(self.device) / (1024 * 1024),
            max_allocated_mb=torch.cuda.max_memory_allocated(self.device) / (1024 * 1024),
            timestamp=time.time(),
        )

    def start(self):
        if self.device:
            torch.cuda.reset_peak_memory_stats(self.device)
        self._running = True
        self._peak_mb = 0.0

    def record(self):
        if not self._running:
            return
        snap = self.snapshot()
        self.history.append(snap)
        if snap.allocated_mb > self._peak_mb:
            self._peak_mb = snap.allocated_mb

    def stop(self) -> Dict[str, Any]:
        self._running = False
        return {
            "peak_allocated_mb": self._peak_mb,
            "peak_reserved_mb": max(
                (s.reserved_mb for s in self.history), default=0.0
            ),
            "num_snapshots": len(self.history),
        }

    def clear(self):
        self.history.clear()
        self._peak_mb = 0.0
        if self.device:
            torch.cuda.reset_peak_memory_stats(self.device)
            torch.cuda.empty_cache()


class GradientAccumulator:
    """Accumulates gradients across micro-batches on CPU to save GPU VRAM.

    Instead of keeping gradients in GPU across micro-steps, we accumulate
    them on CPU after each backward pass.
    """

    def __init__(self, params: list[torch.nn.Parameter]):
        self.params = params
        self.accumulated = [
            torch.zeros_like(p.data, device="cpu") for p in params
        ]

    def add(self):
        for acc, param in zip(self.accumulated, self.params):
            if param.grad is not None:
                acc.add_(param.grad.data.detach().cpu())

    def apply_and_zero(self):
        for param, acc in zip(self.params, self.accumulated):
            if param.grad is not None:
                param.grad.data.copy_(acc.to(param.device))
            else:
                param.grad = acc.to(param.device)
            acc.zero_()

    def normalized_grad(self) -> float:
        total = 0.0
        for acc in self.accumulated:
            total += acc.norm().item() ** 2
        return total ** 0.5


def clear_gpu_memory(device: str = "cuda:0"):
    """Aggressive GPU memory cleanup."""
    gc.collect()
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
        torch.cuda.synchronize(device)


def estimate_model_vram(
    num_params: int, dtype_bytes: int = 2, optimizer_factor: float = 3.0
) -> float:
    """Estimate VRAM for model weights + optimizer states.

    Args:
        num_params: total parameter count
        dtype_bytes: bytes per param (2=fp16, 4=fp32)
        optimizer_factor: optimizer memory multiplier
                          (2 for SGD, 3 for AdamW, 4 for AdamW+fp32 states)

    Returns:
        Estimated VRAM in MB.
    """
    model_mb = num_params * dtype_bytes / (1024 * 1024)
    opt_mb = num_params * dtype_bytes * optimizer_factor / (1024 * 1024)
    return model_mb + opt_mb


def memory_report(monitor: VRAMMonitor) -> str:
    """Human-readable memory report."""
    stats = monitor.stop()
    return (
        f"VRAM Peak: {stats['peak_allocated_mb']:.1f} MB allocated, "
        f"{stats['peak_reserved_mb']:.1f} MB reserved | "
        f"{stats['num_snapshots']} snapshots"
    )