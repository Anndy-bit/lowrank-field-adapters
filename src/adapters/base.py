"""
Base classes for S³ adapters.

Abstract interface and utilities shared by SVMO, NMF, and STB.
"""

import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Dict, Any


class BaseAdapter(ABC, nn.Module):
    """Abstract base for all S³ adapters.

    Defines the minimal contract: every adapter must report its
    parameter counts and VRAM estimate for frugal training.
    """

    @abstractmethod
    def trainable_param_count(self) -> int:
        """Number of parameters that receive gradients."""
        ...

    @abstractmethod
    def vram_estimate_mb(self, dtype_size: int = 2) -> float:
        """Peak VRAM in MB for this adapter during forward+backward."""
        ...

    def param_summary(self) -> Dict[str, Any]:
        return {
            "trainable": self.trainable_param_count(),
            "total": sum(p.numel() for p in self.parameters()),
            "vram_mb": self.vram_estimate_mb(),
        }


class FrozenWeightStore(nn.Module):
    """Holds frozen pretrained weights for one transformer layer.

    Weights are stored as buffers (no grad) and can be moved between
    CPU and GPU during layer-by-layer training.

    Args:
        layer_weights: dict mapping name → torch.Tensor for each weight matrix
                       in the transformer layer (Q,K,V,O,up,gate,down weights + biases)
    """

    def __init__(self, layer_weights: Dict[str, nn.Parameter]):
        super().__init__()
        self.weight_names = []
        self.total_bytes = 0
        for name, param in layer_weights.items():
            buffer = param.data.detach().clone()
            self.register_buffer(name, buffer)
            self.weight_names.append(name)
            self.total_bytes += buffer.numel() * buffer.element_size()

    @property
    def total_mb(self) -> float:
        return self.total_bytes / (1024 * 1024)

    def to_device(self, device):
        for name in self.weight_names:
            buf = getattr(self, name)
            setattr(self, name, buf.to(device))
        return self

    def cpu(self):
        return self.to_device("cpu")

    def cuda(self, device=None):
        return self.to_device("cuda" if device is None else device)

    def extra_repr(self) -> str:
        return f"matrices={self.weight_names}, total={self.total_mb:.1f}MB"


def count_total_params(module: nn.Module, trainable_only: bool = False) -> int:
    if trainable_only:
        return sum(p.numel() for p in module.parameters() if p.requires_grad)
    return sum(p.numel() for p in module.parameters())


def vram_estimate_all(modules: list[BaseAdapter], dtype_size: int = 2) -> float:
    return sum(m.vram_estimate_mb(dtype_size) for m in modules)