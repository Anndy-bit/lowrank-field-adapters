"""VRAM memory tests — verify peak usage predictions.

Run with: python -m pytest tests/test_memory.py -v
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest


class TestVRAMEstimates:
    def test_svmo_vram_estimate(self):
        from src.adapters.svmo import SVMOAdapter
        import torch

        weight = torch.randn(4096, 4096)
        svmo = SVMOAdapter(weight, k=128, hidden_dim=32)
        est = svmo.vram_estimate_mb(dtype_size=2)
        assert est < 5.0
        assert est > 0.1

    def test_svmo_trainable_params_small(self):
        from src.adapters.svmo import SVMOAdapter
        import torch

        weight = torch.randn(4096, 4096) * 0.02
        svmo = SVMOAdapter(weight, k=128, hidden_dim=32)
        n = svmo.trainable_param_count()
        assert n < 1500

    def test_nmf_vram_estimate(self):
        from src.adapters.nmf import NMFFlow

        flow = NMFFlow(dim=4096, bottleneck_dim=8, T=1.0, N_steps=4)
        est = flow.vram_estimate_mb(dtype_size=2)
        assert est < 1.0
        assert est > 0.01

    def test_stb_vram_estimate(self):
        from src.adapters.stb import STBBridge
        import torch

        U_k = torch.randn(4096, 128)
        bridge = STBBridge(U_k, k=128, beta=0.5)
        est = bridge.vram_estimate_mb(dtype_size=2)
        assert est < 5.0

    def test_total_vram_target(self):
        import torch
        from src.adapters.svmo import SVMOAdapter
        from src.adapters.nmf import NMFFlow
        from src.adapters.stb import STBBridge

        svmo_ests = 0
        for _ in range(7):
            w = torch.randn(4096, 4096)
            svmo = SVMOAdapter(w, k=128, hidden_dim=32)
            svmo_ests += svmo.vram_estimate_mb(2)

        nmf_ests = 0
        for _ in range(2):
            flow = NMFFlow(dim=4096, bottleneck_dim=8)
            nmf_ests += flow.vram_estimate_mb(2)

        U_k = torch.randn(4096, 128)
        bridge = STBBridge(U_k, k=128, beta=0.5)
        stb_est = bridge.vram_estimate_mb(2)

        adapter_total = svmo_ests + nmf_ests + stb_est
        weight_layer_mb = 7 * 4096 * 4096 * 2 / (1024 * 1024)

        peak_vram = adapter_total + weight_layer_mb + 10 + 50
        assert peak_vram < 2000, f"Peak VRAM {peak_vram:.0f}MB exceeds 2GB"


class TestMemoryUtils:
    def test_estimate_model_vram(self):
        from src.utils.memory_utils import estimate_model_vram

        est = estimate_model_vram(num_params=8_000_000_000, dtype_bytes=2)
        assert 1000 < est < 100000

    def test_clear_gpu_memory_noop(self):
        from src.utils.memory_utils import clear_gpu_memory

        clear_gpu_memory("cpu")

    def test_gradient_accumulator_noop(self):
        from src.utils.memory_utils import GradientAccumulator
        import torch

        params = [torch.nn.Parameter(torch.randn(10))]
        acc = GradientAccumulator(params)
        params[0].grad = torch.randn(10)
        acc.add()
        acc.apply_and_zero()
        assert params[0].grad is not None