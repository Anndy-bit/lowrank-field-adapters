"""Unit tests for STB adapter."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest


@pytest.fixture
def stb_module():
    from src.adapters.stb import SpectralCoupling, STBBridge, create_stb_bridge_from_svmo
    return {
        "SpectralCoupling": SpectralCoupling,
        "STBBridge": STBBridge,
        "create_stb_bridge_from_svmo": create_stb_bridge_from_svmo,
    }


class TestSpectralCoupling:
    def test_init(self, stb_module):
        c = stb_module["SpectralCoupling"](k=128, head_dim=32, beta=0.5)
        assert c.k == 128
        assert c.beta == 0.5

    def test_forward_shape(self, stb_module):
        import torch
        c = stb_module["SpectralCoupling"](k=128, head_dim=32, beta=0.5)
        s = torch.randn(4, 128)
        sigma_log = torch.randn(128)
        out = c(s, sigma_log)
        assert out.shape == (4, 128)

    def test_identity_ish(self, stb_module):
        import torch
        c = stb_module["SpectralCoupling"](k=64, head_dim=16, beta=0.1)
        s = torch.randn(3, 64)
        sigma_log = torch.randn(64)
        out = c(s, sigma_log)
        delta = (out - s).norm(dim=-1).mean()
        assert delta < 5.0

    def test_gradient_flow(self, stb_module):
        import torch
        c = stb_module["SpectralCoupling"](k=64, head_dim=16, beta=0.5)
        s = torch.randn(3, 64)
        sigma_log = torch.randn(64)
        out = c(s, sigma_log)
        loss = out.sum()
        loss.backward()
        grads = [p.grad is not None for p in c.parameters()]
        assert any(grads)


class TestSTBBridge:
    def test_init(self, stb_module):
        import torch
        U_k = torch.randn(512, 128)
        bridge = stb_module["STBBridge"](U_k, k=128, beta=0.5)
        assert bridge.k == 128
        assert bridge.beta == 0.5

    def test_forward_shape(self, stb_module):
        import torch
        U_k = torch.randn(512, 128)
        bridge = stb_module["STBBridge"](U_k, k=128, beta=0.5)
        h = torch.randn(4, 512)
        sigma_log = torch.randn(128)
        h_out = bridge(h, sigma_log)
        assert h_out.shape == (4, 512)

    def test_spectral_signature(self, stb_module):
        import torch
        U_k = torch.randn(512, 64)
        bridge = stb_module["STBBridge"](U_k, k=64, beta=0.5)
        h = torch.randn(4, 512)
        s = bridge.spectral_signature(h)
        assert s.shape == (4, 64)

    def test_gradient_flow(self, stb_module):
        import torch
        U_k = torch.randn(256, 64)
        bridge = stb_module["STBBridge"](U_k, k=64, beta=0.5)
        h = torch.randn(3, 256)
        sigma_log = torch.randn(64)
        h_out = bridge(h, sigma_log)
        loss = h_out.sum()
        loss.backward()
        has_grad = any(
            p.grad is not None for p in bridge.coupling.parameters()
        )
        assert has_grad

    def test_u_k_frozen(self, stb_module):
        import torch
        U_k = torch.randn(256, 64)
        bridge = stb_module["STBBridge"](U_k, k=64, beta=0.5)
        h = torch.randn(3, 256)
        sigma_log = torch.randn(64)
        h_out = bridge(h, sigma_log)
        loss = h_out.sum()
        loss.backward()
        assert bridge.U_k.requires_grad is False or bridge.U_k.grad is None


class TestCreateBridgeFromSVMO:
    def test_basic(self, stb_module):
        import torch
        from src.adapters.svmo import SVMOAdapter, create_svmo_from_linear
        lin = torch.nn.Linear(256, 256)
        svmo = create_svmo_from_linear(lin, k=32)
        bridge = stb_module["create_stb_bridge_from_svmo"](svmo, beta=0.3)
        assert bridge.k == 32
        assert bridge.beta == 0.3