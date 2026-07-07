"""Unit tests for NMF adapter."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest


@pytest.fixture
def nmf_module():
    from src.adapters.nmf import VelocityField, NMFFlow, NMFAdjoint, create_nmf_pair
    return {
        "VelocityField": VelocityField,
        "NMFFlow": NMFFlow,
        "NMFAdjoint": NMFAdjoint,
        "create_nmf_pair": create_nmf_pair,
    }


class TestVelocityField:
    def test_init(self, nmf_module):
        vf = nmf_module["VelocityField"](dim=512, bottleneck_dim=8)
        assert vf.dim == 512
        assert vf.bottleneck_dim == 8

    def test_forward_shape(self, nmf_module):
        import torch
        vf = nmf_module["VelocityField"](dim=512, bottleneck_dim=8)
        h = torch.randn(4, 512)
        out = vf(h, t=0.5)
        assert out.shape == (4, 512)

    def test_init_near_zero(self, nmf_module):
        import torch
        vf = nmf_module["VelocityField"](dim=512, bottleneck_dim=8)
        h = torch.randn(4, 512)
        out = vf(h, t=0.0)
        assert out.abs().max() < 1.0


class TestNMFFlow:
    def test_init(self, nmf_module):
        flow = nmf_module["NMFFlow"](dim=512, bottleneck_dim=8, T=1.0, N_steps=4)
        assert flow.T == 1.0
        assert flow.N_steps == 4
        assert flow.dt == 0.25

    def test_forward_rk4(self, nmf_module):
        import torch
        flow = nmf_module["NMFFlow"](dim=512, bottleneck_dim=8, T=1.0, N_steps=4)
        h = torch.randn(2, 512)
        h_out = flow(h)
        assert h_out.shape == (2, 512)
        assert torch.isfinite(h_out).all()

    def test_forward_euler(self, nmf_module):
        import torch
        flow = nmf_module["NMFFlow"](dim=256, bottleneck_dim=4, T=0.5, N_steps=2, solver="euler")
        h = torch.randn(3, 256)
        h_out = flow(h)
        assert h_out.shape == (3, 256)

    def test_identity_init(self, nmf_module):
        import torch
        flow = nmf_module["NMFFlow"](dim=256, bottleneck_dim=4, T=1.0, N_steps=4)
        h = torch.randn(4, 256)
        h_out = flow(h)
        delta = (h_out - h).norm(dim=-1).mean()
        assert delta < 2.0, f"Identity init delta too large: {delta.item()}"

    def test_deformation(self, nmf_module):
        import torch
        flow = nmf_module["NMFFlow"](dim=256, bottleneck_dim=4, T=1.0, N_steps=4)
        h = torch.randn(4, 256)
        delta = flow.deformation(h)
        assert delta.shape == (4, 256)
        assert delta.norm() > 0

    def test_gradient_flow(self, nmf_module):
        import torch
        flow = nmf_module["NMFFlow"](dim=128, bottleneck_dim=4, T=1.0, N_steps=4)
        h = torch.randn(2, 128)
        h_out = flow(h)
        loss = h_out.sum()
        loss.backward()
        has_grad = any(
            p.grad is not None for p in flow.field.parameters()
        )
        assert has_grad

    def test_create_pair(self, nmf_module):
        f1, f2 = nmf_module["create_nmf_pair"](dim=768, bottleneck_dim=8)
        assert f1.dim == 768
        assert f2.dim == 768
        assert f1 is not f2


class TestNMFAdjoint:
    def test_init(self, nmf_module):
        adj = nmf_module["NMFAdjoint"](dim=256, bottleneck_dim=4, T=1.0, N_steps=4)
        assert adj.dim == 256
        assert adj.T == 1.0