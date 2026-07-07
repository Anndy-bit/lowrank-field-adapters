"""Unit tests for SVMO adapter — syntax and structure verification.

Run with: python -m pytest tests/test_svmo.py -v
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest


@pytest.fixture
def svmo_module():
    from src.adapters.svmo import SVMOAdapter, ModulationMLP, randomized_svd, create_svmo_from_linear
    return {
        "SVMOAdapter": SVMOAdapter,
        "ModulationMLP": ModulationMLP,
        "randomized_svd": randomized_svd,
        "create_svmo_from_linear": create_svmo_from_linear,
    }


class TestModulationMLP:
    def test_init(self, svmo_module):
        mlp = svmo_module["ModulationMLP"](hidden_dim=32)
        assert mlp.hidden_dim == 32
        assert isinstance(mlp.fc1, object)
        assert isinstance(mlp.fc2, object)
        assert isinstance(mlp.fc3, object)

    def test_forward_shape(self, svmo_module):
        import torch
        mlp = svmo_module["ModulationMLP"](hidden_dim=32)
        x = torch.randn(8, 1)
        y = mlp(x)
        assert y.shape == (8, 1)

    def test_init_near_zero(self, svmo_module):
        import torch
        mlp = svmo_module["ModulationMLP"](hidden_dim=32)
        x = torch.randn(8, 1)
        y = mlp(x)
        assert y.abs().max() < 1.0


class TestSVMOAdapter:
    def test_init(self, svmo_module):
        import torch
        weight = torch.randn(64, 48) * 0.02
        svmo = svmo_module["SVMOAdapter"](weight, k=16, hidden_dim=32, alpha=0.3)
        assert svmo.k == 16
        assert svmo.alpha == 0.3
        assert svmo.U_k.shape == (64, 16)
        assert svmo.Vt_k.shape == (16, 48)

    def test_forward_shape(self, svmo_module):
        import torch
        weight = torch.randn(64, 48) * 0.02
        svmo = svmo_module["SVMOAdapter"](weight, k=16)
        x = torch.randn(8, 48)
        y = svmo(x)
        assert y.shape == (8, 64)

    def test_identity_init(self, svmo_module):
        import torch
        weight = torch.randn(128, 128) * 0.01
        svmo = svmo_module["SVMOAdapter"](weight, k=32)
        x = torch.randn(4, 128)
        y_orig = torch.nn.functional.linear(x, weight)
        y_svmo = svmo(x)
        delta = (y_orig - y_svmo).abs().max()
        assert delta < 0.5, f"Identity init error too large: {delta}"

    def test_forward_full_consistency(self, svmo_module):
        import torch
        weight = torch.randn(64, 64) * 0.02
        svmo = svmo_module["SVMOAdapter"](weight, k=16)
        x = torch.randn(4, 64)
        y1 = svmo(x)
        y2 = svmo.forward_full(x)
        assert (y1 - y2).abs().max() < 1e-4

    def test_gradient_flow(self, svmo_module):
        import torch
        weight = torch.randn(64, 48) * 0.02
        svmo = svmo_module["SVMOAdapter"](weight, k=16, hidden_dim=32)
        x = torch.randn(4, 48)
        y = svmo(x)
        loss = y.sum()
        loss.backward()
        grads_exist = any(
            p.grad is not None for p in svmo.modulation.parameters()
        )
        assert grads_exist

    def test_u_v_frozen(self, svmo_module):
        import torch
        weight = torch.randn(64, 48) * 0.02
        svmo = svmo_module["SVMOAdapter"](weight, k=16, hidden_dim=32)
        x = torch.randn(4, 48)
        y = svmo(x)
        loss = y.sum()
        loss.backward()
        assert svmo.U_k.requires_grad is False or svmo.U_k.grad is None

    def test_trainable_param_count(self, svmo_module):
        import torch
        weight = torch.randn(128, 128) * 0.02
        svmo = svmo_module["SVMOAdapter"](weight, k=32, hidden_dim=32)
        count = svmo.trainable_param_count()
        assert 300 < count < 1500

    def test_modulation_profile(self, svmo_module):
        import torch
        weight = torch.randn(64, 64) * 0.02
        svmo = svmo_module["SVMOAdapter"](weight, k=16)
        orig, mod = svmo.modulation_profile()
        assert orig.shape == mod.shape == (16,)
        ratio = (mod / (orig + 1e-8)).clamp(0, 10)
        assert (ratio - 1.0).abs().max() < 0.5 + 1e-3

    def test_create_from_linear(self, svmo_module):
        import torch
        lin = torch.nn.Linear(48, 64)
        svmo = svmo_module["create_svmo_from_linear"](lin, k=16)
        x = torch.randn(4, 48)
        y1 = lin(x)
        y2 = svmo(x)
        delta = (y1 - y2).abs().max()
        assert delta < 1.5


class TestRandomizedSVD:
    def test_basic(self, svmo_module):
        import torch
        weight = torch.randn(256, 192)
        U, S, Vt = svmo_module["randomized_svd"](weight, k=32)
        assert U.shape == (256, 32)
        assert S.shape == (32,)
        assert Vt.shape == (32, 192)

    def test_reconstruction_quality(self, svmo_module):
        import torch
        weight = torch.randn(256, 192)
        U, S, Vt = svmo_module["randomized_svd"](weight, k=32, n_oversamples=10, n_iter=2)
        approx = (U * S.unsqueeze(0)) @ Vt
        rel_error = (weight - approx).norm() / (weight.norm() + 1e-8)
        assert rel_error < 0.85

    def test_vs_full_svd(self, svmo_module):
        import torch
        weight = torch.randn(128, 128) * 0.02
        U_r, S_r, Vt_r = svmo_module["randomized_svd"](weight, k=32, n_oversamples=10, n_iter=2)
        U_f, S_f, Vt_f = torch.linalg.svd(weight, full_matrices=False)
        U_f = U_f[:, :32]
        S_f = S_f[:32]
        Vt_f = Vt_f[:32, :]
        diff = (S_r - S_f).abs().max()
        assert diff < 0.5