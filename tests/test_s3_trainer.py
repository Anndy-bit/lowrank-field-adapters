"""Verify S3Trainer: end-to-end learning + layer-swap gradient equivalence (CPU)."""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from src.adapters.s3_block import wrap_model_with_s3
from src.training.s3_trainer import S3Trainer, S3TrainConfig


def build():
    cfg = Qwen2Config(vocab_size=256, hidden_size=64, intermediate_size=128,
                      num_hidden_layers=3, num_attention_heads=8, num_key_value_heads=2,
                      max_position_embeddings=128, tie_word_embeddings=False)
    torch.manual_seed(0)
    m = Qwen2ForCausalLM(cfg)
    blocks = wrap_model_with_s3(m, svmo_k=16, svmo_hidden=16, nmf_bottleneck=8)
    return m, blocks


class ListLoader:
    def __init__(self, batches): self.batches = batches
    def __iter__(self): return iter(self.batches)
    def __len__(self): return len(self.batches)


def test_end_to_end_learning():
    m, blocks = build()
    cfg = S3TrainConfig(learning_rate=1e-2, max_epochs=6, grad_accum_steps=1,
                        gradient_checkpointing=False, layer_swap=False, amp=False)
    tr = S3Trainer(m, blocks, cfg, device="cpu")
    torch.manual_seed(1)
    batches = [{"input_ids": torch.randint(0, 256, (4, 24))}]
    loader = ListLoader(batches * 8)

    def loss_now():
        m.eval()
        with torch.no_grad():
            l = tr.forward_loss(batches[0]["input_ids"])
        m.train()
        return float(l)

    l0 = loss_now()
    tr.train(loader)
    l1 = loss_now()
    print(f"[trainer] loss {l0:.4f} -> {l1:.4f}")
    assert l1 < l0 - 0.05, "trainer did not reduce loss"


def test_layerswap_grad_equivalence():
    """Grads with layer_swap must equal grads without it (CPU: swap moves cpu->cpu)."""
    m1, b1 = build()
    m2, b2 = build()
    m2.load_state_dict(m1.state_dict())  # identical params

    ids = torch.randint(0, 256, (2, 20))
    cfg_a = S3TrainConfig(gradient_checkpointing=False, layer_swap=False, amp=False)
    cfg_b = S3TrainConfig(gradient_checkpointing=False, layer_swap=True, amp=False)
    ta = S3Trainer(m1, b1, cfg_a, device="cpu")
    tb = S3Trainer(m2, b2, cfg_b, device="cpu")

    la = ta.forward_loss(ids); la.backward()
    lb = tb.forward_loss(ids); lb.backward()

    print(f"[swap] loss resident={float(la):.6f}  swap={float(lb):.6f}")
    assert abs(float(la) - float(lb)) < 1e-5, "swap changed the forward"

    ga = {n: p.grad.clone() for n, p in m1.named_parameters() if p.grad is not None}
    gb = {n: p.grad for n, p in m2.named_parameters() if p.grad is not None}
    assert set(ga) == set(gb), "different trainable params"
    max_diff = max(float((ga[n] - gb[n]).abs().max()) for n in ga)
    print(f"[swap] trainable-grad max|Δ|={max_diff:.2e}  (params={len(ga)})")
    assert max_diff < 1e-5, f"swap changed gradients: {max_diff}"


if __name__ == "__main__":
    print("=" * 60)
    test_end_to_end_learning(); print("END-TO-END LEARNING: PASS")
    print("=" * 60)
    test_layerswap_grad_equivalence(); print("LAYER-SWAP EQUIVALENCE: PASS")
    print("=" * 60)
