"""Correctness harness for the rebuilt S³ core (CPU, tiny Qwen2).

Two guarantees the old implementation failed:
  1. IDENTITY PARITY: with all adapters at init, the S3-wrapped model must
     reproduce the frozen base model's loss (Δ < 1e-3).
  2. LEARNING: a few optimizer steps on a fixed batch must drive the loss DOWN.

If both pass on the tiny model, the same code path scales to Qwen2.5-7B.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import copy
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from src.adapters.s3_block import wrap_model_with_s3


def build_tiny():
    cfg = Qwen2Config(
        vocab_size=256, hidden_size=64, intermediate_size=128,
        num_hidden_layers=3, num_attention_heads=8, num_key_value_heads=2,
        max_position_embeddings=128, tie_word_embeddings=False,
    )
    torch.manual_seed(0)
    return Qwen2ForCausalLM(cfg).eval()


def test_identity_parity():
    torch.manual_seed(0)
    base = build_tiny()
    ids = torch.randint(0, 256, (2, 16))
    with torch.no_grad():
        base_loss = float(base(ids, labels=ids).loss)

    s3 = copy.deepcopy(base)
    wrap_model_with_s3(s3, svmo_k=16, svmo_hidden=16, svmo_alpha=0.3,
                       nmf_bottleneck=8, nmf_N=4, stb_beta=0.5)
    s3.eval()
    with torch.no_grad():
        s3_loss = float(s3(ids, labels=ids).loss)

    delta = abs(base_loss - s3_loss)
    print(f"[parity] base={base_loss:.6f}  s3_init={s3_loss:.6f}  |Δ|={delta:.2e}")
    assert delta < 1e-3, f"identity parity FAILED: Δ={delta}"


def test_learning():
    torch.manual_seed(0)
    base = build_tiny()
    s3 = copy.deepcopy(base)
    wrap_model_with_s3(s3, svmo_k=16, svmo_hidden=16, svmo_alpha=0.3,
                       nmf_bottleneck=8, nmf_N=4, stb_beta=0.5)
    s3.train()

    trainable = [p for p in s3.parameters() if p.requires_grad]
    n_train = sum(p.numel() for p in trainable)
    n_total = sum(p.numel() for p in s3.parameters())
    print(f"[learning] trainable={n_train:,} / {n_total:,} ({100*n_train/n_total:.3f}%)")

    # overfit a single fixed batch — loss MUST fall
    ids = torch.randint(0, 256, (4, 24))
    opt = torch.optim.AdamW(trainable, lr=1e-2)
    losses = []
    for step in range(40):
        opt.zero_grad()
        loss = s3(ids, labels=ids).loss
        loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        opt.step()
        losses.append(float(loss))
        if step % 10 == 0:
            print(f"  step {step:2d}: loss={float(loss):.4f}  gnorm={float(gnorm):.3f}")
    print(f"[learning] loss {losses[0]:.4f} -> {losses[-1]:.4f}")
    assert torch.isfinite(torch.tensor(losses)).all(), "NaN/inf in loss"
    assert losses[-1] < losses[0] - 0.1, f"loss did not decrease: {losses[0]}->{losses[-1]}"


if __name__ == "__main__":
    print("=" * 60)
    test_identity_parity()
    print("PARITY: PASS")
    print("=" * 60)
    test_learning()
    print("LEARNING: PASS")
    print("=" * 60)
