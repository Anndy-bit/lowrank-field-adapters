"""USF layer-major (formalismo_streaming.md §6, Thm 4) — exactness test.

Thm 4 claims: inverting the loops (keep unit i resident, push all G micro-batches
through it) yields gradients IDENTICAL to batch-major, because gradient
accumulation is a sum and sums are order-invariant.

This test is the empirical check of that claim: same model, same data, same seed,
batch-major vs layer-major -> gradients must match.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from src.adapters.s3_block import wrap_model_with_s3
from src.training.s3_trainer import S3Trainer, S3TrainConfig


def build(seed=0):
    cfg = Qwen2Config(vocab_size=256, hidden_size=64, intermediate_size=128,
                      num_hidden_layers=3, num_attention_heads=8, num_key_value_heads=2,
                      max_position_embeddings=128, tie_word_embeddings=False)
    torch.manual_seed(seed)
    m = Qwen2ForCausalLM(cfg)
    # sbs_p=1.0 -> STB always applied: removes the only stochastic branch so the
    # two orderings are compared on identical computation.
    blocks = wrap_model_with_s3(m, svmo_k=16, svmo_hidden=16, nmf_bottleneck=8, sbs_p=1.0)
    return m, blocks


def make_batches(G=4, B=2, S=16, seed=1):
    g = torch.Generator().manual_seed(seed)
    out = []
    for _ in range(G):
        ids = torch.randint(1, 256, (B, S), generator=g)
        attn = torch.ones(B, S, dtype=torch.long)
        out.append({"input_ids": ids, "attention_mask": attn})
    return out


def grads_of(model):
    return {n: p.grad.detach().clone() for n, p in model.named_parameters()
            if p.requires_grad and p.grad is not None}


def test_layer_major_equals_batch_major():
    batches = make_batches()
    G = len(batches)

    # --- batch-major: one step_streaming per micro-batch, grads accumulate ---
    m1, b1 = build()
    t1 = S3Trainer(m1, b1, S3TrainConfig(amp=False, grad_accum_steps=G, ckpt_offload=False),
                   device="cpu")
    m1.train()
    for b in batches:
        t1.step_streaming(b)
    g_bm = grads_of(m1)

    # --- layer-major: one step over all G micro-batches ---
    m2, b2 = build()
    m2.load_state_dict(m1.state_dict())
    t2 = S3Trainer(m2, b2, S3TrainConfig(amp=False, grad_accum_steps=G, ckpt_offload=True),
                   device="cpu")
    m2.train()
    t2.step_layer_major(batches)
    g_lm = grads_of(m2)

    assert set(g_bm) == set(g_lm), "different trainable params"
    max_diff = max(float((g_bm[n] - g_lm[n]).abs().max()) for n in g_bm)
    rel = max_diff / max(1e-12, max(float(g_bm[n].abs().max()) for n in g_bm))
    print(f"[Thm 4] params={len(g_bm)}  max|Δgrad| = {max_diff:.3e}  (rel {rel:.3e})")
    assert max_diff < 1e-6, f"layer-major changed the gradients: {max_diff}"


def test_ckpt_offload_is_neutral():
    """Cor 2.1: moving checkpoints to CPU must not change anything."""
    batches = make_batches()
    m1, b1 = build()
    t1 = S3Trainer(m1, b1, S3TrainConfig(amp=False, grad_accum_steps=len(batches),
                                         ckpt_offload=False), device="cpu")
    m1.train(); t1.step_layer_major(batches); g_on = grads_of(m1)

    m2, b2 = build()
    m2.load_state_dict(m1.state_dict())
    t2 = S3Trainer(m2, b2, S3TrainConfig(amp=False, grad_accum_steps=len(batches),
                                         ckpt_offload=True), device="cpu")
    m2.train(); t2.step_layer_major(batches); g_off = grads_of(m2)

    max_diff = max(float((g_on[n] - g_off[n]).abs().max()) for n in g_on)
    print(f"[Cor 2.1] checkpoint offload max|Δgrad| = {max_diff:.3e}")
    assert max_diff < 1e-9, f"offload changed gradients: {max_diff}"


if __name__ == "__main__":
    print("=" * 64)
    test_layer_major_equals_batch_major(); print("THM 4 (layer-major == batch-major): PASS")
    print("=" * 64)
    test_ckpt_offload_is_neutral(); print("COR 2.1 (ckpt offload neutral): PASS")
    print("=" * 64)
