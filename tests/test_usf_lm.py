"""Verify the lm_eval backend scores exactly what a plain forward would (CPU).

The risk this file exists for is silent wrongness. A scoring backend that
mis-indexes by one position, or lets right-padding leak into attention, still
returns plausible log-likelihoods and still produces a benchmark table — just a
wrong one. So the tests compare against an independent reference computed with
the model's own forward, on requests of DIFFERENT lengths so that padding is
actually exercised rather than assumed away.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, Qwen2Config, Qwen2ForCausalLM

from src.adapters.s3_block import wrap_model_with_s3
from src.training.s3_trainer import S3Trainer, S3TrainConfig
from src.eval.usf_lm import USFLM


class Req:
    """Stands in for an lm_eval Instance; the backend only reads `.args`."""
    def __init__(self, ctx, cont):
        self.args = (ctx, cont)


def build():
    tok = AutoTokenizer.from_pretrained("gpt2")
    tok.pad_token = tok.eos_token
    cfg = Qwen2Config(vocab_size=len(tok), hidden_size=64, intermediate_size=128,
                      num_hidden_layers=3, num_attention_heads=8, num_key_value_heads=2,
                      max_position_embeddings=512, tie_word_embeddings=False)
    torch.manual_seed(0)
    m = Qwen2ForCausalLM(cfg)
    m.eval()
    blocks = wrap_model_with_s3(m, svmo_k=16, svmo_hidden=16, nmf_bottleneck=8)
    tr = S3Trainer(m, blocks, S3TrainConfig(gradient_checkpointing=False,
                                            layer_swap=False, amp=False,
                                            ckpt_offload=False),
                   device="cpu")
    return m, tok, tr


def ref_loglikelihood(model, tok, ctx, cont):
    """Reference: score the pair with the model's own forward, unpadded."""
    ctx_ids = tok.encode(ctx, add_special_tokens=False)
    whole = tok.encode(ctx + cont, add_special_tokens=False)
    cont_ids = whole[len(ctx_ids):]
    with torch.no_grad():
        logits = model(torch.tensor([whole[:-1]])).logits
    lo = len(ctx_ids) - 1
    lp = F.log_softmax(logits[0, lo:lo + len(cont_ids)].float(), dim=-1)
    tgt = torch.tensor(cont_ids)
    ll = float(lp.gather(-1, tgt.unsqueeze(-1)).sum())
    greedy = bool((lp.argmax(-1) == tgt).all())
    return ll, greedy


PAIRS = [                       # deliberately unequal lengths -> real padding
    ("The capital of France is", " Paris"),
    ("A much longer piece of context that runs on for a while before it", " ends"),
    ("Q: 2+2?\nA:", " four"),
    ("Short", " one"),
    ("Water boils at one hundred degrees", " celsius and freezes at zero"),
]


def test_loglikelihood_matches_plain_forward():
    m, tok, tr = build()
    lm = USFLM(tr, tok, chunk_requests=64, batch_size=4)
    got = lm.loglikelihood([Req(c, k) for c, k in PAIRS])

    for (ctx, cont), (ll, greedy) in zip(PAIRS, got):
        ref_ll, ref_greedy = ref_loglikelihood(m, tok, ctx, cont)
        assert abs(ll - ref_ll) < 1e-3, f"{ctx!r}: {ll} vs reference {ref_ll}"
        assert greedy == ref_greedy


def test_padding_does_not_leak():
    """Same request scored alone vs batched with a much longer one."""
    m, tok, tr = build()
    short = Req("Short", " one")
    long = Req("A much longer piece of context that runs on for a while before it", " ends")

    alone = USFLM(tr, tok, batch_size=1).loglikelihood([short])[0]
    together = USFLM(tr, tok, batch_size=4).loglikelihood([short, long])[0]
    assert abs(alone[0] - together[0]) < 1e-3, (
        f"padding leaked: {alone[0]} alone vs {together[0]} batched")


def test_chunking_is_invariant():
    """Thm 4: the loop order is free, so chunk size must not move the numbers."""
    m, tok, tr = build()
    reqs = [Req(c, k) for c, k in PAIRS]
    one_read = USFLM(tr, tok, chunk_requests=64, batch_size=8).loglikelihood(reqs)
    many_reads = USFLM(tr, tok, chunk_requests=1, batch_size=1).loglikelihood(reqs)
    for (a, _), (b, _) in zip(one_read, many_reads):
        assert abs(a - b) < 1e-4, f"chunking changed the score: {a} vs {b}"


def test_forward_layer_major_matches_eval_loss():
    """The new forward must agree with the batch-major one it replaces."""
    m, tok, tr = build()
    torch.manual_seed(3)
    ids = torch.randint(0, 1000, (2, 16))
    batch = {"input_ids": ids, "attention_mask": torch.ones_like(ids)}

    ref = tr.eval_loss(batch)
    H, labels = tr.forward_layer_major([batch])
    head = tr.lm_head.weight
    logits = F.linear(H[0].to(head.device, dtype=head.dtype), head)
    got = float(F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(),
                                labels[0].reshape(-1).to(head.device), ignore_index=-100))
    assert abs(got - ref) < 1e-4, f"layer-major {got} != batch-major {ref}"


def test_encode_pair_survives_boundary_merge():
    """enc(ctx)+enc(cont) != enc(ctx+cont) when the tokeniser merges across the
    seam; the split must follow the joint encoding, which is what the model reads."""
    m, tok, tr = build()
    lm = USFLM(tr, tok)
    for ctx, cont in PAIRS + [("hel", "lo world"), ("test ", "ing")]:
        ctx_ids, cont_ids = lm._encode_pair(ctx, cont)
        n_spaces = len(ctx) - len(ctx.rstrip())
        joint = tok.encode(ctx.rstrip() + (ctx[-n_spaces:] if n_spaces else "") + cont,
                           add_special_tokens=False)
        assert ctx_ids + cont_ids == joint, f"{ctx!r}|{cont!r} lost the seam"
        assert len(cont_ids) > 0
