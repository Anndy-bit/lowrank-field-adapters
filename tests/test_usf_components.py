"""USF components 4/5/6 (formalismo_streaming.md) — correctness tests.

  Thm 3 (§5)  granularity: streaming attn|mlp separately must not change gradients.
  Prop 5 (§7) prefetch:    reading ahead must not change gradients.
  §9          auto-selection: pick the coarsest granularity that fits the budget.

Every claim in the formalism gets a test — the same policy used for Thm 1 / Thm 4.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from src.adapters.s3_block import wrap_model_with_s3
from src.training.s3_trainer import S3Trainer, S3TrainConfig
from src.training.streaming_loader import (
    GRANULARITIES, SUPPORTED_GRANULARITIES, unit_sizes, choose_granularity)


def build(seed=0):
    cfg = Qwen2Config(vocab_size=256, hidden_size=64, intermediate_size=128,
                      num_hidden_layers=3, num_attention_heads=8, num_key_value_heads=2,
                      max_position_embeddings=128, tie_word_embeddings=False)
    torch.manual_seed(seed)
    m = Qwen2ForCausalLM(cfg)
    blocks = wrap_model_with_s3(m, svmo_k=16, svmo_hidden=16, nmf_bottleneck=8, sbs_p=1.0)
    return m, blocks


def batches(G=2, B=2, S=16, seed=1):
    g = torch.Generator().manual_seed(seed)
    return [{"input_ids": torch.randint(1, 256, (B, S), generator=g),
             "attention_mask": torch.ones(B, S, dtype=torch.long)} for _ in range(G)]


class FakeStreamer:
    """Streamer with the real granularity plan but no disk: the tiny model's
    weights are already resident, so load/unload are no-ops. This isolates the
    *segmentation* logic (Thm 3) from the I/O."""

    def __init__(self, granularity="layer"):
        self.granularity = granularity
        self.loads = 0

    def units(self, idx):
        return list(range(len(GRANULARITIES[self.granularity])))

    def load(self, block, idx, unit=None):
        self.loads += 1

    def unload(self, block, idx, unit=None):
        pass

    def prefetch(self, idx, unit=None):
        pass


def grads_of(model):
    return {n: p.grad.detach().clone() for n, p in model.named_parameters()
            if p.requires_grad and p.grad is not None}


def _run(granularity, bs):
    m, b = build()
    tr = S3Trainer(m, b, S3TrainConfig(amp=False, grad_accum_steps=len(bs), ckpt_offload=True),
                   device="cpu", frozen_streamer=FakeStreamer(granularity))
    m.train()
    tr.step_layer_major(bs)
    return m, grads_of(m), tr.frozen_streamer.loads


def test_subunit_forward_matches_monolithic():
    """Thm 3: blk.forward_attn -> blk.forward_mlp must equal blk.forward.

    This is what licenses sub-layer streaming: the split is a re-association of
    the same composition, so the peak VRAM drops to max(|W_attn|,|W_mlp|) without
    touching the maths.
    """
    m, blocks = build()
    m.eval()
    blk = blocks[0]
    torch.manual_seed(3)
    h = torch.randn(2, 16, 64)
    pos = m.model.rotary_emb(h, torch.arange(16).unsqueeze(0))
    mask = torch.triu(torch.full((16, 16), float("-inf")), diagonal=1).view(1, 1, 16, 16)
    with torch.no_grad():
        whole = blk(h, attention_mask=mask, position_embeddings=pos, stb_apply=True)
        split = blk.forward_mlp(
            blk.forward_attn(h, attention_mask=mask, position_embeddings=pos, stb_apply=True))
    d = float((whole - split).abs().max())
    print(f"[Thm 3] |forward - (forward_attn->forward_mlp)| = {d:.3e}")
    assert d < 1e-6, f"sub-unit split changed the forward: {d}"


def test_granularity_hierarchy_is_monotone():
    """Thm 3: max_u|W_u| shrinks monotonically, and each plan is a real partition."""
    cfg = Qwen2Config(hidden_size=4096, intermediate_size=14336,
                      num_attention_heads=32, num_key_value_heads=8)
    sizes = {g: unit_sizes(cfg, g) for g in GRANULARITIES}
    print("[Thm 3] unit sizes (MB): " + ", ".join(f"{g}={s/2**20:.1f}" for g, s in sizes.items()))
    assert sizes["matrix"] <= sizes["sublayer"] <= sizes["layer"]
    for g, plan in GRANULARITIES.items():
        flat = [p for unit in plan for p in unit]
        assert len(flat) == 7 and len(set(flat)) == 7, f"{g} is not a partition"


def test_sublayer_granularity_preserves_gradients():
    """Thm 3: splitting a block into attn|mlp units must not change gradients."""
    bs = batches()
    m1, g_layer, loads_layer = _run("layer", bs)
    m2, g_sub, loads_sub = _run("sublayer", bs)
    assert set(g_layer) == set(g_sub)
    max_diff = max(float((g_layer[n] - g_sub[n]).abs().max()) for n in g_layer)
    print(f"[Thm 3] layer vs sublayer: max|Δgrad| = {max_diff:.3e} | "
          f"loads {loads_layer} -> {loads_sub} (2x units, same I/O bytes)")
    assert max_diff < 1e-6, f"granularity changed gradients: {max_diff}"
    # sublayer must issue exactly 2x the loads (fwd+bwd over 2 units per block)
    assert loads_sub == 2 * loads_layer


def test_prefetch_preserves_gradients():
    """Prop. 5: prefetching changes WHEN a tensor is read, never its value."""
    class PrefetchingFake(FakeStreamer):
        def __init__(self, g="layer"):
            super().__init__(g)
            self.prefetches = 0

        def prefetch(self, idx, unit=None):
            self.prefetches += 1

    bs = batches()
    m1, g_plain, _ = _run("layer", bs)

    m2, b2 = build()
    st = PrefetchingFake("layer")
    tr = S3Trainer(m2, b2, S3TrainConfig(amp=False, grad_accum_steps=len(bs), ckpt_offload=True),
                   device="cpu", frozen_streamer=st)
    m2.train()
    tr.step_layer_major(bs)
    g_pf = grads_of(m2)

    max_diff = max(float((g_plain[n] - g_pf[n]).abs().max()) for n in g_plain)
    print(f"[Prop 5] prefetch: max|Δgrad| = {max_diff:.3e} | prefetch calls = {st.prefetches}")
    assert max_diff < 1e-9, f"prefetch changed gradients: {max_diff}"
    assert st.prefetches > 0, "prefetch was never issued"


def test_auto_granularity_selection():
    """§9: pick the coarsest unit that fits; refuse honestly when nothing does."""
    cfg = Qwen2Config(hidden_size=3584, intermediate_size=18944,
                      num_attention_heads=28, num_key_value_heads=4, head_dim=128)
    layer_b = unit_sizes(cfg, "layer")
    sub_b = unit_sizes(cfg, "sublayer")
    assert sub_b < layer_b, "Thm 3: finer granularity must not be larger"

    # generous budget -> coarsest ('layer')
    assert choose_granularity(cfg, layer_b + 2**30, verbose=False) == "layer"
    # budget between the two -> 'sublayer'
    assert choose_granularity(cfg, sub_b + 1024, verbose=False) == "sublayer"
    # impossible budget -> honest failure, not a silent wrong answer
    try:
        choose_granularity(cfg, 1024, verbose=False)
        raise AssertionError("should have refused")
    except RuntimeError as e:
        assert "not implemented" in str(e) or "No supported granularity" in str(e)
    print(f"[§9] 7B: layer={layer_b/2**30:.2f}GB sublayer={sub_b/2**30:.2f}GB | "
          f"supported={SUPPORTED_GRANULARITIES}")


def test_scaling_table_matches_formalism():
    """Cor 3.1: the published per-model numbers must come from the code."""
    llama70 = Qwen2Config(hidden_size=8192, intermediate_size=28672,
                          num_attention_heads=64, num_key_value_heads=8, head_dim=128)
    layer_gb = unit_sizes(llama70, "layer") / 2**30
    sub_gb = unit_sizes(llama70, "sublayer") / 2**30
    print(f"[Cor 3.1] Llama-3-70B: layer={layer_gb:.2f}GB sublayer={sub_gb:.2f}GB")
    assert 1.5 < layer_gb < 1.7, layer_gb      # formalism says 1.59 GB
    assert 1.2 < sub_gb < 1.4, sub_gb          # formalism says 1.31 GB
    # 70B fits a 4GB card at layer granularity -> the headline claim
    assert layer_gb < 3.94


if __name__ == "__main__":
    print("=" * 64)
    test_subunit_forward_matches_monolithic(); print("THM 3 (sub-unit split == monolithic): PASS")
    test_granularity_hierarchy_is_monotone(); print("THM 3 (hierarchy monotone): PASS")
    test_sublayer_granularity_preserves_gradients(); print("THM 3 (granularity exact): PASS")
    print("=" * 64)
    test_prefetch_preserves_gradients(); print("PROP 5 (prefetch exact): PASS")
    print("=" * 64)
    test_auto_granularity_selection(); print("§9 (auto-selection): PASS")
    print("=" * 64)
    test_scaling_table_matches_formalism(); print("COR 3.1 (scaling table): PASS")
    print("=" * 64)
