"""
Disk-streaming loader for S³ — the true "7B on 2-4GB VRAM" path (formalismo §1.5).

The frozen base is NEVER held whole in RAM or VRAM. The model skeleton is built
on the `meta` device (zero memory); each transformer layer's frozen weights are
read lazily from the HuggingFace safetensors shards, materialised on the GPU only
while that layer runs, then dropped back to `meta`. Embeddings and the LM head
(the two big vocab matrices) stay on CPU. Adapters live permanently on GPU.

No new download: reads directly from ~/.cache/huggingface. SVD factors come from
disk too (svd_dir), so wrapping never touches the (meta) base weights.
"""

import glob
import json
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Optional

import torch
import torch.nn as nn
from safetensors import safe_open


def find_snapshot(model_name: str) -> str:
    """Locate the local snapshot dir for a cached HF model (offline)."""
    cache = os.path.expanduser("~/.cache/huggingface/hub")
    slug = "models--" + model_name.replace("/", "--")
    snaps = sorted(glob.glob(os.path.join(cache, slug, "snapshots", "*")))
    if not snaps:
        raise FileNotFoundError(f"No local snapshot for {model_name} under {cache}")
    for s in snaps:
        if os.path.exists(os.path.join(s, "model.safetensors.index.json")) or \
           glob.glob(os.path.join(s, "*.safetensors")):
            return s
    return snaps[-1]


class ShardReader:
    """Lazily reads individual tensors from sharded safetensors (kept mmap'd)."""

    def __init__(self, snapshot_dir: str):
        self.dir = snapshot_dir
        idx = os.path.join(snapshot_dir, "model.safetensors.index.json")
        if os.path.exists(idx):
            self.weight_map = json.load(open(idx))["weight_map"]
        else:  # single-shard model
            shard = os.path.basename(glob.glob(os.path.join(snapshot_dir, "*.safetensors"))[0])
            with safe_open(os.path.join(snapshot_dir, shard), framework="pt") as f:
                self.weight_map = {k: shard for k in f.keys()}
        self._handles: Dict[str, object] = {}

    def _handle(self, shard: str):
        if shard not in self._handles:
            self._handles[shard] = safe_open(os.path.join(self.dir, shard), framework="pt", device="cpu")
        return self._handles[shard]

    def has(self, name: str) -> bool:
        return name in self.weight_map

    def get(self, name: str) -> torch.Tensor:
        return self._handle(self.weight_map[name]).get_tensor(name)


def build_streaming_model(model_name: str):
    """Build the model skeleton on `meta` (zero memory) from the cached config."""
    from accelerate import init_empty_weights
    from transformers import AutoConfig, AutoModelForCausalLM

    cfg = AutoConfig.from_pretrained(model_name)
    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(cfg, dtype=torch.float16)
    model.eval()
    return model, cfg


def _set_weight(module: nn.Module, attr: str, tensor: Optional[torch.Tensor]):
    if tensor is None:
        return
    setattr(module, attr, nn.Parameter(tensor, requires_grad=False))


# frozen linears inside a wrapped S3Block: attribute path -> checkpoint suffix
_PROJ = [
    ("self_attn", "q_proj"), ("self_attn", "k_proj"),
    ("self_attn", "v_proj"), ("self_attn", "o_proj"),
    ("mlp", "gate_proj"), ("mlp", "up_proj"), ("mlp", "down_proj"),
]

# ---- granularity (formalismo_streaming.md §5, Thm 3) -----------------------
# Nested partitions of a layer's weights. Refining reduces peak VRAM at equal
# total I/O; §9 picks the COARSEST one that fits (coarsest = fewest loads = fastest).
GRANULARITIES = {
    "layer":    [_PROJ],                                   # 1 unit  = whole block
    "sublayer": [_PROJ[:4], _PROJ[4:]],                    # 2 units = attn | mlp
    "matrix":   [[p] for p in _PROJ],                      # 7 units = one per matrix
}


def unit_sizes(config, granularity: str):
    """Bytes of the largest unit under `granularity`, from a HF config (fp16).

    Model-agnostic: derives the shapes from the config, so it works for any
    transformer of this family without loading a single weight.
    """
    d = config.hidden_size
    ffn = config.intermediate_size
    kv = getattr(config, "num_key_value_heads", config.num_attention_heads)
    hd = getattr(config, "head_dim", d // config.num_attention_heads)
    numel = {"q_proj": d * d, "o_proj": d * d, "k_proj": kv * hd * d, "v_proj": kv * hd * d,
             "gate_proj": ffn * d, "up_proj": ffn * d, "down_proj": d * ffn}
    return max(sum(numel[p] for _, p in unit) * 2 for unit in GRANULARITIES[granularity])


# Granularities the forward can actually exploit. 'matrix' is NOT here: HF's
# attention consumes q,k,v,o together inside `self_attn`, so streaming them one
# at a time would require reimplementing the attention internals (exactly what
# broke the original code). It stays in GRANULARITIES for the size math only.
SUPPORTED_GRANULARITIES = ("layer", "sublayer")


def choose_granularity(config, vram_budget_bytes: int, reserve_bytes: int = 0,
                       prefetch: bool = False, verbose: bool = True) -> str:
    """§9 Algorithm 9.1 — pick the coarsest granularity that fits the budget.

    Prop. 9.2: total I/O is invariant to granularity (Thm 3) but each `load` pays
    a fixed latency, so the coarsest partition that fits minimises time.

    Model-agnostic: reads only the config, so the same call works for 7B or 70B.
    """
    factor = 2 if prefetch else 1        # Prop. 5: double buffering keeps 2 units resident
    for g in SUPPORTED_GRANULARITIES:
        need = unit_sizes(config, g) * factor
        if need + reserve_bytes <= vram_budget_bytes:
            if verbose:
                print(f"[USF] granularity='{g}': unit {need/2**30:.2f}GB + reserve "
                      f"{reserve_bytes/2**30:.2f}GB <= budget {vram_budget_bytes/2**30:.2f}GB")
            return g
    theo = unit_sizes(config, "matrix") * factor
    raise RuntimeError(
        f"No supported granularity fits: smallest supported unit is "
        f"{unit_sizes(config,'sublayer')*factor/2**30:.2f}GB (sublayer) but the budget is "
        f"{(vram_budget_bytes-reserve_bytes)/2**30:.2f}GB.\n"
        f"Per-matrix streaming would need {theo/2**30:.2f}GB and WOULD fit, but it is not "
        f"implemented: it requires reimplementing HF's attention internals "
        f"(see formalismo_streaming.md §5, limitation note).")


class FrozenStreamer:
    """Streams frozen weights disk -> GPU (load) and GPU -> meta (unload).

    Granularity (Thm 3): `granularity` selects the streamable unit —
    'layer' (1 unit/block), 'sublayer' (attn | mlp), 'matrix' (7 units).
    A finer granularity lowers peak VRAM at identical total I/O, which is what
    makes 70B/405B fit (Cor. 3.2).

    `load/unload/prefetch` take an optional `unit` index; `unit=None` means the
    whole block (backwards-compatible with the layer granularity).
    """

    def __init__(self, reader: ShardReader, device, granularity: str = "layer"):
        self.r = reader
        self.device = device
        self.granularity = granularity
        self._plan = GRANULARITIES[granularity]

    def units(self, idx: int):
        """Indices of the streamable units of block `idx` under this granularity."""
        return list(range(len(self._plan)))

    def _projs(self, unit):
        return _PROJ if unit is None else self._plan[unit]

    @staticmethod
    def _linear(block, sub, proj):
        """The frozen Linear whose weight we stream — SVMOLinear.base (when SVMO
        wraps it) or the raw HF Linear (ablations with -SVMO / LoRA)."""
        m = getattr(getattr(block, sub), proj)
        return getattr(m, "base", m)

    def _norms_of(self, unit):
        """RMSNorm weights ride with the unit that consumes them (tiny: ~7 KB)."""
        if unit is None:
            return ("input_layernorm", "post_attention_layernorm")
        if self.granularity == "layer":
            return ("input_layernorm", "post_attention_layernorm")
        # sublayer/matrix: attention units carry input_layernorm, mlp units the other
        subs = {s for s, _ in self._plan[unit]}
        return ("input_layernorm",) if "self_attn" in subs else ("post_attention_layernorm",)

    def load(self, block: nn.Module, idx: int, unit=None):
        pre = f"model.layers.{idx}."
        for sub, proj in self._projs(unit):
            base = self._linear(block, sub, proj)
            _set_weight(base, "weight", self.r.get(f"{pre}{sub}.{proj}.weight").to(self.device))
            bname = f"{pre}{sub}.{proj}.bias"
            if self.r.has(bname):
                _set_weight(base, "bias", self.r.get(bname).to(self.device))
        for norm in self._norms_of(unit):
            _set_weight(getattr(block, norm), "weight",
                        self.r.get(f"{pre}{norm}.weight").to(self.device))

    def unload(self, block: nn.Module, idx: int, unit=None):
        meta = torch.device("meta")
        for sub, proj in self._projs(unit):
            base = self._linear(block, sub, proj)
            for attr in ("weight", "bias"):
                p = getattr(base, attr, None)
                if isinstance(p, torch.Tensor):
                    setattr(base, attr, nn.Parameter(p.data.to(meta), requires_grad=False))
        for norm in self._norms_of(unit):
            m = getattr(block, norm)
            if isinstance(m.weight, torch.Tensor):
                m.weight = nn.Parameter(m.weight.data.to(meta), requires_grad=False)

    def prefetch(self, idx: int, unit=None):
        """No-op for the plain streamer (see PrefetchStreamer)."""
        return


class PrefetchStreamer(FrozenStreamer):
    """FrozenStreamer + double buffering (formalismo_streaming.md §7, Prop. 5).

    While unit `i` computes, unit `i+1` is read from disk on a worker thread into
    pinned CPU memory and copied on a side CUDA stream. If t_io(i+1) <= t_cmp(i)
    the I/O is fully hidden and the step becomes compute-bound.

    Cost (Prop. 5): two units resident instead of one -> the VRAM bound of Thm 2
    becomes 2*max_u|W_u|. Exactness is untouched: prefetching changes *when* a
    tensor is read, never its value.
    """

    def __init__(self, reader: ShardReader, device, granularity: str = "layer",
                 enabled: bool = True):
        super().__init__(reader, device, granularity)
        self.enabled = enabled and torch.cuda.is_available()
        self._pool = ThreadPoolExecutor(max_workers=1) if self.enabled else None
        self._future = None
        self._key = None             # (idx, unit) currently in flight
        self._stream = torch.cuda.Stream(device=device) if self.enabled else None

    def _read_cpu(self, idx: int, unit):
        """Read one unit's tensors into pinned CPU memory (worker thread, no CUDA)."""
        pre = f"model.layers.{idx}."
        out = []
        for sub, proj in self._projs(unit):
            out.append(((sub, proj, "weight"),
                        self.r.get(f"{pre}{sub}.{proj}.weight").pin_memory()))
            bname = f"{pre}{sub}.{proj}.bias"
            if self.r.has(bname):
                out.append(((sub, proj, "bias"), self.r.get(bname).pin_memory()))
        for norm in self._norms_of(unit):
            out.append(((norm, None, "weight"), self.r.get(f"{pre}{norm}.weight").pin_memory()))
        return out

    def prefetch(self, idx: int, unit=None):
        """Kick off the disk read for (idx, unit) on a worker thread (non-blocking)."""
        if not self.enabled:
            return
        key = (idx, unit)
        if self._key == key:
            return
        if self._future is not None:      # drop an unused in-flight read
            self._future.cancel()
        self._future = self._pool.submit(self._read_cpu, idx, unit)
        self._key = key

    def load(self, block: nn.Module, idx: int, unit=None):
        if not self.enabled:
            return super().load(block, idx, unit)
        key = (idx, unit)
        if self._key != key or self._future is None:
            self.prefetch(idx, unit)      # not prefetched (first unit / miss)
        tensors = self._future.result()   # wait only for what's left of the read
        self._future, self._key = None, None
        with torch.cuda.stream(self._stream):
            for (a, b, attr), t in tensors:
                mod = getattr(getattr(block, a), b) if b else getattr(block, a)
                mod = getattr(mod, "base", mod)
                _set_weight(mod, attr, t.to(self.device, non_blocking=True))
        torch.cuda.current_stream(self.device).wait_stream(self._stream)


def materialize_shared(model, reader: ShardReader, embed_device, head_device, norm_device):
    """Materialise the always-needed pieces: embeddings, LM head, final norm, rotary."""
    _set_weight(model.model.embed_tokens, "weight", reader.get("model.embed_tokens.weight").to(embed_device))
    _set_weight(model.model.norm, "weight", reader.get("model.norm.weight").to(norm_device))

    head_name = "lm_head.weight" if reader.has("lm_head.weight") else "model.embed_tokens.weight"
    _set_weight(model.lm_head, "weight", reader.get(head_name).to(head_device))

    # rotary embedding has no shard weights (derived from config) — rebuild real buffers
    cfg = model.config
    rot_cls = type(model.model.rotary_emb)
    model.model.rotary_emb = rot_cls(config=cfg).to(norm_device)
    return model
