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


class FrozenStreamer:
    """Streams one S3Block's frozen weights disk -> GPU (load) and GPU -> meta (unload)."""

    def __init__(self, reader: ShardReader, device):
        self.r = reader
        self.device = device

    @staticmethod
    def _linear(block, sub, proj):
        """The frozen Linear whose weight we stream — either SVMOLinear.base
        (when SVMO wraps it) or the raw HF Linear (ablations with -SVMO)."""
        m = getattr(getattr(block, sub), proj)
        return getattr(m, "base", m)

    def load(self, block: nn.Module, idx: int):
        pre = f"model.layers.{idx}."
        for sub, proj in _PROJ:
            base = self._linear(block, sub, proj)
            _set_weight(base, "weight", self.r.get(f"{pre}{sub}.{proj}.weight").to(self.device))
            bname = f"{pre}{sub}.{proj}.bias"
            if self.r.has(bname):
                _set_weight(base, "bias", self.r.get(bname).to(self.device))
        for norm in ("input_layernorm", "post_attention_layernorm"):
            _set_weight(getattr(block, norm), "weight", self.r.get(f"{pre}{norm}.weight").to(self.device))

    def unload(self, block: nn.Module, idx: int):
        meta = torch.device("meta")
        for sub, proj in _PROJ:
            base = self._linear(block, sub, proj)
            for attr in ("weight", "bias"):
                p = getattr(base, attr, None)
                if isinstance(p, torch.Tensor):
                    setattr(base, attr, nn.Parameter(p.data.to(meta), requires_grad=False))
        for norm in ("input_layernorm", "post_attention_layernorm"):
            m = getattr(block, norm)
            m.weight = nn.Parameter(m.weight.data.to(meta), requires_grad=False)


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
