"""
S3Trainer — correct, verifiable training loop for the rebuilt S³ core.

Replaces the broken FrugalTrainer path (1-token context + hand-rolled backward
that fed each layer its own output). This trainer:

  * runs a FULL sequence (real autoregressive context) through the model;
  * drives the layer loop explicitly so RoPE / causal mask / final norm / lm_head
    come from the frozen HF model and are applied correctly;
  * uses standard autograd (loss.backward) — no manual per-layer gradient hand-off;
  * supports gradient checkpointing per S3Block to cut activation VRAM;
  * supports LAYER SWAP: each block's *frozen* buffers are moved to GPU only while
    that block runs, then back to CPU. Trainable adapter params stay resident on
    GPU so the optimizer/AdamW state never changes device.

The layer-swap path is verified (tests/test_s3_trainer.py) to produce gradients
identical to the resident path, so it is a pure memory optimization.
"""

from dataclasses import dataclass
from typing import List

import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


@dataclass
class S3TrainConfig:
    learning_rate: float = 1e-3
    weight_decay: float = 0.01
    betas: tuple = (0.9, 0.999)
    max_epochs: int = 3
    warmup_ratio: float = 0.05
    max_grad_norm: float = 1.0
    grad_accum_steps: int = 8
    gradient_checkpointing: bool = True
    layer_swap: bool = False
    amp: bool = True
    amp_dtype: torch.dtype = torch.float16
    log_interval: int = 10
    # --- USF (formalismo_streaming.md) ---
    layer_major: bool = False     # Thm 4: invert loops -> I/O divided by grad_accum_steps
    ckpt_offload: bool = True     # Cor 2.1: checkpoints to CPU -> VRAM independent of depth
    # --- S3-OPT (formalismo_optimizacion) coupling ---
    # Only correctness-preserving optimizations are wired live. Others are
    # accepted and logged so both formalisms stay coupled without risking learning.
    use_nfr: bool = True          # NFR (Thm 9): progressive ODE steps N=2 -> N=4 by epoch
    nfr_initial_N: int = 2
    nfr_full_N: int = 4
    use_sbs: bool = True          # SBS (Thm 10): STB applied with prob sbs_p (set at wrap time)
    use_sgc: bool = True          # SGC (Thm 8): inherent to residual SVMO (grad flows through k-dim bottleneck)
    extra_opts: tuple = ()        # names of other requested S3-OPT flags (logged only)


class S3Trainer:
    def __init__(self, model, blocks: nn.ModuleList, config: S3TrainConfig,
                 device: str = "cuda:0", frozen_streamer=None):
        self.model = model
        self.blocks = blocks
        self.cfg = config
        self.device = torch.device(device)
        self._is_cuda = self.device.type == "cuda"
        self.frozen_streamer = frozen_streamer
        self._streaming = frozen_streamer is not None

        self.embed = model.model.embed_tokens
        self.rotary = model.model.rotary_emb
        self.norm = model.model.norm
        self.lm_head = model.lm_head

        # When the model was loaded with accelerate (device_map / quantized offload),
        # its modules carry dispatch hooks and some weights live on `meta`. We must
        # NOT move them by hand; accelerate streams them per forward. In that mode we
        # only place the adapters on GPU and let the model's own forward run.
        self._accelerate_managed = not self._streaming and (
            getattr(model, "hf_device_map", None) is not None
            or getattr(model, "is_quantized", False)
            or getattr(model, "is_loaded_in_4bit", False)
            or getattr(model, "is_loaded_in_8bit", False)
            or any(hasattr(m, "_hf_hook") for m in model.modules())
            or any(p.device.type == "meta" for p in model.parameters())
        )
        self._fwd_ms = self._bwd_ms = 0.0
        self._disk_mb_per_step = 0.0
        if self._streaming:
            # embeddings + LM head already placed on CPU by the loader; move only
            # the small always-on pieces and the adapters to GPU.
            if self._is_cuda:
                self.norm.to(self.device)
                self.rotary.to(self.device)
                self._adapters_to_gpu()
            # estimate bytes streamed from disk per step (fp16 base weights, fwd+bwd)
            frozen = 0
            for blk in self.blocks:
                for p in blk.parameters():
                    if not p.requires_grad:
                        frozen += p.numel()
            self._disk_mb_per_step = frozen * 2 * 2 / 1024**2  # fp16, two passes
        elif self._accelerate_managed:
            if self.cfg.layer_swap:
                print("[S3Trainer] accelerate/quantized model detected -> layer_swap "
                      "disabled (accelerate already streams weights per layer).")
                self.cfg.layer_swap = False
            self._adapters_to_gpu()
            if self.cfg.gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
                model.config.use_cache = False
                # non-reentrant: works even though every base input is frozen
                model.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs={"use_reentrant": False})
        elif self._is_cuda:
            self._resident_to_gpu()

        # trainable adapter params live permanently on GPU
        self.trainable: List[nn.Parameter] = [p for p in model.parameters() if p.requires_grad]

        # A "base" (no-adapter) model has no trainable params — valid for
        # eval-only use (forward_layer_major/eval_loss), just not for training.
        self.opt = torch.optim.AdamW(
            self.trainable, lr=config.learning_rate,
            betas=config.betas, weight_decay=config.weight_decay,
        ) if self.trainable else None
        # lower init scale -> fewer skipped "warmup" steps at the start of fp16 training
        self.scaler = torch.amp.GradScaler(
            "cuda", enabled=config.amp and self._is_cuda,
            init_scale=2.0 ** 12, growth_interval=200)
        n = sum(p.numel() for p in self.trainable)
        tot = sum(p.numel() for p in model.parameters())
        print(f"[S3Trainer] trainable={n:,} / {tot:,} ({100*n/tot:.4f}%)  "
              f"layer_swap={config.layer_swap}  grad_ckpt={config.gradient_checkpointing}")
        self._log_s3opt()

    def _log_s3opt(self):
        active = []
        if self.cfg.use_nfr:
            active.append(f"NFR(N:{self.cfg.nfr_initial_N}->{self.cfg.nfr_full_N})")
        if self.cfg.use_sbs:
            active.append("SBS")
        if self.cfg.use_sgc:
            active.append("SGC(inherent)")
        line = "[S3Trainer] S3-OPT active: " + (", ".join(active) if active else "none")
        if self.cfg.extra_opts:
            line += "  | requested (experimental, not altering gradients): " + ", ".join(self.cfg.extra_opts)
        print(line)

    def _apply_nfr(self, epoch: int):
        """NFR (Thm 9): fewer ODE steps in early epochs, full precision later."""
        if not self.cfg.use_nfr:
            return
        N = self.cfg.nfr_initial_N if epoch <= 1 else self.cfg.nfr_full_N
        for blk in self.blocks:
            for nmf in (getattr(blk, "nmf_attn", None), getattr(blk, "nmf_mlp", None)):
                if nmf is not None:
                    nmf.N_steps = N
                    nmf.dt = nmf.T / N

    # ---- device management ------------------------------------------------
    def _adapters_to_gpu(self):
        """Accelerate-managed base: move ONLY the trainable adapter tensors (and the
        frozen SVD factors they need) to the GPU. Base weights are left to accelerate."""
        if not self._is_cuda:
            return
        for blk in self.blocks:
            for p in blk.parameters():
                if p.requires_grad and p.device.type != "meta":
                    p.data = p.data.to(self.device)
            for m in blk.modules():
                for bname, buf in list(m._buffers.items()):
                    if buf is not None and buf.device.type != "meta":
                        m._buffers[bname] = buf.to(self.device)

    def _resident_to_gpu(self):
        """Move always-on pieces (embeddings/rotary/norm/lm_head + trainable adapters) to GPU.
        If layer_swap is off, whole blocks go to GPU too."""
        self.embed.to(self.device)
        self.norm.to(self.device)
        self.lm_head.to(self.device)
        for blk in self.blocks:
            if self.cfg.layer_swap:
                # only trainable params to GPU; frozen buffers stay on CPU
                for name, p in blk.named_parameters():
                    if p.requires_grad:
                        p.data = p.data.to(self.device)
            else:
                blk.to(self.device)

    def _swap_block_frozen(self, blk: nn.Module, to_device):
        """Move a block's FROZEN tensors (frozen params + all buffers) to a device.
        Trainable params (requires_grad=True) are never touched — they stay
        resident on GPU so the optimizer state never changes device."""
        for p in blk.parameters():
            if not p.requires_grad:
                p.data = p.data.to(to_device)
        for m in blk.modules():
            for bname, buf in list(m._buffers.items()):
                if buf is not None:
                    m._buffers[bname] = buf.to(to_device)

    # ---- forward ----------------------------------------------------------
    def _run_block(self, idx, blk, hidden, attn_mask, pos_emb):
        if self._streaming and self._is_cuda:
            self.frozen_streamer.load(blk, idx)
        elif self.cfg.layer_swap and self._is_cuda:
            self._swap_block_frozen(blk, self.device)
        out = blk(hidden, attention_mask=attn_mask, position_embeddings=pos_emb)
        if self._streaming and self._is_cuda:
            self.frozen_streamer.unload(blk, idx)
        elif self.cfg.layer_swap and self._is_cuda:
            self._swap_block_frozen(blk, torch.device("cpu"))
        return out

    def forward_loss(self, input_ids: torch.Tensor) -> torch.Tensor:
        if self._accelerate_managed:
            # Inputs must land on the device where the embedding actually lives
            # (accelerate may keep it on CPU); its hooks align every later module.
            emb_w = self.model.model.embed_tokens.weight
            dev = self.device if emb_w.device.type == "meta" else emb_w.device
            input_ids = input_ids.to(dev)
            # HF shifts labels internally for causal LM.
            return self.model(input_ids=input_ids, labels=input_ids).loss

        labels = input_ids[:, 1:].contiguous().to(self.device)
        inp = input_ids[:, :-1].contiguous()
        B, S = inp.shape

        # embeddings live on CPU in streaming mode -> embed there, then move to GPU
        emb_dev = self.embed.weight.device
        hidden = self.embed(inp.to(emb_dev)).to(self.device)

        # autocast unifies fp16 base weights and fp32 adapter weights on the GPU
        with torch.amp.autocast("cuda", enabled=self.cfg.amp and self._is_cuda,
                                dtype=self.cfg.amp_dtype):
            pos_ids = torch.arange(S, device=self.device).unsqueeze(0)
            pos_emb = self.rotary(hidden, pos_ids)
            mask = torch.full((S, S), float("-inf"), device=self.device, dtype=hidden.dtype)
            mask = torch.triu(mask, diagonal=1).view(1, 1, S, S)

            for idx, blk in enumerate(self.blocks):
                if self.cfg.gradient_checkpointing and self.model.training:
                    hidden = checkpoint(self._run_block, idx, blk, hidden, mask, pos_emb,
                                        use_reentrant=False)
                else:
                    hidden = self._run_block(idx, blk, hidden, mask, pos_emb)

            hidden = self.norm(hidden)

        head_w = self.lm_head.weight
        head_dev = head_w.device                     # CPU in streaming mode
        logits = F.linear(hidden.to(head_dev, dtype=head_w.dtype), head_w)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(),
                               labels.reshape(-1).to(head_dev))
        return loss.to(self.device)

    def _sbs_decide(self, blk):
        if getattr(blk, "stb", None) is None:
            return False
        return blk.sbs_p >= 1.0 or (torch.rand(()) < blk.sbs_p)

    # ---- streaming helpers (granularity + prefetch aware) -------------------
    def _units(self, i):
        """Streamable units of block i under the active granularity (Thm 3)."""
        return self.frozen_streamer.units(i) if self.frozen_streamer is not None else []

    def _load(self, blk, i, unit=None):
        if self.frozen_streamer is not None:
            self.frozen_streamer.load(blk, i, unit)

    def _unload(self, blk, i, unit=None):
        if self.frozen_streamer is not None:
            self.frozen_streamer.unload(blk, i, unit)

    def _prefetch(self, i, unit=None):
        """Prop. 5: kick off the next unit's disk read while this one computes."""
        if self.frozen_streamer is not None and hasattr(self.frozen_streamer, "prefetch"):
            if 0 <= i < len(self.blocks):
                self.frozen_streamer.prefetch(i, unit)

    def _granularity(self):
        return getattr(self.frozen_streamer, "granularity", "layer") if self.frozen_streamer else "layer"

    def _segments(self):
        """Streamable *segments* = (block_idx, unit, kind), in execution order.

        'layer'    -> 1 segment/block (whole block).
        'sublayer' -> 2 segments/block: attention | MLP, so peak VRAM becomes
                      max(|W_attn|,|W_mlp|) instead of |W_layer| (Thm 3).

        Finer than sublayer (per-matrix) would need reimplementing HF's attention
        internals — q,k,v,o are consumed together inside `self_attn` — so it is
        NOT supported here (see formalismo_streaming.md §5, limitation note).
        """
        g = self._granularity()
        segs = []
        for i in range(len(self.blocks)):
            if g == "layer":
                segs.append((i, None, "full"))
            else:
                segs.append((i, 0, "attn"))
                segs.append((i, 1, "mlp"))
        return segs

    def _run_seg(self, seg, h, mask, pos_emb, stb_apply, nmf_N=None):
        """Execute one segment. Composition over segments == blk.forward()."""
        i, _unit, kind = seg
        blk = self.blocks[i]
        if kind == "full":
            return blk(h, attention_mask=mask, position_embeddings=pos_emb,
                       stb_apply=stb_apply, nmf_N_steps=nmf_N)
        if kind == "attn":
            return blk.forward_attn(h, attention_mask=mask, position_embeddings=pos_emb,
                                    stb_apply=stb_apply, nmf_N_steps=nmf_N)
        return blk.forward_mlp(h, nmf_N_steps=nmf_N)

    def _run_unit(self, blk, i, h, mask, pos_emb, stb_apply, nmf_N=None, nxt=None):
        """Run block `i` honouring the active granularity (Thm 3) + prefetch (Prop 5).

        'layer'  -> load whole block, run, free.
        'sublayer'/'matrix' -> load attention weights, run attn, free; then the
        MLP weights, run mlp, free. Peak VRAM = max(|W_attn|,|W_mlp|).
        Composition is identical to blk.forward(), so gradients are unchanged.
        """
        g = self._granularity()
        if g == "layer":
            self._load(blk, i)
            if nxt is not None:
                self._prefetch(nxt)                       # overlap next block's read
            out = blk(h, attention_mask=mask, position_embeddings=pos_emb,
                      stb_apply=stb_apply, nmf_N_steps=nmf_N)
            self._unload(blk, i)
            return out
        # --- sub-unit path: attention first, then MLP ---
        self._load(blk, i, unit=0)
        self._prefetch(i, unit=1)                          # MLP weights while attn computes
        h = blk.forward_attn(h, attention_mask=mask, position_embeddings=pos_emb,
                             stb_apply=stb_apply, nmf_N_steps=nmf_N)
        self._unload(blk, i, unit=0)
        self._load(blk, i, unit=1)
        if nxt is not None:
            self._prefetch(nxt, unit=0)                    # next block's attn weights
        h = blk.forward_mlp(h, nmf_N_steps=nmf_N)
        self._unload(blk, i, unit=1)
        return h

    def _causal(self, S, dtype):
        mask = torch.full((S, S), float("-inf"), device=self.device, dtype=dtype)
        return torch.triu(mask, diagonal=1).view(1, 1, S, S)

    def _prep(self, batch, dtype):
        """Unpack a batch (tensor or {input_ids, attention_mask}) into
        (inp, labels, additive_mask). Padding is masked in attention AND ignored
        in the loss (-100), so batch>1 with right-padding is correct."""
        if isinstance(batch, dict):
            ids, attn = batch["input_ids"], batch.get("attention_mask")
        else:
            ids, attn = batch, None
        inp = ids[:, :-1].contiguous()
        B, S = inp.shape
        labels = ids[:, 1:].contiguous().to(self.device)
        mask = self._causal(S, dtype)                      # [1,1,S,S]
        if attn is not None:
            attn_inp = attn[:, :-1].to(self.device)        # valid input positions
            attn_lab = attn[:, 1:].to(self.device)         # valid label positions
            labels = labels.masked_fill(attn_lab == 0, -100)
            mask = mask.expand(B, 1, S, S).clone()
            mask = mask.masked_fill((attn_inp == 0)[:, None, None, :], float("-inf"))
        return inp, labels, mask

    @torch.no_grad()
    def eval_loss(self, batch) -> float:
        """Memory-bounded no-grad loss (for base/after perplexity). One layer at a time."""
        ac = dict(device_type="cuda", enabled=self.cfg.amp and self._is_cuda, dtype=self.cfg.amp_dtype)
        with torch.amp.autocast(**ac):
            inp, labels, mask = self._prep(batch, self.amp_dtype_or(torch.float32))
            S = inp.shape[1]
            h = self.embed(inp.to(self.embed.weight.device)).to(self.device)
            mask = mask.to(h.dtype)
            pos_emb = self.rotary(h, torch.arange(S, device=self.device).unsqueeze(0))
            for i, blk in enumerate(self.blocks):
                if self._streaming and self._is_cuda:
                    h = self._run_unit(blk, i, h, mask, pos_emb, True,
                                       nxt=i + 1 if i + 1 < len(self.blocks) else None)
                else:
                    h = blk(h, attention_mask=mask, position_embeddings=pos_emb, stb_apply=True)
            h = self.norm(h)
        head_w = self.lm_head.weight
        logits = F.linear(h.to(head_w.device, dtype=head_w.dtype), head_w)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(),
                               labels.reshape(-1).to(head_w.device), ignore_index=-100)
        return float(loss)

    def amp_dtype_or(self, fallback):
        return self.cfg.amp_dtype if (self.cfg.amp and self._is_cuda) else fallback

    @torch.no_grad()
    def forward_layer_major(self, batches):
        """Forward-only with the loops INVERTED — Thm 4 applied to inference.

        `eval_loss` reads all Σ|W_i| bytes to serve ONE batch. Here a unit is
        loaded once and every batch is pushed through it, so a whole evaluation
        costs a single model read instead of one per batch. There is no backward,
        so unlike `step_layer_major` nothing needs saving for recomputation: the
        only resident state is one hidden tensor per batch, offloaded per Cor 2.1.

        STB is applied deterministically (as in `eval_loss`), never sampled — an
        evaluation must be a function of the input, not of the RNG.

        Returns the post-norm hidden states (CPU) and labels, one per batch; the
        caller projects through lm_head, which stays off this loop because the
        vocab matrix dwarfs a decoder layer.
        """
        G = len(batches)
        emb_dev = self.embed.weight.device
        ac = dict(device_type="cuda", enabled=self.cfg.amp and self._is_cuda,
                  dtype=self.cfg.amp_dtype)

        H, labels, masks, pos_embs = [], [], [], []
        with torch.amp.autocast(**ac):
            for b in batches:
                inp, lab, mask = self._prep(b, self.amp_dtype_or(torch.float32))
                h = self.embed(inp.to(emb_dev)).to(self.device)
                S = inp.shape[1]
                pe = self.rotary(h, torch.arange(S, device=self.device).unsqueeze(0))
                H.append(self._ckpt_store(h)); labels.append(lab)
                masks.append(mask.to(h.dtype)); pos_embs.append(pe)

            segs = self._segments()
            for si, seg in enumerate(segs):
                i, unit, _kind = seg
                blk = self.blocks[i]
                self._load(blk, i, unit)
                if si + 1 < len(segs):
                    nxt = segs[si + 1]
                    self._prefetch(nxt[0], nxt[1])
                for j in range(G):
                    h = self._ckpt_load(H[j])
                    h = self._run_seg(seg, h, masks[j], pos_embs[j], True)
                    H[j] = self._ckpt_store(h)
                self._unload(blk, i, unit)

            for j in range(G):
                H[j] = self._ckpt_store(self.norm(self._ckpt_load(H[j])))
        return H, labels

    # ---- USF layer-major (formalismo_streaming.md §6, Thm 4) -----------------
    def _ckpt_store(self, t):
        """Cor. 2.1: checkpoints live on CPU so VRAM stays independent of depth.

        Synchronous on purpose. `non_blocking=True` only overlaps with the host
        when the OTHER side of the copy is pinned memory, which this is not; the
        streamer's own weight loads run on a side CUDA stream to overlap disk I/O
        with compute (Prop. 5), and an async copy of `t` here has no ordering
        guarantee against that side stream. Measured effect: with non_blocking,
        the loaded tensor could be read before its data landed, producing
        exactly-zero hidden states (caught by comparing against `eval_loss` on
        the real 7B model). 28 small synchronous copies per forward is immeasurable
        next to the weight I/O this method exists to amortize.
        """
        return t.to("cpu") if self.cfg.ckpt_offload else t

    def _ckpt_load(self, t):
        return t.to(self.device) if self.cfg.ckpt_offload else t

    def step_layer_major(self, batches) -> float:
        """One optimizer step over G micro-batches with the loops INVERTED
        (formalismo_streaming.md §6): keep unit `i` resident and push all G
        micro-batches through it, then move on.

        I/O per optimizer step drops from G·2Σ|W_i| to 2Σ|W_i| — a factor G —
        while the gradients stay IDENTICAL (Thm 4: gradient accumulation is a
        sum, and sums are order-invariant; every VJP is evaluated at the same
        point as in batch-major).

        Cost: G sets of activations per unit boundary, offloaded to CPU (Cor 2.1).
        """
        G = len(batches)
        emb_dev = self.embed.weight.device
        ac = dict(device_type="cuda", enabled=self.cfg.amp and self._is_cuda, dtype=self.cfg.amp_dtype)

        def _sync():
            if self._is_cuda:
                torch.cuda.synchronize(self.device)

        # ---- prep every micro-batch (each may have its own S / padding) ----
        H, labels, masks, pos_embs = [], [], [], []
        t_fwd0 = time.time()
        with torch.no_grad(), torch.amp.autocast(**ac):
            for b in batches:
                inp, lab, mask = self._prep(b, self.amp_dtype_or(torch.float32))
                h = self.embed(inp.to(emb_dev)).to(self.device)
                S = inp.shape[1]
                pe = self.rotary(h, torch.arange(S, device=self.device).unsqueeze(0))
                H.append(h); labels.append(lab); masks.append(mask.to(h.dtype)); pos_embs.append(pe)

            # ---- FORWARD, layer-major over SEGMENTS ----
            # Thm 4 x Thm 3 x Prop 5 composed: load a segment ONCE, push all G
            # micro-batches through it, prefetch the next one meanwhile, free it.
            segs = self._segments()
            saved = [[None] * G for _ in segs]
            flags = [[None] * G for _ in segs]
            for si, seg in enumerate(segs):
                i, unit, kind = seg
                blk = self.blocks[i]
                self._load(blk, i, unit)
                if si + 1 < len(segs):
                    nxt = segs[si + 1]
                    self._prefetch(nxt[0], nxt[1])
                for j in range(G):
                    saved[si][j] = self._ckpt_store(H[j])
                    flags[si][j] = self._sbs_decide(blk) if kind in ("full", "attn") else None
                    H[j] = self._run_seg(seg, H[j], masks[j], pos_embs[j], flags[si][j])
                self._unload(blk, i, unit)
        _sync(); self._fwd_ms = (time.time() - t_fwd0) * 1000

        # ---- tail per micro-batch -> cotangent at the last block ----
        t_bwd0 = time.time()
        head_w = self.lm_head.weight
        grads, total_loss = [], 0.0
        for j in range(G):
            h_last = H[j].detach().requires_grad_(True)
            with torch.amp.autocast(**ac):
                normed = self.norm(h_last)
            logits = F.linear(normed.to(head_w.device, dtype=head_w.dtype), head_w)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(),
                                   labels[j].reshape(-1).to(head_w.device), ignore_index=-100)
            self.scaler.scale(loss.to(self.device) / G).backward()
            grads.append(h_last.grad)
            total_loss += float(loss)
        H = None

        # ---- BACKWARD, layer-major over SEGMENTS (reverse order) ----
        for si in range(len(segs) - 1, -1, -1):
            seg = segs[si]
            i, unit, _kind = seg
            blk = self.blocks[i]
            self._load(blk, i, unit)
            if si - 1 >= 0:
                prv = segs[si - 1]
                self._prefetch(prv[0], prv[1])
            for j in range(G):
                x = self._ckpt_load(saved[si][j]).detach().requires_grad_(True)
                with torch.amp.autocast(**ac):
                    out = self._run_seg(seg, x, masks[j], pos_embs[j], flags[si][j])
                out.backward(grads[j])
                grads[j] = x.grad
                del x, out
                saved[si][j] = None
            self._unload(blk, i, unit)
        _sync(); self._bwd_ms = (time.time() - t_bwd0) * 1000
        return total_loss / G

    def step_streaming(self, batch) -> float:
        """Memory-bounded forward+backward: one layer resident at a time in BOTH
        passes (formalismo §4.5). Gradients are exact — each block's local
        Jacobian is applied to the incoming grad, standard reverse-mode, just
        checkpointed with the frozen weights re-streamed from disk per layer."""
        emb_dev = self.embed.weight.device
        ac = dict(device_type="cuda", enabled=self.cfg.amp and self._is_cuda, dtype=self.cfg.amp_dtype)

        def _sync():
            if self._is_cuda:
                torch.cuda.synchronize(self.device)
        t_fwd0 = time.time()

        # 1) no-grad forward, saving each layer's INPUT + its SBS decision
        with torch.no_grad(), torch.amp.autocast(**ac):
            inp, labels, mask = self._prep(batch, self.amp_dtype_or(torch.float32))
            S = inp.shape[1]
            hidden = self.embed(inp.to(emb_dev)).to(self.device)
            mask = mask.to(hidden.dtype)
            pos_ids = torch.arange(S, device=self.device).unsqueeze(0)
            pos_emb = self.rotary(hidden, pos_ids)
            saved, flags = [], []
            h = hidden
            for i, blk in enumerate(self.blocks):
                saved.append(h)
                flag = self._sbs_decide(blk)
                if self.frozen_streamer is not None:
                    h = self._run_unit(blk, i, h, mask, pos_emb, flag,
                                       nxt=i + 1 if i + 1 < len(self.blocks) else None)
                else:
                    h = blk(h, attention_mask=mask, position_embeddings=pos_emb, stb_apply=flag)
                flags.append(flag)
        _sync(); self._fwd_ms = (time.time() - t_fwd0) * 1000

        # 2) tail (norm + LM head on CPU + loss) -> grad wrt last block output
        t_bwd0 = time.time()
        h_last = h.detach().requires_grad_(True)
        with torch.amp.autocast(**ac):
            normed = self.norm(h_last)
        head_w = self.lm_head.weight
        logits = F.linear(normed.to(head_w.device, dtype=head_w.dtype), head_w)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(),
                               labels.reshape(-1).to(head_w.device), ignore_index=-100)
        self.scaler.scale(loss.to(self.device) / self.cfg.grad_accum_steps).backward()
        grad = h_last.grad

        # 3) backward layer-by-layer, re-streaming weights, freeing after each
        for i in range(len(self.blocks) - 1, -1, -1):
            blk = self.blocks[i]
            x = saved[i].detach().requires_grad_(True)
            with torch.amp.autocast(**ac):
                if self.frozen_streamer is not None:
                    out = self._run_unit(blk, i, x, mask, pos_emb, flags[i],
                                         nxt=i - 1 if i - 1 >= 0 else None)
                else:
                    out = blk(x, attention_mask=mask, position_embeddings=pos_emb, stb_apply=flags[i])
            out.backward(grad)
            grad = x.grad
            del x, out
            saved[i] = None
        _sync(); self._bwd_ms = (time.time() - t_bwd0) * 1000
        return float(loss)

    def _train_layer_major(self, dataloader, monitor=None, logger=None):
        """USF layer-major loop: group G micro-batches per optimizer step so the
        model is streamed ONCE per step instead of G times (Thm 4)."""
        import math as _math
        self.model.train()
        G = max(1, self.cfg.grad_accum_steps)
        n_steps = (len(dataloader) // G) * self.cfg.max_epochs
        gstep = 0
        t0 = time.time()
        print(f"[S3Trainer] USF layer-major: G={G} micro-batches/step -> I/O divided by {G}")

        for epoch in range(self.cfg.max_epochs):
            self._apply_nfr(epoch)
            self.opt.zero_grad(set_to_none=True)
            group = []
            for bidx, batch in enumerate(dataloader):
                group.append(batch)
                if len(group) < G:
                    continue

                toks = sum(b["input_ids"].shape[0] * b["input_ids"].shape[1] for b in group)
                self.tokens_cum = getattr(self, "tokens_cum", 0) + toks
                if self._is_cuda:
                    torch.cuda.reset_peak_memory_stats(self.device)
                t_step = time.time()
                loss = self.step_layer_major(group)
                group = []

                self.scaler.unscale_(self.opt)
                grad_comp = logger.grad_components() if logger is not None else None
                grad_total = (sum(v ** 2 for v in grad_comp.values()) ** 0.5) if grad_comp else ""
                torch.nn.utils.clip_grad_norm_(self.trainable, self.cfg.max_grad_norm)
                scale_before = self.scaler.get_scale()
                self.scaler.step(self.opt)
                self.scaler.update()
                skipped = self.scaler.get_scale() < scale_before
                self.opt.zero_grad(set_to_none=True)
                for g in self.opt.param_groups:
                    g["lr"] = self._lr_at(gstep, max(1, n_steps))
                gstep += 1

                step_ms = (time.time() - t_step) * 1000
                vram_peak = torch.cuda.max_memory_allocated(self.device) / 1024**2 if self._is_cuda else 0.0
                if logger is not None:
                    import psutil
                    row = {
                        "wall_s": round(time.time() - t0, 2), "epoch": epoch, "opt_step": gstep,
                        "batch_idx": bidx, "seq_len": group[0]["input_ids"].shape[1] if group else -1,
                        "tokens_cum": self.tokens_cum, "loss": round(loss, 5),
                        "perplexity": round(_math.exp(min(loss, 30)), 4),
                        "lr": self.opt.param_groups[0]["lr"],
                        "grad_total": round(grad_total, 5) if grad_total != "" else "",
                        "scaler_scale": self.scaler.get_scale(), "skipped": skipped,
                        "fwd_ms": round(self._fwd_ms, 1), "bwd_ms": round(self._bwd_ms, 1),
                        "step_ms": round(step_ms, 1),
                        "vram_alloc_mb": round(torch.cuda.memory_allocated(self.device) / 1024**2, 1) if self._is_cuda else 0,
                        "vram_peak_mb": round(vram_peak, 1),
                        "cpu_ram_mb": round(psutil.Process().memory_info().rss / 1024**2, 1),
                        # Thm 4: the model is streamed once per optimizer step, not G times
                        "disk_read_mb": round(getattr(self, "_disk_mb_per_step", 0), 1),
                    }
                    if grad_comp:
                        row.update({f"grad_{k}": round(v, 5) for k, v in grad_comp.items()})
                    logger.log_step(row)
                    if gstep == 1 or gstep % max(1, self.cfg.log_interval) == 0:
                        logger.log_layer_grads(gstep)

                if gstep % max(1, self.cfg.log_interval) == 0 or gstep == 1:
                    print(f"[E{epoch+1} step{gstep}] loss={loss:.4f} vram_peak={vram_peak:.0f}MB "
                          f"fwd={self._fwd_ms:.0f}ms bwd={self._bwd_ms:.0f}ms")
                    if monitor is not None:
                        monitor.record_step(step=gstep, epoch=epoch, batch_idx=bidx, micro_step=-1,
                                            loss=loss, step_time_ms=step_ms, vram_mb=vram_peak,
                                            lr=self.opt.param_groups[0]["lr"], seq_len=0)

    # ---- training ---------------------------------------------------------
    def _lr_at(self, step, total):
        warmup = max(1, int(total * self.cfg.warmup_ratio))
        if step < warmup:
            return self.cfg.learning_rate * step / warmup
        prog = (step - warmup) / max(1, total - warmup)
        return self.cfg.learning_rate * 0.5 * (1 + math.cos(math.pi * prog))

    def train(self, dataloader, monitor=None, logger=None):
        if self._streaming and self.cfg.layer_major:
            return self._train_layer_major(dataloader, monitor, logger)
        import math as _math
        self.model.train()
        total_steps = len(dataloader) * self.cfg.max_epochs // max(1, self.cfg.grad_accum_steps)
        gstep = 0
        t0 = time.time()
        for epoch in range(self.cfg.max_epochs):
            self._apply_nfr(epoch)
            self.opt.zero_grad(set_to_none=True)
            for bidx, batch in enumerate(dataloader):
                bsz, seq_len = batch["input_ids"].shape[0], batch["input_ids"].shape[1]
                self.tokens_cum = getattr(self, "tokens_cum", 0) + bsz * seq_len
                if self._is_cuda:
                    torch.cuda.reset_peak_memory_stats(self.device)
                t_step = time.time()
                if self._streaming:
                    # manual memory-bounded fwd+bwd (loss already scaled + backprop'd inside)
                    loss = self.step_streaming(batch)
                else:
                    with torch.amp.autocast("cuda", enabled=self.cfg.amp and self._is_cuda, dtype=self.cfg.amp_dtype):
                        loss = self.forward_loss(batch["input_ids"])
                    self.scaler.scale(loss / self.cfg.grad_accum_steps).backward()
                    loss = float(loss)

                grad_comp, grad_total, skipped = None, "", ""
                if (bidx + 1) % self.cfg.grad_accum_steps == 0:
                    self.scaler.unscale_(self.opt)
                    if logger is not None:
                        grad_comp = logger.grad_components()
                        grad_total = sum(v ** 2 for v in grad_comp.values()) ** 0.5
                    torch.nn.utils.clip_grad_norm_(self.trainable, self.cfg.max_grad_norm)
                    scale_before = self.scaler.get_scale()
                    self.scaler.step(self.opt)
                    self.scaler.update()
                    skipped = self.scaler.get_scale() < scale_before
                    self.opt.zero_grad(set_to_none=True)
                    for g in self.opt.param_groups:
                        g["lr"] = self._lr_at(gstep, total_steps)
                    gstep += 1
                    if logger is not None and (gstep == 1 or gstep % max(1, self.cfg.log_interval) == 0):
                        logger.log_layer_grads(gstep)

                step_ms = (time.time() - t_step) * 1000
                vram_peak = torch.cuda.max_memory_allocated(self.device) / 1024**2 if self._is_cuda else 0.0
                vram_alloc = torch.cuda.memory_allocated(self.device) / 1024**2 if self._is_cuda else 0.0

                if logger is not None:
                    import psutil
                    lr_now = self.opt.param_groups[0]["lr"]
                    row = {
                        "wall_s": round(time.time() - t0, 2), "epoch": epoch, "opt_step": gstep,
                        "batch_idx": bidx, "seq_len": seq_len, "tokens_cum": self.tokens_cum,
                        "loss": round(loss, 5), "perplexity": round(_math.exp(min(loss, 30)), 4),
                        "lr": lr_now, "grad_total": round(grad_total, 5) if grad_total != "" else "",
                        "scaler_scale": self.scaler.get_scale(), "skipped": skipped,
                        "fwd_ms": round(getattr(self, "_fwd_ms", 0), 1),
                        "bwd_ms": round(getattr(self, "_bwd_ms", 0), 1), "step_ms": round(step_ms, 1),
                        "vram_alloc_mb": round(vram_alloc, 1), "vram_peak_mb": round(vram_peak, 1),
                        "cpu_ram_mb": round(psutil.Process().memory_info().rss / 1024**2, 1),
                        "disk_read_mb": round(getattr(self, "_disk_mb_per_step", 0), 1),
                    }
                    if grad_comp:
                        row.update({f"grad_{k}": round(v, 5) for k, v in grad_comp.items()})
                    logger.log_step(row)

                if bidx % self.cfg.log_interval == 0:
                    print(f"[E{epoch+1} B{bidx}] loss={float(loss):.4f} vram_peak={vram_peak:.0f}MB "
                          f"fwd={getattr(self,'_fwd_ms',0):.0f}ms bwd={getattr(self,'_bwd_ms',0):.0f}ms")
                    if monitor is not None:
                        monitor.record_step(step=gstep, epoch=epoch, batch_idx=bidx,
                                            micro_step=-1, loss=float(loss),
                                            step_time_ms=step_ms, vram_mb=vram_peak,
                                            lr=self.opt.param_groups[0]["lr"], seq_len=seq_len)
