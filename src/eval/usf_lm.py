"""An `lm_eval` backend that scores a USF-streamed model layer-major.

Why this exists at all: `lm_eval`'s stock HF backend needs the model resident, or
else leans on accelerate's per-forward disk offload, which re-reads every weight
for every batch. On a 4 GB card serving a 7B model that re-read dominates —
~28 s of disk per batch against ~8 s of compute. Thm 4 says the loop order is
free to change because the result does not depend on it, so this backend loads a
unit once and pushes *every* scoring request through it. The model is read once
per chunk rather than once per batch.

It also exists because S³ cannot be evaluated any other way. LoRA merges into the
frozen weights, so `lm_eval --model hf --model_args peft=...` handles it; S³ does
not, because NMF is a neural ODE over hidden states and no reparameterisation
folds a nonlinear flow into a weight matrix. Both adapters go through this path,
which has the side benefit of making the comparison strictly like-for-like.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from lm_eval.api.model import LM

__all__ = ["USFLM"]


class USFLM(LM):
    """Scores `lm_eval` requests through an `S3Trainer`'s streamed forward.

    Args:
        trainer: a built `S3Trainer` (streaming or resident; both work).
        tokenizer: the model's tokenizer.
        chunk_requests: how many scoring requests share one model read. Larger
            amortises the read further but holds more activations in host RAM;
            the default keeps that under roughly 1 GB at benchmark sequence
            lengths while making the read cost negligible.
        batch_size: sequences per forward inside a chunk.
        max_length: hard cap on context+continuation tokens.
    """

    def __init__(self, trainer, tokenizer, chunk_requests: int = 2048,
                 batch_size: int = 16, max_length: int = 2048):
        super().__init__()
        self.trainer = trainer
        self.tok = tokenizer
        self.chunk_requests = chunk_requests
        self.batch_size = batch_size
        self.max_length = max_length
        self._head = trainer.lm_head.weight
        self._head_gpu = None   # lazily moved to device on first score (below)

    # ---- tokenisation ------------------------------------------------------
    def _encode_pair(self, context: str, continuation: str):
        """Split a (context, continuation) pair into token ids.

        Encoding the two strings separately is wrong: a tokeniser may merge the
        characters straddling the boundary, so `enc(ctx) + enc(cont)` need not
        equal `enc(ctx + cont)` and the continuation would be scored at a
        position the model never sees. We encode the concatenation — the exact
        string the model reads — and locate the boundary by finding how much of
        the joint encoding the context's own encoding still agrees with.
        """
        n_spaces = len(context) - len(context.rstrip())
        if n_spaces > 0:                       # trailing space belongs to the
            continuation = context[-n_spaces:] + continuation   # continuation
            context = context[:-n_spaces]

        whole = self.tok.encode(context + continuation, add_special_tokens=False)
        ctx = self.tok.encode(context, add_special_tokens=False)

        split = len(ctx)
        for i in range(len(ctx)):
            if whole[i] != ctx[i]:
                split = i
                break
        if split == 0:                          # never score with empty context
            split = 1
        return whole[:split], whole[split:]

    # ---- scoring -----------------------------------------------------------
    @torch.no_grad()
    def _score_chunk(self, items):
        """Score one chunk of (idx, ctx_ids, cont_ids) with ONE model read."""
        batches, meta = [], []
        for k in range(0, len(items), self.batch_size):
            group = items[k:k + self.batch_size]
            seqs = [it[1] + it[2] for it in group]
            width = max(len(s) for s in seqs)
            ids = torch.full((len(group), width), self.tok.pad_token_id, dtype=torch.long)
            attn = torch.zeros((len(group), width), dtype=torch.long)
            for r, s in enumerate(seqs):
                ids[r, :len(s)] = torch.tensor(s, dtype=torch.long)
                attn[r, :len(s)] = 1
            batches.append({"input_ids": ids, "attention_mask": attn})
            meta.append(group)

        # ONE model read serves every batch above (Thm 4).
        H, _labels = self.trainer.forward_layer_major(batches)

        # Project on the streamer's device, not CPU: by this point every layer
        # has been unloaded (only one was ever resident), so the ~1GB fp16 vocab
        # matrix comfortably fits the freed budget. Measured effect: 1.15s/item
        # on CPU vs 45ms/item on this GPU (~25x) -- the difference between
        # minutes and hours over a full task's items.
        if self._head_gpu is None:
            dev = self.trainer.device
            self._head_gpu = (self._head.to(dev, dtype=self._head.dtype)
                              if dev.type == "cuda" else self._head)
        head = self._head_gpu

        out = {}
        for h_cpu, group in zip(H, meta):
            h = h_cpu.to(head.device, dtype=head.dtype)
            for r, (idx, ctx_ids, cont_ids) in enumerate(group):
                # `_prep` fed the model seq[:-1], so row t of h predicts seq[t+1].
                # The continuation occupies seq[len(ctx) : len(ctx)+len(cont)],
                # hence it is predicted by rows len(ctx)-1 ... len(ctx)+len(cont)-2.
                lo = len(ctx_ids) - 1
                hi = lo + len(cont_ids)
                logits = F.linear(h[r, lo:hi], head).float()
                logprobs = F.log_softmax(logits, dim=-1)
                tgt = torch.tensor(cont_ids, device=logprobs.device)
                ll = float(logprobs.gather(-1, tgt.unsqueeze(-1)).sum())
                greedy = bool((logprobs.argmax(dim=-1) == tgt).all())
                out[idx] = (ll, greedy)
        return out

    def loglikelihood(self, requests):
        items = []
        for idx, req in enumerate(requests):
            ctx, cont = req.args
            ctx_ids, cont_ids = self._encode_pair(ctx, cont)
            if not cont_ids:                     # nothing to score
                items.append((idx, ctx_ids, [self.tok.eos_token_id]))
                continue
            room = self.max_length - len(cont_ids)
            items.append((idx, ctx_ids[-room:] if room > 0 else ctx_ids[-1:], cont_ids))

        # Group by length so padding stays small; padding is masked and costs
        # only compute, but at these sequence lengths it would cost a lot of it.
        items.sort(key=lambda it: len(it[1]) + len(it[2]), reverse=True)

        results = {}
        for k in range(0, len(items), self.chunk_requests):
            results.update(self._score_chunk(items[k:k + self.chunk_requests]))
        return [results[i] for i in range(len(requests))]

    def loglikelihood_rolling(self, requests):
        raise NotImplementedError(
            "USFLM scores fixed (context, continuation) pairs. Rolling likelihood "
            "would re-read the model per window; use loglikelihood tasks."
        )

    def generate_until(self, requests):
        raise NotImplementedError(
            "Autoregressive generation defeats layer-major streaming: each new "
            "token needs a full pass, so the model would be re-read per token. "
            "Only multiple-choice (loglikelihood) tasks are supported."
        )
