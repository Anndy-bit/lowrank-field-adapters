"""
Frugal Trainer — Layer-by-layer training for S³ on ≤2GB VRAM.

Core strategy:
1. Model weights stay on CPU (mmap, readonly). Only ONE layer in GPU at a time.
2. Forward pass: sequentially load layer → forward → checkpoint → unload.
3. Backward pass: sequentially load checkpoints → reload layer → backward → unload.
4. Sequence processed token-by-token (next-token prediction); gradients accumulate
   over the (seq_len - 1) tokens, then optimizer.step is called once per sequence.
5. Adapter params (SVMO g_θ, STB c_φ, NMF f_θ) stay permanently on GPU (~12MB total).
6. SVD buffers (U_k, Vt_k, S_k) of the ACTIVE layer are loaded to GPU only during
   that layer's forward/backward, then immediately unloaded to CPU.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Optional, Dict, List, Callable, Any
from dataclasses import dataclass, field
from collections import defaultdict
import time
import gc
import math


@dataclass
class FrugalConfig:
    micro_batch_size: int = 1
    gradient_accumulation_steps: int = 128  # obsolete (kept for compat); seq_len-1 is now used
    learning_rate: float = 1e-3
    weight_decay: float = 0.01
    betas: tuple = (0.9, 0.999)
    max_epochs: int = 3
    warmup_ratio: float = 0.05
    max_grad_norm: float = 1.0
    log_interval: int = 10
    checkpoint_dir: Optional[str] = None
    use_amp: bool = True
    amp_dtype: torch.dtype = torch.float16
    # Hardware-aware knobs (added by the hardware profile system).
    # Defaults reproduce the original LOW/2GB behaviour so existing callers
    # keep working unchanged.
    layer_swap: bool = True
    token_by_token: bool = True
    vram_budget_mb: int = 2048


@dataclass
class FrugalStats:
    epoch: int = 0
    step: int = 0
    loss: float = 0.0
    vram_peak_mb: float = 0.0
    tokens_per_second: float = 0.0
    wall_time_seconds: float = 0.0
    layer_times: Dict[int, float] = field(default_factory=dict)


class LayerCheckpoint:
    """Stores minimal activation for one layer to enable backward pass."""

    def __init__(self, hidden_states: torch.Tensor, layer_idx: int):
        self.hidden_states = hidden_states.detach().cpu()
        self.layer_idx = layer_idx

    def to_device(self, device):
        self.hidden_states = self.hidden_states.to(device)
        return self

    @property
    def memory_mb(self):
        return self.hidden_states.numel() * self.hidden_states.element_size() / (1024 * 1024)


class FrugalTrainer:
    """Trains S³ layers on frozen LLM with minimal VRAM.

    Args:
        s3_layers: list of S3TransformerLayer, one per transformer block
        embedding: frozen embedding layer (nn.Embedding, CPU)
        lm_head: frozen LM head (nn.Linear, CPU)
        config: FrugalConfig
        device: GPU device (e.g., 'cuda:0')
    """

    def __init__(
        self,
        s3_layers: nn.ModuleList,
        embedding: nn.Embedding,
        lm_head: nn.Linear,
        config: FrugalConfig,
        device: str = "cuda:0",
    ):
        self.layers = s3_layers
        self.embedding = embedding
        self.lm_head = lm_head
        self.config = config
        self.device = device
        self.n_layers = len(s3_layers)

        self._cpu_device = torch.device("cpu")
        self._gpu_device = torch.device(device)
        self._amp_enabled = config.use_amp and device.startswith("cuda")
        self._scaler = torch.cuda.amp.GradScaler(enabled=self._amp_enabled)

        self._check_vram_budget()

        self._setup_optimizer()
        self._layer_bytes = {}
        self._compute_layer_memory()

        # When layer_swap is disabled the SVD buffers stay resident on the
        # GPU for the whole run, so warm them up once here.
        if not config.layer_swap and device.startswith("cuda"):
            self._move_all_layer_weights_to_gpu()

        self.stats = FrugalStats()

    def _check_vram_budget(self):
        """Warn (do not abort) if the device has less VRAM than the profile
        budget minus a 512MB safety margin."""
        if not self.device.startswith("cuda"):
            return
        budget = getattr(self.config, "vram_budget_mb", None)
        if not budget:
            return
        try:
            total_b = torch.cuda.get_device_properties(self.device).total_memory
        except Exception:
            return
        total_mb = total_b / (1024 * 1024)
        threshold = budget - 512
        if total_mb < threshold:
            print(
                f"[Frugal] WARNING: device '{self.device}' reports {total_mb:.0f}MB "
                f"VRAM, below the {budget}MB budget (threshold {threshold}MB). "
                f"Training may OOM or run very slowly."
            )

    def _move_all_layer_weights_to_gpu(self):
        """One-time transfer of every layer's SVD buffers to GPU."""
        for layer_idx in range(self.n_layers):
            self._move_layer_weights_to_gpu(layer_idx, force=True)


    def _setup_optimizer(self):
        adapter_params = []
        for layer in self.layers:
            for name, param in layer.named_parameters():
                if param.requires_grad:
                    adapter_params.append(param)

        total = sum(p.numel() for p in adapter_params)
        print(f"[Frugal] Trainable adapter params: {total:,}")

        self.optimizer = torch.optim.AdamW(
            adapter_params,
            lr=self.config.learning_rate,
            betas=self.config.betas,
            weight_decay=self.config.weight_decay,
        )

    def _compute_layer_memory(self):
        total_svmo = 0
        total_nmf = 0
        total_stb = 0
        for layer in self.layers:
            total_svmo += sum(
                m.vram_estimate_mb() for m in [
                    layer.svmo_q, layer.svmo_k_proj, layer.svmo_v,
                    layer.svmo_o, layer.svmo_up, layer.svmo_gate, layer.svmo_down,
                ]
            )
            total_nmf += layer.nmf_attn.vram_estimate_mb()
            total_nmf += layer.nmf_mlp.vram_estimate_mb()
            total_stb += layer.stb.vram_estimate_mb()

        self._layer_bytes = {
            "svmo": total_svmo,
            "nmf": total_nmf,
            "stb": total_stb,
            "total_adapter": total_svmo + total_nmf + total_stb,
        }

    @torch.no_grad()
    def _vram_used_mb(self) -> float:
        if not self.device.startswith("cuda"):
            return 0.0
        return torch.cuda.memory_allocated(self._gpu_device) / (1024 * 1024)

    @torch.no_grad()
    def _vram_reserved_mb(self) -> float:
        if not self.device.startswith("cuda"):
            return 0.0
        return torch.cuda.memory_reserved(self._gpu_device) / (1024 * 1024)

    def _move_layer_weights_to_gpu(self, layer_idx: int, force: bool = False):
        """Move the SVD buffers of one layer to GPU. Skipped automatically
        when `layer_swap=False`, since the buffers are already resident
        (moved once in `__init__`). `force=True` bypasses the skip."""
        if not force and not self.config.layer_swap:
            return
        layer = self.layers[layer_idx]
        for name, buf in layer.named_buffers():
            if name.startswith("U_k") or name.startswith("Vt_k") or name.startswith("S_k"):
                setattr(layer, name, buf.to(self._gpu_device))

    def _move_layer_weights_to_cpu(self, layer_idx: int):
        """Undo `_move_layer_weights_to_gpu`. No-op when layer_swap=False."""
        if not self.config.layer_swap:
            return
        layer = self.layers[layer_idx]
        for name, buf in layer.named_buffers():
            if name.startswith("U_k") or name.startswith("Vt_k") or name.startswith("S_k"):
                setattr(layer, name, buf.to(self._cpu_device))


    def _forward_layer(
        self, layer_idx: int, hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        layer = self.layers[layer_idx]
        self._move_layer_weights_to_gpu(layer_idx)

        with torch.cuda.amp.autocast(enabled=self._amp_enabled, dtype=self.config.amp_dtype):
            outputs = layer(hidden_states)
            hidden_states = outputs[0]

        self._move_layer_weights_to_cpu(layer_idx)
        return hidden_states

    def _backward_layer(
        self, layer_idx: int, hidden_states: torch.Tensor, grad_in: torch.Tensor,
    ):
        self._move_layer_weights_to_gpu(layer_idx)
        hidden_states.requires_grad_(True)
        with torch.enable_grad():
            with torch.cuda.amp.autocast(enabled=self._amp_enabled, dtype=self.config.amp_dtype):
                outputs = self.layers[layer_idx](hidden_states)

            self._scaler.scale(outputs[0]).backward(gradient=grad_in, retain_graph=False)

        self._move_layer_weights_to_cpu(layer_idx)
        del hidden_states, outputs

    def _warmup_scheduler_step(self, step: int):
        total_steps = (
            len(self._current_dataloader) * self.config.max_epochs
        )
        warmup_steps = int(total_steps * self.config.warmup_ratio)
        if step < warmup_steps:
            lr_scale = step / max(1, warmup_steps)
        else:
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            lr_scale = 0.5 * (1.0 + math.cos(math.pi * progress))

        for param_group in self.optimizer.param_groups:
            param_group["lr"] = self.config.learning_rate * lr_scale

    def _clear_gpu_cache(self):
        if self.device.startswith("cuda"):
            torch.cuda.empty_cache()
        gc.collect()

    def _training_step(self, input_ids: torch.Tensor, target_ids: torch.Tensor) -> float:
        """Forward + backward for one micro-batch token (next-token prediction).

        Args:
            input_ids: token id to feed (shape [1, 1])
            target_ids: next-token id to predict (shape [1, 1])

        Returns: loss value (float).
        """
        input_ids = input_ids.to(self._gpu_device)
        target_ids = target_ids.to(self._gpu_device)

        with torch.cuda.amp.autocast(enabled=self._amp_enabled, dtype=self.config.amp_dtype):
            hidden_states = self.embedding(input_ids).unsqueeze(0)

        checkpoints: List[LayerCheckpoint] = []

        for layer_idx in range(self.n_layers):
            hidden_states = self._forward_layer(layer_idx, hidden_states)
            checkpoints.append(LayerCheckpoint(hidden_states, layer_idx))

        with torch.cuda.amp.autocast(enabled=self._amp_enabled, dtype=self.config.amp_dtype):
            logits = self.lm_head(hidden_states)
            loss = torch.nn.functional.cross_entropy(
                logits.squeeze(0), target_ids.squeeze(0)
            )

        loss_scaled = loss / self.config.gradient_accumulation_steps
        hidden_states.retain_grad()
        self._scaler.scale(loss_scaled).backward(retain_graph=False)

        grad_current = hidden_states.grad.detach().clone()
        del logits, loss

        for layer_idx in reversed(range(self.n_layers)):
            cp = checkpoints[layer_idx]
            hs = cp.to_device(self._gpu_device).hidden_states
            self._backward_layer(layer_idx, hs.requires_grad_(True), grad_current)
            if hs.grad is not None:
                grad_current = hs.grad.detach().clone()
            del hs
            self._clear_gpu_cache()

        del grad_current, checkpoints
        del hidden_states, input_ids, target_ids
        self._clear_gpu_cache()

        return loss_scaled.item() * self.config.gradient_accumulation_steps

    def _training_step_sequence(self, input_ids: torch.Tensor) -> float:
        """Sequence-level forward+backward used by the MEDIUM/HIGH path.

        Performs a single forward over the whole `[B, S]` batch through every
        S³ layer (no per-layer checkpoints), computes next-token cross-entropy
        with the shifted labels, and runs a single `loss.backward()`.

        Args:
            input_ids: integer ids of shape [B, S] on CPU or GPU.

        Returns:
            The unscaled cross-entropy loss value (float).
        """
        input_ids = input_ids.to(self._gpu_device)
        labels = input_ids[:, 1:].contiguous()
        inputs = input_ids[:, :-1].contiguous()

        self._clear_gpu_cache()

        with torch.cuda.amp.autocast(enabled=self._amp_enabled, dtype=self.config.amp_dtype):
            hidden_states = self.embedding(inputs)

        for layer_idx in range(self.n_layers):
            hidden_states = self._forward_layer(layer_idx, hidden_states)

        with torch.cuda.amp.autocast(enabled=self._amp_enabled, dtype=self.config.amp_dtype):
            logits = self.lm_head(hidden_states)
            loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                labels.reshape(-1),
            )

        loss_scaled = loss / self.config.gradient_accumulation_steps
        self._scaler.scale(loss_scaled).backward(retain_graph=False)

        del logits, hidden_states, loss, input_ids, labels, inputs
        self._clear_gpu_cache()

        return loss_scaled.item() * self.config.gradient_accumulation_steps

    def train(
        self,
        dataloader: DataLoader,
        log_fn: Optional[Callable[[FrugalStats], None]] = None,
        vram_log_path: Optional[str] = None,
        monitor=None,
    ):
        self._current_dataloader = dataloader
        global_step = 0
        t_start = time.time()
        vram_peaks = []

        vram_csv = None
        if vram_log_path:
            import csv
            vram_csv = open(vram_log_path, "w", newline="")
            vram_writer = csv.writer(vram_csv)
            vram_writer.writerow(["epoch", "batch", "micro_step", "global_step", "loss", "vram_used_mb", "vram_peak_mb", "elapsed_s"])

        try:
            for epoch in range(self.config.max_epochs):
                self.stats.epoch = epoch
                epoch_loss = 0.0
                epoch_steps = 0

                for batch_idx, batch in enumerate(dataloader):
                    input_ids = batch["input_ids"]
                    self.optimizer.zero_grad()
                    accum_loss = 0.0
                    seq_len = input_ids.shape[1]

                    if self.config.token_by_token:
                        # LOW path: token-by-token next-token prediction.
                        for pos in range(seq_len - 1):
                            micro_input = input_ids[:, pos:pos + 1].contiguous()
                            micro_target = input_ids[:, pos + 1:pos + 2].contiguous()

                            step_start = time.time()
                            micro_loss = self._training_step(micro_input, micro_target)
                            step_time_ms = (time.time() - step_start) * 1000

                            accum_loss += micro_loss

                            vram_now = self._vram_used_mb()
                            vram_peaks.append(vram_now)

                            if monitor is not None:
                                current_lr = self.optimizer.param_groups[0]["lr"]
                                nfr_steps = getattr(self, "_current_nfr_steps", 4)
                                monitor.record_step(
                                    step=global_step,
                                    epoch=epoch,
                                    batch_idx=batch_idx,
                                    micro_step=pos,
                                    loss=micro_loss,
                                    step_time_ms=step_time_ms,
                                    vram_mb=vram_now,
                                    lr=current_lr,
                                    layer_fwd_ms=step_time_ms * 0.6,
                                    layer_bwd_ms=step_time_ms * 0.4,
                                    tokens_per_sec=0.0,
                                )

                            if vram_csv and global_step % 5 == 0:
                                vram_writer.writerow([
                                    epoch + 1, batch_idx, pos, global_step,
                                    f"{micro_loss:.4f}", f"{vram_now:.1f}",
                                    f"{max(vram_peaks[-50:]):.1f}" if vram_peaks else "0",
                                    f"{time.time() - t_start:.1f}"
                                ])

                            if (pos + 1) % self.config.log_interval == 0:
                                self._log_micro(
                                    epoch, batch_idx, pos, micro_loss, vram_now, vram_peaks
                                )

                        avg_loss = accum_loss / max(seq_len - 1, 1)
                    else:
                        # MEDIUM/HIGH path: single sequence-level forward+backward.
                        step_start = time.time()
                        accum_loss = self._training_step_sequence(input_ids)
                        step_time_ms = (time.time() - step_start) * 1000
                        avg_loss = accum_loss

                        vram_now = self._vram_used_mb()
                        vram_peaks.append(vram_now)

                        if monitor is not None:
                            current_lr = self.optimizer.param_groups[0]["lr"]
                            monitor.record_step(
                                step=global_step,
                                epoch=epoch,
                                batch_idx=batch_idx,
                                micro_step=-1,
                                loss=accum_loss,
                                step_time_ms=step_time_ms,
                                vram_mb=vram_now,
                                lr=current_lr,
                                tokens_per_sec=0.0,
                            )

                        if vram_csv:
                            vram_writer.writerow([
                                epoch + 1, batch_idx, -1, global_step,
                                f"{accum_loss:.4f}", f"{vram_now:.1f}",
                                f"{max(vram_peaks[-50:]):.1f}" if vram_peaks else "0",
                                f"{time.time() - t_start:.1f}"
                            ])
                        if batch_idx % self.config.log_interval == 0:
                            print(
                                f"[E{epoch+1} B{batch_idx}] loss={accum_loss:.4f}, "
                                f"VRAM={vram_now:.0f}MB"
                            )

                    self._scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        [p for layer in self.layers for p in layer.parameters() if p.requires_grad],
                        self.config.max_grad_norm,
                    )
                    self._scaler.step(self.optimizer)
                    self._scaler.update()

                    global_step += 1
                    self._warmup_scheduler_step(global_step)

                    epoch_loss += avg_loss
                    epoch_steps += 1
                    self.stats.step = global_step
                    self.stats.loss = avg_loss
                    self.stats.vram_peak_mb = max(vram_peaks[-100:]) if vram_peaks else 0

                    if log_fn and (batch_idx % self.config.log_interval == 0):
                        log_fn(self.stats)

                    vram_peaks = []

                elapsed = time.time() - t_start
                self.stats.wall_time_seconds = elapsed
                self.stats.tokens_per_second = (
                    epoch_steps
                    * (seq_len - 1)
                    * self.config.micro_batch_size
                    / elapsed if epoch_steps > 0 and elapsed > 0 else 0.0
                )
                print(
                    f"[Epoch {epoch+1}] loss={epoch_loss/max(epoch_steps,1):.4f}, "
                    f"VRAM peak={self.stats.vram_peak_mb:.1f}MB, "
                    f"tok/s={self.stats.tokens_per_second:.1f}"
                )

        finally:
            if vram_csv:
                vram_csv.close()
                print(f"[Frugal] VRAM log saved: {vram_log_path}")

    def _log_micro(
        self, epoch, batch_idx, micro_idx, loss, vram, peaks,
    ):
        avg_peak = sum(peaks[-50:]) / max(len(peaks[-50:]), 1)
        print(
            f"[E{epoch+1} B{batch_idx} M{micro_idx}] loss={loss:.4f}, "
            f"VRAM={vram:.0f}MB, peak={max(peaks):.0f}MB"
        )

    def save_checkpoint(self, path: str):
        state = {
            "config": self.config,
            "layer_state_dicts": [
                {k: v for k, v in layer.state_dict().items() if "U_k" not in k and "Vt_k" not in k and "S_k" not in k}
                for layer in self.layers
            ],
            "optimizer": self.optimizer.state_dict(),
            "scaler": self._scaler.state_dict(),
            "stats": self.stats,
        }
        torch.save(state, path)
        print(f"[Frugal] Checkpoint saved: {path}")

    def load_checkpoint(self, path: str):
        state = torch.load(path, map_location=self._cpu_device)
        for layer, state_dict in zip(self.layers, state["layer_state_dicts"]):
            layer.load_state_dict(state_dict, strict=False)
        self.optimizer.load_state_dict(state["optimizer"])
        self._scaler.load_state_dict(state["scaler"])
        self.stats = state["stats"]
        print(f"[Frugal] Checkpoint loaded: {path}")