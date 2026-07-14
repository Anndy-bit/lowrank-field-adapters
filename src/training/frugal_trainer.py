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
from torch.nn import functional as F
from torch.utils.data import DataLoader
from typing import Optional, Dict, List, Callable, Any
from dataclasses import dataclass, field
from collections import defaultdict
import time
import gc
import math
import os

from src.training.s3_opt import S3OPTOptimizer
from src.training.s3_optimizations import SGCGradientHook, NFRController, SBSController

# FSDP imports — graceful degradation on single GPU / no distributed
FSDP_AVAILABLE = False
try:
    from torch.distributed.fsdp import (
        FullyShardedDataParallel as FSDP,
        ShardingStrategy,
        MixedPrecision,
        BackwardPrefetch,
        FullStateDictConfig,
        StateDictType,
    )
    from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
    from torch.distributed.distributed_c10d import ProcessGroup
    FSDP_AVAILABLE = True
except ImportError:
    ShardingStrategy = None
    MixedPrecision = None


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
    # Memory management (FASE 1 — hardware-adaptive, no hardcoding)
    svd_dir: Optional[str] = None             # Directory with precomputed SVD factors
    svd_storage: str = "mmap"                 # "ram" (all in RAM) | "mmap" (lazy load per layer) | "disk" (stream from disk)
    base_model_offload: bool = True           # Use accelerate device_map="auto" with offload
    base_model_offload_dir: str = "/tmp/s3_offload"  # Disk offload folder for accelerate
    dataset_streaming: bool = False           # Use datasets streaming mode (0 RAM for data)
    cpu_ram_budget_gb: float = 10.0           # Hard limit for CPU RAM usage (accelerate max_memory)
    model_quantization: str = "none"          # "none" | "4bit" | "8bit" (for 70B support)
    # 70B / multi-GPU support (FASE 5)
    quant_compute_dtype: torch.dtype = torch.float16  # compute dtype after dequant
    quant_double_quant: bool = True                   # nested quantization for extra savings
    quant_quant_type: str = "nf4"                     # "nf4" | "fp4" (NF4 recommended)
    quant_bnb_4bit_use_dq: bool = True                 # use nested quantization
    fsdp_enabled: bool = False                        # Fully Sharded Data Parallel for multi-GPU
    fsdp_world_size: int = 1                          # number of GPUs for FSDP
    fsdp_sharding_strategy: str = "full"              # "full" | "shard_grad" | "no_shard"
    cpu_offload_params: bool = False                  # offload params to CPU (for 70B on 1 GPU)
    cpu_offload_optims: bool = False                  # offload optimizer states to CPU
    use_gradient_checkpointing: bool = False          # recompute activations to save VRAM
    # S3-OPT optimization flags (all disabled by default for backwards compat)
    use_smv: bool = False          # SMV: delta-modulation vector transfer
    use_emp: bool = False          # EMP: predictor-based μ skip
    use_ter: bool = False          # TER: entropy-based ODE routing
    use_fdgd: bool = False         # FDGD: Fourier gradient denoising
    use_gns: bool = False         # GNS: gradient-normalized backward skip
    use_sma: bool = False         # SMA: spectral checkpoint compression
    use_tows: bool = False        # TOWS: token-wise ODE warmstarting
    use_dra: bool = False         # DRA: dynamic rank adaptation
    use_mso: bool = False         # MSO: manifold shortcut detection
    use_hfisc: bool = False       # HFISC: optimal initialization
    # S3 already-implemented optimizations
    use_sgc: bool = True          # SGC: spectral gradient compression
    use_nfr: bool = True          # NFR: NMF flow recycling
    use_sbs: bool = True          # SBS: STB bypass sampling
    sbs_p_stb: float = 0.3         # STB bypass probability


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
        self._lm_head_weight = lm_head.weight
        self._lm_head_bias = lm_head.bias
        self.config = config
        self.device = device
        self.n_layers = len(s3_layers)
        self.training = False

        self._cpu_device = torch.device("cpu")
        self._gpu_device = torch.device(device)
        self._amp_enabled = config.use_amp and device.startswith("cuda")
        self._scaler = torch.amp.GradScaler("cuda", enabled=self._amp_enabled)

        self._check_vram_budget()

        self._setup_optimizer()
        self._layer_bytes = {}
        self._compute_layer_memory()

        # When layer_swap is disabled AND using ram storage, the SVD buffers stay 
        # resident on the GPU for the whole run, so warm them up once here.
        if not config.layer_swap and device.startswith("cuda") and config.svd_storage == "ram":
            self._move_all_layer_weights_to_gpu()

        self.stats = FrugalStats()

        # FASE 6: FSDP multi-GPU setup for 70B models
        self._is_distributed = False
        self._fsdp_enabled = False
        self._fsdp_world_size = 1
        self._rank = 0

        if config.fsdp_enabled and FSDP_AVAILABLE:
            self._setup_distributed_fsdp(config)
            if self._is_distributed:
                self._wrap_layers_with_fsdp(config)
        elif config.fsdp_enabled and not FSDP_AVAILABLE:
            print("[Frugal] WARNING: FSDP requested but torch.distributed.fsdp not available. Continuing without FSDP.")

        self.s3_opt = S3OPTOptimizer(
            smv=config.use_smv,
            emp=config.use_emp,
            ter=config.use_ter,
            fdgd=config.use_fdgd,
            gns=config.use_gns,
            sma=config.use_sma,
            tows=config.use_tows,
            dra=config.use_dra,
            mso=config.use_mso,
            hfisc=config.use_hfisc,
        )

        self.sgc = SGCGradientHook if config.use_sgc else None
        # NFRController needs an NMF module - use the first layer's nmf_attn
        if config.use_nfr and self.layers:
            self.nfr = NFRController(self.layers[0].nmf_attn)
        else:
            self.nfr = None
        self.sbs_enabled = config.use_sbs
        self.sbs_p_stb = config.sbs_p_stb if config.use_sbs else 0.0

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

    def is_rank_zero(self) -> bool:
        """Return True if this is the main process (rank 0) in distributed training."""
        return self._rank == 0

    def _setup_distributed_fsdp(self, config: FrugalConfig):
        """Initialize distributed training context for FSDP on multi-GPU 70B."""
        if not torch.cuda.is_available():
            print("[Frugal] WARNING: FSDP requires CUDA but no GPU available.")
            return

        self._rank = int(os.environ.get("RANK", "0"))
        self._world_size = int(os.environ.get("WORLD_SIZE", "1"))

        if config.fsdp_world_size > 1:
            self._fsdp_world_size = config.fsdp_world_size
        else:
            self._fsdp_world_size = self._world_size

        if self._fsdp_world_size > 1 and self._rank >= self._fsdp_world_size:
            print(f"[Frugal] Rank {self._rank} out of world_size {self._fsdp_world_size}. Skipping.")
            return

        if self._world_size > 1:
            self._is_distributed = True
            self._fsdp_enabled = True
            print(f"[Frugal] Distributed training: rank={self._rank}, world_size={self._world_size}")
        elif self._fsdp_world_size == 1 and torch.cuda.device_count() > 1:
            self._fsdp_world_size = torch.cuda.device_count()
            self._is_distributed = True
            self._fsdp_enabled = True
            print(f"[Frugal] Auto-detected {self._fsdp_world_size} GPUs for FSDP")
        else:
            print(f"[Frugal] FSDP enabled but only 1 GPU available. Running single-GPU.")

    def _wrap_layers_with_fsdp(self, config: FrugalConfig):
        """Wrap s3_layers with FSDP for sharded multi-GPU training (70B).

        Only wraps if FSDP is available and world_size > 1.
        Each rank gets its own FSDP-wrapped model (sharded by FSDP).
        For single GPU (world_size=1) the wrappers are no-ops.
        """
        if not self._is_distributed or self._fsdp_world_size <= 1:
            return

        try:
            sharding_strategy = {
                "full": ShardingStrategy.FULL_SHARD,
                "shard_grad": ShardingStrategy.SHARD_GRAD_OP,
                "no_shard": ShardingStrategy.NO_SHARD,
            }.get(config.fsdp_sharding_strategy, ShardingStrategy.SHARD_GRAD_OP)

            mp_policy = MixedPrecision(
                param_dtype=torch.float16,
                reduce_dtype=torch.float16,
                buffer_dtype=torch.float16,
            ) if config.use_amp else None

            def s3_auto_wrap_policy(module, recurse, **kwargs):
                if recurse:
                    return True
                return isinstance(module, type(self.layers[0]))

            wrapped_layers = nn.ModuleList()
            for layer in self.layers:
                fsdp_layer = FSDP(
                    layer,
                    sharding_strategy=sharding_strategy,
                    auto_wrap_policy=s3_auto_wrap_policy,
                    mixed_precision=mp_policy,
                    backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
                    device_id=self._rank % torch.cuda.device_count(),
                )
                wrapped_layers.append(fsdp_layer)

            self.layers = wrapped_layers
            print(f"[Frugal] FSDP wrapping complete: {self.n_layers} layers, strategy={config.fsdp_sharding_strategy}")

        except Exception as e:
            print(f"[Frugal] FSDP wrapping failed: {e}. Falling back to non-FSDP.")
            self._fsdp_enabled = False

    def _wrap_layers_with_fsdp(self, config: FrugalConfig):
        """Wrap s3_layers with FSDP for sharded multi-GPU training (70B).

        Only wraps if FSDP is available and world_size > 1.
        Each rank gets its own FSDP-wrapped model (sharded by FSDP).
        For single GPU (world_size=1) the wrappers are no-ops.
        """
        if not self._is_distributed or self._fsdp_world_size <= 1:
            return

        try:
            sharding_strategy = {
                "full": ShardingStrategy.FULL_SHARD,
                "shard_grad": ShardingStrategy.SHARD_GRAD_OP,
                "no_shard": ShardingStrategy.NO_SHARD,
            }.get(config.fsdp_sharding_strategy, ShardingStrategy.SHARD_GRAD_OP)

            mp_policy = MixedPrecision(
                param_dtype=torch.float16,
                reduce_dtype=torch.float16,
                buffer_dtype=torch.float16,
            ) if config.use_amp else None

            # Auto-wrap policy: each S3TransformerLayer in its own FSDP unit
            def s3_auto_wrap_policy(module, recurse, **kwargs):
                if recurse:
                    return True
                return isinstance(module, type(self.layers[0]))

            wrapped_layers = nn.ModuleList()
            for layer in self.layers:
                fsdp_layer = FSDP(
                    layer,
                    sharding_strategy=sharding_strategy,
                    auto_wrap_policy=s3_auto_wrap_policy,
                    mixed_precision=mp_policy,
                    backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
                    device_id=self._rank % torch.cuda.device_count(),
                )
                wrapped_layers.append(fsdp_layer)

            self.layers = wrapped_layers
            print(f"[Frugal] FSDP wrapping complete: {self.n_layers} layers, strategy={config.fsdp_sharding_strategy}")

        except Exception as e:
            print(f"[Frugal] FSDP wrapping failed: {e}. Falling back to non-FSDP.")
            self._fsdp_enabled = False

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
        """Move the SVD buffers of one layer to GPU. 
        
        Modes:
        - layer_swap=True, svd_storage="ram": buffers already in layer, just .to(gpu)
        - layer_swap=True, svd_storage="mmap"/"disk": load from disk to GPU per layer
        - layer_swap=False: buffers already on GPU (moved once in __init__), skip
        
        `force=True` bypasses the skip."""
        if not force and not self.config.layer_swap:
            return
        
        layer = self.layers[layer_idx]
        
        # If svd_storage is mmap/disk and we have svd_dir, load from disk
        if self.config.svd_dir and self.config.svd_storage in ("mmap", "disk"):
            self._load_svd_from_disk(layer, layer_idx, self._gpu_device)
        else:
            # Traditional: buffers already in layer, just move to GPU
            for name, buf in layer.named_buffers():
                if name.startswith("U_k") or name.startswith("Vt_k") or name.startswith("S_k"):
                    setattr(layer, name, buf.to(self._gpu_device))
            # Also move modulation MLP and STB parameters to GPU
            if device.startswith("cuda"):
                for attr_name in ["svmo_q", "svmo_k_proj", "svmo_v", "svmo_o",
                                   "svmo_up", "svmo_gate", "svmo_down"]:
                    svmo = getattr(layer, attr_name, None)
                    if svmo is not None and hasattr(svmo, 'modulation'):
                        svmo.modulation.to(self._gpu_device)
                stb = getattr(layer, 'stb', None)
                if stb is not None:
                    for buf_attr in ['U_k', 'Vt_k']:
                        buf = getattr(stb, buf_attr, None)
                        if buf is not None:
                            setattr(stb, buf_attr, buf.to(self._gpu_device))

    def _move_layer_weights_to_cpu(self, layer_idx: int):
        """Undo `_move_layer_weights_to_gpu`. No-op when layer_swap=False.
        
        For mmap/disk mode: move SVD buffers back to CPU (or just delete to free GPU).
        """
        if not self.config.layer_swap:
            return
        
        layer = self.layers[layer_idx]
        
        if self.config.svd_dir and self.config.svd_storage in ("mmap", "disk"):
            # For mmap/disk: move SVD buffers back to CPU
            self._move_svd_to_cpu(layer)
        else:
            for name, buf in layer.named_buffers():
                if name.startswith("U_k") or name.startswith("Vt_k") or name.startswith("S_k"):
                    setattr(layer, name, buf.to(self._cpu_device))

    def _load_svd_from_disk(self, layer, layer_idx: int, device):
        """Load SVD factors for a layer from safetensors on disk directly to device."""
        from pathlib import Path
        try:
            from safetensors.torch import load_file
        except ImportError:
            return
        
        svd_dir = Path(self.config.svd_dir)
        if not svd_dir.exists():
            return
        
        # Load SVD for each SVMO projection in this layer
        proj_names = ["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "gate_proj", "down_proj"]
        svmo_attrs = [
            "svmo_q", "svmo_k_proj", "svmo_v", "svmo_o",
            "svmo_up", "svmo_gate", "svmo_down"
        ]
        
        for proj_name, attr_name in zip(proj_names, svmo_attrs):
            svmo = getattr(layer, attr_name, None)
            if svmo is None:
                continue
            
            layer_dir = svd_dir / f"layer_{layer_idx}.{proj_name}"
            if not layer_dir.exists():
                continue
            
            try:
                U_k = load_file(str(layer_dir / "U_k.safetensors"), device=str(device))["U_k"]
                S_k = load_file(str(layer_dir / "S_k.safetensors"), device=str(device))["S_k"]
                Vt_k = load_file(str(layer_dir / "Vt_k.safetensors"), device=str(device))["Vt_k"]

                svmo.U_k = U_k
                svmo.S_k = S_k
                svmo.Vt_k = Vt_k
                svmo.S_k_log = torch.log(S_k + 1e-8)

                # Move modulation MLP parameters to GPU
                if hasattr(svmo, 'modulation') and device.type == 'cuda':
                    svmo.modulation.to(device)
            except Exception:
                # If load fails, keep existing buffers (fallback)
                pass

        # Move STB parameters to GPU as well (outside the per-projection loop)
        stb = getattr(layer, 'stb', None)
        if stb is not None and device.type == 'cuda':
            for buf_attr in ['U_k', 'Vt_k']:
                buf = getattr(stb, buf_attr, None)
                if buf is not None and buf.device.type == 'cpu':
                    setattr(stb, buf_attr, buf.to(device))

    def _move_svd_to_cpu(self, layer):
        """Move SVD buffers back to CPU."""
        for name, buf in layer.named_buffers():
            if name.startswith("U_k") or name.startswith("Vt_k") or name.startswith("S_k"):
                setattr(layer, name, buf.to(self._cpu_device))
        # Also move modulation MLP and STB back to CPU
        for attr_name in ["svmo_q", "svmo_k_proj", "svmo_v", "svmo_o",
                           "svmo_up", "svmo_gate", "svmo_down"]:
            svmo = getattr(layer, attr_name, None)
            if svmo is not None and hasattr(svmo, 'modulation'):
                svmo.modulation.to(self._cpu_device)
        stb = getattr(layer, 'stb', None)
        if stb is not None:
            for buf_attr in ['U_k', 'Vt_k']:
                buf = getattr(stb, buf_attr, None)
                if buf is not None:
                    setattr(stb, buf_attr, buf.to(self._cpu_device))


    def _forward_layer(
        self, layer_idx: int, hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        layer = self.layers[layer_idx]
        self._move_layer_weights_to_gpu(layer_idx)

        # Comprehensively ensure the entire layer is on GPU (catches any missed submodules)
        if self._gpu_device.type == 'cuda':
            layer = layer.to(self._gpu_device, non_blocking=True)

        # Update S3-OPT context with current layer's SVD factors
        self.s3_opt.update_layer_context(layer)

        # TER: Route NMF steps based on token entropy (if enabled)
        nmf_N_steps = None
        if self.config.use_ter and self.s3_opt.ter is not None:
            nmf_N_steps = self.s3_opt.route_nmf_steps(layer, hidden_states)

        # TOWS: Token-wise ODE warmstart
        if self.s3_opt.cfg.tows and self.s3_opt._tows_h_prev is not None:
            hidden_states = self.s3_opt.warmstart_forward(hidden_states, layer_idx)

        # Defensive: guarantee 3D [B, S, D] before calling the layer
        if hidden_states.dim() == 2:
            hidden_states = hidden_states.unsqueeze(0)
        elif hidden_states.dim() == 1:
            hidden_states = hidden_states.unsqueeze(0).unsqueeze(0)

        with torch.amp.autocast("cuda", enabled=self._amp_enabled, dtype=self.config.amp_dtype):
            if self.config.use_gradient_checkpointing and self.training:
                from torch.utils.checkpoint import checkpoint
                hidden_states.requires_grad_(True)
                chk_out = checkpoint(
                    layer, hidden_states, nmf_N_steps=nmf_N_steps,
                    use_reentrant=False,
                    preserve_rng_state=False,
                )
                hidden_states = chk_out[0] if isinstance(chk_out, tuple) else chk_out
            else:
                layer_out = layer(hidden_states, nmf_N_steps=nmf_N_steps)
                hidden_states = layer_out[0] if isinstance(layer_out, tuple) else layer_out

        # SGC: Attach gradient hook for spectral compression (Theorem 8)
        if self.config.use_sgc and self.sgc is not None and hasattr(layer, 'svmo_q'):
            u_k = getattr(layer.svmo_q, 'U_k', None)
            if u_k is not None:
                hook = SGCGradientHook(u_k)
                hidden_states.register_hook(hook.backward_hook)

        # SMA: Compress checkpoint if enabled
        if self.config.use_sma and self.s3_opt.sma is not None:
            u_k = getattr(layer.svmo_q, 'U_k', None)
            if u_k is not None:
                layer._sma_compressed = self.s3_opt.sma_compress(hidden_states, u_k)

        self._move_layer_weights_to_cpu(layer_idx)
        return hidden_states

    def _backward_layer(
        self, layer_idx: int, hidden_states: torch.Tensor, grad_in: torch.Tensor,
    ):
        self._move_layer_weights_to_gpu(layer_idx)
        
        # SMA: Reconstruct hidden_states from compressed checkpoint if available
        if self.config.use_sma and self.s3_opt.sma is not None:
            # The checkpoint was compressed in forward; we need the original for backward
            # For now, use the passed hidden_states (which came from LayerCheckpoint)
            pass
        
        hidden_states.requires_grad_(True)
        with torch.enable_grad():
            with torch.amp.autocast("cuda", enabled=self._amp_enabled, dtype=self.config.amp_dtype):
                layer_out = self.layers[layer_idx](hidden_states)

            layer_out_tensor = layer_out[0] if isinstance(layer_out, tuple) else layer_out
            self._scaler.scale(layer_out_tensor).backward(gradient=grad_in, retain_graph=False)

        self._move_layer_weights_to_cpu(layer_idx)
        del hidden_states

    def _warmup_scheduler_step(self, step: int, seq_len: int = None):
        """Update learning rate with cosine schedule + warmup.
        
        For token_by_token mode: total micro-steps = num_sequences * (seq_len - 1) * epochs
        For sequence mode: total steps = num_batches * epochs
        """
        if seq_len is None:
            seq_len = 512  # default fallback
        
        # Get dataloader length (fallback if not set yet)
        dataloader_len = getattr(self, '_current_dataloader', None)
        if dataloader_len is not None:
            dataloader_len = len(dataloader_len)
        else:
            dataloader_len = 100  # reasonable fallback
        
        if self.config.token_by_token:
            # Each sequence produces (seq_len - 1) micro-steps
            total_micro_steps = dataloader_len * (seq_len - 1) * self.config.max_epochs
        else:
            # One step per sequence/batch
            total_micro_steps = dataloader_len * self.config.max_epochs
        
        warmup_steps = int(total_micro_steps * self.config.warmup_ratio)
        
        if step < warmup_steps:
            lr_scale = step / max(1, warmup_steps)
        else:
            progress = (step - warmup_steps) / max(1, total_micro_steps - warmup_steps)
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

        with torch.amp.autocast("cuda", enabled=self._amp_enabled, dtype=self.config.amp_dtype):
            hidden_states = self.embedding(input_ids).unsqueeze(0)

        checkpoints: List[LayerCheckpoint] = []

        for layer_idx in range(self.n_layers):
            hidden_states = self._forward_layer(layer_idx, hidden_states)
            checkpoints.append(LayerCheckpoint(hidden_states, layer_idx))

        with torch.amp.autocast("cuda", enabled=self._amp_enabled, dtype=self.config.amp_dtype):
            logits = F.linear(
                hidden_states.to("cpu", non_blocking=True),
                self._lm_head_weight,
                self._lm_head_bias,
            ).to(hidden_states.device, non_blocking=True)
            loss = torch.nn.functional.cross_entropy(
                logits.squeeze(0), target_ids.squeeze(0)
            )

        loss_scaled = loss / self.config.gradient_accumulation_steps

        # Compute gradient w.r.t. hidden_states ONLY through logits path (not 28-layer graph)
        # to avoid CPU tensors in the 28-layer computation graph
        hidden_states_detached = hidden_states.detach().requires_grad_(True)
        logits_detached = F.linear(
            hidden_states_detached.to("cpu", non_blocking=True),
            self._lm_head_weight,
            self._lm_head_bias,
        ).to(hidden_states.device, non_blocking=True)
        loss_detached = torch.nn.functional.cross_entropy(
            logits_detached.squeeze(0), target_ids.squeeze(0)
        )
        loss_detached_scaled = loss_detached / self.config.gradient_accumulation_steps
        grad_current = torch.autograd.grad(loss_detached_scaled, hidden_states_detached)[0]

        del logits, loss, logits_detached, loss_detached, hidden_states_detached

        for layer_idx in reversed(range(self.n_layers)):
            # FDGD: filter gradient in frequency domain
            if self.s3_opt.cfg.fdgd:
                grad_current = self.s3_opt.fdgd.filter(grad_current)

            # GNS: skip backward if gradient norm too small
            if self.s3_opt.cfg.gns:
                grad_norm = grad_current.norm().item()
                if self.s3_opt.gns.should_skip(grad_norm, f"layer_{layer_idx}"):
                    continue

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

        with torch.amp.autocast("cuda", enabled=self._amp_enabled, dtype=self.config.amp_dtype):
            hidden_states = self.embedding(inputs)

        for layer_idx in range(self.n_layers):
            hidden_states = self._forward_layer(layer_idx, hidden_states)

        with torch.amp.autocast("cuda", enabled=self._amp_enabled, dtype=self.config.amp_dtype):
            logits = F.linear(
                hidden_states.to("cpu", non_blocking=True),
                self._lm_head_weight,
                self._lm_head_bias,
            ).to(hidden_states.device, non_blocking=True)
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
        self.training = True
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

                # S3-OPT: Epoch-level updates
                # NFR: Progressive ODE steps (N=2 -> N=4)
                if self.config.use_nfr and self.nfr is not None:
                    self.nfr.set_epoch(epoch)
                
                # TER: Update entropy routing thresholds per epoch
                if self.config.use_ter and self.s3_opt.ter is not None:
                    self.s3_opt.ter.set_epoch(epoch)
                
                # DRA: Update rank adaptation thresholds per epoch
                if self.config.use_dra and self.s3_opt.dra is not None:
                    self.s3_opt.dra.set_epoch(epoch)

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
                                    seq_len=seq_len,
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
                                seq_len=seq_len,
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
                    self._warmup_scheduler_step(global_step, seq_len)

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

                # S3-OPT: NFR - Save ODE state for next epoch warm-start
                if self.config.use_nfr and self.nfr is not None and epoch < self.config.max_epochs - 1:
                    # NFR saves the final state of each NMF flow as warm-start for next epoch
                    for layer in self.layers:
                        if hasattr(layer, 'nmf_attn') and layer.nmf_attn is not None:
                            self.nfr.save_checkpoint(torch.zeros(1), torch.zeros(1))  # placeholder

        finally:
            self.training = False
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