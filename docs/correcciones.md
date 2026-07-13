# Plan de Correcciones y Optimización — S³ para Poor Compute Universal

**Objetivo:** Hacer que S³ entrene 7B (y escale a 70B) en **cualquier hardware** — desde GTX 1050 2GB/4GB + 10GB RAM hasta A100 80GB — **sin hardcodear nada**, usando las 3 tiers (LOW/MEDIUM/HIGH) ya definidas y activando S3-OPT de forma real en el training loop.

---

## 1. DIAGNÓSTICO REAL (Lo que NO funciona hoy)

| Componente | Problema | Impacto |
|---|---|---|
| `hybrid.py` `_build_from_pretrained` | No recibe/usa `svmo_k`, `svmo_hidden`, `svmo_alpha`; `mlp` undefined | **Crash al instanciar layers** |
| `SVMOAdapter.__init__` | Recalcula SVD con `torch.linalg.svd` full (no randomized) cada layer | **Ignora `make svd` offline**, CPU RAM ×2 |
| `FrugalTrainer` | Carga **todo el modelo base (14GB) + SVD factors (8GB) en CPU RAM** a la vez | **OOM en 11GB libre** |
| `S3OPTOptimizer` | 13 optimizaciones instanciadas pero **sus métodos no se llaman** en fwd/bwd | **Código muerto** — 0% integración real |
| `token_by_token` + `grad_accum_steps=128` | LR scheduler usa `len(dataloader)` ≠ micro-steps reales | **Warmup/cosine decay rotos** |
| `SVMOAdapter.k` | Buffer fijo, no mutable | **DRA inoperativo** |
| `NMFFlow.N_steps` | Fijo en `__init__`, no se modifica en runtime | **TER inoperativo** |
| 70B support | No sharding, no disk offload, no quantization | **Imposible** |

---

## 2. PRINCIPIOS DE DISEÑO (No negociables)

1. **CERO HARDCODEO** — Todo parámetro viene de `FrugalConfig` / `HardwareProfile` / YAML
2. **HARDWARE-ADAPTIVE BY DEFAULT** — El mismo código corre en LOW/MEDIUM/HIGH cambiando solo `tier`
3. **S3-OPT INTEGRADO EN LOOP** — Cada optimización tiene un hook real en forward/backward
4. **MEMORY-FIRST** — Nada vive en RAM/GPU si no es estrictamente necesario *ahora*
5. **STREAMING OVER LOADING** — SVD factors, pesos base, dataset → mmap/disk streaming
6. **THEORY ↔ CODE 1:1** — Si el paper dice "TER routea N por entropía", el código hace `nmf.N_steps = ter.route(H)` en *cada forward*

---

## 3. PLAN DE ACCIÓN — FASES

### FASE 0: Fix Críticos (Bloqueadores) — **Día 1**

| Archivo | Cambio | Por qué |
|---|---|---|
| `src/adapters/hybrid.py:65-80` | `S3TransformerLayer.__init__` recibe y guarda `svmo_k, svmo_hidden, svmo_alpha, nmf_bottleneck, nmf_T, nmf_N, stb_beta, stb_head_dim`; `_build_from_pretrained` los usa | Permite instanciar layers |
| `src/adapters/svmo.py:98-104` | Si `use_randomized_svd=True` **y** existe `svd_dir` con factores precomputados → **cargar desde disco**, no recomputar | Honra `make svd` offline |
| `src/training/training_pipeline.py:141` | Pasa todos los hyperparams S³ desde `config["s3"]` a `create_s3_layer_from_hf` | Conecta YAML → código |

**Test:** `make svd && python -c "from src.training.training_pipeline import build_s3_model; ... "` → no crashea, imprime trainable params por layer.

---

### FASE 1: Memory Management Universal — **Días 2-3**

**Problema:** 14GB modelo + 8GB SVD + 1GB dataset = 23GB > 11GB RAM libre.

**Solución: Streaming por capas con `accelerate` + mmap**

```python
# En FrugalTrainer.__init__ — REEMPLAZA loading actual:
from accelerate import init_empty_weights, load_checkpoint_and_dispatch

# 1. Modelo base: sharding hook
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto",           # accelerate decide CPU/GPU
    max_memory={                 # budget por device
        0: f"{config.vram_budget_mb - 512}MB",  # GPU reservado para adapters
        "cpu": "10GB"            # LÍMITE DURO CPU RAM
    },
    low_cpu_mem_usage=True,
    offload_folder="/tmp/offload",  # spill a disco si pasa
    offload_state_dict=True,
)
```

**SVD factors:** Guardar como `safetensors` por layer (`svd_factors/layer_{i}/U_k.safetensors`) y **mmap-load solo el layer activo**:

```python
# En _move_layer_weights_to_gpu:
from safetensors.torch import load_file
svd_path = Path(self.config.svd_dir) / f"layer_{layer_idx}"
buffers = load_file(svd_path / "svd.safetensors", device="cpu")
for name, buf in buffers.items():
    setattr(layer, name, buf.to(self._gpu_device))
```

**Dataset:** `datasets.load_dataset(..., streaming=True)` + `iterable` DataLoader → **0 RAM** para datos.

**Config unificada (en `FrugalConfig`):**
```python
svd_dir: str = "./svd_factors/"
svd_storage: Literal["ram", "mmap", "disk"] = "mmap"  # auto por tier
base_model_offload: bool = True
dataset_streaming: bool = True
cpu_ram_budget_gb: float = 10.0  # usuario ajusta
```

**Test:** `make train DEVICE=cuda:0` en tu 1050 4GB + 15GB RAM → **no OOM, VRAM < 3.5GB, CPU RAM < 10GB**.

---

### FASE 2: S3-OPT Integration Real — **Días 4-6**

Cada optimización necesita **un hook concreto** en `FrugalTrainer`. No clases aisladas.

#### 2.1 SGC (Theorem 8) — **YA IMPLEMENTADO, solo hookear**
```python
# En _forward_layer, después de layer(hidden_states):
if self.config.use_sgc and hasattr(layer, "svmo_q"):
    U_k = layer.svmo_q.U_k
    hidden_states.register_hook(SGCGradientHook(U_k).backward_hook)
```

#### 2.2 GNS (Gradient-Normalized Skip) — **Hook real en backward**
```python
# En _backward_layer, ANTES de backward:
if self.config.use_gns:
    grad_norm = grad_current.norm().item()
    if self.s3_opt.gns.should_skip(grad_norm, f"layer_{layer_idx}"):
        self._move_layer_weights_to_cpu(layer_idx)
        return  # SKIP backward completo
```

#### 2.3 TER (Token Entropy Routing) — **Cambia N_steps en runtime**
```python
# En _forward_layer, ANTES de NMF:
if self.config.use_ter and self.s3_opt.ter:
    # Necesita bottleneck activation → hook intermedio en NMFFlow
    H = self.s3_opt.ter.compute_entropy(nmf_bottleneck_act)
    N = self.s3_opt.ter.route(H)
    layer.nmf_attn.N_steps = N
    layer.nmf_attn.dt = layer.nmf_attn.T / N
    layer.nmf_mlp.N_steps = N
    layer.nmf_mlp.dt = layer.nmf_mlp.T / N
```

#### 2.4 SMA (Spectral Checkpoint Compression) — **Replace LayerCheckpoint**
```python
# En _forward_layer:
if self.config.use_sma and self.s3_opt.sma:
    u_k = layer.svmo_q.U_k
    s_compressed = self.s3_opt.sma_compress(hidden_states, u_k)
    checkpoints.append(SMACompressedCheckpoint(s_compressed, layer_idx, u_k))
# En _backward_layer:
if isinstance(cp, SMACompressedCheckpoint):
    hs = self.s3_opt.sma_reconstruct(cp.compressed, cp.U_k)
```

#### 2.5 NFR (NMF Flow Recycling) — **Ya listo, solo llamar**
```python
# En train(), después de epoch:
if self.config.use_nfr and epoch > 0:
    for nfr in self.s3_opt.nfr_controllers:
        nfr.set_epoch(epoch)
```

#### 2.6 SBS (STB Bypass) — **Ya en STBBridge.forward via `enable_sbs`**
Verificar que `layer.enable_sbs = config.use_sbs` y `layer.sbs_p = config.sbs_p_stb`.

#### 2.7 DRA (Dynamic Rank Adaptation) — **Requiere k mutable**
```python
# En SVMOAdapter: k como property con setter que re-slicea U_k, S_k, Vt_k
# En FrugalTrainer: después de cada epoch, si config.use_dra:
if self.config.use_dra and self.s3_opt.dra:
    for layer_idx, layer in enumerate(self.layers):
        s = self.s3_opt.dra.compute_s(layer.svmo_q.S_k)
        if self.s3_opt.dra.should_adjust(s):
            new_k = self.s3_opt.dra.adjust_k(s)
            layer.svmo_q.set_rank(new_k)
            layer.svmo_k_proj.set_rank(new_k)
            # ... todos los SVMO del layer
            layer.stb.set_k(new_k)  # STB usa mismo k
```

#### 2.8 EMP, SMV, TOWS, MSO, FDGD, HFISC — **Evaluar ROI real**
- **EMP/SMV:** Diseñados para multi-GPU PCIe transfer → **single GPU: desactivar en LOW/MEDIUM**
- **TOWS:** Warmstart entre tokens → requiere `NMFFlow.set_warmstart(h_prev)` → implementar si TER+NFR no bastan
- **MSO:** Shortcut detection → añadir `MSOController.detect_shortcut` en `NMFFlow.forward`
- **FDGD:** Ya hookeable en backward (línea 348-349 `frugal_trainer.py`) → **activar**
- **HFISC:** Solo init → llamar en `create_s3_layer_from_hf` si `config.use_hfisc`

**Config gating por tier (ya en `hardware_profiles.py`):**
```python
# LOW:  TODAS activadas menos EMP/SMV (single GPU)
# MEDIUM: TODAS activadas
# HIGH:   Solo speedups (EMP, TER, TOWS, MSO, HFISC, SBS) — sin memory savers
```

**Test:** `make train DEVICE=cuda:0` → logs muestran `[GNS] skipped layer_5`, `[TER] N=2 for token`, `[SMA] compressed 4096→128`, VRAM < budget.

---

### FASE 3: Token-by-Token LR Scheduler Fix — **Día 7**

**Problema:** `gradient_accumulation_steps=128` hardcodeado pero en `token_by_token=True` acumulas `seq_len-1` micro-steps por secuencia.

**Fix en `FrugalConfig`:**
```python
# Eliminar gradient_accumulation_steps hardcodeado
# Calcular steps reales:
effective_accum_steps = (seq_len - 1) * micro_batch_size  # token-by-token
# O para sequence mode:
effective_accum_steps = gradient_accumulation_steps * micro_batch_size
```

**En `_warmup_scheduler_step`:**
```python
tokens_per_seq = seq_len - 1 if self.config.token_by_token else seq_len
total_micro_steps = len(dataloader) * tokens_per_seq * max_epochs
warmup_steps = int(total_micro_steps * warmup_ratio)
```

---

### FASE 4: 70B Readiness (Arquitectura, no implementación full) — **Día 8**

**Cambios estructurales para escalar:**
1. **SVD sharding:** `svd_factors/layer_{i}/rank_{k}/U_k.safetensors` → cargar solo rank activo (DRA)
2. **Base model quantization:** `load_in_4bit=True` + `bnb_4bit_compute_dtype=fp16` para 70B en 24GB VRAM
3. **FSDP/DeepSpeed integration:** `device_map="auto"` ya usa `accelerate` → compatible
4. **Config extensible:** `model_size: Literal["7b", "13b", "70b"]` → auto-setea `d_model`, `n_layers`, `svmo_k_default`

---

## 4. INNOVACIONES "FUERA DE LA CAJA" (Para Poor Compute Real)

Estas NO están en el paper/código actual. Son ideas para **explotar la estructura S³**:

### 4.1 **SVMO Rank-1 Updates en vez de MLP**
> En vez de MLP `g_θ: 1→H→H→1` por cada σ, **aprender vector `δ ∈ R^k` directamente**:
> ```python
> # params: k (vs H²≈1024). Update: σ'_i = σ_i * (1 + α * tanh(δ_i))
> # Gradiente: ∂L/∂δ_i = ∂L/∂σ'_i * α * σ_i * sech²(δ_i)
> ```
> **Ahorro:** 32x menos params SVMO, forward 10x más rápido, **igual expressividad** (Teorema 1: error O(1/√H) → aquí H=k).

### 4.2 **NMF como Low-Rank LoRA Dinámico**
> `f_θ(h,t) = W_out tanh(W_in [h;t])` con `W_in: (d+1)×r, W_out: r×d` → **equivale a LoRA rank-r en el residual stream** pero **continuo en tiempo**.
> **Idea:** Compartir `W_in, W_out` entre NMF_attn y NMF_mlp → **half params**. O hacer `r` adaptativo por token (TER ya da N, extiéndelo a `r`).

### 4.3 **STB como Attention-Free Spectral Mixing**
> `STB(h) = U_k Attention(s, σ_log) ≈ U_k (s ⊙ σ_mod)` si `Q=K=I` (no learnable).
> **Test ablation:** `stb_beta=0.5` vs `stb_beta=0` vs **fixed mixing `s * σ_mod`** → si no hay degradación, **elimina 12K params/layer**.

### 4.4 **Fused Kernel: SVMO + Attention + NMF en 1 launch**
> En lugar de:
> ```
> q = svmo_q(norm) → attn → svmo_o → nmf_attn → norm → svmo_up/gate/down → nmf_mlp
> ```
> **Un kernel Triton** que haga todo el layer S³ sin materializar intermedios en VRAM.
> **Impacto:** 2-3x speedup en 1050, **elimina activation checkpoints**.

### 4.5 **Progressive SVD Rank durante Training (DRA++ )**
> Empezar `k=256` (exploración) → reducir a `k=64` (explotación) basado en `s(t) = σ_{k+1}/Σσ_i`.
> **Teoría:** DRA ya lo hace, pero **recompute SVD offline cada N epochs** para actualizar `U_k, V_k` → captura direcciones nuevas.

### 4.6 **CPU-GPU Pipeline Asíncrono (Double Buffering)**
> Mientras GPU computa `layer_i`, CPU **prefetch + SVD load + data load** `layer_{i+1}` en stream separado.
> ```python
> # En FrugalTrainer: dos CUDA streams
> self._compute_stream = torch.cuda.Stream()
> self._load_stream = torch.cuda.Stream()
> # _forward_layer lanza en compute_stream, _move_layer_weights_to_gpu en load_stream
> ```
> **Elimina bubbles de layer swap** → 30% más throughput.

---

## 5. MÉTRICAS DE ÉXITO (Definition of Done)

| Métrica | Target LOW (1050 4GB, 10GB RAM) | Target MEDIUM (3060 12GB) | Target HIGH (4090 24GB) |
|---|---|---|---|
| **VRAM peak** | < 3.5 GB | < 10 GB | < 20 GB |
| **CPU RAM peak** | < 9 GB | < 16 GB | < 32 GB |
| **Time/epoch (Alpaca 52k)** | < 4 h | < 1 h | < 20 min |
| **S3-OPT active** | 10/13 | 13/13 | 6/13 (speedups only) |
| **Trainable params** | 4.83M (0.06%) | 4.83M | 4.83M |
| **Perplexity (eval)** | ≤ baseline LoRA r=8 | ≤ baseline LoRA r=8 | ≤ baseline LoRA r=8 |
| **OOM crashes** | 0 | 0 | 0 |

---

## 6. ORDEN DE EJECUCIÓN PRÁCTICO

```bash
# 1. Fix blockers
#    Editar hybrid.py, svmo.py, training_pipeline.py

# 2. Test instanciación
python -c "
from src.training.training_pipeline import build_s3_model
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-7B-Instruct', device_map='cpu')
trainer = build_s3_model(model, './svd_factors/qwen2.5-7b/', {'s3': {...}}, 'cuda:0')
print('OK:', sum(p.numel() for l in trainer.layers for p in l.parameters() if p.requires_grad))
"

# 3. Memory profile (dry run, no train)
python -c "
from src.training.frugal_trainer import FrugalTrainer, FrugalConfig
# ... setup minimal ...
trainer = FrugalTrainer(..., config=FrugalConfig(layer_swap=True, token_by_token=True, vram_budget_mb=3500), device='cuda:0')
print('VRAM init:', torch.cuda.memory_allocated()/1e9)
"

# 4. Integrar S3-OPT hooks (FASE 2)

# 5. Train real en tu 1050
make train DEVICE=cuda:0 CONFIG=experiments/configs/s3_standard.yaml

# 6. Validar métricas → iterar
```

---

## 7. RIESGOS Y MITIGACIONES

| Riesgo | Probabilidad | Mitigación |
|---|---|---|
| `accelerate` device_map rompe layer swap manual | Media | Usar `accelerate` SOLO para offload base model; layer swap de adapters sigue manual |
| `safetensors` mmap lento en HDD | Baja | `offload_folder` en `/tmp` (tmpfs RAM) o NVMe; pre-load next layer async |
| DRA cambia k → STB/Attention shape mismatch | Alta | `set_rank()` re-slicea todos los buffers SVD + STB.U_k + attention heads en **un solo paso atómico** |
| TER N_steps dinámico rompe `NFRController` | Media | `NFRController` lee `nmf.N_steps` en runtime, no en init |
| 70B no cabe ni con 4-bit + offload | Media | Target real: 13B en 1050, 70B en 24GB+ (MEDIUM/HIGH) |

---

## 8. CONCLUSIÓN: ¿ES POSIBLE LO QUE BUSCAS?

**SÍ, la teoría S³ es correcta y suficiente para 7B en 2GB VRAM teórico.**
**PERO el código actual tiene 3 gaps fatales:**
1. **Memory loading strategy** ignora límites RAM reales → fix con streaming + accelerate
2. **S3-OPT no integrado** → 13 optimizaciones = 0% efecto real → fix con hooks en loop
3. **Hardcodeos y bugs** en hybrid.py → fix mecánicos

**Con la FASE 0-3 resueltas:** Tu 1050 4GB + 10GB RAM **entrena 7B S³ en ~3-4h/epoch** con todas las optimizaciones activas.

**Para 70B:** Requiere FASE 4 (quantization + sharding) → **target MEDIUM/HIGH tier**, no LOW. El framework ya está preparado (3 tiers), solo falta implementar el path.

---

## 9. PRÓXIMO PASO INMEDIATO

¿Empezamos por **FASE 0** (fix hybrid.py + svmo.py loading)? Dame el OK y edito los archivos en orden.

O si prefieres, **FASE 1** (memory streaming) primero — pero sin FASE 0 el código no instancia.

Tu llamada. El plan está listo para ejecutar.