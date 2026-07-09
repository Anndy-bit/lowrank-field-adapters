# S3-OPT: Mejoras, Sinergias e Innovacion

> Documento de laboratorio. Ideas locas, implementaciones inexistentes, innovacion real.
> "Quien dice que esto tiene que hacerse asi? Y si de esta otra forma que nadie lo ha inventado?"

## 1. Resumen de Estado Actual

### Flags de S3-OPT por Tier (Configuracion Automatica)

| Optimizacion | LOW (1050 2GB) | MEDIUM (3060-5090Ti 8-16GB) | HIGH (4090 32GB+) |
|---|---|---|---|
| SMV (vector transfer) | ON | ON | OFF |
| EMP (MLP skip) | ON | ON | ON |
| TER (ODE routing) | ON | ON | ON |
| FDGD (gradient filter) | ON | ON | OFF |
| GNS (backward skip) | ON | ON | OFF |
| SMA (spectral checkpoints) | ON | ON | OFF |
| TOWS (token warmstart) | ON | ON | ON |
| DRA (dynamic rank) | ON | ON | OFF |
| MSQ (shortcut detection) | ON | ON | ON |
| HFISC (init) | ON | ON | ON |
| SGC (gradient compr) | ON | ON | OFF |
| NFR (flow recycling) | ON | ON | OFF |
| SBS (STB bypass) | ON (p=0.3) | ON (p=0.3) | ON (p=0.3) |

---

## 2. Sinergias Entre Optimizaciones

### 2.1 GNS + DRA: Reduccion Dinamica de Compute

**Combinacion:** GNS detecta capas de bajo gradiente, DRA detecta capas de bajo rank.

```
Capas con gradiente BAJO + rank BAJO → skip completo (backward + forward reducido)
Capas con gradiente ALTO + rank ALTO → full compute
```

**Sinergia real:**
- GNS: skip ~20-30% de backward passes (capas con ||g|| < ε·max)
- DRA: reduce k de 128→64 en capas donde s(t) < 0.1
- COMBINADO: capas que cumplen ambos criterios se reducen 4x en compute

**Codigo de integracion propuesto:**
```python
# En _training_step backward loop
for layer_idx in reversed(range(n_layers)):
    grad_norm = grad_current.norm().item()
    sigma = svd_sigmas[layer_idx]
    s_t = dra.compute_s(sigma)
    
    # Skip si AMBOS condiciones se cumplen
    if gns.should_skip(grad_norm, layer_idx) and s_t < 0.1:
        # Skip completo: no backward, no forward
        continue
    elif gns.should_skip(grad_norm, layer_idx) or s_t < 0.05:
        # Skip parcial: backward pero con rank reducido
        grad_current = fdgd.filter(grad_current)
        backward_with_reduced_rank(layer_idx, hs, grad_current, k_reduced=64)
    else:
        # Full compute
        grad_current = fdgd.filter(grad_current)
        _backward_layer(layer_idx, hs.requires_grad_(True), grad_current)
```

### 2.2 TER + TOWS: Routing Inteligente por Token

**Combinacion:** TER decide N (pasos ODE), TOWS decide si hacer warmstart.

```
Token de alta entropia (H > 1.0) → N=4 + warmstart (mas compute, mas precision)
Token de baja entropia (H < 0.5) → N=1 + no warmstart (menos compute)
Token de entropia media → N=2
```

**Sinergia:** TOWS y TER comparten la senal de entropia. TOWS usa coherencia de representacion para decidir warmstart. TER usa entropia de la bottleneck NMF.

**Potencial speedup:** 40-60% reduccion en ODE compute en sequences con tokens de baja entropia (los mas comunes en instruccion-following).

### 2.3 SMA + GNS + DRA: VRAM Manager Triple

**Combinacion:** Los tres manejan el tamano de lo que se guarda en VRAM.

```
SMA: comprime checkpoints (4096→128 por layer) — factor 16x
GNS: decide cuales layers necesitan checkpoint (skip si no hay gradiente)
DRA: adapta el rank de los factores U_k/V_k por layer
```

**Sinergia en VRAM:**
- Sin optimizacion: cada layer checkpoint = 4096 floats = 16KB
- Con SMA: 128 floats = 0.5KB (32x reduccion)
- Con GNS: solo se guarda checkpoint si el gradiente es significativo
- Con DRA: U_k/V_k mas pequenos cuando k se reduce

**Total VRAM reduction potencial:** 32x * cobertura * factor de rank

### 2.4 EMP + TER: Prediccion de Microcache

**Combinacion:** EMP predice μ desde Adam moments, TER decide complejidad del token.

```
Si EMP predice "skip" (H bajo, estable) + TER dice N=1:
  → Skip completo del MLP para ese token
Si EMP predice "no-skip" + TER dice N=4:
  → Full MLP execution
```

**Innovacion:** EMP y TER usan señales complementarias:
- EMP: H del token actual (basado en Adam moments)
- TER: H de la bottleneck NMF (basado en activación)

Si ambos coinciden en "baja complejidad" → skip agresivo.

---

## 3. Ideas Locas (Out of the Box)

### 3.1 "SVD como hash" — Compresion Radical de Embeddings

**Idea:** Los embeddings de tokens pueden comprimirse usando la base espectral U_k del primer layer como "hash codes".

**Teoria:**
- h_emb ∈ R^d (e.g. 4096)
- s_emb = h_emb @ U_k ∈ R^k (e.g. 128) — esto ES el checkpoint
- Si dos tokens tienen s_emb muy cercano, tienen similar "representacion spectral"

**Innovacion:**
```python
def spectral_nearest_token(h_emb, U_k, embedding_table):
    """Encuentra el token mas cercano en base espectral."""
    s = h_emb @ U_k  # Proyeccion espectral
    s_table = embedding_table @ U_k  # Todas las proyecciones
    distances = (s_table - s).norm(dim=-1)
    return distances.argmin()

# Usa spectral nearest para:
# 1. Inicializacion de hidden states (mejor que embedding directo)
# 2. Fallback cuando embedding no esta en cache
# 3. Deteccion de tokens semanticamente similares
```

**Potencial:** 32x compresion de embedding cache con perdida minima de precision.

### 3.2 "Gradiente como senal de aprendizaje" — GNS Invertido

**Idea convencional:** Skip backward cuando el gradiente es pequeno (el layer ya aprendio lo suficiente).

**Idea loca:** AMPLIAR el gradiente cuando es pequeno, para forzar aprendizaje acelerado en capas "perezosas".

```python
class GNSInverted(GNSController):
    """GNS invertido: cuando gradiente es pequeno, AMPLIARlo en vez de skippear.
    
    Esto fuerza al layer perezoso a aprender mas aggressively.
    """
    def should_amplify(self, grad_norm, threshold=0.001):
        return grad_norm < threshold
    
    def amplify_grad(self, grad, amplification=5.0):
        # Redimensiona el gradiente pequeno para forzar aprendizaje
        return grad * amplification

# Uso:
if gns.should_amplify(grad_norm):
    grad = gns.amplify_grad(grad)
    _backward_layer(layer_idx, hs, grad)
else:
    _backward_layer(layer_idx, hs, grad)
```

**Teoria:** Las capas con gradientes pequenos podrian beneficiarse de "overlearning" momentaneo para escapar de minimos locales.

### 3.3 "ODE como VAE" — Variational ODE

**Idea:** Tratar la ODE del NMF como un VAE, con z como latent space.

```
q(z | h) = N(f_θ(h), σ²I)  # Encoder (el NMF mismo)
p(h | z) = N(g_φ(z), σ²I)   # Decoder (reconstruccion)
```

**Innovacion:** La entropia del latent z (calculada por TER) mide incertidumbre del modelo. Si H es alta, el modelo esta incierto → N=4 pasos (exploracion). Si H es baja, el modelo esta seguro → N=1 paso (explotacion).

Esto es un "Variational Inference for Neural ODEs" — nadie lo ha publicado para NMF.

### 3.4 "Tokenizer Espectral" — Tokenizacion en Espacio Latente

**Idea:** Usar U_k para crear un "tokenizador spectral" que discretiza el espacio latente en celdas.

```python
class SpectralTokenizer:
    """Tokenizador spectral: discretiza el espacio latente en celdas.
    
    En vez de tokens de NLP, usa "tokens espectrales" que codifican
    la posicion en el espacio de los primeros k componentes singulares.
    """
    def __init__(self, U_k, n_bins=256, k=128):
        self.U_k = U_k
        self.k = k
        # Cuantizacion de cada dimension en n_bins niveles
        self.bin_edges = [torch.linspace(-3, 3, n_bins+1) for _ in range(k)]
    
    def encode(self, h):
        """Codifica h en token espectral (k-dimensional integer)."""
        s = h @ self.U_k  # Proyeccion
        # Cuantiza cada dimension
        tokens = torch.zeros(self.k, dtype=torch.long)
        for i in range(self.k):
            tokens[i] = torch.bucketize(s[i], self.bin_edges[i])
        return tokens
    
    def decode(self, tokens):
        """Decodifica token espectral a h aproximado."""
        # Centros de bin
        s_approx = torch.zeros(self.k)
        for i in range(self.k):
            s_approx[i] = (self.bin_edges[i][tokens[i]] + self.bin_edges[i][tokens[i]+1]) / 2
        return s_approx @ self.U_k.t()
```

**Potencial:** Compresion 32x para storage de hidden states, reconstruccion con perdida controlada.

### 3.5 "Transfer Learning Inverso" — De 7B a 1050

**Idea loca:** Si el 7B tiene conocimiento y la 1050 tiene eficiencia, por que no "transferir" el conocimiento del 7B frozen a los adapters de la 1050 usando solo estadisticas spectrales?

```
U_k_7B (frozen) contiene la estructura de singular vectors del modelo.
U_k_7B_escalado = α * U_k_7B + β (learnable)
→ Los adapters en la 1050 aprenden a proyectar en la base spectral del 7B
→ El 7B "enseña" la estructura de singular vectors
```

**Teoria:** El 7B frozen sabe cuales direcciones del espacio latente son importantes (alta varianza en Σ_k). La 1050 puede beneficiarse de esta informacion sin transferir el modelo completo.

### 3.6 "Micro-batching Adaptativo" — MSO Dinamico

**Idea:** En vez de micro_batch_size fijo, adaptar dinámicamente segun complejidad de la secuencia.

```python
class AdaptiveMicroBatch:
    """Ajusta micro_batch_size segun complejidad promedio de la secuencia."""
    def compute_sequence_complexity(self, input_ids):
        # Complejidad = varianza de embeddings normalizados
        emb = self.embedding(input_ids)
        complexity = emb.std(dim=-1).mean().item()
        return complexity
    
    def recommend_batch_size(self, sequence_complexity, max_bs=32):
        # Complejidad alta → batch pequeno (mas granular)
        # Complejidad baja → batch grande (mas throughput)
        base = max_bs
        scale = max(1, int(max_bs * (1 - sequence_complexity)))
        return min(scale, max_bs)
```

**Sinergia:** Con TOWS y TER, las secuencias de baja complejidad pueden processarse con batch_size grande + N=1. Las de alta complejidad con batch_size pequeno + N=4.

---

## 4. Mejoras Concretas para GTX 1050 (2GB)

### 4.1 Prioridad Alta (Implementar Ahora)

**1. SMA + GNS Integration:**
```python
# En _training_step: solo guardar SMA checkpoints para layers no-skipped
for layer_idx in range(n_layers):
    if gns.should_skip(grad_norm, f"layer_{layer_idx}"):
        checkpoints[layer_idx] = None  # No guardar nada
    else:
        h = hidden_states.clone()
        checkpoints[layer_idx] = sma_compress(h, layer.U_k)
```

**2. DRA + SMV Integration:**
```python
# Cuando DRA reduce k, actualizar los buffers de SMV
# U_k y V_k se hacen mas pequenos → delta-modulation mas pequena
if dra.should_adjust(s_t):
    new_k = dra.adjust_k(s_t)
    smv.update_k(new_k)  # Reduce tamano de delta transfer
    svmo.set_rank(new_k)  # Actualiza rank del adapter
```

**3. EMP + TER Coherence:**
```python
# Emp predice H, Ter usa H. Si EMP y TER coinciden, skip agresivo.
emp_H = emp.predict_H(adam_moments)
ter_H = ter.compute_entropy(nmf_activation)

if emp_H < 0.5 and ter_H < 0.5:
    # Skip agresivo: N=1 + no MLP + no STB
    nmf.set_N_steps(1)
    svmo.set_mode("skip")
elif emp_H > 1.5 or ter_H > 1.5:
    # Full compute
    nmf.set_N_steps(4)
    svmo.set_mode("full")
```

### 4.2 Prioridad Media (Implementar Despues)

**4. TOWS + Embedding Cache:**
```python
# Cache de embeddings proyectados (SMA-style)
class EmbeddingCache:
    """Cache de embeddings usando projeccion espectral como key."""
    def __init__(self, U_k, max_size=1000):
        self.cache = {}  # s_emb → (token_id, h_original)
        self.max_size = max_size
        self.U_k = U_k
    
    def lookup(self, h):
        s = h @ self.U_k
        key = tuple(s.round().tolist())  # Discretizado
        if key in self.cache:
            return self.cache[key]
        return None
    
    def store(self, h, token_id):
        s = h @ self.U_k
        key = tuple(s.round().tolist())
        if len(self.cache) > self.max_size:
            self.cache.popitem()
        self.cache[key] = (token_id, h)
```

**5. NFR + DRA Synergy:**
```python
# NFR guarda estados intermedios. DRA decide cual rank usar para guardar.
# Estados intermedios se guardan con rank k_actual (no full k)
if nfr.should_store(intermediate_state):
    k_eff = dra.k_current  # Puede ser menor que k=128
    s_inter = intermediate_state @ U_k[:, :k_eff]
    nfr.store(s_inter, k_eff)
```

### 4.3 Prioridad Baja (Ideas para Investigar)

**6. "Sparse Forward" — Solo layers importantes:**
```python
# Forward pass solo en layers con alta "importancia"
# Importancia medida por GNS skip count historico
importance_scores = gns.get_layer_importance()  # [n_layers]
threshold = importance_scores.median()
active_layers = (importance_scores > threshold).nonzero()

# Skip de capas de baja importancia: usar identidad
for layer_idx in range(n_layers):
    if importance_scores[layer_idx] < threshold:
        h = h  # Skip layer, identidad implicita
    else:
        h = forward_layer(layer_idx, h)
```

**7. "Mixture of Experts Spectral":**
```python
# No MLP completo. MLP solo en componentes singulares importantes.
# Componentes de bajo singular value → skip
sigma_normalized = S_k / S_k.sum()
important_mask = sigma_normalized > 0.01  # Solo top 90% de singular values
z = h @ V_k[important_mask]  # Proyectar a subespacio importante
z_mod = z * (1 + mlp_small(z))  # MLP solo en subespacio reducido
h_out = z_mod @ V_k[important_mask].t()  # Reconstruir
```

---

## 5. Metricas de Success

Para medir si las optimizaciones estan funcionando:

```
VRAM Peak (MB):     torch.cuda.max_memory_allocated() / 1024**2
Tokens/second:      steps * batch_size * seq_len / wall_time
S3-OPT coverage:    skip_count / total_backward_passes
DRA effective k:    dra.k_current (promedio por layer)
TOWS hit rate:      warmstart_count / total_tokens
SMA compression:    uncompressed_bytes / compressed_bytes
```

---

## 6. Notas de Laboratorio

### Observacion 1: TER sesga hacia N=4
TER-4 sesga sistematicamente hacia N=4 por O(1/N). Esto es MEDIBLE.
- Mide H_NMF para tokens de alta confianza (logits de alto pico)
- Compara H_NMF con la entropia real del tokenizer
- Si H_NMF > H_real por margen constante → el sesgo es sistemático

### Observacion 2: EMP Fisher != Hessian (fase temprana)
En fase temprana, los Adam moments (m_t, v_t) no son Fisher information.
- m_t ≈ gradiente promedio (sesgado)
- v_t ≈ gradiente^2 promedio (aproximacion de Fisher)
- Esto hace que EMP sea impreciso en early training
- Fix: usar EMP solo despues de epoch 1

### Observacion 3: DRA Q debe ser adaptativa
La Q en FDGD no puede ser fija.
- Q fija = sobre-suavizado o infra-suavizado
- Q adaptativa = medida real de energia en frecuencias altas
- Implementar: Q = f(energy_high_freq / energy_total)

### Observacion 4: SMA compression ratio depende de k/d
- k=128, d=4096 → 3.1% del espacio original
- k=64, d=4096 → 1.6% del espacio original
- k=32, d=4096 → 0.8% del espacio original
- Esto es la deduccion de singular values, NO perdida de informacion

### Observacion 5: TOWS necesita "coherence threshold" adaptive
El threshold de coherencia no puede ser fijo.
- secuencias de alta entropia → threshold bajo (facil encontrar coherencia)
- secuencias de baja entropia → threshold alto (coherencia menos util)
- Fix: threshold = f(entropia_promedio_de_la_sequence)

---

## 7. Roadmap de Implementacion

### Fase 1: Core (YA IMPLEMENTADO)
- [x] 10 optimizaciones en s3_opt.py
- [x] Integracion en frugal_trainer.py
- [x] Flags en FrugalConfig
- [x] Hardware profiles con S3-OPT auto-configurados
- [x] train_s3.py con YAML config
- [x] experiments/run_training.py con profile
- [x] tests pasando

### Fase 2: Sinergias (PROXIMA)
- [ ] GNS + DRA combined skip logic
- [ ] TER + TOWS coherence sharing
- [ ] EMP + TER "skip agresivo" mode
- [ ] SMA + GNS checkpoint skipping

### Fase 3: Innovacion
- [ ] Spectral tokenizer (3.4)
- [ ] Variational ODE (3.3)
- [ ] GNS invertido (3.2)
- [ ] Adaptive micro-batching (3.6)

### Fase 4: Tuning Fino
- [ ] TER bias measurement (nota 1)
- [ ] EMP phase-aware enable (nota 2)
- [ ] FDGD adaptive Q (nota 3)
- [ ] TOWS adaptive threshold (nota 5)

---

*Ultima actualizacion: Julio 2026*
*Proyecto: lowrank-field-adapters*
*Objetivo: Fine-tuning de 7B en GTX 1050 2GB en tiempo razonable*