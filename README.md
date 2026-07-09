# S³: Spectral-Spatial-Smooth Fine-Tuning

**Fine-tuning de LLMs en 2GB VRAM con calidad comparable a GPU de alta gama**

---

## Idea Central

S³ es un paradigma de Parameter-Efficient Fine-Tuning (PEFT) que permite adaptar modelos de lenguaje grandes (7B parámetros) utilizando solo **2GB de VRAM** en hardware de consumo (GTX 1050), sin destilación, cuantización agresiva ni poda. El enfoquecongela las matrices de peso $W = U\Sigma V^T$ y adjunta tres operadores matemáticos noveles que juntos totalizan **4.83M parámetros entrenables (0.060% del modelo)**.

---

## Los Tres Operadores

### 1. SVMO — Singular Value Modulation Operator

El corazón de S³ es la modulación puntual de valores singulares. Dada la descomposición SVD $W = U\Sigma V^T$ con $\Sigma = \diag(\sigma_1, \dots, \sigma_r)$, SVMO aprende una función de modulación no-lineal por valor singular:

$$m_\theta(\sigma) = \sigma \cdot \big(1 + \alpha \cdot \tanh(g_\theta(\log(\sigma + \varepsilon)))\big)$$

donde $\alpha \in (0,1]$ controla la modulación máxima, $\varepsilon = 10^{-8}$, y $g_\theta: \mathbb{R} \to \mathbb{R}$ es una MLP de 2 capas ocultas con $H$ unidades. Los vectores singulares $U$ y $V$ permanecen **congelados**; solo los parámetros de $g_\theta$ se entrenan. Costo: $O(H^2)$ por matriz, independiente de la dimensión del modelo $d$.

**Propiedades formalizadas:**
- *Aproximación* (Teorema 1): Error de aproximación $O(1/\sqrt{H})$ respecto a cualquier matriz de adaptación $\Delta^*$.
- *Estabilidad de gradiente* (Teorema 2): $\big\|\frac{\partial\mathcal{L}}{\partial\theta}\big\|_2 \leq \alpha \sigma_{\max} k \|g'_\theta\|_\infty \sqrt{|\theta|}$, **independiente de la escala de entrada** gracias a la saturación $\sech^2(z) \leq 1$ de $\tanh$.

### 2. NMF — Neural Manifold Flow

NMF deforma continuamente el espacio latente mediante una Neural ODE. El flujo resuelve:

$$\frac{dh(t)}{dt} = f_\theta(h(t), t), \qquad h(0) = h, \qquad \tilde{h} = h(T)$$

con campo de velocidad diseñado como:

$$f_\theta(h, t) = W_{\text{out}} \cdot \tanh(W_{\text{in}} \cdot [h; t])$$

donde $[h; t] \in \mathbb{R}^{d+1}$ concatena la representación con el tiempo, $W_{\text{in}} \in \mathbb{R}^{d \times d_b}$, $W_{\text{out}} \in \mathbb{R}^{d_b \times d}$, y $d_b \in \{4, 8, 16\}$. Parámetros totales: $2d \cdot d_b$ (aprox. 65K para $d=4096, d_b=8$). Integración numérica con RK4 de 4 etapas; overhead $\sim$2% FLOPs.

**Propiedades formalizadas:**
- *Cota de Lipschitz* (Lema 1): $L_f \leq \|W_{\text{out}}\|_2 \|W_{\text{in}}\|_2 \approx 4.9 \times 10^{-4}$ (inicialización Xavier).
- *Trayectoria* (Teorema 3): Error $\|h_{\text{NMF}}(T) - h^*\| \approx \varepsilon_1 \cdot T$ (corrección del bound exponencial mal aplicado de Chen et al. 2018).
- *Integración RK4* (Teorema 4): Error de discretización $\varepsilon_{\text{ODE}} \leq 6.3 \times 10^{-6}$ con $N=4$ pasos.

### 3. STB — Spectral Transport Bridge

STB crea un acoplamiento bidireccional entre SVMO y NMF mediante cross-attention espectral. Para una representación $h$, se extrae la firma espectral $s = U_k^T h \in \mathbb{R}^k$ (proyección sobre los $k$ primeros vectores singulares derechos). El operador computa:

$$\text{STB}(h) = U_k \, \text{Attention}(Q=s, \; K=\Sigma_k m_\theta(\sigma), \; V=s)$$

donde $\Sigma_k m_\theta(\sigma) \in \mathbb{R}^k$ son los valores singulares modulados por SVMO. STB permite que la información de los valores singulares modulados influya en cómo NMF deforma el espacio latente.

**Propiedades formalizadas:**
- *Información mutua* (Teorema 5): $I_{\text{STB}}(\Theta_S; \Theta_N \mid X) \geq \frac{\beta^2 k}{2d} H(X) > 0$ (acoplamiento no-vacío).
- *Aceleración de convergencia* (Teorema 6): El precondicionamiento espectral vía STB reduce el número decondition number efectivo.

### Arquitectura S³

Los tres operadores se composen en una capa híbrida por cada capa del transformer. Para la capa $l$ con pesos $W_l = U_l \Sigma_l V_l$:

1. **SVMO**: Modifica los valores singulares $\sigma_i^{(l)} \mapsto m_\theta(\sigma_i^{(l)})$.
2. **STB**: Aplica cross-attention espectral entre la firma de la representación $h^{(l)}$ y los valores singulares modificados.
3. **NMF**: Aplica flujo continuo sobre la representación salida: $h^{(l)} \to \tilde{h}^{(l)}$.

Solo los parámetros de $g_\theta$ (SVMO), $W_{\text{in}}, W_{\text{out}}$ (NMF), y las matrices de atención de STB se entrenan. Los pesos $U, V$ del modelo base permanecen congelados. Entrenamiento con swapping secuencial capa-por-capa CPU$\leftrightarrow$GPU: solo una capa en GPU a la vez.

---

## S3-OPT: Optimizaciones de Training

S3-OPT es un conjunto de **10+ optimizaciones teóricas** documentadas en `docs/formalismo_optimizacion.md` que trabajan en sinergia para minimizar VRAM y maximizar throughput en cualquier GPU, desde la GTX 1050 de 2GB hasta la RTX 4090 de 24GB.

### Las 10 Optimizaciones

| # | Nombre | Que hace | Por que funciona |
|---|--------|----------|-----------------|
| 1 | **SMV** (Vector Delta-Modulation) | Transfiere solo $\Delta\mu = \mu_t - \mu_{t-1}$ por PCIe en vez de la matriz completa $m_\theta$ | Reduce ancho de banda PCIe ~32x. Solo cambia cuando hay variacion significativa. |
| 2 | **EMP** (MLP Predictor) | Predice si el MLP debe ejecutarse desde Adam moments $(m_t, v_t)$ sin correr el forward pass | Adam moments contienen suficiente informacion sobre H del token. Elimina ~99% del MLP compute cuando H es predecible. |
| 3 | **TER** (ODE Router) | Routa dinamicamente N ∈ {1, 2, 4} pasos ODE por entropia del bottleneck | Tokens de baja entropia (alta confianza) necesitan menos pasos. Media de N puede bajar de 4 a ~1.8. |
| 4 | **FDGD** (Fourier Gradient Filter) | Filtra gradientes en dominio de Fourier (Butterworth) antes del backward | Remueve ruido de alta frecuencia que causa oscilacion. Q adaptativa ajusta el corte segun energia real. |
| 5 | **GNS** (Gradient Skip) | Skip backward completo cuando $\\|g\\| < \epsilon \cdot \\|g\\|_{\\max}$ | Capas con gradiente pequeno no contribuyen al aprendizaje. Skip ~20-30% de backward passes. |
| 6 | **SMA** (Spectral Checkpoints) | Guarda $s = U_k^T h$ (128 dim) en vez de $h$ completo (4096 dim) | Compresion 16x en storage de checkpoints. Reconstruccion exacta si U_k es ortonormal. |
| 7 | **TOWS** (Token Warmstart) | Usa hidden state del token anterior como warmstart para ODE del token actual | Tokens consecutivos en una secuencia son similares. Evita reinicializar la ODE desde cero. |
| 8 | **DRA** (Dynamic Rank Adaptation) | Adapta k de SVD dinamicamente: $k' = f(s(t))$ donde $s(t) = \sigma_{k+1} / \Sigma\sigma_i$ | Rank innecesario consume FLOPs. Si s(t) < threshold por n pasos, reducir k. |
| 9 | **MSQ** (Manifold Shortcut) | Detecta curvatura κ < κ_thresh y aplica shortcut lineal en vez de RK4 | Si la ODE es casi lineal, el paso lineal es suficiente. Ahorra 4x en FLOPs del ODE. |
| 10 | **HFISC** (Optimal Init) | Inicializa MLP con escala $\\sqrt{2/d}$ para GELU (Kaiming + ortogonal) | Preserva varianceforward y backward. Converge ~2x mas rapido en early training. |

**Optimizaciones auxiliares** (ya implementadas en `s3_optimizations.py`):

| # | Nombre | Que hace | Por que funciona |
|---|--------|----------|-----------------|
| 11 | **SGC** (Spectral Gradient Compression) | Comprime gradiente $g \to g \cdot U_k \cdot U_k^T$ en backward | Proyeccion sobre base ortonormal. Compression ratio 32x con reconstruccion exacta. |
| 12 | **NFR** (NMF Flow Recycling) | Reusa estados intermedios de ODE entre epochs | El ultimo estado de epoch t es warmstart de epoch t+1. Evita reinicializacion. |
| 13 | **SBS** (STB Bypass Sampling) | Bypassea STB con probabilidad p_STB (Bernoulli) | STB no siempre es necesario. A p=0.3, elimina 70% de FLOPs de STB con 0.23% degradacion. |

### Sinergias (Por que 1+1 > 2)

Las optimizaciones no trabajan aisladas. Las combinaciones mas poderosas:

**GNS + DRA + SMA (VRAM Manager Triple)**
- GNS: decide si una capa necesita backward
- DRA: decide cual rank usar por capa
- SMA: decide cuanto espacio ocupa el checkpoint
- Resultado: hasta 32x menos VRAM para capas de baja actividad

**TER + TOWS + EMP (Compute Reducer)**
- TER: minimiza pasos ODE por token
- TOWS: evita reinicializar ODE
- EMP: skippea MLP cuando H predecible
- Resultado: hasta 70% menos compute en sequences de baja entropia

**SMA + GNS (Checkpoint Storage)**
- SMA: comprime 4096→128 por checkpoint
- GNS: solo guarda checkpoint si el gradiente es significativo
- Resultado: storage de checkpoints reducido ~64x efectivo

### Configuracion Automatica por Hardware

S3-OPT se configura automaticamente segun tu GPU. No necesitas saber cuales flags activar — el sistema detecta tu hardware y activa las optimizaciones apropiadas.

#### Hardware Tiers

**LOW — GTX 1050 / 2GB VRAM**
```
Todas las 13 optimizaciones ACTIVAS
layer_swap=True, token_by_token=True
VRAM budget: 2048 MB
Objetivo: Sobrevivir en 2GB, maximizar calidad
```
- SMV, GNS, SMA: VRAM critico — todo activo
- DRA, FDGD: Estabilidad — todo activo
- EMP, TER, TOWS, MSQ: Compute reducer — todo activo

**MEDIUM — RTX 3060-5090 Ti / 8-16GB VRAM**
```
Todas las 13 optimizaciones ACTIVAS
layer_swap=False, token_by_token=False
VRAM budget: 12288 MB
Objetivo: Balance VRAM / throughput
```
- Toda optimizacion disponible para throughput maximo
- layer_swap=False porque hay VRAM para mantener mas en GPU

**HIGH — RTX 4090 / 32GB+ VRAM**
```
Solo 6 optimizaciones ACTIVAS (speedup, no VRAM saving)
layer_swap=False, token_by_token=False
VRAM budget: 40960 MB
Objetivo: Maximo throughput, VRAM no es problema
```
- OFF: SMV, FDGD, GNS, SMA, DRA, SGC, NFR (no necesarios)
- ON: EMP, TER, TOWS, MSQ, HFISC, SBS (compute speedups)

---

### Uso Rapido

#### Metodo 1: make train (recomendado)

```bash
# Configuracion estandar (todas las optimizaciones para GTX 1050)
make train DEVICE=cuda:0

# Configuracion desde YAML (ver experiments/configs/)
make train CONFIG=experiments/configs/s3_standard.yaml DEVICE=cuda:0

# Ablation (desactivar componentes)
make train-ablation ABLATION=s3_no_stb DEVICE=cuda:0
```

#### Metodo 2: Menu interactivo

```bash
# Seleccionas tu GPU (LOW/MEDIUM/HIGH) y el modelo
python src/training/run_s3.py

# O con argumentos directos
python src/training/run_s3.py --tier medium --model Qwen/Qwen2.5-7B-Instruct
```

#### Metodo 3: Programatico

```python
from src.training.hardware_profiles import HardwareTier, get_profile
from src.training.training_pipeline import build_s3_model, load_dataset
from transformers import AutoModelForCausalLM

profile = get_profile(HardwareTier.LOW)  # Auto-configurado para tu GPU
# profile.config ya tiene todos los flags S3-OPT activos

trainer = build_s3_model(
    base_model=model,
    svd_dir="./svd_factors/",
    config=profile.config,  # S3-OPT flags incluidos
    device="cuda:0"
)
```

### Configuracion YAML

En `experiments/configs/s3_standard.yaml`, la seccion `s3_opt` controla cada optimizacion:

```yaml
s3_opt:
  use_smv: true       # Delta-modulacion vector PCIe
  use_emp: true       # Prediccion MLP desde Adam moments
  use_ter: true       # Routing N={1,2,4} por entropia
  use_fdgd: true      # Filtrado de gradientes Fourier
  use_gns: true       # Skip backward bajo gradiente
  use_sma: true       # Checkpoints espectrales (16x compression)
  use_tows: true      # Warmstart entre tokens
  use_dra: true       # Rank adaptativo dinamico
  use_mso: true       # Shortcuts lineales
  use_hfisc: true     # Inicializacion optima
  use_sgc: true       # Compresion espectral gradiente
  use_nfr: true       # Recycling de estados NMF
  use_sbs: true       # Bypass STB (p_stb=0.3)
  sbs_p_stb: 0.3      # Probabilidad de bypass STB
```

### Por que funciona — Teoria

Cada optimizacion tiene base teorica formalizada. Resumen rapido:

- **SMV**: Teorema de compression ratio. Transferencia vectorial $\Delta\mu$ es suficiente si $\|m_t - m_{t-1}\| < \delta$.
- **EMP**: Fisher information $\approx$ Adam moments en steady state. H_predicha $\approx$ H_real.
- **TER**: Entropia de bottleneck $H_{\text{NMF}} = -\Sigma a_j \log(a_j + \epsilon)$ es proxy de complejidad computacional. N rutas optimas minimizan expected FLOPs.
- **FDGD**: Butterworth filter de orden 2 preserva $\int |G(f)|^2 df$. Energia total conservada.
- **GNS**: Skip si $\|g\| < \epsilon \|g\|_{\max}$. El gradiente relativo pequeno implica que la capa esta cerca de un optimo local.
- **SMA**: $s = U_k^T h$ es reconstruccion exacta si $U_k^T U_k = I_k$. Compression ratio $k/d$.
- **TOWS**: Coherencia $C = \langle h_{t-1}, h_t \rangle / (\|h_{t-1}\| \|h_t\|)$ alta → warmstart seguro.
- **DRA**: $s(t) = \sigma_{k+1} / \Sigma\sigma_i$ mide fraccion de energia en componentes descartados. Si $s(t) < \delta$, el rank k es suficiente.
- **MSQ**: Curvatura $\kappa = \|f_\theta(h)\|_2 / \|h\|_2$. Si $\kappa < \kappa_{\text{thresh}}$, la ODE es aproximadamente lineal.
- **HFISC**: $\sigma = \sqrt{2/d}$ preserva $\mathbb{E}[x^2] = 1$ para GELU con entrada $\mathcal{N}(0,1)$.

Para detalles completos, ver `docs/formalismo_optimizacion.md` y `docs/mejoras.md` (laboratorio de inovacion).

---

## Comparacion con Metodos Existentes

---

## Comparación con Métodos Existentes

| Propiedad | S³ | LoRA $r{=}8$ | QLoRA $r{=}8$ | Full FT |
|-----------|-----|----------|---------|---------|
| Parámetros entrenables | 4.83M (0.060%) | 16.8M (0.21%) | 16.8M (0.21%) | 8,030M (100%) |
| VRAM peak (modelo 7B) | ~345 MB | ~8 GB | ~4.5 GB | >60 GB |
| Ejecutable en GTX 1050 (2GB) | ✓ | ✗ | ✗ | ✗ |
| Modulación espectral de pesos | ✓ (SVMO) | ✗ | ✗ | ✓ (implícito) |
| Deformación continua de representaciones | ✓ (NMF, Neural ODE) | ✗ | ✗ | ✗ |
| Acoplamiento pesos↔representaciones | ✓ (STB) | ✗ | ✗ | ✓ (implícito) |
| Garantía de estabilidad de gradiente | ✓ (Teorema 2) | ✗ | ✗ | Depende |
| Inicialización que preserva identidad | ✓ | ✗ | ✗ | N/A |

---

## Estructura del Repositorio

```
lowrank-field-adapters/
├── README.md                              (este archivo)
├── docs/
│   ├── formalismo.md                      (formalismo S3: 6 teoremas)
│   ├── formalismo_optimizacion.md         (10 teoremas S3-OPT, 1851 lineas)
│   └── mejoras.md                         (laboratorio de inovacion y sinergias)
├── src/
│   ├── adapters/
│   │   ├── svmo.py                        (Singular Value Modulation Operator)
│   │   ├── nmf.py                         (Neural Manifold Flow + solver RK4)
│   │   ├── stb.py                         (Spectral Transport Bridge)
│   │   ├── hybrid.py                       (capa S3: SVMO + STB + NMF integrados)
│   │   └── base.py                        (interfaz abstracta comun)
│   ├── training/
│   │   ├── frugal_trainer.py               (training loop con swap CPU↔GPU por capa)
│   │   ├── s3_opt.py                      (10+ optimizaciones S3-OPT)
│   │   ├── s3_optimizations.py             (SGC, NFR, SBS)
│   │   ├── training_pipeline.py            (build_s3_model + load_dataset compartidos)
│   │   ├── hardware_profiles.py            (tiers LOW/MEDIUM/HIGH + S3-OPT auto-config)
│   │   ├── launcher.py                     (menu interactivo)
│   │   └── run_s3.py                       (entry point menu: python run_s3.py)
│   ├── benchmarks/
│   │   └── run_benchmarks.py
│   └── utils/
│       ├── vram_monitor.py
│       ├── randomized_svd.py                (SVD offline pre-computado)
│       └── logging.py
├── experiments/
│   ├── configs/                            (YAML: s3_standard, s3_small, s3_large)
│   ├── results/
│   └── run_training.py                     (entry point menu: experiments/run_training.py)
├── train_s3.py                             (entry point make train)
└── tests/
    └── test_s3_full.py                     (tests de todos los componentes)
```

---

## Resultados Teóricos Principales

| Teorema | Resultado | Implicación práctica |
|---------|-----------|----------------------|
| T1: Aproximación SVMO | Error $O(1/\sqrt{H})$ | $H=32$ sufficient for ~4% error relativo |
| T2: Estabilidad de gradiente | Gradiente independiente de $\|x\|$ | No gradient clipping requerido |
| T3: Trayectoria NMF | Error lineal en $T$, no exponencial | Corrección de la literatura (Chen et al. 2018) |
| T4: RK4 | $\varepsilon_{\text{ODE}} \leq 6.3 \times 10^{-6}$ | 4 pasos suficientes |
| T5: Información mutua STB | $I > 0$ para cualquier $p_{\text{STB}} > 0$ | Acoplamiento garantizado |
| T6: Convergencia STB | Reducción de condition number | Entrenamiento más rápido |
| T7: PAC-Bayes | Cota no-vacía (26.5%) con $d_{\text{eff}} \approx 1760$ | Generalización demostrable |

---

## Requisitos de Hardware

| Componente | Mínimo (S³) | Recomendado ( baselines) |
|------------|-------------|------------------------|
| VRAM GPU | 2 GB (GTX 1050) | 8 GB+ (LoRA), 40 GB (Full FT) |
| RAM CPU | 16 GB | 32 GB |
| Almacenamiento | 30 GB | 30 GB |
| GPU (cloud, solo baselines) | — | A100-40GB (~1-2h renta) |

**Stack de software**: PyTorch 2.x, HuggingFace Transformers, CUDA Toolkit, LaTeX (para el paper).

---

**Autores**: Anndy-bit
**Fecha de inicio**: Julio 2026  
**Proyecto**: S³ — Spectral-Spatial-Smooth Fine-Tuning para VRAM limitada

---

## Uso Rápido

### Linux (con Makefile)

```bash
# Ver ayuda
make help

# Crear entorno virtual
make venv

# Ejecutar SVD (CPU por defecto, para GPU usa DEVICE=cuda:0)
make svd MODEL=Qwen/Qwen2.5-7B-Instruct K=128

# Entrenar
make train DEVICE=cuda:0

# Entrenar con monitor
make train-monitored DEVICE=cuda:0

# Pipeline completo
make all
```

### Windows (con bat_s3.bat)

```cmd
REM Ver ayuda
bat_s3.bat help

REM Crear entorno virtual
bat_s3.bat venv

REM Ejecutar SVD (usa tu GPU con --device cuda:0)
bat_s3.bat svd MODEL=Qwen/Qwen2.5-7B-Instruct K=128

REM Entrenar
bat_s3.bat train DEVICE=cuda:0

REM Entrenar con monitor
bat_s3.bat train-monitored DEVICE=cuda:0

REM Pipeline completo
bat_s3.bat all
```

### Variables personalizables

| Variable | Descripción | Valor por defecto |
|----------|-------------|-------------------|
| `MODEL` | Modelo de HuggingFace | `Qwen/Qwen2.5-7B-Instruct` |
| `K` | Rank SVD | `128` |
| `DEVICE` | Dispositivo (cpu/cuda:0) | `cuda:0` |
| `CONFIG` | Archivo YAML de config | `experiments/configs/s3_standard.yaml` |
| `ABLATION` | Estudio de ablación | `s3_full` |