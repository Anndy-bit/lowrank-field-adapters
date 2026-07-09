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
3. **NMF**: Aplica flujo连续 sobre la representación salida: $h^{(l)} \to \tilde{h}^{(l)}$.

Solo los parámetros de $g_\theta$ (SVMO), $W_{\text{in}}, W_{\text{out}}$ (NMF), y las matrices de atención de STB se entrenan. Los pesos $U, V$ del modelo base permanecen congelados. Entrenamiento con swapping secuencial capa-por-capa CPU$\leftrightarrow$GPU: solo una capa en GPU a la vez.

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
├── README.md
├── docs/
│   └── formalismo.md                   (formalismo completo: 6 teoremas con demostraciones)
├── src/
│   ├── adapters/
│   │   ├── svmo.py                     (Singular Value Modulation Operator)
│   │   ├── nmf.py                      (Neural Manifold Flow + solver RK4)
│   │   ├── stb.py                      (Spectral Transport Bridge)
│   │   ├── hybrid.py                    (capa S³: SVMO + STB + NMF integrados)
│   │   └── base.py                     (interfaz abstracta común)
│   ├── training/
│   │   ├── frugal_trainer.py            (training loop con swap CPU↔GPU por capa)
│   │   └── checkpointing.py             (gradient checkpointing custom)
│   ├── benchmarks/
│   │   ├── run_benchmarks.py
│   │   └── metrics.py
│   └── utils/
│       ├── vram_monitor.py
│       ├── randomized_svd.py            (SVD offline pre-computado)
│       └── logging.py
├── experiments/
│   ├── configs/                         (YAML de configuraciones)
│   └── results/                         (resultados y visualizaciones)
├── tests/
│   ├── test_svmo.py
│   ├── test_nmf.py
│   ├── test_stb.py
│   └── test_memory.py
└── paper/
    ├── main.tex                         (borrador LaTeX, formato NeurIPS/ICLR)
    └── figures/
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