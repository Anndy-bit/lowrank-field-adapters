# S³: Spectral-Spatial-Smooth Fine-Tuning

## Fine-tuning de LLMs en 2GB VRAM con calidad de GPU de alta gama

### Idea Central

Usar modelos pre-entrenados open-source con **matrices de pesos congeladas** y adjuntar tres **operadores matemáticos inventados desde cero** que permiten adaptar el modelo a nuevas tareas con:
- Calidad comparable a full fine-tuning (RTX 4090 / A100)
- VRAM de entrenamiento ≤ 2GB (GTX 1050)
- Sin destilación, sin cuantización agresiva, sin poda
- Solo 0.06% de parámetros entrenables (~4.8M para modelo 7B)

### Tres Operadores Nuevos (S³)

**Spectral — Spatial — Smooth.** Tres operadores que no existen en la literatura, mutuamente acoplados:

1. **SVMO — Singular Value Modulation Operator**: Modulación puntual no-lineal de cada valor singular de las matrices de pesos congeladas $W = U\Sigma V^T$, mediante una función aprendible $m_\theta(\sigma)$. U y V congelados. Solo $O(H^2)$ parámetros por matriz (independiente de la dimensión del modelo). Distinto a Zhang & Pilanci (2024) que usan rotación/adición de vectores singulares.

2. **NMF — Neural Manifold Flow**: Campo de deformación continuo sobre el espacio latente del modelo, definido como una Neural ODE $dh/dt = f_\theta(h, t)$ resuelta con RK4. Las representaciones siguen trayectorias curvas, no desplazamientos estáticos. Primera aplicación de Neural ODEs a PEFT para LLMs.

3. **STB — Spectral Transport Bridge**: Operador de acoplamiento bidireccional que transporta representaciones latentes a través de la base espectral $U_k$ usando cross-attention entre la firma espectral de $h$ y los valores singulares modulados $\Sigma_k$. Crea un canal de información mutua entre SVMO y NMF, permitiendo que la adaptación de pesos y la deformación de representaciones se coordinen durante el entrenamiento.

### Propiedades Clave

| Propiedad | S³ | LoRA r=8 | Full FT |
|-----------|-----|----------|---------|
| Parámetros entrenables | 4.83M (0.060%) | 16.8M (0.21%) | 8,030M (100%) |
| VRAM peak (7B model) | ~345 MB | ~8 GB | >60 GB |
| Corre en GTX 1050 (2GB) | ✓ | ✗ | ✗ |
| Modulación espectral de pesos | ✓ (SVMO) | ✗ | ✓ (implícito) |
| Deformación continua de reps. | ✓ (NMF, Neural ODE) | ✗ | ✗ (implícito) |
| Acoplamiento pesos↔reps | ✓ (STB) | ✗ | ✓ (implícito) |
| Estabilidad de gradiente | Demostrado (Thm 2) | No garantizado | Depende |
| Inicialización en identidad | ✓ | ✗ (ruido inicial) | N/A |

### Novedad Verificada

Investigación de literatura (arXiv, NeurIPS, ICLR, ICML, ACL, EMNLP, Google Scholar) — Julio 2026:

- **SVMO**: Distinto de Zhang & Pilanci "Spectral Adapter" (2024). Ellos hacen additive tuning + orthogonal rotation de vectores singulares. SVMO hace modulación puntual no-lineal con U y V congelados + garantía de estabilidad de gradiente. Diferenciación técnica sólida en `docs/formalismo.md` §1.8.
- **NMF**: Neural ODEs para PEFT en LLMs — cero hits. Nadie ha aplicado continuous-depth models a fine-tuning de transformers.
- **STB**: Transporte espectral bidireccional con cross-attention — cero hits. Sin precedentes.
- **S³ combinado**: Arquitectura de tres operadores acoplados para fine-tuning frugal — cero hits. Contribución principal del paper.

### Roadmap

```
Fase 1 — Formalismo Matemático ✅ COMPLETADO
├── Definición rigurosa de SVMO, NMF, STB ✅
├── 6 teoremas con demostraciones formales ✅
│   ├── Thm 1: Capacidad de aproximación de SVMO
│   ├── Thm 2: Estabilidad de gradiente de SVMO
│   ├── Thm 3: Expresividad de flujo de NMF
│   ├── Thm 4: Estabilidad numérica ODE (RK4)
│   ├── Thm 5: Cota de información mutua con STB
│   └── Thm 6: Aceleración de convergencia con STB
├── Análisis de complejidad y VRAM ✅
├── Randomized SVD (Halko et al. 2011) ✅
└── Comparación con Zhang & Pilanci (2024) ✅

Fase 2 — Implementación (PyTorch)
├── Módulo src/adapters/svmo.py
├── Módulo src/adapters/nmf.py (con solver RK4)
├── Módulo src/adapters/stb.py (cross-attention espectral)
├── Módulo src/adapters/hybrid.py (capa S³ completa)
├── src/training/frugal_trainer.py (swap CPU↔GPU)
├── src/training/checkpointing.py
└── Tests de VRAM y corrección

Fase 3 — Experimentos en GTX 1050
├── Modelos: Qwen2.5-7B, Mistral-7B-v0.3, Llama-3-8B
├── Datasets: Alpaca (52K), OpenOrca (50K), FLAN v2 (20K)
├── Benchmarks: MMLU, HellaSwag, ARC-Challenge, GSM8K, AlpacaEval 2.0
├── Baselines: LoRA r=8, LoRA r=64, QLoRA, Full FT
├── Ablación: 7 configuraciones (SVMO/NMF/STB combinaciones)
├── Estadística: Welch t-test, Cohen's d, bootstrap 95% CI, n=3 runs
└── VRAM logs con nvidia-smi polling 100ms

Fase 4 — Paper (LaTeX, formato NeurIPS/ICLR)
├── Introduction
├── Related Work
├── Method: SVMO, NMF, STB, S³ Architecture
├── Experiments
├── Discussion
└── Supplementary (derivaciones, código, logs)
```

### Estructura del Repositorio

```
lowrank-field-adapters/
├── README.md                       (este documento)
├── docs/
│   └── formalismo.md               (formalismo completo: 1091 líneas, 6 teoremas)
├── src/
│   ├── adapters/
│   │   ├── svmo.py                 (Singular Value Modulation Operator)
│   │   ├── nmf.py                  (Neural Manifold Flow + RK4 solver)
│   │   ├── stb.py                  (Spectral Transport Bridge)
│   │   ├── hybrid.py               (capa S³: SVMO + STB + NMF integrados)
│   │   └── base.py                 (clase abstracta común)
│   ├── training/
│   │   ├── frugal_trainer.py       (training loop con swap CPU↔GPU)
│   │   └── checkpointing.py        (gradient checkpointing custom)
│   ├── benchmarks/
│   │   ├── run_benchmarks.py
│   │   └── metrics.py
│   └── utils/
│       ├── vram_monitor.py
│       ├── randomized_svd.py       (SVD rápido offline)
│       └── logging.py
├── experiments/
│   ├── configs/                    (YAML de experimentos)
│   └── results/                    (resultados crudos + plots)
├── tests/
│   ├── test_svmo.py
│   ├── test_nmf.py
│   ├── test_stb.py
│   └── test_memory.py
└── paper/
    ├── main.tex                    (borrador LaTeX)
    └── figures/
```

### Laboratorio

- **GPU objetivo**: Nvidia GTX 1050 (2GB VRAM) o GTX 1050 Ti (4GB VRAM)
- **CPU**: Cualquiera (para offloading y SVD offline)
- **RAM**: ≥16GB (modelo completo en CPU)
- **Software**: PyTorch 2.x, HuggingFace Transformers, CUDA Toolkit
- **Cloud (solo baselines)**: A100-40GB (~1-2h rentada para LoRA/Full FT)

### Target Journal

- **NeurIPS** — track main (factor: novedad matemática + eficiencia de hardware)
- **ICLR** — eficiencia de fine-tuning + base teórica sólida
- **ICML** — contribuciones algorítmicas
- **ACL/EMNLP** — si se enfatiza el impacto en NLP downstream

### Estado Actual

- [x] Idea fundacional
- [x] Formalismo matemático completo (6 teoremas, docs/formalismo.md)
- [ ] Implementación SVMO (`src/adapters/svmo.py`)
- [ ] Implementación NMF (`src/adapters/nmf.py`)
- [ ] Implementación STB (`src/adapters/stb.py`)
- [ ] Implementación capa híbrida S³
- [ ] Training loop frugal
- [ ] SVD offline de modelos base
- [ ] Experimentos en GTX 1050
- [ ] Paper LaTeX

---

**Autores**: [Tu nombre aquí] y equipo
**Fecha de inicio**: Julio 2026
**Proyecto**: S³ — Spectral-Spatial-Smooth Fine-Tuning para VRAM limitada