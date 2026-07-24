# Evidencia experimental — Paper 1 (USF: Unit-Streamed Fine-tuning)

Consolidación de todas las mediciones que respaldan `docs/formalismo_streaming.md`.
Hardware: **NVIDIA GTX 1050 (3.94 GB)**, 8 cores, 15.5 GB RAM, NVMe.
Modelo: **Qwen2.5-7B-Instruct** (7.62B params, 28 capas, fp16).
Fecha: Julio 2026 · Rama: `feat/streaming-formalism`

---

## 1. Teorema 1 — Exactitud del streaming (VALIDADO)

| Escenario | Comparación | Resultado |
|---|---|---|
| Tiny Qwen2 (CPU) | streaming vs autograd residente | **max\|Δgrad\| = 0.00e+00** (150 tensores) |

**Fuente:** `tests/test_s3_trainer.py::test_layerswap_grad_equivalence`
**Conclusión:** el streaming de pesos no altera los gradientes. No es aproximación.

---

## 2. Teorema 4 — Layer-major (VALIDADO EN 7B REAL) ⭐

| Escenario | Comparación | Resultado |
|---|---|---|
| Tiny Qwen2 (CPU) | layer-major vs batch-major | **max\|Δgrad\| = 0.00e+00** (150 tensores) |
| **Qwen2.5-7B (GTX 1050)** | layer-major vs batch-major, **mismo modelo** | **max\|Δgrad\| = 0.0000e+00** (1400 tensores) |

Escala de referencia: `max|grad| = 7.55e+02` → el error relativo es **exactamente 0**.

**Fuentes:** `tests/test_layer_major.py::test_layer_major_equals_batch_major` (tiny),
`results/thm4_exactness_7b.log` (7B real).
**Conclusión:** invertir los bucles reduce el I/O por un factor `G` **sin coste alguno en
corrección**. El Teorema 4 pasa de propuesto a demostrado empíricamente a escala real.

---

## 3. Corolario 2.1 — Descarga de checkpoints a CPU (VALIDADO)

| Comparación | Resultado |
|---|---|
| ckpt en VRAM vs ckpt en CPU | **max\|Δgrad\| = 0.00e+00** |

**Fuente:** `tests/test_layer_major.py::test_ckpt_offload_is_neutral`

---

## 4. Rendimiento — batch-major vs layer-major (7B real, medido)

Configuración: `B=4`, `S=256`, `G=4`, fp16, streaming por capa, 16 ejemplos.

| Camino | Tiempo | Por ejemplo | VRAM pico |
|---|---|---|---|
| Batch-major | 457.5 s | **28.6 s/ej** | 2524 MB |
| **Layer-major** | **327.4 s** | **20.5 s/ej** | **2219 MB** |
| **Ganancia** | — | **1.40×** | **−305 MB** |

**Fuente:** `results/layermajor_validation.log`

**Predicción del modelo de coste (§8 del formalismo) para `G=4`:**
`T_bm = G·(T_io+T_cmp) = 4·(45+57) = 408 s` · `T_lm = T_io + G·T_cmp = 45+4·57 = 273 s`
→ speedup predicho **1.49×**, medido **1.40×**. **Error del modelo: 6%.**
La diferencia se explica por el coste de transferir los checkpoints GPU↔CPU.

**Nota:** con `G=8` el modelo predice ~1.7× (el término `T_io` se amortiza más). No medido.

---

## 5. Teorema 2 — Cota de VRAM (VALIDADO)

| Adapter | Params entrenables | VRAM pico medido |
|---|---|---|
| S³ full | 3,896,452 (0.0511%) | **2269 MB** |
| LoRA r=8 | 20,185,088 | **1606 MB** |
| LoRA r=2 | 5,046,272 | 1598 MB |
| LoRA r=1 | 2,523,136 | 1315 MB |

**Corolario 1.2 (independencia del adapter): CONFIRMADO** — la misma infraestructura corre S³ y
LoRA. La diferencia de VRAM es exactamente el término `|θ|·s_opt + A` del Teorema 2 (S³ mantiene
residentes los factores SVD y los estados de la ODE).

**Fuente:** `results/ablations_baselines_20260716_024551/comparison.csv`

---

## 6. Modelo de coste (§8) — validación

Parámetros calibrados: `BW = 0.54 GB/s`, `Φ = 0.7 TFLOP/s`.

| Magnitud | Modelo | Medido | Error |
|---|---|---|---|
| I/O por paso (batch-major) | 24.3 GB | 24.9 GB | 2% |
| `T_ej` (batch=4, batch-major) | 25.6 s | 26.9 s | **5%** |
| Speedup layer-major (G=4) | 1.49× | 1.40× | **6%** |

**Conclusión:** el modelo es **predictivo**, no descriptivo. Las proyecciones a 70B/405B (§11)
se apoyan en él.

---

## 7. Proposición 10.1 — Separación de viabilidad frente a cuantización (VALIDADO)

| Método | Contabilidad | Resultado en 3.94 GB |
|---|---|---|
| 4-bit NF4 (bitsandbytes) | capas 3.25 GB + embed 1.09 + lm_head 1.09 = **5.43 GB** | **CUDA OOM** ❌ |
| **USF** | `max_u\|W_u\|` = 0.43 GB + `\|θ\|` + A | **2.27 GB — corre** ✅ |

**Evidencia:** intento real de carga en 4-bit → `torch.OutOfMemoryError`. Además, el offload
parcial de bitsandbytes falla por un bug de tensores `meta`.
**Conclusión:** en este hardware no es una preferencia, es una **separación de viabilidad**.

---

## 8. Demostración de aprendizaje end-to-end (7B, GTX 1050)

| Métrica | Base (init) | Después | Δ |
|---|---|---|---|
| Perplejidad (100 held-out) | 9.52 | **4.46** | **−53%** |
| eval loss | 2.48 | 1.71 | −0.78 |
| Loss de entrenamiento | 5.03 | 0.82 (mín) | −4.21 |

Corrida: 1100 ejemplos de Alpaca, 1 época, 275 pasos, **4 h 16 min**, VRAM pico 2233 MB,
0 pasos saltados por overflow fp16.
**Fuente:** `results/run_20260714_225107/` (steps.csv, vram_trace.csv 114.766 muestras, figuras).

---

## 9. Estado de cada afirmación del formalismo

| Afirmación | Estado |
|---|---|
| Thm 1 — exactitud del streaming | ✅ **validado** (tiny) |
| Thm 2 — cota de VRAM | ✅ **validado** (S³ y LoRA, 7B) |
| Cor 1.2 — independencia del adapter | ✅ **validado** (S³ + LoRA) |
| Cor 2.1 — checkpoints a CPU | ✅ **validado** |
| Thm 3 — jerarquía de granularidad | ⬜ implementado, **sin validar** |
| **Thm 4 — layer-major** | ✅ **validado en 7B real (Δ=0)** |
| Prop 5 — prefetch | ⬜ implementado, **sin validar** |
| §8 — modelo de coste | ✅ **validado** (error 5-6%) |
| §9 — selección automática de granularidad | ⬜ implementado, **sin validar** |
| Prop 10.1 — separación vs cuantización | ✅ **validado** (OOM real) |
| Cor 3.2 — 405B en 4 GB | ⬜ **teórico** (bloqueado por descarga de 748 GB) |

---

## 10. Lo que falta para cerrar el Paper 1

1. **Validar Prop 5 (prefetch)** — medir el solapamiento real. ~20 min GPU.
2. **Validar Thm 3 + §9 (granularidad y auto-selección)** — comprobar que sub-capa y por-matriz
   dan los mismos gradientes y el pico de VRAM predicho. ~20 min GPU.
3. **(Opcional, gran impacto)** una corrida de **70B** — convertiría el Cor. 3.2 de teórico a
   demostrado. Bloqueado por la descarga de 128 GB.
