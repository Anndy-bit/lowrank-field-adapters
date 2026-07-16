# Paper 1 — USF · Outline y trazabilidad de datos

> **Formato:** IEEE (revista) · **Idiomas:** EN (principal) + ES (traducción)
> **Estado del material:** todos los datos ya medidos. Nada pendiente de correr.
> **Regla de oro:** cada número del paper tiene aquí su fuente. Si no está aquí, no entra.

---

## 1. Título — opciones

| # | Título | Tono |
|---|---|---|
| **A** ⭐ | **Unit-Streamed Fine-Tuning: Decoupling Model Size from GPU Memory with Exact Gradients** | Sobrio + el claim es de por sí llamativo. **Recomendado.** |
| B | *One Layer at a Time: Exact-Gradient Fine-Tuning of 7B Language Models on a Consumer 4 GB GPU* | Más directo/llamativo |
| C | *Memory Is Not the Barrier: Exact Fine-Tuning of Large Language Models on Commodity Hardware* | Tesis como título |

**Por qué A:** en revista IEEE el título debe decir *qué se hizo*, y "Decoupling Model Size from GPU
Memory" **es** la afirmación contraintuitiva. No necesita adjetivos: el resultado es el gancho.
El viejo título ("on 2 GB Consumer GPUs") **hay que retirarlo**: medimos 2.2 GB de pico, no cabe
en 2 GB, y la tarjeta es de 4 GB. Sería el primer ataque de un revisor.

---

## 2. Abstract (borrador, solo números reales)

> Fine-tuning a 7B-parameter language model conventionally requires 8–16 GB of GPU memory, since
> the frozen base must remain resident throughout forward and backward. We show that this
> requirement is an artifact of implementation, not a mathematical necessity. Reverse-mode
> differentiation composes **local** vector–Jacobian products: computing the gradient at layer *i*
> needs only that layer's input, its weights, and the incoming cotangent — never the other *L−1*
> layers. We introduce **Unit-Streamed Fine-Tuning (USF)**, which keeps exactly one weight unit
> resident at a time, streamed from disk, and prove that the resulting gradients are **exact**
> (Thm 1), not approximate. The consequence is a memory bound that is **independent of model depth
> and of total parameter count**: `VRAM = f(max_u |W_u|)` (Thm 2) — only the largest single unit
> matters. We further prove that inverting the loop order (**layer-major**, Thm 4) divides disk I/O
> by the gradient-accumulation factor *G* while leaving gradients bit-identical, changing the
> regime from I/O-bound to compute-bound. On a **GTX 1050 (3.94 GB)** we fine-tune Qwen2.5-7B with
> a **peak of 2.2 GB**, reducing held-out perplexity from 9.52 to 4.46; layer-major is verified
> gradient-exact on the real 7B (max|Δ| = 0 over 1400 tensors) and yields a measured 1.40×
> speed-up. A predictive cost model matches measurements within 6%. USF is **adapter-agnostic**:
> we run both a spectral adapter and LoRA unchanged. Under the same budget, 4-bit quantization
> does **not** fit (5.43 GB required), making this a feasibility separation rather than a
> preference. The bound predicts that Llama-3-70B fits the same 4 GB card at the implemented
> granularity (1.59 GB/layer); we state this as a corollary and delimit precisely what remains
> unverified.

**Longitud objetivo:** ~250 palabras. Recortar en la versión final.

---

## 3. Contribuciones (las 5 que se defienden)

1. **Un resultado de independencia** (Thm 2 + Cor 2.2): la VRAM de fine-tuning no depende de `L`
   ni de `Σ|W_i|`, solo de `max_u |W_u|`. Rompe un acoplamiento asumido universalmente.
2. **Exactitud demostrada y verificada** (Thm 1, Thm 4): no es una aproximación con degradación
   — `max|Δgrad| = 0`, incluido **en el 7B real**.
3. **Layer-major** (Thm 4): reordenar la acumulación divide el I/O por `G`. Carece de sentido si
   el modelo cabe en memoria — por eso no está en la literatura. Cambia el régimen a compute-bound.
4. **Modelo de coste predictivo** (§8): error 5–6% frente a medición; convierte las proyecciones
   a 70B en predicciones fundamentadas, no en extrapolación.
5. **Un sistema agnóstico al adapter y al modelo**: misma infraestructura para S³ y LoRA;
   selección automática de granularidad según la VRAM disponible.

---

## 4. Mapa de secciones → fuente de cada dato

| § | Sección | Contenido | **Fuente de los datos** |
|---|---|---|---|
| I | Introduction | La barrera 8–16 GB; la observación de la localidad del VJP; contribuciones | — |
| II | Related Work | ZeRO-Infinity, accelerate, activation checkpointing, QLoRA. **Qué es nuevo y qué no** | `formalismo_streaming.md` §13 |
| III | Preliminaries | Notación, unidad transmisible, granularidad, Alg. USF | `formalismo_streaming.md` §2 |
| IV | **Exactness** | Thm 1 + prueba; numérica fp16 | §3 · test `test_s3_trainer::test_layerswap_grad_equivalence` (Δ=0, 150 tensores) |
| V | **Memory bound** | Thm 2, Cor 2.1 (⊥ profundidad), Cor 2.2 (⊥ tamaño) | §4 · medido: S³ 2233 MB, LoRA r8 1606 MB |
| VI | **Granularity** | Thm 3, jerarquía, tabla por modelo, **límite honesto de per-matrix** | §5 · `test_usf_components` (Δ=0) · tabla desde código |
| VII | **Layer-major** | Thm 4 + prueba, Prop. 4.2 (elección de G) | §6 · **Δ=0 en 7B (1400 tensores)** · 1.40× medido |
| VIII | Prefetch | Prop. 5, condición de solapamiento | §7 · `test_usf_components::test_prefetch...` (Δ=0) |
| IX | **Cost model** | Fórmula, régimen, **validación 5–6%** | §8 · predicho 25.6 s/ej vs medido 26.9 |
| X | Auto-granularity | Alg. 9.1, Prop. 9.2 | §9 · `test_auto_granularity_selection` |
| XI | **Experiments** | Setup, exactitud, VRAM, speedup, aprendizaje, vs cuantización, agnosticismo | `results/USF_paper1_evidence.md` (todo) |
| XII | Scaling analysis | Tabla 7B/70B/405B; **qué está demostrado vs propuesto** | §11 + §5 nota de limitación |
| XIII | Limitations | Tiempo vs memoria; disco; `\|θ\|≪\|W\|`; per-matrix no implementado; 405B teórico ×2 | §14 |
| XIV | Conclusion | — | — |

---

## 5. Tablas y figuras del paper (todas con datos reales)

> **Numeración real:** LaTeX numera por orden de aparición. El diagrama de planificación cae en
> §Layer-major (antes que Experimentos), así que es la **Fig. 1**, no la 3. Las tablas quedaron en
> el orden de abajo. Ambos idiomas coinciden.

| Elemento | Contenido | Fuente |
|---|---|---|
| **Tabla I** | Escalado: 7B/8B/13B/70B/405B × granularidad × ¿cabe en 4 GB? | `test_usf_components::test_scaling_table` |
| **Tabla II** | Modelo de coste: predicho vs medido (I/O, T_ej, speedup) — error ≤6% | §8.1 |
| **Tabla III** | Exactitud: Δgrad por teorema (Thm 1, 4, Cor 2.1, Thm 3, Prop 5) — todos 0.00e+00 | 56 tests |
| **Tabla IV** | VRAM (3 medidas: alloc/resv/smi) y params, 4 adapters | `ablations_baselines_*/*/vram_trace.csv` |
| **Tabla V** | Layer-major: batch-major 28.6 s/ej vs layer-major 20.5 s/ej; VRAM 2524→2219 | `layermajor_validation.log` |
| **Tabla VI** | Alcance de afirmaciones: 7B demostrado / 70B predicho / 405B teórico | §11 + §5 |
| **Fig. 1** ✅ | Diagrama TikZ: batch-major vs layer-major (por qué I/O ÷ G) | dibujado en el `.tex` |
| **Fig. 2** ✅ | Traza de VRAM, 114.766 muestras @100 ms, 3 medidas + capacidad | `make_figures.py` ← `run_20260714_225107/vram_trace.csv` |
| **Fig. 3** ✅ | Curva de aprendizaje, 138 pasos de optimizador + ppl reservada | `make_figures.py` ← `.../steps.csv` |

**Figuras:** vectoriales (PDF), regenerables con `./.venv/bin/python paper/usf/make_figures.py`.
Se emiten en dos idiomas — `figures/` (EN) y `figures/es/` (ES) — desde los mismos CSV, para que
los rótulos de ejes no queden en inglés dentro del paper español. Los PNG de `results/*/figures/`
**no** se usan: llevan títulos con jerga interna («RQ1») y son de mapa de bits.

---

## 6. Cómo se enmarca el 7B / 70B / 405B (la escalera de honestidad)

Se dedica una subsección explícita, con esta tabla literal:

| Modelo | Afirmación | Base |
|---|---|---|
| **Qwen2.5-7B** | **Demostrado empíricamente** | Corrido: 2.2 GB pico, ppl 9.52→4.46, Δgrad=0 |
| **Llama-3-70B** | **Predicho por la cota (Thm 2), no ejecutado** | 1.59 GB/capa < 3.94 GB con la granularidad **ya implementada**. Falta el modelo (128 GB) |
| **Llama-3-405B** | **Teórico; requiere una extensión no implementada** | Necesita granularidad por-matriz (1.62 GB): la atención de HF consume q,k,v,o juntos |

> Esto es lo que da impacto **sin exponerse**: se demuestra el 7B, se da la cota que predice el
> 70B **con la maquinaria que ya corre**, y se especifica exactamente qué falta para el 405B.
> Un revisor acepta extensiones teóricas claramente etiquetadas; lo que rechaza es venderlas como
> hechas.

---

## 7. Qué NO entra en este paper (y por qué)

- **Los operadores S³** (SVMO/NMF/STB): son el **Paper 2**. Aquí S³ aparece solo como *uno de los
  dos adapters* que demuestran el agnosticismo (Cor 1.2), sin reclamar nada sobre su calidad.
- **Las ablaciones RQ4, la comparación S³ vs LoRA en ppl**: Paper 2 (y le faltan n≥3 semillas).
- **S³-OPT**: Paper 3, no viable aún.
- **PAC-Bayes**: la cota es vacua. Fuera.

---

## 8. Riesgos del paper y cómo se cubren

| Riesgo (lo que preguntará un revisor) | Respuesta en el paper |
|---|---|
| "El offloading ya existe (ZeRO-Infinity)" | §II: régimen distinto (base congelada, 1 GPU de consumo, sin estados que shardear) + la cota formal + layer-major, que carece de sentido si el modelo cabe |
| "¿Exactitud solo en un modelo juguete?" | Thm 4 verificado **en el 7B real** (1400 tensores). Thm 1 solo puede verificarse donde el modelo quepa residente — se dice explícitamente |
| "Es demasiado lento para ser útil" | §IX + §XIII: lo admitimos. USF cambia memoria por tiempo. Régimen práctico: 7B–13B |
| "¿Por qué no cuantizar?" | Prop 10.1: **no cabe** (5.43 GB). Separación de viabilidad, con OOM medido |
| "El 70B no lo corrieron" | §XII: etiquetado como predicción de la cota, no como resultado |

---

## 9. Plan de escritura

1. ✅ Este outline
2. ✅ `main.tex` (EN) — 8 pág, 0 errores, 0 refs rotas, 0 avisos bibtex
3. ✅ Todas las secciones EN (14 §, 4 thm + 4 cor + 3 prop, 6 demostraciones)
4. ✅ Figuras: TikZ (Fig. 1) + 2 vectoriales desde CSV (Figs. 2–3), en EN y ES
5. ✅ `main_es.tex` — 9 pág, mismo estado; paridad de datos verificada

**Pendiente:** bloque de autor (ahora «Anonymous Author(s)» / «Autor(es) Anónimo(s)»).

---

## 10. Correcciones aplicadas durante la escritura

Cosas que estaban mal y se arreglaron. Se dejan anotadas porque varias vinieron de datos
generados por nuestro propio código, no del paper:

| Qué | Estaba | Es | Origen |
|---|---|---|---|
| **Pasos de optimizador** | 275 | **138** (275 = micro-lotes; `grad_accum=2`) | `report.md` etiquetó mal la columna; el paper lo heredó |
| **Umbral compute-bound** | `2sΦ/BW` → 2.6e3 tok | **`sΦ/(3·BW)`** → 8.6e2 tok | Se perdió el factor 6 de los FLOPs al despejar. La fórmula impresa y el número ni coincidían entre sí. Contradecía el propio §Prefetch (48 s vs 57 s) |
| **VRAM reportada** | «pico 2.2 GB», a secas | **alloc 2233 / resv 2514 / smi 2891** | 2.2 GB era solo el asignador; la ocupación real con contexto CUDA es 2.9 GB (71.7 % de la tarjeta). La afirmación sobrevive bajo la medida estricta |
| **`\crefname{figure}`** | ausente en ambos | `Fig.` / `Figura` | El cuerpo decía «Figure 1» y el pie «Fig. 1»/«Figura 1» |
| **Tablas en ES** | «Cuadro» (babel) | **«Tabla»** (`es-tabla`) | El cuerpo decía «Tabla II», el pie «Cuadro II» |
| **`\algorithmicto`** | `\renewcommand` | eliminado | No existe en `algpseudocode`; era un error de LaTeX oculto |

> **Lección de método:** el log en español sale en ISO-8859 por los acentos, así que `grep` lo
> trataba como binario y **silenciaba su salida** — un error de LaTeX pasó desapercibido y por poco
> se reporta el paper como limpio. Verificar siempre con `grep -a`. Y `grep -c` sale con código 1
> cuando cuenta 0, lo que rompe cadenas `&&` a mitad.

Verificado: la aritmética de la cuantización (5.43 GB), el tamaño en disco (15.23 GB), las
proyecciones de escalado (4.0 / 41.4 / 243.8 h) y `ppl = exp(loss)` (error rel. 2.4e-5, por eso la
Fig. 3 no duplica el eje) se recomputaron desde los datos y cuadran.