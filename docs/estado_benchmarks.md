# Estado: benchmarks y Paper 2 — 2026-07-16

Handoff de la sesión del 16/07/2026. Rama `feat/streaming-formalism`.

---

## 1. Hecho y verificado

| Qué | Dónde | Estado |
|---|---|---|
| **Paper 1 (USF), inglés** | `paper/usf/main.tex` | 7 pág, compila, 0 errores, 0 refs sin resolver, 0 avisos bibtex, 28/28 citas |
| **Paper 1, español** | `paper/usf/main_es.tex` | Escrito. **Sin verificar compilación** |
| **`forward_layer_major`** | `src/training/s3_trainer.py` | Thm 4 aplicado a inferencia. Una lectura del modelo sirve a *todos* los batches |
| **`USFLM`** (backend `lm_eval`) | `src/eval/usf_lm.py` | Sirve base/S³/LoRA por el mismo camino |
| **Tests del wrapper** | `tests/test_usf_lm.py` | 5 tests. **61 pasan en la suite completa**, sin regresiones |
| **Datasets de benchmark** | `~/.cache/huggingface/` | 155 MB. 7 tareas OK |

### Los datasets (ya descargados)
`boolq` (aps/super_glue) · `piqa` (baber/piqa) · `hellaswag` (Rowan/hellaswag) ·
`winogrande` (allenai/winogrande) · `arc_easy` + `arc_challenge` (allenai/ai2_arc) ·
`openbookqa` (allenai/openbookqa) · `gsm8k` (openai/gsm8k) · `mmlu` (hails/mmlu_no_train)

**SIQA está rota**: `allenai/social_i_qa` es un dataset de script y `datasets` 5.0 ya no
los soporta (bajó 12,5 kB, sin datos). Haría falta un repo en parquet.

---

## 2. El hallazgo que cambia el Paper 2

**SVMO no está muerto. Yo lo medí mal.** (Corrección hecha tras un desafío del usuario.)

### Dos cosas que invalidan la lectura ingenua del ablation

1. **STB depende estructuralmente de SVMO** (`s3_block.py:137-139`):
   ```python
   if enable_stb and enable_svmo:
       self.stb = STBResidual(self.self_attn.q_proj.U_k, ...)
   ```
   STB se construye *desde los factores SVD de SVMO*. La variante `svmo_only` no es
   "SVMO aislado": es **SVMO sin el operador que consume su base**. Y `stb_only` es
   imposible por construcción.

2. **Los `extra_opts` (TER, DRA, MSO, TOWS, HFISC, FDGD, GNS, EMP, SMV) NO se pasan en
   `run_ablations.py`** — solo en `run_s3_train.py`. Los ablations no corren la misma
   configuración que el run titular del paper.

### La comparación justa (datos ya existentes)

| Vía | Params | Δppl |
|---|---:|---:|
| `no_nmf` = **SVMO+STB** (espectral completa, SBS on) | 684.740 | **−5,73** |
| `nmf_b2` = **NMF solo** (NFR on) | 802.928 | **−5,42** |

**La vía espectral gana con 15% menos parámetros.** La tesis de sinergia está respaldada.

⚠️ La comparación que NO hay que hacer (y que yo hice, mal): `svmo_h64` (853K, SVMO sin
STB) vs `nmf_b2`. Eso enfrenta un método amputado contra uno entero.

**Por qué SVMO solo no mueve la aguja** (teoría, no bug): `Δσ = S_k · α · tanh(g)` con
α=0,3 solo **reescala** ±30% los valores singulares existentes. Está confinado al
subespacio singular que ya existe: repondera, no rota, no crea direcciones. Su valor no
está en su propio Δσ — está en **fabricar la base U_k sobre la que STB opera**.

### Acción pendiente (0 GPU)
Re-reportar `comparison.md` **por vías** (espectral / flujo / ambas), no por operadores
sueltos. Tal como está, cualquier revisor llega a la conclusión errónea.

---

## 3. Presupuesto de tiempo — y mis tres fallos de estimación

**Tokens reales de la suite** (contados, no estimados — `scratchpad/count_tokens.py`):

| Tarea | Peticiones | Tokens |
|---|---:|---:|
| boolq | 1000 | 148.196 |
| hellaswag | 2000 | 86.245 |
| arc_challenge | 1997 | 75.475 |
| arc_easy | 1998 | 62.714 |
| piqa | 1000 | 35.644 |
| openbookqa | 2000 | 30.226 |
| winogrande | 1000 | 21.763 |
| **TOTAL** (limit=500) | **10.995** | **460.263** |

Media real: **42 tokens/petición** — son 0-shot y cortísimas.

### Historial de estimaciones (para calibrar cuánto fiarse)
1. "~1 h/tarea, ~8 h/adapter" → **falso**, asumí 256 tok/petición sin verificar.
2. "45,9 tok/s → 2,8 h/adapter" → es el **techo de cómputo**, solo alcanzable con
   amortización total.
3. Medido con `eval_loss` a batch 8: **11,3 tok/s → 11,3 h/adapter**. Pero ese régimen
   está dominado por I/O (384 tok por lectura de 15,2 GB).

**El modelo de coste sí funciona**: predijo 36,5 s para esa config, medido 34,0 s → **7%
de error**, coherente con el 6% que afirma el Paper 1.

### La predicción sin verificar
| Tokens por lectura | tok/s predicho | Suite/adapter |
|---:|---:|---:|
| 384 (`eval_loss`, **medido**) | 11,3 | 11,3 h |
| 6.144 | 37,9 | 3,4 h |
| 460.263 (wrapper, **sin medir**) | 45,8 | **2,8 h** |

**Esta tabla es una predicción, no una medición.** Es lo que quedó pendiente de validar.

---

## 4. Por qué murieron las mediciones

Los procesos en segundo plano son **hijos de la sesión de Claude Code**. Al cerrar VS
Code, la sesión muere y el kernel se lleva a los hijos. Murieron así el barrido de batch
y el benchmark del wrapper. No fue OOM ni timeout.

**Solución** (`tmux`/`screen` NO están instalados; `setsid` sí):
```bash
cd "/home/gula/Documentos/Proyectos Github/lowrank-field-adapters"
setsid nohup ./.venv/bin/python -u SCRIPT.py > /tmp/run.log 2>&1 < /dev/null &
echo "PID $!"
```
Seguimiento: `tail -f /tmp/run.log` · Matar: `kill <PID>`

**Trampa aparte:** `eval_loss` proyecta el vocabulario sobre la secuencia entera
(`[B, S, 152000]` fp32). A batch 128 son 3,66 GB, ×2 por el `.float()`. No sirve para
medir a batch grande. `USFLM` no tiene ese problema — solo proyecta las posiciones de la
continuación.

---

## 5. Próximos pasos, en orden

| # | Qué | GPU | Nota |
|---|---|---|---|
| 1 | **Medir `USFLM`** en openbookqa (limit=100) | ~15 min | `scratchpad/bench_wrapper.py`. Mide corrección (Thm 4: trocear no mueve logprobs) *y* coste (amortización). **Decide todo lo demás.** |
| 2 | **Re-reportar ablation por vías** | **0** | Reanálisis puro. Urgente: los datos actuales inducen a error |
| 3 | Benchmarks: 7 tareas × {base, S³, LoRA r8} | ~8,4 h* | *si el paso 1 confirma la amortización |
| 4 | Barrido de invariancia del solver ODE (N=1,2,4,8,16) | ~1 h | Test **falsable**: si NMF aprende un *flujo*, la salida debe ser ~invariante al nº de pasos del solver. Si oscila, no es una ODE. Sin reentrenar |
| 5 | n≥3 semillas | ~15 h | Significancia. El margen S³ (4,46) vs LoRA r8 (4,54) es de 1 semilla |
| 6 | Verificar compilación de `main_es.tex` | 0 | Nunca se compiló |

### Decisión metodológica cerrada
**Los benchmarks se miden de forma tradicional**, pese a que S³ es no lineal. Un
benchmark es una sonda de caja negra sobre la *salida*; el mecanismo le es invisible por
diseño, y esa invisibilidad es lo que hace válida la comparación con LoRA. Darle métrica
propia a S³ mataría el Paper 2. La no-linealidad es la *hipótesis*, no una exención.

Lo que sí añade valor por encima: diagnósticos de mecanismo (el paso 4).
