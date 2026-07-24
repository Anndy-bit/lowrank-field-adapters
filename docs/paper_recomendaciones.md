## Estado (noche 2026-07-18/19, trabajo autónomo)

`paper/main.tex` fue reescrito con datos reales: autoría real, VRAM/params
corregidos en todo el documento (título, abstract, tablas, conclusión), cita
fabricadas eliminadas de Related Work (7 `\bibitem` placeholder sin rellenar
que aparecían como si fueran reales — riesgo serio, ver §Related Work),
sección de Experimentos reemplazada con las 3 tablas reales (benchmark 7
tareas, ablation por vías, invariancia ODE), notas de honestidad agregadas
después de Thm 5-6 y Thm 7. Compila limpio (14 páginas, 0 errores, 0 refs sin
resolver, 24/24 citas reales). Pendiente: la sección IV-F (Statistical
Significance) queda con un marcador explícito de "en progreso" hasta que
termine el barrido de semillas (`scripts/run_seed_sweep.py`, corriendo en
background) — NO citar significancia de este borrador hasta que esa sección
se actualice con datos reales.

# Recomendaciones para el Paper 2 (S³) — audiencia real y honestidad

> No son cambios de redacción todavía. Es la justificación que hay que incorporar
> cuando se escriba/revise la motivación, para que aguante la primera pregunta
> obvia de un revisor. Fecha: 2026-07-18.

---

## 1. La pregunta que cualquier revisor va a hacer: "¿por qué no usar Colab?"

Google Colab da gratis una GPU T4 de **16GB**. Con eso, LoRA estándar sobre un
7B ya entra cómodo, con las herramientas de siempre (HuggingFace PEFT), sin
streaming por disco, sin la complejidad de SVMO/STB/NMF. **Si alguien tiene
acceso real y estable a Colab, S³ no le da nada que Colab+LoRA no le dé ya.**

La introducción actual (`paper/main.tex`) dice que la barrera de hardware
"excludes independent researchers... in resource-constrained settings" — eso,
tal cual, es refutable: la GPU gratis existe y es accesible para casi
cualquiera con cuenta de Google. Hay que reemplazar esa afirmación genérica
por la razón honesta y específica de **para quién Colab no es una opción**:

| Razón | Por qué Colab no sirve ahí |
|---|---|
| **Internet inestable o sin acceso confiable** | Colab necesita conexión sostenida; se desconecta si se cae la sesión. Una corrida de horas (o las 15-17h que corrimos hoy sin supervisión) no sobrevive una conexión intermitente. |
| **Los datos no pueden salir de la máquina** | Salud, legal, gobierno, datos propietarios — mandar los datos de fine-tuning a la nube de un tercero no es negociable en muchos contextos reales. |
| **Restricción geográfica** | Hay países donde el acceso a servicios de Google está bloqueado o muy limitado. |
| **Corridas largas sin límite de sesión** | El tier gratis de Colab tiene tope de horas y se desconecta por inactividad — no sirve para entrenamientos sostenidos de días. |
| **Costo marginal a largo plazo** | Uso sostenido (un laboratorio, un producto) en Colab Pro/Pro+ acumula costo recurrente; una GPU propia barata corriendo indefinidamente termina siendo más barata. |

**La audiencia real de S³ es más angosta que "cualquiera sin mucha plata".**
Es gente para quien ninguna de esas rutas de nube —ni siquiera la gratis—
está genuinamente disponible. Vale la pena decirlo así de explícito en vez de
dejar la afirmación genérica actual, que un revisor tumba con una sola
pregunta.

---

## 2. Tiempo y costo: local vs nube (números de referencia)

### Local, medido (no estimado)

`results/run_20260714_225107/`: 1,100 ejemplos (hasta 256 tokens c/u),
fine-tuning S³ completo en GTX 1050 → **4.26 h** (258 ejemplos/hora).

Extrapolación lineal (aproximada — el cuello de botella es lectura de disco,
podría no escalar perfecto a mayor escala):

| Tamaño del dataset | Tiempo local (S³, GTX 1050) |
|---|---:|
| 1,100 ejemplos | 4.3 h (medido) |
| 5,000 ejemplos | ~19 h |
| 10,000 ejemplos | ~1.6 días |
| 50,000 ejemplos | ~8-9 días |

Coincide con lo ya anotado en `docs/mejoras_futuras.md`: una época completa
sobre un dataset grande es del orden de semanas.

### Nube rentada (RunPod / Vast.ai / Lambda Labs — no Colab)

⚠️ **Sin verificar en vivo** — el mercado de spot/rental fluctúa por hora.
Confirmar precios actuales antes de presupuestar nada en serio.

- RTX 4090 (24GB): orden de **$0.30-0.50/hora**
- A100 (40-80GB): orden de **$1-2.5/hora**

Con esa VRAM el modelo completo cabe (no hace falta streaming de disco), se
usa LoRA estándar con batches grandes, y el throughput sube de los ~5 tok/s
medidos localmente a cientos o miles de tok/s. El mismo trabajo de 10,000
ejemplos que localmente toma ~1.6 días, rentado probablemente **termina en
menos de una hora**, por **unos pocos dólares**.

A diferencia de Colab gratis, un rental pagado **no tiene límite de sesión
forzado** — corre mientras se pague. El límite real ahí es presupuesto, no
tiempo.

### Veredicto

Si alguien puede pagar 2-5 dólares y tiene acceso a esas plataformas
(tarjeta, verificación, país no restringido), rentar es casi siempre la
decisión más racional para el entrenamiento en sí — la comparación no es
"días contra horas", es **días contra minutos, por unos dólares**. El
"poor compute" real, el que S³ tiene sentido para, es para quien
genuinamente no puede pagar ni eso, o tiene una razón de privacidad/
soberanía de datos que se lo impide.

---

## 3. Recomendación de framing para la introducción/motivación

Reemplazar la afirmación genérica de "barrera de hardware excluye
investigadores de bajos recursos" por una que nombre explícitamente **las
condiciones** bajo las cuales ni la nube gratuita resuelve el problema
(tabla de la sección 1). Eso convierte una afirmación fácil de refutar en
una acotada y defendible.

---

## 3.5. Discrepancia sin resolver: ppl base 11.96 (15 jul) vs 9.52 (18 jul)

Los 2 puntos nuevos de ablation (`nmf_b3`, `no_nmf_h64`, corridos hoy 18/07 a las
11:35) midieron **ppl base = 9.52**, distinto del **~11.96** de las 4 variantes
originales corridas el 15/07 (`results/ablations_capacity_20260715_211509/`).

**Descartado como causa** (verificado, no supuesto):
- Modelo base distinto: **no** — params congelados idénticos en ambas corridas
  (7,615,616,512 exacto, verificado restando `total - trainable` en ambos logs).
- Código de `eval_loss`/`run_ablations.py` distinto: **no** — `git diff` contra
  el último commit solo muestra mi propia adición de las 2 variantes nuevas al
  diccionario `CAPACITY_SUITE`, nada más.
- Versiones de librerías (`torch`/`transformers`/`datasets`/etc.): **no** —
  idénticas en ambos `provenance.json`.
- Dataset (`tatsu-lab/alpaca`): **no** — mismo cache, mismo hash, mismo
  `last modified` en ambos logs.
- El fix de `_ckpt_store`/`_ckpt_load` (`non_blocking` race condition) que
  estaba sin commitear en `s3_trainer.py`: **no** — ese fix solo toca
  `forward_layer_major`/`step_layer_major`; `eval_loss` (lo que usa
  `run_ablations.py` para ppl base/after) no pasa por ahí en absoluto.

**No resuelto.** No tuve tiempo/GPU disponible (el barrido de semillas la
estaba usando) para aislar la causa con una corrida de control. Las corridas
`capacity_seed43` y `capacity_seed44` que el barrido nocturno va a ejecutar
SÍ incluyen las 6 variantes (las 4 originales + `nmf_b3` + `no_nmf_h64`) en
la misma sesión — eso da un cruce apples-to-apples que debería aclarar si el
9.52 o el 11.96 es el valor real, o si varía por sesión. **Revisar esos
resultados antes de dar por buena cualquiera de las dos tablas de ablation
por vías en el paper.**

## 4. Otras honestidades pendientes (ya identificadas, resumen)

No son nuevas — vienen de una revisión anterior de `docs/formalismo*.md` y
quedan aquí para no perderlas de vista antes de someter:

- **PAC-Bayes (Thm 7) da un bound vacuo** (>1) pero se presenta como
  garantía. Hay que atenuar el lenguaje o mover a apéndice/future work.
- **STB (Thm 5-6)**: el bound es real pero minúsculo (κ mejora 0.8%,
  MI≈0.0039·H(X)) y se vende como central. La evidencia fuerte de que STB
  aporta es la empírica (`results/ablation_pathways.md`), no el bound en sí.
- **`formalismo_optimizacion.md`**: EMP-1 dimensionalmente inconsistente,
  TER-3 crece T^{7/2}, GNS-1 cambia de argumento a mitad de la prueba,
  SMV-6 usa un término que diverge. Varias de las "10 optimizaciones
  S3-OPT" nunca estuvieron realmente enganchadas al forward/backward
  (quedaron como scaffolding aceptado-y-logueado, no como optimización real).
- **Recomendación general**: vender el resultado de ingeniería medido (piso
  de VRAM bajado de 8GB a 2-4GB, calidad a la par de LoRA, verificado con
  datos reales) — no el conteo de teoremas. Es lo que aguanta revisión.
