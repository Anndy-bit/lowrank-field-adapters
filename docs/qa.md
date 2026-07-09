# Preguntas y Respuestas — S3

> Preguntas que un experto podria hacer sobre S3 y S3-OPT.

---

## Q: Por que no usar LoRA que es mas simple?

**A:** LoRA entrena matrices A y B de baja rango (r=8). Eso son **16.8M parametros** (0.21% del modelo). S3 entrena solo **4.83M parametros** (0.06%).

Ademas, LoRA necesita guardar gradientes de A y B en GPU durante backward. S3 no — los gradientes van a las capas pequenas de g_theta en SVMO y f_theta en NMF.

LoRA es re-escalado lineal: W + BA. S3 es **deformacion no-lineal continua + modulacion espectral + puente cross-attention**. La diferencia conceptual es significativa.

---

## Q: Como escala a 70B?

**A:** Para 70B parametros:
- Frozen weights: 140GB (en RAM, no VRAM)
- Factores por capa: ~200MB por capa (70 capas)
- Parametros entrenables: ~48M (0.06%)
- VRAM peak: **~2GB** con S3-OPT activo

El layer swap toma mas tiempo (70 capas vs 32), pero el proceso es el mismo. Tiempo estimado: ~8-12 horas por epoch en GTX 1050, 3 epochs = ~30-36 horas. Aceptable.

---

## Q: Cual es el bottleneck principal?

**A:** El layer swap de los factores U, V entre CPU y GPU:
- U_k: (4096, 128) = 8MB
- V_k: (4096, 128) = 8MB  
- Total por capa: ~16MB
- Para 32 capas: ~500MB por forward pass completo

SMV (S3-OPT) reduce esto usando delta-modulation: solo transfiere los cambios en mu, no los factores completos.

---

## Q: No seria mas rapido hacer todo en CPU?

**A:** No, porque:
1. La GPU es ~50x mas rapida que CPU para algebra lineal
2. Solo 1 capa en GPU a la vez significa que el transfer overhead es pequeno (~5% del tiempo total)
3. El training loop seria dominated por compute, no por transfer

El throughput real de S3 en GPU es limitado por swap, no por compute. S3-OPT reduce el overhead de swap.

---

## Q: Por que 128 para el rank SVD?

**A:** Los primeros 128 singular values capturan ~95% de la energia de las matrices de peso de transformers. Es un sweet spot:

| k   | Compression | Energia capturada |
|-----|-------------|-------------------|
| 64  | 64x         | ~88%              |
| 128 | 32x         | ~95%              |
| 256 | 16x         | ~98%              |

DRA (S3-OPT) permite reducir k dinamicamente a 64 cuando s(t) es baixo, alcanzando compression 64x.

---

## Q: NMF no es muy costoso para lo que aporta?

**A:** El overhead es ~2% de los FLOPs por capa (4 pasos RK4 con MLP pequeno). A cambio:

1. Deformacion continua del espacio latente (no un solo salto)
2. Warmstart de TOWS aprovecha coherencia entre tokens
3. TER adapta el numero de pasos a la complejidad del token

El overhead es **paralelo** al forward pass, no secuencial. No hay penalizacion linear.

---

## Q: Frozen = no aprende?

**A:** No. Los 4.83M parametros entrenables aprendem a modular la estructura frozen del modelo.

Es como ensenarle a alguien a hablar un idioma nuevo sin cambiar como funciona su cerebro (frozen), pero si cambia como procesa y deforma la informacion.

SVMO modula los valores singulares (la "escala" de cada direccion).
NMF deforma el espacio latente.
STB conecta ambos mecanismos.

El modelo frozen provee la estructura. Los operadores entrenables aprendem a controlarla.

---

## Q: Como se compara con QLoRA?

**A:** QLoRA usa 4-bit quantization + LoRA:
- Ruido de cuantizacion en los pesos frozen
- LoRA sigue necesitando 16.8M parametros (vs 4.83M de S3)
- S3 preserva precision completa en los pesos frozen

S3 no cuantiza. La adaptacion es pura modulacion espectral sin perdida de precision.

---

## Q: Cual es el error mas grande que la gente hace al pensar en S3?

**A:** Pensar que "es como LoRA pero mas complicado". No lo es.

LoRA es re-escritura lineal: W + BA. S3 es deformacion no-lineal continua + modulacion espectral + puente cross-attention espectral.

Otro error: pensar que frozen = no aprende. Los operadores entrenables aprendem a controlar la estructura frozen de manera sutil pero significativa.

---

*Ultima actualizacion: Julio 2026*
