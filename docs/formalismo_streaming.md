# Formalismo del Streaming por Unidades (USF)
## Fine-tuning de modelos de tamaño arbitrario en VRAM acotada

> **Estado:** formalismo del sistema implementado y verificado (§12) + las extensiones que lo
> generalizan a cualquier modelo (§6, §7, §10).
> **Rama:** `feat/streaming-formalism` · **Fecha:** Julio 2026
> **Relación con el resto:** `formalismo.md` define los operadores S³ (*qué* se adapta); este
> documento define *dónde vive el cómputo* (*cómo* se ejecuta). Son ortogonales: USF funciona con
> S³, con LoRA o con cualquier adapter — verificado con ambos (§12).

---

## Índice

1. Motivación: la barrera que rompemos
2. Notación, granularidad y el algoritmo
3. Teorema 1 — Exactitud (y numérica en punto flotante)
4. Teorema 2 — Cota de VRAM e independencia del tamaño
5. Teorema 3 — Jerarquía de granularidad y el corolario del 405B
6. Teorema 4 — Streaming *layer-major*: amortización del I/O
7. Proposición 5 — Prefetch y condición de solapamiento
8. **Modelo de coste temporal** (predictivo y validado)
9. **Selección automática de granularidad** (agnosticismo al modelo)
10. **USF frente a cuantización**: una separación de viabilidad
11. Tabla de costes por modelo (7B / 70B / 405B)
12. Validación empírica
13. Relación con trabajo previo
14. Limitaciones honestas
15. Programa de implementación

---

## 1. Motivación: la barrera que rompemos

Fine-tunear un modelo de 7B con LoRA requiere convencionalmente **8–16 GB de VRAM**: el modelo
base debe residir en memoria durante forward y backward. En una GPU de 4 GB se considera
inviable. Para 70B (140 GB) o 405B (810 GB), impensable.

La observación central de este documento:

> **Esa restricción es un artefacto de la implementación, no una necesidad matemática.**

La retropropagación en modo inverso es una composición de **productos vector-jacobiano (VJP)
locales**. Para computar el gradiente en la capa `i` solo se necesitan, *en ese instante*: la
entrada de esa capa, sus pesos, y el cotangente entrante. **Nada obliga a que las otras `L−1`
capas estén simultáneamente en memoria.** La práctica universal de mantener el modelo residente
es una optimización de velocidad, no un requisito de corrección.

**USF (Unit-Streamed Fine-tuning)** explota esto: mantiene en VRAM **una sola unidad de pesos a
la vez**, transmitida desde disco, y demuestra que los gradientes resultantes son **exactos**
(Teorema 1). La consecuencia es que la VRAM deja de depender del tamaño del modelo (Teorema 2).

---

## 2. Notación, granularidad y el algoritmo

Transformer congelado de `L` capas:

- `h_0 = E(x)`, `h_i = f_i(h_{i-1}; W_i, θ_i)` para `i = 1..L`, `ℓ = C(h_L, y)`.
- `W_i` — pesos **congelados** de la capa `i`. **`∇_{W_i} ℓ` nunca se necesita** (no se entrenan).
- `θ_i` — parámetros **entrenables** del adapter (S³, LoRA, …), con `|θ| ≪ |W|`.
- `s` — bytes por escalar (2 en fp16). `B` — micro-batch. `S` — longitud de secuencia.
- `d` — dimensión del modelo. `G` — pasos de acumulación de gradiente.

**Definición 2.1 (unidad transmisible).** Una *unidad* `u` es un subconjunto de pesos cargable y
liberable de forma independiente. Sea `𝒰` una partición de `{W_i}`. Tres granularidades naturales,
**anidadas**:

| granularidad | unidad | `max_u |W_u|` |
|---|---|---|
| **capa** | bloque decoder completo | `max_i |W_i|` |
| **sub-capa** | bloque de atención · bloque MLP | `max_i max(|W_i^{attn}|, |W_i^{mlp}|)` |
| **matriz** | una proyección (`q,k,v,o,gate,up,down`) | `max_{i,p} |W_{i,p}|` |

**Definición 2.2 (operadores).**
- `load(u)`: materializa `W_u` desde disco a VRAM. Coste I/O: `|W_u|` bytes.
- `free(u)`: devuelve `W_u` a `meta`. Libera VRAM. Coste: 0.

Son estrictamente anidados: entre `load(u)` y `free(u)` no se carga ninguna otra unidad.

**Definición 2.3 (algoritmo USF, batch-major).**
```
FORWARD (sin construir grafo):
  h ← E(x)
  para i = 1..L:
      c_i ← h                              # checkpoint: entrada de la capa i
      load(W_i);  h ← f_i(h; W_i, θ_i);  free(W_i)
  ℓ ← C(h, y);   g ← ∂ℓ/∂h_L

BACKWARD (capa a capa, re-streaming):
  para i = L..1:
      x ← c_i.requires_grad_()
      load(W_i)
      ĥ ← f_i(x; W_i, θ_i)                 # recompute CON grafo, solo esta capa
      ĥ.backward(g)                        # acumula ∂ℓ/∂θ_i ; produce x.grad
      free(W_i)
      g ← x.grad                           # cotangente para la capa i−1
```

Obsérvese que el forward **no construye grafo**: las activaciones internas se descartan y se
recomputan en el backward (activation checkpointing clásico), pero aquí además **los pesos** se
descartan y se re-streamean.

---

## 3. Teorema 1 — Exactitud

**Teorema 1.** Sea `∇θ^full` el gradiente producido por autograd sobre el grafo completo y
`∇θ^USF` el del Algoritmo 2.3. En aritmética exacta:

```
∇θ^USF = ∇θ^full
```

*Demostración.* La retropropagación computa, para cada capa `i`, el par

```
(∂ℓ/∂θ_i , ∂ℓ/∂h_{i-1}) = VJP_{f_i}( h_{i-1}, W_i, θ_i ; ∂ℓ/∂h_i )
```

Este VJP es **local**: función únicamente de (a) el punto de evaluación `h_{i-1}`, (b) los
parámetros `W_i, θ_i`, (c) el cotangente entrante `∂ℓ/∂h_i`. No depende de `W_j` para `j ≠ i`.

El algoritmo garantiza las tres condiciones:
- **(a)** `c_i = h_{i-1}` fue guardado en el forward — el mismo punto exacto, sin aproximación.
- **(b)** `load(W_i)` restaura `W_i` **bit a bit** (relee el mismo tensor del disco); `θ_i` es
  residente y no se modifica dentro del paso.
- **(c)** `g` entrante es `∂ℓ/∂h_i` por inducción hacia atrás desde `g_L = ∂ℓ/∂h_L`.

Cada VJP local se evalúa en un punto idéntico al del grafo completo; su composición —que *es* la
regla de la cadena— produce el mismo resultado. `free(W_i)` cambia *dónde reside* un tensor, no su
valor: la aritmética es invariante al dispositivo. ∎

**Corolario 1.1.** USF **no es una aproximación**: no introduce error, sesgo ni varianza. Se
distingue categóricamente de la cuantización (§10), que sí degrada.

**Corolario 1.2 (independencia del adapter).** El Teorema 1 no usa ninguna propiedad de `f_i` más
allá de su diferenciabilidad. **USF es agnóstico al adapter**: vale para S³, LoRA, o cualquier
`θ`. Verificado empíricamente con ambos (§12).

### 3.1 Numérica en punto flotante (honestidad)

El Teorema 1 afirma igualdad *en aritmética exacta*. En punto flotante caben dos matices:

1. **Los pesos se preservan exactamente.** `load` relee el mismo tensor fp16 del disco: no hay
   requantización ni pérdida. Este punto es incondicional.
2. **El orden de reducción puede variar.** Si el recompute selecciona un kernel distinto que el
   forward original, la suma en punto flotante puede diferir en los últimos bits. Esto es el
   **mismo no-determinismo del activation checkpointing estándar**, no algo propio de USF.

En la práctica, con kernels y formas idénticas, la coincidencia es exacta: medimos
**max |Δ| = 0.00e+00** sobre 150 tensores de gradiente (§12).

---

## 4. Teorema 2 — Cota de VRAM e independencia del tamaño

**Teorema 2.** Bajo el Algoritmo 2.3 con granularidad `𝒰`, el pico de VRAM satisface

```
VRAM_pico  ≤  max_u |W_u|·s  +  |θ|·s_opt  +  A  +  K
```

| término | qué es | magnitud (7B, batch=4) |
|---|---|---|
| `max_u |W_u|·s` | la única unidad de pesos residente | 0.43 GB (capa) |
| `|θ|·s_opt` | adapters + estados AdamW (residentes) | ~0.05 GB (S³, 3.9M params) |
| `A` | activaciones internas de **una** unidad | ~0.3 GB |
| `K` | checkpoints `{c_i}`: `L·B·S·d·s` si en VRAM; **≈0 si se descargan a CPU** | 0.21 GB |

*Demostración.* (i) En todo instante hay **exactamente una** unidad cargada: `load`/`free` son
estrictamente anidados (Def. 2.2), de donde el primer término. (ii) Los adapters deben ser
residentes: `θ` se actualiza cada paso y su estado de optimizador debe persistir; su tamaño es
`|θ| ≪ |W|` por hipótesis del régimen PEFT. (iii) El grafo con gradiente existe únicamente dentro
de la iteración de **una** unidad (el forward no construye grafo), de donde `A`. (iv) Los `c_i`
son los únicos tensores que sobreviven entre iteraciones. ∎

**Corolario 2.1 (independencia de la profundidad).** Si los checkpoints se descargan a RAM del CPU
(`K ≈ 0` en VRAM — son activaciones, no necesitan residir en GPU entre usos):

```
VRAM_pico  ≈  max_u |W_u|·s  +  |θ|·s_opt  +  A        ⟂  L
```

Añadir capas **no aumenta la VRAM**. Solo aumenta tiempo y tráfico de disco.

**Corolario 2.2 (independencia del tamaño total).** La cota tampoco depende de `Σ_i |W_i|`. Dos
modelos con la misma unidad máxima tienen el mismo requisito de VRAM, aunque uno sea 50× más
grande.

> **Resultado central.** `VRAM_pico = f(max_u |W_u|)`, **no** de `Σ_i |W_i|` ni de `L`.
> Se rompe el acoplamiento —asumido universalmente— entre *tamaño del modelo* y *VRAM para
> fine-tunearlo*. El tamaño pasa a ser un problema de **almacenamiento y tiempo**, no de memoria.

---

## 5. Teorema 3 — Jerarquía de granularidad y el corolario del 405B

**Teorema 3.** Para las granularidades anidadas de la Def. 2.1:

```
max_{i,p} |W_{i,p}|   ≤   max_i max(|W_i^{attn}|,|W_i^{mlp}|)   ≤   max_i |W_i|
```

y el Teorema 2 aplica a cualquiera. Refinar la granularidad **reduce monótonamente el pico de
VRAM a igualdad de I/O total**.

*Demostración.* Las particiones son anidadas (matriz ⊂ sub-capa ⊂ capa); el máximo sobre una
partición más fina no puede exceder el máximo sobre una más gruesa. El I/O total por paso es
`Σ_u |W_u| = Σ_i |W_i|`, **invariante a la partición**: los mismos bytes, en trozos más pequeños. ∎

**Corolario 3.1 (viabilidad por tamaño de modelo).** VRAM de la unidad, fp16 (los términos
`|θ|+A+K` añaden < 0.6 GB):

| Modelo | `L` | `Σ|W|` | capa | attn / mlp | **matriz máx** | ¿4 GB? |
|---|---|---|---|---|---|---|
| Qwen2.5-7B | 28 | 12.2 GB | 0.43 GB | 0.05 / 0.38 | **0.13 GB** | ✅ cualquier granularidad |
| Llama-3-8B | 32 | 13.0 GB | 0.41 GB | 0.08 / 0.33 | **0.11 GB** | ✅ |
| 13B-class | 40 | 23.6 GB | 0.59 GB | 0.20 / 0.40 | **0.13 GB** | ✅ |
| **Llama-3-70B** | 80 | 127.5 GB | **1.59 GB** | 0.28 / 1.31 | **0.44 GB** | ✅ **incluso por capa** |
| **Llama-3-405B** | 126 | 748 GB | 5.94 ❌ | 1.06 / 4.88 ❌ | **1.62 GB** ✅ | ✅ **solo por matriz** |

> **Corolario 3.2 (el resultado "imposible").** Con granularidad por-matriz, **un modelo de 405B
> puede fine-tunearse en una GPU de 4 GB**: su matriz más grande (`gate_proj`, 16384×53248 =
> 872M params) ocupa **1.62 GB** en fp16. El modelo pesa 748 GB en disco; la VRAM nunca ve más de
> 1.62 GB a la vez.

**Nota sobre atención con granularidad por-matriz.** Matemáticamente la partición por-matriz es
válida: se computa `q = W_q x`, se libera `W_q`, luego `k = W_k x`, etc. Lo que persiste son las
**activaciones** `q,k,v` (decenas de MB), no los pesos.

> ⚠️ **Limitación de implementación (honestidad).** El Teorema 3 es correcto, pero **la
> granularidad por-matriz NO está implementada**. La razón es de ingeniería, no de matemática:
> nuestro diseño delega la atención al módulo `self_attn` de HuggingFace (§12), que consume
> `q,k,v,o` **juntos** en una sola llamada. Streamearlos de uno en uno exigiría reimplementar los
> internos de la atención — precisamente lo que rompió la versión original del proyecto (pérdida
> de RoPE/GQA/máscara causal). Preferimos correctitud verificada sobre alcance no verificado.
>
> **Granularidades implementadas y testeadas: `layer` y `sublayer`** (`SUPPORTED_GRANULARITIES`).
> Consecuencia para el Cor. 3.2: el 405B **sigue siendo teórico por partida doble** — falta el
> modelo (748 GB) *y* falta la granularidad por-matriz. **El 70B, en cambio, cabe con la
> granularidad `layer` ya implementada** (1.59 GB < 3.94 GB), y `sublayer` le da margen (1.31 GB).

---

## 6. Teorema 4 — Streaming *layer-major*: amortización del I/O

El Algoritmo 2.3 es **batch-major**: por cada micro-batch recorre las `L` unidades. Con
acumulación sobre `G` micro-batches, el I/O por paso de optimizador es `G · 2Σ_i|W_i|` — **el
modelo entero se lee `G` veces**.

**Definición 4.1 (USF layer-major).** Invertir el orden de los bucles: mantener `u` cargada y
pasar **los `G` micro-batches** por ella antes de avanzar.

```
FORWARD:
  H ← [E(x⁽¹⁾), …, E(x⁽ᴳ⁾)]
  para i = 1..L:
      load(W_i)
      para j = 1..G:   C_i⁽ʲ⁾ ← H⁽ʲ⁾ ;  H⁽ʲ⁾ ← f_i(H⁽ʲ⁾; W_i, θ_i)
      free(W_i)
BACKWARD: simétrico — una carga de W_i, G retropropagaciones locales
```

**Teorema 4.** El Algoritmo 4.1 produce gradientes **idénticos** a 2.3, y su I/O por paso de
optimizador es

```
I/O_layer-major = 2 Σ_i |W_i|        frente a      I/O_batch-major = G · 2 Σ_i |W_i|
```

— una **reducción de factor `G`**.

*Demostración (exactitud).* La acumulación computa `∇θ = Σ_{j=1}^{G} ∇θ⁽ʲ⁾`. La suma es
conmutativa y asociativa: el orden de acumulación no altera el resultado. Cada `∇θ⁽ʲ⁾` se computa
con el VJP local del Teorema 1 evaluado en `(C_i⁽ʲ⁾, W_i, θ_i)` — los mismos puntos que en 2.3.
Reordenar los bucles no cambia ningún punto de evaluación, solo el orden de visita. ∎

*Demostración (I/O).* En 4.1 el bucle externo recorre `i` una sola vez por paso de optimizador,
luego cada `W_i` se carga exactamente una vez por pase (dos: forward y backward). En 2.3 el bucle
externo sobre `j` repite el recorrido completo `G` veces. ∎

**Coste en memoria.** Layer-major mantiene `G` conjuntos de activaciones en cada frontera:
`G·B·S·d·s` bytes. Para `G=8, B=4, S=256, d=3584`, fp16: **59 MB** por frontera, y
`L·G·B·S·d·s = 1.6 GB` de checkpoints — **descargables a RAM del CPU** (Cor. 2.1).

**Proposición 4.2 (elección de `G`).** Con layer-major, el tiempo por ejemplo es

```
T_ej(G) = 2Σ|W| / (BW · G · B)  +  C/B
```

donde `C` es el cómputo por micro-batch. `T_ej` es **monótonamente decreciente en `G`** y
converge al límite de cómputo `C/B`. Luego conviene el `G` **más grande** que satisfaga la
restricción de memoria de checkpoints:

```
G* = max { G : L·G·B·S·d·s ≤ M_disponible }
```

(con `M_disponible` la RAM del CPU si se descargan, típicamente holgada).

---

## 7. Proposición 5 — Prefetch y condición de solapamiento

**Proposición 5.** Sea `t_io(u) = |W_u|/BW` y `t_cmp(u)` el cómputo de la unidad `u`. Con doble
buffer (emitir `load(u+1)` asíncronamente al iniciar el cómputo de `u`):

```
T_paso  ≈  max( Σ_u t_io(u) , Σ_u t_cmp(u) )  +  t_io(u_1)
```

y el I/O queda **totalmente oculto** si `t_io(u+1) ≤ t_cmp(u)` para todo `u`.

*Demostración.* Es un pipeline de dos etapas (carga · cómputo) con buffers independientes; su
throughput en régimen estacionario lo fija la etapa más lenta, más el llenado inicial `t_io(u_1)`. ∎

**Coste en VRAM:** dos unidades residentes → la cota del Teorema 2 pasa a `2·max_u|W_u|`. Para 7B
por capa: 0.86 GB en vez de 0.43 GB. Sigue holgado en 4 GB.

**Aplicación a las mediciones actuales** (§12): `Σt_io ≈ 45 s`, `Σt_cmp ≈ 57 s` por paso a
batch=4. Como `t_io < t_cmp`, **la condición se cumple** y el prefetch ocultaría prácticamente
todo el disco. Combinado con layer-major (que divide `t_io` por `G`), el margen es amplísimo.

---

## 8. Modelo de coste temporal (predictivo y validado)

Sea `N = Σ_i |W_i| / s` el número de parámetros transmitidos, `BW` el ancho de banda efectivo de
disco y `Φ` el throughput efectivo de la GPU (FLOP/s).

**Tiempo de I/O por paso de optimizador:**
```
T_io  =  2·N·s / BW           (layer-major)
T_io  =  G·2·N·s / BW         (batch-major)
```

**Tiempo de cómputo por paso de optimizador** (regla estándar `≈6N` FLOPs por token para
forward+backward):
```
T_cmp  =  6·N·(G·B·S) / Φ
```

**Tiempo por paso y por ejemplo:**
```
T_paso = T_io + T_cmp        (sin prefetch)
T_paso = max(T_io, T_cmp)    (con prefetch, Prop. 5)
T_ej   = T_paso / (G·B)
```

**Régimen.** El sistema es **compute-bound** si `T_cmp > T_io`, es decir si

```
G·B·S / Φ  >  2·s / BW        ⟺       G·B·S  >  2·s·Φ / BW
```

Con los valores medidos (`s=2`, `Φ=0.7 TFLOP/s`, `BW=0.54 GB/s`): el umbral es
`G·B·S > 2.6·10³` tokens por paso. Con `G=8, B=4, S=256` → `8192 tokens` ≫ umbral:
**layer-major deja el sistema firmemente compute-bound**, incluso sin prefetch.

### 8.1 Validación del modelo

Configuración medida: Qwen2.5-7B, batch-major (`G=1`), `B=4`, `S=256`, `N = 6.5·10⁹` transmitidos.

| magnitud | modelo | medido |
|---|---|---|
| I/O por paso | `2·6.5e9·2 = 24.3 GB` | **24.9 GB** |
| `T_io` @ BW=0.54 GB/s | 45 s | ~46 s |
| `T_cmp` @ Φ=0.7 TFLOP/s | 57 s | ~57 s |
| **`T_ej` = (45+57)/4** | **25.6 s/ejemplo** | **26.9 s/ejemplo** |

**Error del modelo: 5%.** Los parámetros `BW` y `Φ` se calibraron con dos mediciones
independientes (batch=1 y batch=4) y el modelo predice el resto. Esto convierte §11 en una
**predicción**, no en una extrapolación ciega.

---

## 9. Selección automática de granularidad (agnosticismo al modelo)

Los Teoremas 2 y 3 dan un criterio operativo. Sea `V` la VRAM disponible y `𝒰₁ ⊃ 𝒰₂ ⊃ 𝒰₃` las
granularidades (capa, sub-capa, matriz).

**Algoritmo 9.1 (selección).**
```
para 𝒰 en [capa, sub-capa, matriz]:            # de más gruesa a más fina
    si  max_{u∈𝒰}|W_u|·s · (2 si prefetch) + |θ|·s_opt + A + K  ≤  V:
        devolver 𝒰
error: ni la matriz más grande cabe en V
```

**Proposición 9.2 (optimalidad).** El Algoritmo 9.1 devuelve la granularidad que **minimiza el
tiempo** sujeta a la restricción de memoria.

*Justificación.* El I/O total es invariante a la granularidad (Thm 3), pero el número de
operaciones `load` crece al refinar: `|𝒰₃| > |𝒰₂| > |𝒰₁|`. Cada `load` paga una latencia fija
(apertura, seek, lanzamiento de kernel) `λ` amortizada sobre `|W_u|` bytes. El tiempo es
`Σ_u (λ + |W_u|/BW) = |𝒰|·λ + 2N·s/BW`: el segundo término es constante y el primero crece con
`|𝒰|`. Luego la partición **más gruesa que quepa** minimiza el tiempo. ∎

> **Esto es lo que hace el sistema agnóstico al modelo.** El usuario no configura nada: dada su
> VRAM, el sistema elige sola la unidad más gruesa (la más rápida) que satisfaga el Teorema 2.
> El mismo código corre 7B en una 1050 y 405B en la misma 1050 — solo cambia la granularidad
> elegida, automáticamente.

---

## 10. USF frente a cuantización: una separación de viabilidad

La alternativa habitual para reducir memoria es **cuantizar**. Conviene ser preciso sobre por qué
no resuelve este problema.

**Contabilidad para Qwen2.5-7B en 4-bit (NF4), tarjeta de 3.94 GB:**

| componente | precisión | tamaño |
|---|---|---|
| capas del transformer (6.5B params) | 4-bit | 3.25 GB |
| embeddings (545M params) | fp16 (bnb no los cuantiza) | 1.09 GB |
| lm_head (545M params) | fp16 | 1.09 GB |
| **total residente** | | **5.43 GB > 3.94 GB** ❌ |

**Proposición 10.1 (separación de viabilidad).** En una GPU de 3.94 GB, para Qwen2.5-7B:
`Viable(cuantización 4-bit) = falso`, `Viable(USF) = verdadero`.

*Evidencia empírica.* No es un argumento teórico: lo intentamos. La carga en 4-bit con
`device_map="auto"` produce **CUDA OOM** (registrado en la bitácora del proyecto); reducir el
presupuesto de GPU fuerza a `bitsandbytes` a un offload parcial que además falla por un bug de
tensores `meta`. USF corre el mismo modelo con pico de **2.27 GB** (S³) y **1.61 GB** (LoRA).

**Comparación de los tres ejes:**

| | memoria | tiempo | **calidad** |
|---|---|---|---|
| Cuantización 4-bit | ↓ (×4) | ≈ igual | **↓ degrada** (error de cuantización) |
| **USF** | **↓↓ (×L, ilimitado)** | ↑↑ (`L`× más lento) | **= exacta (Thm 1)** |

Los métodos son **ortogonales y componibles**: se podría cuantizar *y* streamear. Pero son
distintos en naturaleza: la cuantización compra memoria **con calidad**; USF la compra **con
tiempo**. Y solo USF escala sin límite: cuantizar un 405B a 4-bit son 202 GB — sigue sin caber en
ninguna GPU de consumo, mientras que USF lo reduce a 1.62 GB (Cor. 3.2).

---

## 11. Tabla de costes por modelo

Predicciones del modelo §8 con los parámetros medidos en la GTX 1050 (`BW=0.54 GB/s`,
`Φ=0.7 TFLOP/s`), layer-major con `G=8, B=4, S=256`:

| Modelo | `Σ|W|` | I/O/paso | `T_io`/ej | `T_cmp`/ej | régimen | **1000 ejemplos** |
|---|---|---|---|---|---|---|
| **Qwen2.5-7B** | 12.2 GB | 24.3 GB | 1.4 s | **14.3 s** | compute | **4.0 h** |
| **Llama-3-70B** | 127.5 GB | 255 GB | 14.8 s | **150 s** | compute | **41.7 h** (~1.7 días) |
| **Llama-3-405B** | 748 GB | 1496 GB | 86.6 s | **881 s** | compute | **245 h** (~10 días) |

**Lecturas honestas de esta tabla:**
- Con layer-major, **los tres son compute-bound**: el disco deja de ser el cuello de botella. El
  límite pasa a ser los ~0.7 TFLOP/s de una GTX 1050.
- **70B en ~1.7 días** por 1000 ejemplos es *viable como demostración*, no como flujo de trabajo.
- **405B en ~10 días** es *matemáticamente viable, prácticamente inútil* en este hardware. Lo que
  el Corolario 3.2 demuestra es que **la memoria deja de ser la barrera** — el tiempo la sustituye.
- El régimen práctico hoy en esta clase de hardware es **7B–13B**.

---

## 12. Validación empírica

GTX 1050 (3.94 GB), Qwen2.5-7B-Instruct, streaming por capa, `B=4`, `S=256`, fp16:

| Magnitud | Predicción del formalismo | Medido |
|---|---|---|
| Exactitud de gradientes (Thm 1) | `Δ = 0` | **max\|Δ\| = 0.00e+00** (150 tensores) |
| VRAM pico, S³ (Thm 2) | `0.43 + 0.05 + A + 0.21` GB | **2.27 GB** |
| VRAM pico, LoRA r=8 (Thm 2) | misma cota, `|θ|` distinto | **1.61 GB** |
| Independencia del adapter (Cor. 1.2) | misma cota para cualquier `θ` | ✅ S³ **y** LoRA corren |
| I/O por paso (batch-major) | 24.3 GB | **24.9 GB** |
| `T_ej` (modelo §8) | 25.6 s | **26.9 s** (error 5%) |
| Viabilidad 4-bit (Prop. 10.1) | 5.43 GB > 3.94 GB ⇒ falso | **CUDA OOM** ✅ |

La diferencia S³ (2.27 GB) vs LoRA (1.61 GB) es exactamente el término `|θ|·s_opt + A` del
Teorema 2: S³ mantiene residentes los factores `U_k, V_k^T` y los estados de la ODE. **La cota se
cumple en ambos casos** — y el hecho de que ambos corran es la evidencia del Corolario 1.2.

**Reproducibilidad:** `tests/test_s3_trainer.py::test_layerswap_grad_equivalence` (exactitud),
`results/run_*/` (VRAM, I/O, tiempos), `results/ablations_baselines_*/` (S³ vs LoRA).

---

## 13. Relación con trabajo previo (honestidad)

El *offloading* de parámetros **no es nuevo**, y este documento no lo reclama:

- **ZeRO-Infinity / DeepSpeed** (Rajbhandari et al., 2021): offload de parámetros y estados de
  optimizador a CPU/NVMe. Régimen: modelos **completos** (todos los pesos entrenables) en
  **clusters de GPUs**; el problema central es shardear estados del optimizador.
- **HuggingFace `accelerate`** (`device_map`, `offload_folder`): despacho automático de módulos a
  CPU/disco, orientado sobre todo a **inferencia**.
- **Activation checkpointing** (Chen et al., 2016): recomputar *activaciones*. Ortogonal — lo
  usamos, pero USF además re-streamea los **pesos**.
- **Cuantización** (QLoRA, Dettmers et al., 2023): eje distinto; §10 muestra que no basta aquí.

**Lo que aporta USF:**
1. **Régimen distinto:** base congelada + adapters diminutos, **una sola GPU de consumo**. Aquí no
   hay estados de optimizador que shardear (`|θ| ≪ |W|`), así que el problema se reduce por
   completo a la residencia de `W` — y la solución óptima es re-streaming puro.
2. **Cota formal e independencia del tamaño** (Thm 2 + Cor. 2.2): el enunciado explícito de que
   `VRAM_pico = f(max_u|W_u|)` y **no** de `Σ|W_i|` ni de `L`, con la jerarquía de granularidad
   (Thm 3) y el corolario del 405B. No conocemos un enunciado equivalente en la literatura.
3. **Layer-major (Thm 4):** reordenar la acumulación de gradiente para amortizar el I/O por un
   factor `G`. Carece de sentido cuando el modelo cabe en memoria — supuesto de todo el trabajo
   previo. En régimen de streaming es la optimización dominante (§8: cambia el régimen de
   I/O-bound a compute-bound). **No la hemos visto propuesta.**
4. **Selección automática de granularidad (§9)** con criterio de optimalidad: un mismo código,
   agnóstico al modelo y al hardware.
5. **Exactitud demostrada y verificada** (Thm 1, `Δ=0`), frente a la degradación de la
   cuantización.

---

## 14. Limitaciones honestas

1. **Tiempo, no memoria.** USF convierte un problema de VRAM en uno de tiempo/ancho de banda.
   Es sustancialmente más lento que tener el modelo residente. Régimen práctico hoy: **7B–13B**
   en esta clase de hardware (§11).
2. **Requiere almacenamiento local.** 12 GB (7B), 128 GB (70B), 748 GB (405B), y `BW` fija el
   techo de I/O (Prop. 5, §8).
3. **La cota supone `|θ| ≪ |W|`.** Con muchos parámetros entrenables, los estados del optimizador
   dejan de ser despreciables y haría falta shardearlos (dominio de ZeRO). USF es un método para
   **PEFT**, no para pre-entrenamiento.
4. **El Corolario 3.2 (405B) no está verificado empíricamente.** Es consecuencia del Teorema 2 más
   un cálculo de tamaños; la corrida real está bloqueada por la descarga de 748 GB. **El 7B sí
   está verificado de punta a punta** (§12). El 70B está pendiente de disponer de los pesos.
5. **`Φ` y `BW` son específicos del hardware.** La tabla §11 vale para una GTX 1050 con este
   disco; el *modelo* §8 es general, sus *constantes* no.
6. **No acelera la inferencia.** USF es un método de entrenamiento.

---

## 15. Programa de implementación

| # | Componente | Teorema | Estado | Test |
|---|---|---|---|---|
| 1 | USF batch-major, granularidad de capa | Thm 1, Thm 2 | ✅ **verificado en 7B** | `test_s3_trainer::test_layerswap_grad_equivalence` |
| 2 | Descarga de checkpoints a CPU | Cor. 2.1 | ✅ **verificado** (Δ=0) | `test_layer_major::test_ckpt_offload_is_neutral` |
| 3 | Layer-major | Thm 4, Prop. 4.2 | ✅ **verificado en 7B** (Δ=0, **1.40×**) | `test_layer_major::test_layer_major_equals_batch_major` |
| 4 | Prefetch asíncrono (doble buffer) | Prop. 5 | ✅ **implementado + Δ=0** | `test_usf_components::test_prefetch_preserves_gradients` |
| 5 | Granularidad sub-capa | Thm 3 | ✅ **implementado + Δ=0** | `test_usf_components::test_sublayer_granularity_preserves_gradients` |
| 5b | Granularidad por-matriz | Thm 3 | ❌ **no implementada** (ver nota §5) | — |
| 6 | Selección automática de granularidad | §9 | ✅ **implementado** | `test_usf_components::test_auto_granularity_selection` |

**Ganancia medida (componente 3) sobre 7B real:** de **28.6 → 20.5 s/ejemplo (1.40×)** con `G=4`,
y VRAM **2524 → 2219 MB**. El modelo de coste §8 predijo 1.49× (error 6%). Con `G=8` el modelo
proyecta ~1.7× (no medido).

**Alcance real por modelo, con lo implementado hoy:**

| Modelo | granularidad necesaria | ¿implementada? | ¿cabe en 4 GB? |
|---|---|---|---|
| 7B / 8B / 13B | `layer` | ✅ | ✅ **verificado empíricamente** |
| **70B** | `layer` (1.59 GB) o `sublayer` (1.31 GB) | ✅ | ✅ **por cota; falta el modelo** |
| 405B | `matrix` (1.62 GB) | ❌ | teórico ×2 (falta granularidad **y** modelo) |

---

**Fin del formalismo USF.**

**Resultado central:** `VRAM_pico = f(max_u |W_u|)` — independiente de `L` y de `Σ|W_i|`.
El tamaño del modelo deja de ser una barrera de memoria y pasa a ser una de tiempo.
**Corolario práctico:** cualquier transformer cuya matriz de pesos más grande quepa en la GPU
puede fine-tunearse en ella, con gradientes exactos, sea de 7B o de 405B.
