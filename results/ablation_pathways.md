# Ablation de S³, reportado por VÍAS — 2026-07-16

Reanálisis de datos ya medidos. **No hay ninguna corrida nueva aquí.** Lo que cambia es
la unidad de análisis: vías acopladas en lugar de operadores sueltos.

Fuentes: `ablations_20260715_162039/` · `ablations_capacity_20260715_211509/`
Misma semilla y mismos datos en todas las variantes. Base ppl ≈ 11,96.

---

## 1. Por qué el reporte por operadores es inválido

En `s3_block.py:137-139`:

```python
if enable_stb and enable_svmo:
    self.stb = STBResidual(self.self_attn.q_proj.U_k, self.self_attn.q_proj.k, ...)
```

**STB se construye desde los factores SVD de SVMO.** Consecuencias:

- `svmo_only` **no aísla SVMO** — le quita el operador que consume su base. No mide "SVMO
  solo": mide *la vía espectral rota*.
- `stb_only` **no existe ni puede existir**. No hay simetría entre los operadores.

Leer la tabla por operadores lleva a concluir que SVMO no aporta (Δppl −0,32 con 226K
params; −0,31 con 853K). **Esa conclusión es un artefacto del ablation, no un hecho sobre
SVMO.**

## 2. Las vías

| Vía | Operadores | Acoplamiento |
|---|---|---|
| **Espectral** | SVMO + STB | STB *requiere* SVMO (usa su U_k) |
| **Flujo** | NMF | Independiente |

Descomposición de cada variante medida:

| Variante | Espectral | Flujo | Params | Δppl |
|---|---|---|---:|---:|
| `svmo_only` | ✗ rota (SVMO sin STB) | — | 225.988 | −0,32 |
| `svmo_h64` | ✗ rota (SVMO sin STB) | — | 853.188 | −0,31 |
| `no_nmf` | ✓ **685K** | — | 684.740 | **−5,73** |
| `nmf_b1` | — | ✓ 401K | 401.464 | −5,13 |
| `nmf_b2` | — | ✓ 803K | 802.928 | −5,42 |
| `nmf_only` | — | ✓ 3.212K | 3.211.712 | −6,40 |
| `full_nmf_b1` | ✓ 685K | ✓ 401K | 1.086.204 | **−6,28** |
| `no_stb` | ✗ SVMO solo 226K | ✓ 3.212K | 3.437.700 | −6,42 |
| `full` | ✓ 685K | ✓ 3.212K | 3.896.452 | **−6,65** |

Los params suman exacto en las tres composiciones (685K+401K=1.086K ✓ ·
226K+3.212K=3.438K ✓ · 685K+3.212K=3.896K ✓), lo que permite comparaciones controladas.

---

## 3. Tres tests, tres respuestas

### Test 1 — ¿STB es lo que activa a SVMO?

Añadiendo cada cosa a la **misma** vía de flujo (`nmf_only`, 3.212K, −6,40):

| Añadido | Params extra | Δppl resultante | Ganancia | Eficiencia |
|---|---:|---:|---:|---:|
| SVMO **solo** (`no_stb`) | +225.988 | −6,42 | **+0,02** | 0,09 ppl/M |
| SVMO **+ STB** (`full`) | +684.740 | −6,65 | **+0,25** | 0,36 ppl/M |

**SVMO por su cuenta no aporta nada (+0,02), ni siquiera acompañado de la vía de flujo.
Con STB encima, sí.** La pareja es la unidad indivisible, no los operadores.

Esto llega desde un tercer ángulo independiente y coincide con la teoría: `Δσ =
S_k·α·tanh(g)` con α=0,3 solo **reescala** ±30% los valores singulares existentes. SVMO
está confinado al subespacio singular que ya existe — repondera, no rota, no crea
direcciones nuevas. Su función no es mover σ: es **fabricar la base U_k sobre la que STB
opera**.

### Test 2 — ¿La vía espectral compite con la de flujo?

| Vía | Params | Δppl |
|---|---:|---:|
| **Espectral** (`no_nmf`) | 684.740 | **−5,73** |
| **Flujo** (`nmf_b2`) | 802.928 | −5,42 |

**La vía espectral gana con un 15% menos de parámetros.**

⚠️ La comparación que **no** hay que hacer: `svmo_h64` (853K, espectral *rota*) contra
`nmf_b2`. Enfrenta un método amputado contra uno entero.

### Test 3 — ¿Se complementan? (el experimento controlado)

`full_nmf_b1` tiene **exactamente** la suma de parámetros de ambas vías:

| Config | Params | Δppl |
|---|---:|---:|
| Espectral sola | 684.740 | −5,73 |
| Flujo sola | 401.464 | −5,13 |
| **Ambas** (suma exacta) | **1.086.204** | **−6,28** |

Mejor que cualquiera de las dos por separado. Pero la pregunta real es: *¿mejor que gastar
ese mismo presupuesto en una sola vía?*

Partiendo de flujo a 401K (−5,13), invirtiendo +684.740 params:

| Se invierte en | Δppl final | Ganancia | **Eficiencia** |
|---|---:|---:|---:|
| **Vía espectral** | −6,28 (medido) | +1,15 | **1,68 ppl/M** |
| Más flujo (→1.086K) | ≈−5,63 (interpolado) | +0,50 | 0,73 ppl/M |

**Los parámetros espectrales rinden 2,3× más que gastar lo mismo en más flujo.** Esa es la
sinergia, cuantificada.

> **Honestidad sobre el −5,63:** es interpolación logarítmica entre `nmf_b2` (803K, −5,42)
> y `nmf_only` (3.212K, −6,40). **No existe medición de flujo puro a 1.086K.** Es el hueco
> que cierra el Test 3 sin asteriscos.

### Matiz importante: la ventaja es de presupuesto BAJO

| Presupuesto | Añadir espectral | Añadir más flujo |
|---|---:|---:|
| ~1M (desde flujo 401K) | **1,68 ppl/M** | 0,73 ppl/M |
| ~3,2M (desde flujo 3.212K) | 0,36 ppl/M | 0,41 ppl/M |

A presupuesto alto **empatan**. La vía espectral no domina siempre: domina donde los
parámetros escasean — que resulta ser el régimen del laboratorio.

---

## 4. Lo que falta

| Hueco | Por qué importa | Coste |
|---|---|---|
| **Flujo puro a ~1.086K** | Cierra el Test 3 sin interpolar. Es *el* número de la sinergia | ~35 min |
| **Espectral por encima de 685K** | No hay ni un punto. No se puede trazar su curva | ~35 min |
| **`extra_opts` constantes** | `run_ablations.py` no pasa TER/DRA/MSO/TOWS/HFISC/FDGD/GNS/EMP/SMV; `run_s3_train.py` sí. Los ablations no corren la config del run titular | — |
| **n≥3 semillas** | Todo esto es 1 semilla | ~15 h |

## 5. Qué se puede afirmar hoy

✅ **Sostenible con datos medidos:**
- SVMO sin STB no aporta (+0,02 ppl con 226K params) — verificado por dos vías
- La vía espectral (685K) supera a la de flujo (803K) con 15% menos params
- Combinar ambas vías bate a cualquiera por separado, a suma exacta de params

⚠️ **Sostenible pero con interpolación:**
- Los params espectrales rinden 2,3× más que más flujo, a ~1M de presupuesto

❌ **No sostenible aún:**
- Cualquier afirmación de significancia (n=1)
- Cualquier extrapolación de la vía espectral (un solo punto medido)
