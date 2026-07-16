# Mejoras Futuras — S³

> Ideas fuera del alcance del paper actual, con la evidencia experimental que las motiva.
> Estado: documentadas, **no implementadas**. Candidatas a "Future Work".
> Fecha: Julio 2026.

---

## 0. Por qué existen estas ideas — la evidencia

Las ablaciones (`results/ablations_20260715_162039/` y `results/ablations_capacity_20260715_211509/`)
revelaron tres hechos que apuntan todos en la misma dirección:

**Hecho 1 — NMF gana por diseño, no por tamaño.**
A presupuesto de parámetros prácticamente idéntico:

| variante | params | Δ perplejidad |
|---|---|---|
| `nmf_b2` (solo NMF) | 802 928 | **−5.42** |
| `svmo_h64` (solo SVMO) | 853 188 | **−0.31** |

NMF mejora ~17× más con los mismos parámetros. El confound de capacidad queda descartado.

**Hecho 2 — SVMO solo es arquitectónicamente débil.**
Incluso con 0.85M params, SVMO apenas mueve la perplejidad (11.96 → 11.65). Razón estructural:
con `U` y `V` congelados, SVMO **solo puede reescalar direcciones que ya existen** (±α), nunca crear
direcciones nuevas. Su techo es bajo por construcción, no por falta de capacidad.

**Hecho 3 — la sinergia es real, y vive en el régimen frugal.**

| variante | params | Δ perplejidad |
|---|---|---|
| `nmf_b1` (NMF pequeño solo) | 0.40M | −5.13 |
| `full_nmf_b1` (los 3, NMF pequeño) | 1.09M | **−6.28** |
| `nmf_only` (NMF grande solo) | 3.21M | −6.40 |
| `full` (los 3, NMF grande) | 3.90M | −6.65 |

- Añadir SVMO+STB a un NMF **pequeño**: **+1.15 ppl**.
- Añadir SVMO+STB a un NMF **grande**: solo +0.25 ppl.
- `full_nmf_b1` logra ppl 5.67 con **1/3 de los parámetros** de `nmf_only` (ppl 5.55).

**Conclusión:** los operadores se complementan de verdad, pero el beneficio aparece cuando el
presupuesto es mínimo — justo el régimen que persigue el paper. Cuando NMF es grande, hace todo
solo y los demás sobran. Esto sugiere que **un acoplamiento más fuerte** entre operadores podría
amplificar la sinergia. De ahí las ideas siguientes.

---

## 1. NMF guiado por el espectro (acoplamiento suave)

**Idea.** Hoy el campo de velocidad de la Neural-ODE, `f_θ(h, t)`, deforma `h` a ciegas: no sabe
nada de la estructura espectral de los pesos que procesarán esa representación. Propuesta: darle
la firma espectral como entrada adicional,

```
f_θ(h, t, s)   con   s = U_kᵀ h   (o s concatenado a [h; t])
```

Así SVMO señala "estas direcciones son importantes" y NMF fluye en consecuencia. Los dos
operadores compartirían información real en cada paso de integración, no solo a través del
puente STB.

**Coste.** Bajo — es cambiar la entrada del `VelocityField` de `d+1` a `d+1+k`.
**Riesgo.** Bajo. Añade ~`k·d_b` params por flujo.
**Qué probaría.** Si la sinergia sube respecto a `full_nmf_b1` (−6.28), el acoplamiento explícito
funciona y SVMO deja de ser un pasajero.

---

## 2. NMF que fluye DENTRO de las coordenadas espectrales ⭐ (la más prometedora)

**Idea.** En vez de que la ODE deforme `h` en el espacio crudo (`d = 3584`), que opere sobre la
proyección espectral `s = U_kᵀ h` (`k = 128`):

```
ds/dτ = f_θ(s, τ)      integrar en el subespacio top-k
h_out = h + U_k · (s(T) − s(0))
```

Entonces **SVMO y NMF operan sobre el mismo objeto**: SVMO escala los ejes del espectro, NMF
fluye entre ellos. Acoplamiento total por construcción, no por un puente añadido.

**Ventajas.**
- Frugalidad radical: la ODE vive en dimensión 128 en vez de 3584 → NMF con ~28× menos params.
- Ataca exactamente donde la sinergia demostró que paga (régimen frugal, Hecho 3).
- Unifica la geometría: los tres operadores comparten la base `U_k`.

**Riesgo honesto.** Al restringir el flujo a `span(U_k)`, NMF pierde expresividad: ya no puede
crear direcciones fuera del subespacio top-k. Dado que la energía espectral retenida con k=128
es solo ~25% (ver `svd_factors/metadata.json`), podría degradar la calidad. **Es justo el
experimento que hay que correr** — y un resultado negativo también es informativo: demostraría
que NMF *necesita* expresividad de espacio completo, lo que justifica su diseño actual.

**Implementación sugerida.** Añadir una variante `nmf_spectral` a `run_ablations.py` para
compararla de tú a tú contra `full_nmf_b1` y `nmf_b1` con los mismos datos y semilla.

---

## 3. Un solo operador: "flujo espectral unificado" (la apuesta grande)

**Idea.** Fusionar S³ en **un** operador coherente en vez de tres módulos: una Neural-ODE cuyo
estado son los coeficientes espectrales, cuyo campo de velocidad está condicionado por los valores
singulares modulados (SVMO), integrada a través de los `U`/`V` congelados. Los "tres operadores"
pasarían a ser **tres facetas de un mismo flujo**.

**Por qué importa.** Cambiaría la tesis del paper de *"tres adapters acoplados"* a
*"S³ no son tres adapters — es un flujo espectral unificado"*, que es una historia mucho más
fuerte y difícil de atacar (ya no hay "operadores que aportan poco": hay un solo mecanismo).

**Coste/riesgo.** Alto. Es una arquitectura nueva: necesitaría su propia ablación, baselines y
validación completa. **Es otro paper**, no una sección de este.

---

## 4. Otras mejoras pendientes (menores)

- **Ablación de SVMO con k > 128.** Los factores SVD en disco están calculados a k=128, así que
  `svmo_k=256` queda capado. Requeriría recomputar el SVD (`make svd K=256`) para probar si más
  rango espectral ayuda a SVMO.
- **SVMO como vector δ en vez de MLP** (`docs/correcciones.md` §4.1): aprender `δ ∈ R^k` directo
  (k params) en vez del MLP `g_θ` (H² params). Perfil de capacidad distinto, mucho más barato.
- **Prefetch asíncrono de capas** (double buffering CPU→GPU): mientras la GPU computa la capa `i`,
  cargar la `i+1` en otro stream. Atacaría el cuello de botella de I/O (~25 GB/paso).
- **Cachear el modelo en RAM.** Con ≥32 GB de RAM el 7B fp16 (14 GB) cabría en memoria y el swap
  sería RAM→GPU en vez de disco→GPU (~10-50× más rápido). En la máquina actual (15.5 GB) no cabe.

---

## 5. Por qué NO están en el paper actual

Prioridad de esfuerzo. Lo que bloquea la publicación **no es arquitectura**, es evidencia
comparativa:

1. 🔴 **Falta baseline LoRA** — sin él, un revisor no puede situar el trabajo.
2. 🔴 **Eval de solo 8 ejemplos** — los números son indicativos, no sólidos.
3. 🟡 Faltan benchmarks downstream (MMLU/HellaSwag/ARC/GSM8K) y n≥3 semillas.

Un paper con arquitectura novedosa pero sin baseline se rechaza; uno con historia sólida +
baseline + eval decente tiene opción. Estas ideas quedan como **Future Work** — y de hecho
fortalecen el paper, porque muestran una dirección clara y fundamentada en datos propios.
