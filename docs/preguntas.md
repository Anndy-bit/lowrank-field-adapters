## Preguntas Out-of-the-Box por Tema

### SMV — El bottleneck como protocolo

1. Si EMP predice μ con alta correlación temporal (SMV-7, ρ > 0.5), ¿por qué no tratar μ como una señal 1D y comprimirla entre tokens con un codec diferencial (Δ-modulación) antes de enviarla por PCIe, explotando que el predictor ya conoce el estado anterior?

**Respuesta:** Sí, es correcto y es la extensión natural de SMV-7. Si ρ > 0.5, μ(t) y μ(t+1) son altamente correlacionados. En vez de transferir el vector completo o predecir con un modelo lineal, codificas la diferencia Δμ = μ(t) - μ(t+1) — esto es delta-modulación. Para señales suaves, Δμ tiene magnitud mucho menor que μ, permitiendo 4-8 bits por componente en vez de 16. Es conceptualmente más simple y más robusto que el predictor lineal de SMV-7 porque no depende de que el predictor tenga error pequeño. La única condición es ρ > 0.5, que ya está verificada en los primeros pasos de training.

---

2. ¿Podría Uk residir en GPU no como matriz densa sino como un arreglo cuantizado o estructurado (Toeplitz, circulante) que se descomprima on-the-fly en los tensores de activación, liberando VRAM para que SMV funcione en GPUs de 2GB?

**Respuesta:** La idea es correcta en principio pero la propuesta específica (Toeplitz/circulante) está mal orientada. U_k son vectores singulares de una matriz densa arbitraria — no tienen estructura de traslación invariante. Toeplitz aplica a matrices de convolución, no a factores SVD. Una alternativa más realista: guardar U_k cuantizado en uint8/uint4 con lookup table de reconstrucción. Para k=128, d=4096, uint8 reduce de 4MB a 0.5MB por capa. Para 32 capas: 16MB → 2MB. Esto SÍ cabe en GPUs de 2GB. La descompresión on-the-fly de uint8 tiene overhead bajo porque es solo un lookup + scale.

---

### EMP — La dinámica como subproducto

1. En vez de predecir μ, ¿qué tal si el predictor aprende a predecir el error de su propia predicción anterior, creando una cascada de correctores que elimine la necesidad de ejecutar el MLP excepto cada miles de pasos?

**Respuesta:** Sí, esto es cascada de refinamiento iterativo (iterative refinement). Si el predictor k predice el error del predictor k-1, después de n stages el MLP solo se ejecuta en la stage 0 — las demás stages son predictores locales baratos. El riesgo: si la primera predicción diverge, la cascada diverge. Condición de convergencia: el operador de predicción del error debe ser contractivo (radio espectral < 1). En la práctica, para que funcione miles de pasos, necesitas que la cascada se "congele" cuando el error de predicción cae bajo un threshold. El desafío es cuándo reiniciar la cascada si el régimen de training cambia.

---

2. Si A ≈ I − ηJHJ^T, ¿podrías usar la estimación de segunda orden que Adam ya calcula (su matriz de momentos) como proxy de H, manteniendo A actualizada sin costo O(k³) y convirtiendo EMP en un subproducto gratis del optimizador?

**Respuesta:** Sí, esto es excelente. Adam storea v_t = β₂·v_{t-1} + (1-β₂)·g_t² que aproxima diag(F) donde F es la Fisher Information Matrix. Para redes shallow (como el MLP de modulación), F ≈ H (Hessiana) cerca del óptimo. Entonces la diagonal de v_t aproxima diag(H). Esto permite actualizar A ≈ I − η·diag(v_t) sin costo O(k³) — solo O(k) por paso. Convierte EMP de costoso a basically gratis.

**Caveat importante (no mencionado antes):** La aproximación Fisher ≈ Hessiana es válida SOLO cerca del óptimo. Para cross-entropy (que se usa en LLMs), Fisher = Hessiana solo en el óptimo. En fase temprana de fine-tuning, Fisher y Hessiana son objetos distintos. Para que esta aproximación funcione bien, necesitas asumir que el modelo ya está cerca de un mínimo local — lo cual puede no ser cierto en las primeras epochs. El régimen de validez de esta técnica es asintótico, no early training. **Esta es la mejor pregunta del documento: hace EMP viable en vez de teórico, pero con caveat de régimen.**

---

### TER — La entropía como freno activo

1. Si H_NMF sesga sistemáticamente hacia arriba (TER-4), ¿podrías usar ese sesgo como un freno de seguridad conservador que deliberadamente over-engineere los pasos RK4 para tokens críticos, haciendo que el sesgo sea una fortaleza, no un bug?

**Respuesta:** **CORREGIDO — MI ANÁLISIS ANTERIOR ERA INCORRECTO AL DECIR QUE DEBES MEDIR M_4.** El sesgo de H_NMF es O(1/N) que es ≤0.1 bits (pequeño). Pero más importante: M_4 = max‖f_θ⁽⁴⁾‖ es la derivada cuarta del campo de velocidad de la ODE, que es inaccesible en la práctica — requiere conocer la Hessiana de f_θ, que no se puede computar sin guardar estados intermedios adicionales. El sesgo de H_NMF, aunque pequeño, es el ÚNICO proxy medible de "complejidad del token" cuando M_4 no lo es. El "freno conservador" ya existe en TER-1: si el router subestima N, el error escala con M_4. Pero dado que M_4 es inaccesible, el sesgo de H_NMF es lo que tienes en la práctica. No es ideal, pero es lo que hay. La respuesta correcta es: sí, usa el sesgo como brake porque es el único mecanismo disponible, no porque sea óptimo.

---

2. ¿Qué pasa si el router decide no por token aislado sino por pares de tokens consecutivos, explotando que la complejidad semántica viene en frases, no en palabras sueltas?

**Respuesta:** Sí, tiene razón total. La complejidad semántica es a nivel de frase, no de palabra. El router actual (Definición 4.2) decide por token单体 con H_NMF(h_i). Un router que tome input (H_NMF(h_i), H_NMF(h_{i+1})) o mejor aún, el vector diferencia (h_{i+1} - h_i) proyectado a spectral domain, captura mejor la estructura. Esto no requiere cambiar ningún teorema — solo redefine el feature del router. El riesgo: si la secuencia tienetokensfunction words (preposiciones, artículos) que cambian poco, el par captura esto. Si hay cambio abrupto de tópico entre tokens no consecutivos, el router de pares no lo detecta.

---

### MSO — La curvatura como recurso

1. Si MSO detecta curvatura baja (shortcut válido), ¿podrías usar la inversión de esa señal (curvatura alta detectada) para ordenar a DRA que aumente temporalmente el rank k en esa región del manifold?

**Respuesta:** No, hay confusión conceptual. Curvatura alta de la trayectoria de la ODE significa que f_θ está cambiando rápido, requiriendo más integración numérica. El rank k de SVMO controla cuántos componentes del espectro de W se modulan — es un parámetro de la layer, no de la trayectoria de la ODE. DRA debe aumentar k cuando s(t) = σ_{k+1}/Σσ_i indica que hay energía espectral ignorada, no cuando la ODE tiene alta curvatura. Son métricas ortogonales: curvatura mide la dinámica de f_θ, rank k mide la complejidad espectral de W. No tiene sentido aumentar k porque la ODE sea curvilínea.

---

2. ¿Podría MSO operar en el espacio de Fourier de la trayectoria de la ODE, detectando que los coeficientes de alta frecuencia de h*(t) son nulos para justificar el shortcut espectral?

**Respuesta:** Sí, es conceptualmente elegante. Si h*(τ) es suave, su transformada de Fourier H(ω) tiene coeficientes de alta frecuencia pequeños.Shortcut válido cuando |H(ω > ω_cut)| < ε. MSO-2 usa diferencia finita para estimar curvatura, que es análogo a estimar la segunda derivada (que es la primera frecuencia no-trivial en Fourier). Operar en Fourier directamente es más elegante teóricamente: el shortcut es válido si el 95% de la energía del espectro está en frecuencias < ω_cut. El costo: necesitas samplear la trayectoria h*(τ) en N puntos y computar FFT(N log N), lo cual puede ser mayor que las 4N evaluaciones de f_θ que MSO pretende evitar.

---

### FDGD — El gradiente como espectro temporal

1. En vez de filtrar g en el espacio de parámetros, ¿qué tal si filtras la trayectoria del optimizador en el tiempo (θ_t como señal temporal), haciendo un low-pass de los pesos para eliminar oscilaciones del SGD y convirtiendo FDGD en un estabilizador temporal?

**Respuesta:** No tiene razón. Esto es un malentendido del método. FDGD propone filtrar el gradiente g en el dominio de Fourier para eliminar ruido de alta frecuencia antes del optimizer step — es estándar en optimización (equivalente a preconditioning). Filtrar θ directamente es peligroso: θ_t es la solución del optimizador, no el gradiente. Modificar θ_t filtrándolo equivale a alterar la trayectoria de convergencia, lo que puede causar divergencia. El momentum de Adam ya actúa como filtro temporal de baja frecuencia sobre θ_t — aplicar otro low-pass sobre θ_t es redundante y potencialmente destructivo.

---

2. Si FDGD elimina energía de alta frecuencia del gradiente, ¿podría esa energía "descartada" alimentar la matriz de covarianza Q del filtro de Kalman de EMP, es decir, que el ruido filtrado de FDGD sea la señal de innovación de EMP?

**Respuesta:** Sí, esto es ingeniosamente correcto. El ruido de alta frecuencia que FDGD filtra tiene información sobre la incertidumbre del gradiente — específicamente, mide qué tan errático es el gradiente entre steps. Usarlo como matriz de covarianza Q del proceso en EMP (ruido de innovación) conecta FDGD y EMP de manera elegante: FDGD produce la señal y EMP la consume. La señal Q_t = diag(g²) procesada por FDGD representa la incerteza del modelo dinámico.

**Caveat importante (no mencionado antes):** El ruido de alta frecuencia del gradiente NO es estacionario: en fase temprana de training hay más energía de alta frecuencia (el modelo cambia rápidamente), y en fase tardía el gradiente es más suave. Por lo tanto, Q debe ser adaptativa, no fija. Q_t = α(t)·diag(g_t²) con α(t) decayendo durante training. Si Q es fija, EMP sobre-estima la incertidumbre en fase tardía y converge lentamente. Esto complica EMP pero no invalida la idea — solo requiere que FDGD y EMP se comuniquen sobre el nivel de ruido actual.

---

### GNS — El skip como presupuesto

1. Si GNS skippea el backward de una capa porque su gradiente es pequeño, ¿podrías prestar esos FLOPs salvados a una capa vecina que DRA haya identificado como de alta complejidad espectral, creando un presupuesto de compute dinámico entre capas?

**Respuesta:** Sí, conceptualmente correcto. GNS dice "capa i no necesita backward" (‖g_i‖ pequeño). DRA dice "capa j necesita más capacidad" (s(t) alto). Combinarlas: redistribución dinámica de compute. El scheduler de FLOPs maraca las capas saltadas por GNS y les asigna budget adicional a capas con s(t) alto. El mecanismo práctico: GNS skip flag → scheduler redistribuye compute time/FLOPs → capas con s(t) > δ reciben más recursos. Impacto real en hardware limitado (2GB VRAM, GPU lenta): mejor utilización del budget fijo. Esto es viable y no requiere nuevos teoremas — solo un scheduler que orqueste GNS y DRA.

---

2. ¿Podría GNS usarse en modo inverso durante el forward: si el gradiente de una capa es pequeño, su paso hacia adelante es aproximadamente lineal, por lo que podrías evaluar su NMF con N=1 en vez de N=4?

**Respuesta:** Parcialmente. La intuición es: gradiente pequeño → función de loss plana → la ODE apenas necesita deformar → N=1 basta. Esto es logicamente consistente PERO hay un problema: GNS mira ‖∂L/∂θ_i‖ (gradiente de pesos), no la suavidad de f_θ de la ODE. Estos no están directamente conectados. Una capa podría tener gradientes pequeños pero f_θ muy no-lineal (y viceversa). No hay teorema que conecte ‖g_i‖ con la curvatura de la ODE para esa capa. La pregunta formula una hipótesis interesante que no está en el documento — requiere un nuevo teorema que enlace gradiente de capa con la necesidad de integración de la ODE.

---

### SMA — La memoria como huella

1. Si SMA guarda proyecciones U_k^T h de dimensión k, ¿podrías usar esos vectores como huellas digitales para detectar cuándo dos tokens distintos en la secuencia son funcionalmente equivalentes y compartir su backward?

**Respuesta:** Sí, pero con limitaciones. Tokens funcionalmente equivalentes (ej: "el" en diferentes posiciones) producen representaciones similares bajo U_k^T. Detectar equivalencia funcional vía similaridad de U_k^T h es viable si el threshold se calibra bien. Compartir backward entre tokens diferentes es riesgoso: la pérdida de cada token puede requerir gradientes distintos incluso si las activaciones son similares. Aplicación más realista: detección de near-duplicates para evitar cómputo redundante en secuencias con tokens repetidos, pero sin compartir backward — el gradiente sigue siendo por-token.

---

2. ¿Podría la reconstrucción U_k s de SMA ser el punto de partida para un autoencoder espectral entrenado end-to-end que aprenda a comprimir aún más las activaciones, más allá de la SVD fija?

**Respuesta:** Sí, tiene mérito. U_k s es la mejor reconstrucción lineal de h desde los primeros k vectores singulares. Un autoencoder espectral aprendiera una compresión no-lineal h → encode → decode que minimice ‖h - decode(encode(h))‖². Esto podría comprimir más que la SVD truncada fija porque aprende representaciones específicas del dominio. El riesgo: añade parámetros entrenables (encoder/decoder networks), lo cual contradice la filosofía de S³ de mínima sobrecarga. Un compromiso: encoder de 1 capa lineal + decode de 1 capa lineal, manteniendo O(k) parámetros como SMA, pero aprendiendo representaciones adaptadas al training data en vez de fijas por SVD.

---

### HFISC — La inicialización como geometría

1. ¿Qué pasa si inicializas el MLP de modulación para que su Jacobiano sea una isometría parcial (J^T J = I), forzando que preserve ángulos entre direcciones de gradiente y acelerando la convergencia de EMP al hacer A más cercana a una rotación?

**Respuesta:** **CORREGIDO — MI ANÁLISIS ANTERIOR ERA INCORRECTO.** La condición J^T J = I para un MLP 1→H→H→1 significa que el Jacobiano escalar J = ∂m_θ/∂σ satisface J² = 1, es decir, J = ±1 en todo punto. Esto fuerza el MLP a ser la función identidad m_θ(σ) = σ, eliminando toda capacidad de modulación. No es restrictivo — es destructivo. Isometría parcial NO es viable para esta arquitectura. Para preservar capacidad de modulación, la condición correcta sería ‖J‖ ≈ 1 (norma cercana a 1, no igual a 1), que es lo que HFISC-2 implementa con scale √(2/d). La isometría destruiría el propósito del adapter.

---

2. ¿Podría HFISC calibrarse capa por capa a partir del espectro de la matriz base W, usando σ_max local de cada capa en vez de una escala global, para que cada capa tenga su propia inicialización adaptada?

**Respuesta:** Sí, es una mejora natural de HFISC. HFISC-2 usa escala √(2/d) global, misma para todas las capas. Pero la distribución de σ_j de W varía por capa — algunas tienen espectro empinado (pocos valores grandes), otras plano (muchos valores similares). Calibrar por capa: para capa i, scale_i = √(2/σ_max(i)) donde σ_max(i) es el valor singular máximo de W_i. Capas con σ_max grande necesitan scale más pequeña para mantener J ≈ I. Esto no contradice HFISC-2 — lo generaliza con información adicional del espectro. El overhead: necesitas computar σ_max(W_i) offline, pero esto es O(d³) por capa que ya se hace para SVMO. Costo adicional: negligible.

---

### TOWS — El warmstart como detector

1. Si el error de warmstart entre tokens es Δh·e^{LT}, ¿podrías usar ese error como medida de coherencia semántica y activar TER con más pasos RK4 cuando el error de warmstart explota (cambio de tema, idioma, etc.)?

**Respuesta:** Sí, esto es conceptualmente correcto y elegante. TOWS-1 dice que el error escala como Δh·e^{LT}. Si Δh es grande (token semánticamente diferente del anterior), el error explota. Esto es un detector directo de "cambio de régimen" en la secuencia — más directo que H_NMF que es un proxy. Si Δh·e^{LT} > threshold, ejecutar N=4 en vez de N=1. Comparado con H_NMF: TOWS usa el estado real de la ODE, H_NMF usa la entropía de la bottleneck layer. TOWS es más robusto porque no depende de la arquitectura de f_θ. La condición de activación: threshold ≈ 0.1 (basado en TOWS-1 con L≈0.1, T=1, Δh > 0.11).

---

2. ¿Podría TOWS propagarse en el backward: usar el estado final de la ODE del token t como condición inicial para resolver la ODE adjunta hacia atrás en el token t-1?

**Respuesta:** Sí, esto es estándar en optimal control y Neural ODEs (método adjunto). Para la ODE adjunta λ(t) = ∂L/∂h(t), la condición inicial es λ(T) = ∂L/∂h(T) y se integra hacia atrás. Si resuelves token por token de t=512 → t=1, usar h(t) como warmstart para h(t-1) en backward es análogo a TOWS pero en reversa. El riesgo: para sequences largas (512 tokens), resolver la ODE adjunta token por token con warmstart acumula error numérico en dirección inversa — especialmente porque el adjoint de una ODE puede ser numéricamente inestable. El método adjunto de Neural ODE (Chen et al. 2018) resuelve esto con O(1) memory pero no hace warmstart entre tokens — resuelve la ODE adjunta completa de una vez.

---

### DRA — El rank como moneda

1. Si k_eff diminui, ¿podrías usar los grados de libertad liberados para aumentar la dimensión oculta H del MLP de modulación, manteniendo constante el presupuesto de parámetros entrenables pero redistribuyéndolo?

**Respuesta:** **CORREGIDO — MI ANÁLISIS ANTERIOR ERA INCORRECTO.** El MLP de modulación opera componente a componente: es 1→H→H→1, donde la entrada es un valor singular individual (un escalar, no el vector μ completo). El MLP se aplica a cada σ_j independientemente con los mismos pesos compartidos. Aumentar H de 32 a 64 DUPLICA la capacidad del MLP por cada componente — tiene más parámetros para aprender modulaciones complejas por valor singular. Esto SÍ tiene sentido. Si k baja de 128 a 96, los FLOPs de SVMO para esa capa bajan (menos singular values que proyectar), y esos FLOPs/presupuesto liberados pueden invertirse en un MLP más expressivo. La redistribución es: menos singular values pero cada uno con modulación más rica. Es válida y tiene sentido teóricamente.

---

2. ¿Podría el criterio de estabilidad de DRA (s(t) < δ por n pasos) usarse para activar simultáneamente GNS en esa capa, asumiendo que si el rank es estable, la capa está en un mínimo local y no necesita backward?

**Respuesta:** No, es una falacia. s(t) < δ significa que el (k+1)-ésimo singular value contribuye poco a la reconstrucción — el espectro decae rápidamente. Esto NO implica que la capa esté en un mínimo local de la loss. La capa podría simplemente tener singular values que decaen rápido (geometría del espectro), independientemente de si está convergida. GNS necesita gradientes pequeños para skip, no rank estable. No hay conexión causal entre estabilidad del rank y convergencia de la capa. Usar DRA para activar GNS sería aplicar skip a capas que podrían estar lejos del óptimo solo porque su espectro decae rápido.

---

### Análisis Integrado — El ecosistema como control óptimo

1. El documento asume que las optimizaciones son multiplicativas en ahorro, pero ¿hay alguna interacción destructiva? Por ejemplo, ¿FDGD podría filtrar precisamente las componentes de alta frecuencia que MSO necesita para detectar curvatura?

**Respuesta:** Sí, existe conflicto real y documentado. FDGD filtra componentes de alta frecuencia del gradiente para convergencia más suave. MSO detecta curvatura de la trayectoria vía segunda derivada de h*(τ), que se manifiesta como componentes de alta frecuencia en la derivada. Si FDGD filtra estas frecuencias, MSO pierde la señal de curvatura — podría detectar shortcut válido cuando no lo es, o viceversa. Este conflicto está parcialmente reconocido en la tabla de gaps (FDGD: "la demostración asume simetría del espectro"). Solución potencial: que MSO opere antes de FDGD en el flujo de backward, preservando las frecuencias que necesita. O: que FDGD sea selectivo — solo filtre frecuencias que no son relevantes para curvatura (es decir, filtrar solo las altísimas frecuencias que no aportan a curvatura pero sí a ruido).

---

2. ¿Podría el ecosistema completo S³-OPT interpretarse como un sistema de control óptimo donde μ es la señal de control, EMP es el observador, TER es el actuador variable, y GNS es el mecanismo de ahorro de energía, buscando un Lagrangiano global?

**Respuesta:** Sí, es una reinterpretación válida y útil. La analogía con control óptimo es precisa:
- μ = señal de control (decide cómo modular los singular values)
- EMP = observador (predice el estado del sistema antes del siguiente step)
- TER = actuador variable (N adaptativo — más o menos "fuerza" según necesidad)
- GNS = mecanismo de ahorro de energía (minimiza compute cuando posible)
- Loss = Lagrangiano (minimizar loss sujeto a constraints de VRAM y FLOPs)

El beneficio de esta reinterpretación: permite usar teoría de control óptimo para analizar composición de optimizaciones (ej: estabilidad del sistema controlado, bound de regret como bound de tracking error). La limitación: en control óptimo clásico el sistema es determinista y el control es función del estado. Aquí todo es estocástico (SGD) y las decisiones son probabilísticas (TER routing, GNS skip). La analogía es útil como marco conceptual pero no substituye los teoremas individuales.

---

## 🟨 ORO — Implementar esto tiene impacto medible

### SMV.1 — Δ-modulación de μ entre tokens
Si ρ > 0.5, la correlación entre μ(t) y μ(t+1) es alta. En vez de transferir 16k bits (fp16) o 32k bits (fp32) por token, transferirías diferencias codificadas en 4-8 bits por componente. Eso es una reducción de 2x-4x adicional sobre SMV ya existente. **Impacto real en ancho de banda PCIe.** Vale la pena.

**Relación con preguntas:** Responde directamente a SMV-1. Delta-modulación es la implementación práctica de "codificar diferencias" que SMV-1 propone.

### EMP.2 — Usar los momentos de Adam como proxy de H
Adam ya calcula E[g²] y E[g] (segundo y primer momento). Si A ≈ I − ηJHJ^T, y la diagonal de la matriz de momentos de Adam aproxima la diagonal de H, puedes mantener A actualizada sin costo O(k³) adicional. Convierte EMP de un sistema costoso en un subproducto del optimizador. **Impacto: hace EMP viable en vez de teórico.**

**Relación con preguntas:** Es exactamente la misma idea que EMP-2. Una vez respondida la pregunta, esta aplicación es directa.

### GNS.1 — Presupuesto de FLOPs dinámico entre capas
Si skippeas el backward de una capa (ahorras X FLOPs), redistribuir esos FLOPs a una capa donde DRA detectó alta complejidad es balanceo de carga óptimo. En un sistema con recursos fijos (2GB VRAM, GPU lenta), esto es oro puro. **Impacto: mejor utilización del hardware limitado.**

**Relación con preguntas:** Es la versión implementable de GNS-1. Responde la pregunta directamente con un mecanismo concreto.

### TOWS.1 — Error de warmstart como detector de coherencia semántica
Si Δh·e^{LT} explota entre dos tokens, eso significa que el MLP está procesando algo radicalmente diferente (cambio de tema, token especial, idioma). Usar eso para activar TER con N=4 en vez de N=1 es routing basado en contenido real, no en entropía proxy. **Impacto: más preciso que H_NMF.**

**Relación con preguntas:** Es la respuesta completa a TOWS-1. El teorema TOWS-1 da el bound, ORO lo convierte en mecanismo de detección.

---

## Tabla de Gaps y Demostraciones Pendientes

| Sección | Punto exacto | Qué falta |
|---------|-------------|-----------|
| **SMV** | Teorema SMV-1 | El cálculo del ratio salta entre incluir y excluir 4d_out del denominador. Falta consistencia en si el gradiente ∂L/∂y se transfiere o no en el backward de SMV. |
| **SMV** | Teorema SMV-6 | El bound usa (1+β)^δ que explota numéricamente (1.9^32 ≈ 10^7). Falta un bound realista del drift o una simulación que justifique el 0.3% práctico. |
| **EMP** | Teorema EMP-1 | El bound "δ/(1−ρ)(1+2T)" es dimensionalmente inconsistente y sugiere divergencia lineal. Falta restringir a régimen estacionario o derivar correctamente. |
| **EMP** | Sección 3.2 | Falta la derivación explícita de A = I − ηJHJ^T + O(η²) que justifique por qué el modelo LTI es válido en el régimen de S³. |
| **TER** | Teorema TER-3 | El término de regret de generalización crece como T^(7/2), peor que lineal. Falta corregir la discretización o usar análisis de Lipschitz sin bins. |
| **TER** | Sección 4.2 | Falta un teorema que ligue H_NMF con la curvatura real de la ODE. Actualmente es un postulado sin puente formal. |
| **MSO** | Teorema MSO-1 | Falta un corolario que dé un criterio verificable online solo con evaluaciones de f_θ, sin conocer L ni M a priori. |
| **MSO** | Sección 5.9 | Falta análisis cuantitativo de composición con TER: si TER ya reduce N, ¿cuál es la ganancia marginal real de MSO? |
| **FDGD** | Teorema FDGD-1 | La demostración asume implícitamente que el filtro paso-bajo preserva la esperanza del gradiente. Falta asumir simetría del espectro o centrado en cero explícitamente. |
| **FDGD** | Sección 6.5 | Falta un teorema de costo computacional que justifique la factibilidad de FFT sobre P parámetros en GPUs low-end. |
| **GNS** | Teorema GNS-1 | La demostración es incompleta (cambia de argumento a medio camino). Falta una cota rigurosa o una hipótesis de isotropía explícita. |
| **GNS** | Sección 7.9 | Falta conectar con SMA: qué pasa con los checkpoints espectrales cuando GNS skippea el backward de una capa. |
| **SMA** | Teorema SMA-3 | El bound ‖g − g_k‖ ≤ ‖g‖·σ_{k+1}/σ_k no está bien justificado. Falta clarificar la relación entre gradiente de activación y valores singulares de W. |
| **SMA** | Sección 8.1 | Falta un análisis de error de cuantización fp16 en la reconstrucción U_k s (similar al SMV-3 pero para activaciones). |
| **HFISC** | Teorema 8.2.1 | Confunde la Hessiana de la loss con la Hessiana del MLP de modulación. Son objetos distintos; falta separarlos o definir cuál se está minimizando. |
| **HFISC** | Teorema 8.2.4 | Para d grande, 1/d y 2/d son indistinguibles. Falta un argumento de régimen no-asintótico o un experimento que muestre diferencia práctica. |
| **TOWS** | Teorema TOWS-1 | Falta el corolario de condición de utilidad: ¿cuándo Δh es lo suficientemente pequeño para que el warmstart valga la pena frente a cold-start? |
| **TOWS** | Sección 8.3 | Falta un remark o teorema sobre el impacto de LayerNorm en la continuidad token-a-token. Rompe la suposición de Markov. |
| **DRA** | Teorema DRA-2 | ΔL_s no está acotado en términos de valores singulares descartados. Falta expresar el error en función explícita de σ_{k(t)+1}. |
| **DRA** | Teorema DRA-3 | Falta una definición probabilística o determinista rigurosa de "estabilidad" (s(t) < δ por n pasos). |
| **Integrado** | Sección 9.1 | La reducción compuesta ρ_total asume independencia multiplicativa. Falta un teorema de composición que acote el error acumulado cuando todas las optimizaciones están activas. |
| **Integrado** | Sección 10 | Falta un teorema de "no-regret" global o al menos un argumento de que el error no explota bajo la composición simultánea. |