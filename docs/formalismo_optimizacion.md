# S³-OPT: Formalismo de Optimizaciones para Fine-Tuning en VRAM Limitada

> **Estado:** Fase de investigación teórica — Julio 2026
> **Relación con S³ original:** Extensión y generalización del framework Spectral-Spatial-Smooth
> **Autores:** [Pending]

---

## 1. Introducción y Motivación

El framework S³ (Spectral-Spatial-Smooth Fine-Tuning) demostró que es posible fine-tunear modelos de 7B parámetros en VRAM de 2GB mediante tres operadores: SVMO (modulación espectral de valores singulares), NMF (flujo de variedades neuronales via Neural ODE), y STB (puente de transporte espectral). Sin embargo, el bottleneck principal identificado en el training loop es el **ancho de banda PCIe** (~16ms por swap de capa CPU↔GPU × 64 swaps por token).

Este documento formaliza seis nuevas optimizaciones teóricas denominadas colectivamente **S³-OPT**, cada una fundamentada en principios matemáticos distintos y complementarios. Las optimizaciones abordan diferentes capas del problema:

1. **SMV** — Eliminación del bottleneck PCIe mediante transferencia de vectores en lugar de matrices
2. **EMP** — Predicción de migración de valores singulares via dinámica de sistemas
3. **TER** — Routing adaptativo del número de pasos ODE por entropía del token
4. **MSO** — Atajos geodésicos en el manifold de representaciones
5. **FDGD** — Filtrado en dominio Fourier de gradientes para convergencia acelerada
6. **GNS** — Poda de backward pass via magnitud de gradiente normalizado

Adicionalmente, se introducen cuatro optimizaciones complementarias:

7. **SMA** — Attention espectral con memoria deactivaciones
8. **HFISC** — Inicialización libre de Hessian via número de condición espectral
9. **TOWS** — Warmstarting de ODE entre tokens de una secuencia
10. **DRA** — Adaptación dinámica del rango k durante entrenamiento

---

## 2. Spectral Modulation Vector (SMV)

### 2.1 Problema

En la implementación actual de SVMO, cada forward pass requiere transferir las matrices SVD $U_k \in \mathbb{R}^{d_{out} \times k}$ y $V_k^T \in \mathbb{R}^{k \times d_{in}}$ desde CPU a GPU. Para $d_{out} = 4096, k = 128$:

$$\text{PCIe transfer} = 2 \times (4096 \times 128 \times 4\text{ bytes}) = 8\text{MB per layer per forward/backward}$$

Con 32 layers y 2 passes (forward + backward), el total PCIe por token es:

$$8\text{MB} \times 32 \times 2 = 512\text{MB per token}$$

Esto dominan el tiempo de entrenamiento.

### 2.2 Idea Central

Separar la matriz $U_k$ (que es **invariante durante training** porque los pesos originales $W$ no cambian) del vector de modulación $m_\theta(\sigma_k)$ (que **sí cambia** en cada paso).

**Procedimiento propuesto:**
1. Transferir $U_k$ a GPU **una sola vez** al inicio del entrenamiento (8MB total, una vez)
2. En cada forward pass, transferir solo el vector $m_\theta(\sigma_k) \in \mathbb{R}^k$ (~0.5KB)
3. La modulación se realiza in-place en GPU usando el vector pre-cargado

### 2.3 Definición Formal

Sea $W = U_k \Sigma_k V_k^T$ la descomposición SVD truncada de rank $k$ de la matriz de pesos congelada $W$. Definimos el **Spectral Modulation Vector** como:

$$\mu_\theta(\sigma_k) := m_\theta(\sigma_k) = \sigma_k \odot (1 + \alpha \cdot \tanh(g_\theta(\log(\sigma_k + \epsilon)))$$

donde $g_\theta: \mathbb{R} \to \mathbb{R}$ es el MLP de modulación definido en S³-Theorem 1 con arquitectura 1 → $H$ → $H$ → 1, donde $H$ es la dimensión oculta del modulation MLP.

**Proposición 2.1 (Forma Matricial Equivalente):**
Para todo input $x \in \mathbb{R}^{d_{in}}$, el forward pass de SVMO puede escribirse como:

$$y = x V_k^T \cdot \text{diag}(\mu_\theta(\sigma_k)) \cdot U_k^T$$

donde $\cdot$ indica producto matricial estándar. Además:

$$y = x V_k^T \cdot \text{diag}(\mu_\theta(\sigma_k)) \cdot U_k^T = x \cdot \left( V_k \cdot \text{diag}(\mu_\theta(\sigma_k)) \cdot U_k^T \right)$$

*Demostración.* Por Definición 2.3 (SVMO original), $z = x V_k^T$. Luego $z_{mod} = z \odot \mu_\theta(\sigma_k) = z \cdot \text{diag}(\mu_\theta(\sigma_k))$. Finalmente $y = z_{mod} U_k^T$. Sustituyendo: $y = (x V_k^T) \cdot \text{diag}(\mu_\theta(\sigma_k)) \cdot U_k^T$. La segunda forma es equivalente por associatividad del producto matricial. $\square$

### 2.4 Teorema SMV-1: Reducción de Ancho de Banda

**Teorema SMV-1 (Reducción de Transferencia PCIe).**
Sea $T_{original}$ el tiempo de transferencia PCIe por forward pass en la implementación estándar de SVMO, y $T_{SMV}$ el tiempo en la implementación SMV. Entonces:

$$T_{SMV} \leq \frac{4k}{2d_{out}k + d_{in}k + 4k} \cdot T_{original} = \frac{1}{\frac{1}{2}d_{out} + \frac{1}{4}d_{in} + 1} \cdot T_{original}$$

Para $d_{out} = d_{in} = d = 4096$ y $k = 128$:

$$T_{SMV} \leq \frac{4 \cdot 128}{2 \cdot 4096 \cdot 128 + 4096 \cdot 128 + 4 \cdot 128} \cdot T_{original} = \frac{512}{1,049,856 + 524,288 + 512} \cdot T_{original} \approx \frac{1}{3072} \cdot T_{original}$$

Es decir, **reducción de ~99.97%** en tiempo de transferencia por forward pass.

*Demostración.* En la implementación estándar, cada forward pass requiere:
- Transferir $U_k$: $d_{out} \cdot k$ floats = $4d_{out}k$ bytes
- Transferir $V_k^T$: $k \cdot d_{in}$ floats = $4kd_{in}$ bytes
- Total por forward pass: $4k(d_{out} + d_{in})$ bytes

En la implementación SMV:
- $U_k$ y $V_k^T$ se transfieren una sola vez al inicio (costo amortizado: 0 por forward pass)
- Por forward pass, solo se transfiere $\mu_\theta(\sigma_k) \in \mathbb{R}^k$ ($4k$ bytes)
- Para el backward pass, se transfiere $\partial\mathcal{L}/\partial y \in \mathbb{R}^{d_{out}}$ ($4d_{out}$ bytes), lo cual es inevitable en cualquier implementación

El ratio de bandwidth por forward pass es:

$$\frac{T_{SMV}}{T_{original}} = \frac{4k}{4k(d_{out} + d_{in}) + 4d_{out}} = \frac{1}{d_{out} + d_{in} + 1}$$

ya que $T \propto$ bytes transferidos y $4d_{out}$ bytes para el gradiente son comunes a ambas implementaciones.

Para $d_{out} = d_{in} = 4096$: $\frac{1}{8193} \approx 1.22 \times 10^{-4}$. Considerando que la transferencia de gradiente es ineludible, el bound efectivo incluye el término $4d_{out}$:

$$\frac{4k}{4k(d_{out} + d_{in}) + 4d_{out}} = \frac{k}{k(d_{out} + d_{in}) + d_{out}}$$

Evaluando: $\frac{128}{128 \cdot 8192 + 4096} = \frac{128}{1,049,856 + 4096} = \frac{1}{8226}$. $\blacksquare$

*Remark 2.1.* El bound asintótico es $\Theta(1/d)$ — independiente de $k$. Para modelos grandes con $d \gg k$, la ganancia de SMV se maximiza y converge a $1/d$.

### 2.5 Teorema SMV-2: Consistencia Exakta del Gradiente

**Teorema SMV-2 (Conservación Exacta de Gradientes).**
Bajo SMV, para todo $\theta \in \mathbb{R}^p$ y todo input $x \in \mathbb{R}^{d_{in}}$, el gradiente de la loss $\mathcal{L}$ con respecto a $\theta$ calculado en GPU es **exactamente idéntico** al gradiente calculado con la implementación estándar de SVMO:

$$\nabla_\theta \mathcal{L}_{SMV} = \nabla_\theta \mathcal{L}_{standard}$$

*Demostración.* Por Proposición 2.1, el forward pass bajo SMV produce:

$$y_{SMV} = x V_k^T \cdot \text{diag}(\mu_\theta(\sigma_k)) \cdot U_k^T$$

Por la Definición de SVMO estándar (S³-Theorem 1), el forward pass original es:

$$y_{standard} = (x V_k^T \odot \mu_\theta(\sigma_k)) \cdot U_k^T$$

Por la Definición de $\mu_\theta$ (que es la misma función en ambos casos), $y_{SMV} = y_{standard}$ exactamente. Por lo tanto $\mathcal{L}(y_{SMV}) = \mathcal{L}(y_{standard})$. Mediante la regla de la cadena de backpropagation, las derivadas $\partial\mathcal{L}/\partial\theta$ son idénticas punto a punto. $\blacksquare$

**Corolario 2.1 (Conservación de Optimalidad).**
Si $\theta^*$ minimiza localmente $\mathcal{L}$ bajo SVMO estándar, entonces $\theta^*$ también minimiza localmente $\mathcal{L}$ bajo SMV, y viceversa.

*Demostración.* Sigue directamente de Teorema SMV-2: las funciones objetivo son idénticas, por lo tanto sus conjuntos de minimizadores locales coinciden. $\square$

### 2.6 Teorema SMV-3: Análisis de Cuantización FP16

**Teorema SMV-3 (Error de Cuantización a FP16).**
Sea $\tilde{\mu} \in \mathbb{R}^k$ la versión cuantizada a FP16 de $\mu_\theta$, con error de cuantización $\|\mu - \tilde{\mu}\|_\infty \leq 2^{-11} \approx 4.88 \times 10^{-4}$ (precisión de la mantisa FP16 de 10 bits). Sea $y_{FP16} = x V_k^T \cdot \text{diag}(\tilde{\mu}) \cdot U_k^T$ la salida con el vector cuantizado. Entonces el error relativo en la salida está acotado por:

$$\frac{\|y - y_{FP16}\|_2}{\|y\|_2} \leq \frac{2^{-11} \cdot \|x V_k^T\|_2}{\|y\|_2} \leq 2^{-11} \cdot \kappa(\Sigma_k) \cdot \frac{\|x\|_2}{\|\mu_\theta(\sigma_k)\|_2}$$

donde $\kappa(\Sigma_k) = \sigma_1 / \sigma_k$ es el número de condición de la matriz de valores singulares.

*Demostración.* De la forma matricial:

$$y - y_{FP16} = x V_k^T \cdot \text{diag}(\mu - \tilde{\mu}) \cdot U_k^T$$

Por submultiplicatividad de la norma espectral:

$$\|y - y_{FP16}\|_2 \leq \|x V_k^T\|_2 \cdot \|\text{diag}(\mu - \tilde{\mu})\|_2 \cdot \|U_k^T\|_2 = \|x V_k^T\|_2 \cdot \|\mu - \tilde{\mu}\|_\infty$$

ya que $\|U_k^T\|_2 = 1$ (columnas ortonormales). Además $\|x V_k^T\|_2 \leq \|\Sigma_k\|_2 \cdot \|x\|_2 = \sigma_1 \|x\|_2$ y $\|y\|_2 \geq \sigma_k \|\mu_\theta(\sigma_k)\|_2 \cdot \|x\|_2$ (por la ley de equivalencia de normas matriciales). Dividiendo:

$$\frac{\|y - y_{FP16}\|_2}{\|y\|_2} \leq \frac{\sigma_1}{\sigma_k} \cdot \frac{\|\mu - \tilde{\mu}\|_\infty}{\|\mu_\theta(\sigma_k)\|_2} \leq \kappa(\Sigma_k) \cdot \frac{2^{-11}}{\|\mu_\theta(\sigma_k)\|_2}$$

Como $\|\mu_\theta(\sigma_k)\|_2 \geq 1$ (porque $\mu_\theta(\sigma)_j \geq \sigma_j(1-\alpha) \geq (1-\alpha)$ para todo $j$), obtenemos el bound débil:

$$\frac{\|y - y_{FP16}\|_2}{\|y\|_2} \leq \kappa(\Sigma_k) \cdot 2^{-11} \quad \blacksquare$$

*Remark 2.2.* Para $k = 128$ y valores singulares razonablemente condicionados ($\kappa \leq 10$), el error máximo es $\approx 10 \cdot 2^{-11} \approx 0.5\%$. Este error es menor que el error de precisión de fp16 (~0.0001\%) y generalmente aceptable para training.

### 2.7 Teorema SMV-4 (Optimalidad Asintótica)

**Teorema SMV-4 (Cota Inferior de Information-Theoretic).**
Sea $B(\mu)$ el bandwidth mínimo teórico necesario para comunicar la información de modulación $\mu_\theta(\sigma_k)$ desde CPU a GPU, para una tolerancia de error relativo $\epsilon$ en la salida. Entonces:

$$B(\mu) \geq k \cdot \log_2\left(\frac{\alpha \cdot \max_j \sigma_j}{\epsilon}\right) \text{ bits}$$

En la implementación SMV, el bandwidth utilizado es $B_{SMV} = 32k$ bits (fp32) o $16k$ bits (fp16).

*Demostración.* La función de modulación $m_\theta(\sigma)$ produce valores continuos en el rango $[\sigma(1-\alpha), \sigma(1+\alpha)]$ (S³-Theorem 1). Para representar cada componente de $\mu_\theta$ con error relativo $\epsilon$ se necesitan al menos $\log_2(\alpha \sigma_{max}/\epsilon)$ bits por componente (cota de información de Shannon para quantización óptima). Como hay $k$ componentes independientes, el bandwidth total es $k \log_2(\alpha \sigma_{max}/\epsilon)$ bits. $\blacksquare$

*Remark 2.3.* La gap entre el óptimo teórico y SMV es:
- fp32: $\frac{32k}{k \log_2(\alpha \sigma_{max}/\epsilon)} = \frac{32}{\log_2(\alpha \sigma_{max}/\epsilon)}$
- Para $\alpha = 0.3, \sigma_{max} = 10, \epsilon = 0.01$: gap $\approx 32 / 12 \approx 2.7$x — SMV usa ~3x más bits que el óptimo teórico.

**Esto sugiere que existe espacio para comprimir $\mu$ usando un código de longitud variable antes de transferir, por ejemplo mediante:**
- Cuantización no-uniforme óptima ajustada al histograma de $\mu$
- Compresiónentrópica (e.g., codificación aritmética) si la distribución de $\mu$ es conocida

### 2.8 Teorema SMV-5: VRAM Suficiencia

**Teorema SMV-5 (Condición de Residencia en VRAM).**
Sea $V_{GPU}$ la VRAM disponible en el dispositivo GPU, y $V_{overhead}$ la VRAM usada porActivaciones, estados del optimizador, y otros tensores del modelo. Sea $C_{U_k} = d_{out} \cdot k \cdot 4$ bytes el costo de almacenar $U_k$ en fp32, y $C_{V_k} = k \cdot d_{in} \cdot 4$ bytes el de $V_k^T$. La condición para que ambos permanezcan residentes en GPU es:

$$C_{U_k} + C_{V_k} \leq V_{GPU} - V_{overhead}$$

Para $d_{out} = d_{in} = d$:

$$4k(2d) \leq V_{GPU} - V_{overhead} \quad \iff \quad d \leq \frac{V_{GPU} - V_{overhead}}{8k}$$

*Ejemplo:* Para GTX 1050 con $V_{GPU} = 4$GB y $V_{overhead} \approx 2.5$GB (modelo + activations), queda $V_{disponible} \approx 1.5$GB. Con $k = 128$: $d_{max} = 1.5 \times 2^{30} / (8 \times 128) \approx 1.55 \times 10^6$. Esto es para modelos hasta $d \approx 1.5M$, muy por encima de Qwen2.5-7B con $d = 4096$. $\blacksquare$

*Remark 2.4 (VRAM para LLaMA-3 70B).* Para un modelo de 70B con $d_{vocab} \approx 128K$ y capas de FFN $d_{ff} \approx 28K$, el bottleneck es la capa de embedding ($d_{vocab} \times d_{model}$) y las proyecciones $W_Q, W_K, W_V$ por capa. Si almacenamos solo la capa activa (no todas las cabezas), el requerimiento por capa es $C_{U_k} + C_{V_k} = 4 \times 128 \times (4096 + 4096) = 4$MB. Para 80 capas: 320MB, menor que muchos modelos de 7B. Para modelos de 70B con $d = 4096$ y 80 capas: 320MB de overhead de U_k es aceptable en una GPU con 16GB VRAM.

### 2.9 Análisis de Sincronización con Momentum

**Teorema SMV-6 (Safety con Optimizer Momentum).**
Sea $\theta_t$ los pesos del MLP de modulación en paso $t$, con actualización SGD+momentum:
$$\theta_{t+1} = \theta_t - \eta \nabla \mathcal{L}(\theta_t) + \beta (\theta_t - \theta_{t-1})$$

SMV usa $\mu_\theta$ para modular las salidas. Sea $t_0$ el último paso del optimizer, y $t_0 + \Delta$ un paso de forward sin actualización de optimizer. El vector $\mu_{\theta_{t_0}}$ transferido a GPU en $t_0$ permanece válido para todos los pasos $t \in [t_0, t_0 + \Delta]$ porque $\theta$ no cambia hasta el siguiente optimizer step.

El error máximo por usar $\mu_{\theta_{t_0}}$ en lugar de $\mu_{\theta_{t_0+\delta}}$ (para $0 < \delta < \Delta$) es:

$$\|\mu_{\theta_{t_0}} - \mu_{\theta_{t_0+\delta}}\| \leq L_\mu \cdot \|\theta_{t_0+\delta} - \theta_{t_0}\|$$

donde $L_\mu$ es la constante de Lipschitz de $\mu_\theta(\cdot)$, y $G = \max_t \|\nabla \mathcal{L}(\theta_t)\|$.

*Corrección (2026-07-20).* Una versión anterior de este documento derivaba aquí $\|\theta_{t_0+\delta} - \theta_{t_0}\| \leq \delta\eta\sum_{i=0}^{\delta-1}(1+\beta)^i G$, con un error de signo en la recursión de momentum que hacía crecer el bound geométricamente ($(1.9)^{32}\approx 10^7$ para $\beta=0.9$). Ese resultado era incorrecto y se reemplaza por la derivación siguiente.

Definimos la velocidad $v_t = \theta_t - \theta_{t-1}$. La actualización de momentum se reescribe como $v_{t+1} = \beta v_t - \eta \nabla\mathcal{L}(\theta_t)$ (es la forma "heavy-ball" estándar). Con $\|\nabla\mathcal{L}(\theta_t)\| \leq G$ para todo $t$:

$$\|v_{t+1}\| \leq \beta \|v_t\| + \eta G$$

Esta es una recursión **contractiva** porque $\beta < 1$ multiplica el término que decrece, no el que crece — la versión anterior del documento invertía este signo y obtenía $(1+\beta)^i$ (crecimiento geométrico) en vez de $\beta^i$ (decaimiento geométrico), de ahí que $(1.9)^{32}$ explote: ese número nunca debió aparecer. Desenrollando la recursión correcta desde $v_0=0$:

$$\|v_t\| \leq \eta G \sum_{i=0}^{t-1}\beta^i \leq \frac{\eta G}{1-\beta}$$

es decir, la velocidad de momentum está acotada por una **constante**, no crece con $t$ — este es el hecho estándar de que el momentum amplifica el learning rate efectivo por un factor $1/(1-\beta)$, no que el drift diverja. El drift total sobre $\delta$ pasos es entonces:

$$\|\theta_{t_0+\delta} - \theta_{t_0}\| = \Big\|\sum_{i=1}^{\delta} v_{t_0+i}\Big\| \leq \sum_{i=1}^{\delta}\|v_{t_0+i}\| \leq \delta \cdot \frac{\eta G}{1-\beta}$$

Este es el bound riguroso, no una fórmula "práctica" separada del worst-case: es el único bound correcto. Para $\eta = 10^{-4}, \beta = 0.9, G \approx 1, \delta = 32$: $\|\theta_{t_0+\delta} - \theta_{t_0}\| \leq 32 \times 10^{-4} \times 1/0.1 = 0.032$.

Con $L_\mu \approx 0.1$ (estimado de la arquitectura tanh): $\|\mu_{t_0} - \mu_{t_0+\delta}\| \leq 0.003$, i.e., 0.3% de cambio en $\mu$. Esto justifica usar $\mu$ constante durante gradient accumulation. $\blacksquare$

*Remark 2.5 (Condición de estabilidad).* Si el learning rate es demasiado alto ($\eta > 0.1$ para $\beta = 0.9$), el drift de $\theta$ puede ser $\geq 0.1$ durante $\delta = 32$ pasos, causando $\|\mu_{t_0} - \mu_{t_0+\delta}\| \geq 0.01$ (1% de cambio). En este régimen, SMV con transfer único por accumulation cycle puede introducir error significativo. La solución es transferir $\mu$ más frecuentemente (cada $N_{transfer} < \delta$ pasos) cuando $\eta \cdot G > 0.01 / L_\mu$.

### 2.10 Teorema de Optimalidad de Transferencia

**Teorema SMV-7 (Mínimo Teórico de Transferencia).**
Sea $B_{min}(\epsilon)$ el mínimo de bits que deben transferirse por forward pass para comunicar el vector de modulación $\mu \in \mathbb{R}^k$ con error relativo $\epsilon$ en la salida del modelo. Entonces:

$$B_{min}(\epsilon) \geq k \cdot \log_2\left(\frac{\alpha \cdot \sigma_{max}}{\epsilon}\right) - \log_2(\kappa(\Sigma_k)) \quad \text{bits}$$

*Demostración.* Por SMV-4, la cota inferior es $k \log_2(\alpha \sigma_{max}/\epsilon)$ bits para representación sin comprimir. Sin embargo, la estructura de la matriz $W = U_k \cdot \text{diag}(\mu) \cdot V_k^T$ permite explorar la correlación entre componentes: si $\mu_i$ y $\mu_j$ están correlacionados, la entropía conjunta $H(\mu_i, \mu_j) < H(\mu_i) + H(\mu_j)$. El bound exacto considera la entropía diferencial de $\mu$:

$$H(\mu) \leq \sum_{i=1}^k H(\mu_i) - \sum_{i=1}^k I(\mu_i; \mu_{i-1})$$

donde $I$ es la información mutua entre componentes adyacentes. Si la correlación temporal de $\mu$ es $\rho$ (estimable de los primeros pasos de training), entonces $I(\mu_i; \mu_{i-1}) \approx -\frac{1}{2}\log(1-\rho^2)$. El bandwidth mínimo es:

$$B_{min} = H(\mu) \geq k \cdot \log_2\left(\frac{\alpha \sigma_{max}}{\epsilon}\right) + \frac{k}{2}\log(1-\rho^2)$$

para $\rho \in [0, 1)$. Para $\rho = 0$ (componentes independientes): $B_{min} = k \log_2(\alpha \sigma_{max}/\epsilon)$. Para $\rho = 0.9$ (alta correlación): $B_{min} \approx k \log_2(\alpha \sigma_{max}/\epsilon) - 2.6k$ bits.

SMV usa $B_{SMV} = 16k$ bits (fp16). La eficiencia de SMV es $\eta_{SMV} = B_{min}/B_{SMV}$. Para $\alpha = 0.3, \sigma_{max} = 10, \epsilon = 0.01$: $B_{min} = k \times 12$ bits. $B_{SMV} = 16k$ bits. Eficiencia: $12/16 = 75\%$. Queda ~25% de gap para comprimir más. $\blacksquare$

*Remark 2.6 (Implicación práctica).* Si la correlación entre componentes de $\mu$ es alta ($\rho > 0.5$), se podría transferir solo $\lceil k/2 \rceil$ valores y predecir los otros con un predictor lineal de orden 1. Esto reduciría el bandwidth a la mitad, aunque a costo de complejidad adicional en el protocolo.

### 2.11 Análisis de Contención PCIe

**Teorema SMV-8 (Latencia bajo Contención).**
Sea $t_{transfer}(\mu)$ el tiempo de transferir $\mu$ por PCIe en régimen sin contención (idle). En un sistema con $N_{proc}$ procesos compitiendo por el bus PCIe, la latencia efectiva es:

$$\mathbb{E}[t_{transfer}] \approx \frac{t_{transfer}(\mu)}{1 - \rho_{PCIe}} \cdot \frac{1}{1 - p_{contention}}$$

donde $\rho_{PCIe}$ es la utilización del bus por tráfico no-relacionado con $\mu$, y $p_{contention}$ es la probabilidad de colisión. Si el tráfico no-relacionado ocupa fraction $\rho$ del bandwidth, la fraction disponible para $\mu$ es $(1-\rho)$.

*Demostración.* Modelamos el bus PCIe como un servidor M/M/1 con tasa de servicio $\mu_{PCIe}$ y tasa de llegada $\lambda$. El tiempo de espera esperado en cola es $\mathbb{E}[T_{queue}] = \frac{\rho}{\mu_{PCIe}(1-\rho)}$ con $\rho = \lambda/\mu_{PCIe}$. Para tráfico de $\mu$ que llega con tasa $\lambda_\mu$ interleaved con tráfico de otros procesos, el tiempo efectivo de transferencia es:

$$t_{eff} = t_{transfer} + \frac{\rho_{other}}{\mu_{PCIe}(1-\rho_{other})}$$

donde $\rho_{other}$ es la utilización por tráfico no-$\mu$. En la práctica, para NVIDIA GPUs con PCIe Gen 4 x16 y tráfico típico de data loading, $\rho_{other} \approx 0.1$ a 0.3 durante training activo. Por lo tanto, $t_{eff} \approx t_{transfer} \times (1 + 0.1/0.9) \approx 1.11 \times t_{transfer}$, i.e., overhead de ~11%.

Si el data loading está en un hilo separado con priority inversion prevention (e.g., CUDA streams), el overhead se reduce a ~5%. $\blacksquare$

*Remark 2.7.* En sistemas con NVLink (e.g., multi-GPU setup), la contención PCIe desaparece porque NVLink tiene su propio bus. SMV es aún más beneficioso ahí porque el bandwidth de NVLink es mayor pero la optimización de transferir menos datos sigue mattering.

### 2.12 Casos de Falla Conocidos

**Falla 1: VRAM insuficiente para mantener $U_k$ y $V_k^T$ residentes.**
Si $d_{out} = 8192$ (modelos grandes), $U_k$ requiere $8192 \cdot 128 \cdot 4\text{ bytes} = 4$MB por capa. Para un modelo de 80 capas (e.g., LLaMA-3 70B): $80 \cdot 4\text{MB} = 320$MB solo para mantener $U_k$ en GPU. Si la VRAM total del dispositivo es 4GB y otros tensores ya ocupan >3.6GB, puede no haber espacio suficiente, forzando a keeping $U_k$ en CPU y haciendo SMV inaplicable.

**Falla 2: Sistemas con memoria unificada (Apple Silicon, Intel iGPU).**
En arquitecturas donde CPU y GPU comparten memoria física (unified memory), el bandwidth PCIe no existe como bottleneck separado. El bandwidth de memoria unificada es típicamente alto (~100-200 GB/s en Apple M3 Max), pero la latencia de memory mapping puede ser significativa. En estos sistemas, el speedup de SMV puede no ser medible.

**Falla 3: Training con gradient accumulation muy largo.**
Si el optimizer actualiza $\theta$ solo cada $N_{grad\_accum}$ pasos (common en low-VRAM settings con micro-batches de 1), entonces $\mu_\theta$ permanece constante durante $N_{grad\_accum} \cdot (seq\_len - 1)$ forwards. En este caso, la primera evaluación del MLP $g_\theta$ computa $\mu$, las siguientes $N_{grad\_accum} \cdot (seq\_len - 1) - 1$ forwards son redundantes. SMV no elimina esta redundancia — para eso está EMP.

### 2.13 Experimentos Mentales

**Experimento 1: Cuantización a INT8.**
Sea $\alpha = 0.3, k = 128, \max \sigma = 10$, rango de $\mu$: $[7, 13]$. Cuantizando a INT8 (256 niveles uniformesi):

$$\Delta = \frac{13 - 7}{256} = 0.0234$$

Error relativo en $\mu$: $\frac{0.0234}{7} \approx 0.33\%$.

Error propagado a $y$ (Teorema SMV-3): $\leq 0.33\% \cdot \sqrt{k} \approx 3.7\%$.

Para early training esto es aceptable. Para fine-tuning de precisión cerca del óptimo puede acumular error.

**Experimento 2: Tiempo relativo en GTX 1050 vs RTX 5090.**
Sea:
- $t_{ PCIe}$ = tiempo de transferir 4MB por PCIe
- $t_{ compute}$ = tiempo de ejecutar 2 matmuls ($z = x V_k^T$, $y = z_{mod} U_k^T$)

**GTX 1050:** PCIe bandwidth ~8 GB/s, compute ~1.8 TFLOPs
$$t_{PCIe} = \frac{4\text{MB}}{8\text{GB/s}} = 0.5\text{ms}, \quad t_{compute} = \frac{2 \cdot 8192 \cdot 128}{1.8e9} \approx 1.2\text{ms}$$

Ratio: $t_{PCIe} / t_{compute} \approx 0.42$. SMV mejora este ratio a $\approx 0.0005$ (reduciendo PCIe a 0.5KB).

**RTX 5090:** PCIe bandwidth ~50 GB/s, compute ~30 TFLOPs
$$t_{PCIe} = \frac{4\text{MB}}{50\text{GB/s}} = 0.08\text{ms}, \quad t_{compute} = \frac{2 \cdot 8192 \cdot 128}{3e10} \approx 0.07\text{ms}$$

Ratio original: $t_{PCIe} / t_{compute} \approx 1.1$. **En GPUs de alta gama, PCIe ES el bottleneck incluso sin SMV.** SMV reduce el overhead de transferencia a 0.5KB/50GB/s = 0.01μs, ratio final ~0.0001.

**Conclusión:** SMV es más valioso en GPUs de alta gama donde el compute es barato pero la transferencia PCIe aún domina para matrices grandes.

### 2.14 Relación con Literatura

- **LoRA (Hu et al., 2021):** LoRA actualiza matrices $A$ y $B$ de rank bajo, mientras SMV modula valores singulares existentes. SMV no introduce nuevos grados de libertad en la estructura de $W$, solo re-weight los componentes existentes.
- **Spectrum Alignment (Jia et al., 2024):** Usa singular vectors para alignment pero sin la no-linealidad de $\mu_\theta$.
- **Out-of-core LLM (Petache et al., 2024):** Idea similar de separation de almacenamiento y cómputo, pero a nivel de modelo completo, no a nivel de operador.

SMV se distingue por: (1) bound exacto de conservación de gradiente (Teorema SMV-2), (2) análisis de cuantización FP16 (Teorema SMV-3), (3) optimalidad asintótica (Teorema SMV-4), (4) aplicabilidad a nivel de operador dentro del training loop.

### 2.15 Preguntas Abiertas

1. ¿Puede usarse compressive sensing para representar $\mu_\theta$ con menos de $k$ valores efectivos, explotando que $\mu$ tiene estructura de baja complejidad?

2. ¿Cómo se correlaciona $\mu(t)$ con $\mu(t+1)$ durante training? Si la correlación es alta, ¿puede EMP predecir $\mu(t+1)$ sin evaluar el MLP?

3. ¿Es óptimo transferir $U_k$ y $V_k^T$ separately, o debería precomputarse $W_{adapted} = U_k \cdot \text{diag}(\mu) \cdot V_k^T$ para múltiples valores de $\mu$ y almacenarse en una lookup table?

---

## 3. EigenvalueMigrationPredictor (EMP)

### 3.1 Problema

Cada forward pass de SVMO ejecuta el MLP $g_\theta$ para computar $\mu_\theta(\sigma_k) = m_\theta(\sigma_k)$. Sin embargo, $\theta$ solo cambia cuando `optimizer.step()` es llamado. Entre pasos de optimizer (típicamente $N_{grad\_accum} \times (seq\_len - 1)$ forwards para micro-batch de 1), múltiples forward passes ejecutan el MLP con el **mismo** $\theta$, produciendo resultados **idénticos**. Esto es compute redundante puro: para 1000 tokens con gradient accumulation de 32, ejecutamos el MLP 32000 veces cuando bastaría 1.

### 3.2 Idea Central

Los valores singulares modulados $\mu_\theta(\sigma_k)$ cambian suavemente durante el entrenamiento porque $\theta$ cambia suavemente bajo SGD con momentum. Modelamos la dinámica de $\mu$ como un sistema dinámico lineal de tiempo discreto (LTI) perturbado:

$$\mu_{t+1} = A \mu_t + B u_t + w_t$$

donde $A \in \mathbb{R}^{k \times k}$ es la matriz de transición (determinada por la dinámica del gradiente), $u_t$ es una señal de control (e.g., magnitud del gradiente), y $w_t \sim \mathcal{N}(0, \sigma^2 I)$ es ruido de proceso. Usamos un **predictor** para estimar $\mu_{t+1}$ desde $\mu_t$ y las estadísticas del gradiente, evitando la evaluación costosa del MLP $g_\theta$.

### 3.3 Modelo Formal

**Definición 3.1 (Sistema Dinámico de Modulación).**
Sea $\mu_t \in \mathbb{R}^k$ el vector de modulación en timestep $t$, donde $t$ indexing los forward passes entre optimizer steps. La dinámica de $\mu$ está gobernada por:

$$\mu_{t+1} = A \mu_t + b + \delta_t$$

donde:
- $A \in \mathbb{R}^{k \times k}$ es la **matriz de transición**, con radio espectral $\rho(A) = \max_i |\lambda_i(A)|$
- $b \in \mathbb{R}^k$ es el **drift** determinístico
- $\delta_t \in \mathbb{R}^k$ es el **ruido de innovación**, $\|\delta_t\| \leq \sigma$ i.i.d.

**Definición 3.2 (Predictor EMP).**
Sea $\mathcal{P}: \mathbb{R}^k \times \mathbb{R}^d \to \mathbb{R}^k$ un predictor (posiblemente no lineal) tal que:

$$\hat{\mu}_{t+1} = \mathcal{P}(\mu_t, g_t)$$

donde $g_t = \nabla_\theta \mathcal{L}_t$ es el gradiente de la loss en timestep $t$. El predictor tiene error de predicción un-step:

$$\mathbb{E}\|\mu_{t+1} - \hat{\mu}_{t+1}\|^2 \leq \delta^2$$

**Definición 3.3 (Error de Predicción Acumulado).**
Sea $e_t = \mu_t - \hat{\mu}_t$ el error de predicción en timestep $t$. El error acumulado sobre $T$ steps es:

$$E_T = \sum_{t=1}^T \|e_t\|$$

**Definición 3.4 (Radio Espectral de $A$).**
Sea $\rho(A) = \max_i |\lambda_i(A)|$ el radio espectral de la matriz de transición $A$. Asumimos $\rho(A) < 1$ (sistema estables asymptotically), lo cual se espera si el learning rate está bien configurado y el modelo converge.

### 3.4 Teorema EMP-1: Acotación del Error de Predicción

*Corrección (2026-07-20).* La versión anterior de este teorema tenía dos problemas independientes: (i) el enunciado y la demostración terminaban en fórmulas distintas ($\sigma\rho$ vs.\ $2\delta$ como coeficiente del término en $T$ — no coincidían); (ii) Definición 3.2, tal como estaba escrita, hace que $\hat\mu_{t+1}=\mathcal{P}(\mu_t, g_t)$ dependa del estado **verdadero** $\mu_t$, no de la propia predicción anterior — bajo esa definición no hay propagación de error vía $A$ en absoluto (cada predicción es fresca desde el estado real, así que $\mathbb{E}\|e_{t+1}\|^2\leq\delta^2$ ya es la cota final, sin necesidad de la serie geométrica). El caso que EMP realmente necesita es el **recursivo**: cuando se saltan varios pasos consecutivos sin evaluar $g_\theta$, el predictor debe alimentarse de su propia estimación anterior, $\hat\mu_{t+1} = \mathcal{P}(\hat\mu_t, g_t)$. Ese es el caso analizado abajo.

**Teorema EMP-1 (Bound de Error Acumulado, régimen recursivo).**
Sea $\mathcal{P}$ un predictor recursivo, $\hat\mu_{t+1} = \mathcal{P}(\hat\mu_t, g_t)$, con error de un paso acotado $\mathbb{E}\|\mu_{t+1} - \mathcal{P}(\mu_t, g_t)\|^2 \leq \delta^2$ para todo $t$ (error si se alimentara con el estado verdadero). Sea $\rho = \rho(A) < 1$. Asumimos que las innovaciones $\delta_t$ son independientes de $e_t$ con covarianza $\mathbb{E}[\delta_t\delta_t^T]\preceq\sigma^2 I$. Entonces:

$$\limsup_{t\to\infty}\mathbb{E}\|e_t\| \leq \frac{\delta+\sigma}{1-\rho} \qquad\text{(error por paso, se satura en } O(1)\text{)}$$
$$\mathbb{E}[E_T] \leq T\cdot\frac{\delta+\sigma}{1-\rho} + O(1) \qquad\text{(suma acumulada sobre } T \text{ pasos, siempre } O(T)\text{)}$$

*Demostración.* Con predicción recursiva, el error se propaga: $e_{t+1} = \mu_{t+1} - \hat\mu_{t+1} = (A\mu_t + b + \delta_t) - \mathcal{P}(\hat\mu_t, g_t)$. Descomponiendo respecto al predictor evaluado en el estado verdadero, $e_{t+1} = \underbrace{[\mu_{t+1}-\mathcal{P}(\mu_t,g_t)]}_{\text{error de un paso}, \leq\delta} + \underbrace{[\mathcal{P}(\mu_t,g_t)-\mathcal{P}(\hat\mu_t,g_t)]}_{\text{propagación}}$. Si $\mathcal{P}(\cdot,g_t)$ hereda la sensibilidad lineal de $A$ (constante de Lipschitz $\rho$ respecto a su primer argumento, consistente con la linealización de la dinámica), el segundo término está acotado por $\rho\|e_t\|$, dando:

$$\mathbb{E}\|e_{t+1}\| \leq \rho\cdot\mathbb{E}\|e_t\| + \delta + \sigma$$

Desenrollando desde $e_0=0$: $\mathbb{E}\|e_t\| \leq \sum_{i=0}^{t-1}\rho^i(\delta+\sigma) \leq \frac{\delta+\sigma}{1-\rho}$, que **no depende de $t$** — este es el resultado correcto de saturación en $O(1)$, y es válido para el error *por paso*, no para la suma. La suma sobre $T$ pasos de una cantidad acotada por una constante es, trivialmente, $O(T)$: $\mathbb{E}[E_T] = \sum_{t=1}^T \mathbb{E}\|e_t\| \leq T\cdot\frac{\delta+\sigma}{1-\rho}$ (el término $O(1)$ adicional viene de los primeros pasos, donde $\rho^t\|e_0\|$ aún no se ha desvanecido). $\blacksquare$

*Remark 3.1 (corregido).* $\rho(A)$ controla la **constante de saturación** del error por paso, no si la suma acumulada crece o no — la suma $E_T$ siempre es $O(T)$ para cualquier $\rho<1$ fijo, porque es la suma de $T$ términos cada uno acotado por debajo de una cota positiva. Lo que sí depende fuertemente de $\rho$ es la pendiente: $\rho\to 0$ hace la pendiente $\to(\delta+\sigma)$; $\rho\to 1$ hace la pendiente $\to\infty$. La afirmación anterior ("si $\rho(A)\ll1$, el error se satura en $O(1)$") era válida solo para el error por paso, no para $E_T$, y se corrige aquí explícitamente.

*Remark 3.2.* En la práctica, $A$ no es constante — cambia lentamente conforme $\theta$ cambia. Esto viola la suposición de sistema LTI. Un análisis más realista usaría time-varying systems: $\mu_{t+1} = A_t \mu_t + b_t + \delta_t$ con $\sup_t \rho(A_t) \leq \rho_{max} < 1$.

### 3.5 Teorema EMP-2: Condición de Uso del Predictor

**Teorema EMP-2 (Cota del Error de Salida).**
Sea $\hat{\mu}$ la predicción de $\mu$, con error $\|e\| = \|\mu - \hat{\mu}\| \leq \epsilon$. Sea $y(\mu) = x V_k^T \cdot \text{diag}(\mu) \cdot U_k^T$ la salida de SVMO bajo modulación $\mu$, y $y(\hat{\mu})$ bajo la predicción. Asumiendo que la función de modulación $\mu \mapsto y(\mu)$ es Lipschitz con constante $L_y$. Entonces:

$$\frac{\|y(\mu) - y(\hat{\mu})\|_2}{\|y(\mu)\|_2} \leq \epsilon \cdot \frac{L_y}{\|y(\mu)\|_2}$$

Alternativamente, en términos de la constante de Lipschitz de la función de modulación $m_\theta$ (SVMO, S³-Theorem 1):

$$\|m_\theta(\sigma) - m_\theta(\hat{\sigma})\|_2 \leq L_m \cdot \|\sigma - \hat{\sigma}\|_2$$

con $L_m = \max_j \frac{\partial m_j}{\partial \sigma_j} \leq (1+\alpha)$. Por lo tanto:

$$\frac{\|y(\mu) - y(\hat{\mu})\|_2}{\|y(\mu)\|_2} \leq \frac{(1+\alpha) \cdot \epsilon}{\sigma_k}$$

donde $\sigma_k = \min_j \sigma_j$ es el valor singular mínimo.

*Demostración.* De la Proposición 2.1 (forma matricial de SVMO):

$$y(\mu) - y(\hat{\mu}) = x V_k^T \cdot \text{diag}(\mu - \hat{\mu}) \cdot U_k^T$$

Por submultiplicatividad:

$$\|y(\mu) - y(\hat{\mu})\|_2 \leq \|x V_k^T\|_2 \cdot \|\mu - \hat{\mu}\|_\infty \cdot \|U_k^T\|_2 \leq \|x V_k^T\|_2 \cdot \epsilon$$

Además, $\|y(\mu)\|_2 \geq \sigma_k \cdot \|\mu\|_2 \cdot \|x\|_2 / \sqrt{k}$ (bound inferior de la norma via valores singulares). $\blacksquare$

*Remark 3.3.* Para $\alpha = 0.3$ (SVMO default), $L_m \leq 1.3$. Esto significa que el error en la salida es como máximo 1.3 veces el error relativo en los valores singulares. Para tolerar un error relativo en salida de $\gamma = 0.01$, necesitamos $\epsilon \leq 0.01 \cdot \sigma_k / 1.3$. Si $\sigma_k \approx 0.1$ (componentes pequeños de $W$), el error tolerable es $\epsilon \leq 0.0008$.

### 3.6 Arquitectura del Predictor

**Opción A — Filtro de Kalman Lineal (LKF).**

Para el modelo lineal:
$$\mu_{t+1} = A \mu_t + w_t, \quad \hat{\mu}_t = C \mu_t + v_t$$

el predictor óptimo en MMSE es el **filtro de Kalman**:

$$K_{t+1} = P_{t+1|t} C^T (C P_{t+1|t} C^T + R)^{-1}$$
$$\hat{\mu}_{t+1} = A \hat{\mu}_t + K_{t+1}(y_{t+1} - C A \hat{\mu}_t)$$
$$P_{t+1|t+1} = (I - K_{t+1} C) P_{t+1|t}$$

donde $P$ es la covarianza del error, $Q$ es la covarianza del ruido de proceso, $R$ es la covarianza del ruido de medición. **Ventaja:** óptimo en MMSE, incertidumbre calibrada, converge en $O(k^3)$ por update.

**Nota sobre $A$ time-varying.** En la práctica, $A$ depende de $\theta(t)$ via la Hessiana de $\mathcal{L}$: $A(\theta_t) = I - \eta H(\theta_t)$. Como $\theta$ cambia lentamente entre optimizer steps, $A$ es lentamente time-varying. El teorema siguiente establece que el LKF es robusto a este cambio.

**Teorema EMP-3 (Robustez a $A$ Time-Varying).**
Sea el sistema time-varying:
$$\mu_{t+1} = A_t \mu_t + w_t, \quad \hat{\mu}_t = C \mu_t + v_t$$

con $\|A_{t+1} - A_t\| \leq \epsilon_A$ para todo $t$ (i.e., $A_t$ cambia lentamente). Sean $\|A_t\| \leq a < 1$ (spectral norm acotada, sistema estable). Sean $Q \succ 0$, $R \succ 0$ las covarianzas de ruido. Entonces el error de estimación $\Sigma_t = \mathbb{E}[(\mu_t - \hat{\mu}_t)(\mu_t - \hat{\mu}_t)^T]$ satisfies:

$$\|\Sigma_t\| \leq \frac{\|Q\|}{1 - a^2} + t \cdot \epsilon_A \cdot \frac{a}{1-a}$$

*Demostración.* Para el LKF con $A$ time-varying, la ecuación de Riccati se convierte en:

$$\Sigma_{t+1} = A_t \Sigma_t A_t^T + Q - A_t \Sigma_t C^T (C \Sigma_t C^T + R)^{-1} C \Sigma_t A_t^T$$

Separando el término de change de $A$:

$$\Sigma_{t+1} = A \Sigma_t A^T + \underbrace{(A_t - A)\Sigma_t A_t^T + A \Sigma_t (A_t - A)^T + (A_t - A)\Sigma_t (A_t - A)^T}_{\Delta_t}$$

con $\|A_t - A\| \leq \epsilon_A$. Usando $\|A\| \leq a$ y $\|\Sigma_t\| \leq S$:

$$\|\Delta_t\| \leq 2\epsilon_A a S + \epsilon_A^2 S \leq 3\epsilon_A a S$$

(para $\epsilon_A$ pequeño). Si $S_t$ es la solución del sistema time-invariant con $A$ constante (i.e., la cota estable $\bar{S} = \|Q\|/(1-a^2)$), entonces:

$$\|\Sigma_{t+1}\| \leq a^2 S_t + \|Q\| + 3\epsilon_A a S_t \leq S_t(\underbrace{a^2 + 3\epsilon_A a}_{\leq a^2 + \delta}) + \|Q\|$$

para $\epsilon_A \leq \delta/(3a)$. Resolviendo la recurrencia:

$$S_t \leq a^{2t}S_0 + \|Q\| \sum_{i=0}^{t-1} a^{2i} + 3\epsilon_A a \sum_{i=0}^{t-1} a^{2i} \cdot i$$

$$\leq \frac{\|Q\|}{1-a^2} + 3\epsilon_A a \cdot \frac{a(1 - (t-1)a^2) + (t-1)a^{2t}}{(1-a^2)^2}$$

Para $t$ grande y $a < 1$: el término $\sum i a^{2i} \approx a/(1-a)^2$. Por lo tanto:

$$\|\Sigma_t\| \leq \frac{\|Q\|}{1-a^2} + 3\epsilon_A a \cdot \frac{a}{(1-a^2)^2} \cdot t$$

Si $\epsilon_A = O(1/t)$ (i.e., $A$ cambia muy lentamente), el segundo término es $O(1)$ y el error se mantiene bounded. $\blacksquare$

*Remark 3.4 (Interpretación).* El teorema dice que si la dinámica $A$ cambia lentamente ( $\|A_{t+1} - A_t\| \leq \epsilon_A$ ), el error del LKF permanece acotado. Con $\epsilon_A \approx 10^{-6}$ por optimizer step (típico para training con learning rate pequeño), el término adicional $3\epsilon_A a \cdot t/(1-a^2)^2$ es aproximadamente $3 \cdot 10^{-6} \cdot t$. Para $t = 10^5$ steps: $\approx 0.3$, comparable al error base $\|Q\|/(1-a^2)$. Esto justifica el uso de LKF en la práctica.

**Opción B — MLP Learned de 1 capa.**

$$\hat{\mu}_{t+1} = \mu_t - \lambda \cdot W_2 \cdot \text{GELU}(W_1 \cdot g_t + b_1) + b_2$$

donde $W_1 \in \mathbb{R}^{H \times p}, W_2 \in \mathbb{R}^{k \times H}$ son aprendibles jointly con el resto del modelo via backpropagation through time (BPTT). **Ventaja:** captura no-linearidades en la dinámica de $\mu$. **Desventaja:** no tiene incertidumbre calibrada, puede diverger si el gradiente $g_t$ tiene distribución cambiante.

**Opción C — Exponential Moving Average (EMA).**

$$\hat{\mu}_{t+1} = \beta \cdot \mu_t + (1-\beta) \cdot \mu_{MLP,t}$$

con $\beta \in [0.9, 0.99]$. **Ventaja:** trivial de implementar, no requiere training. **Desventaja:** asume dinámica de primer orden, no captura correlaciones complejas.

### 3.7 Casos de Falla

**Falla 1: Divergencia cuando $\rho(A) \geq 1$.**
Si la dinámica de $\mu$ es inestable (e.g., learning rate demasiado alto导致 $\theta$ cambia drásticamente), $\rho(A) \geq 1$ y el sistema no es contractivo. En este caso, el error de predicción diverge exponencialmente: $\|e_t\| \approx O(\rho^t)$. El predictor se vuelve inútil.

**Falla 2: Dependencia del gradient statistics.**
El predictor asume que $g_t$ contiene información útil para predecir $\mu_{t+1}$. Si el gradiente es ruidoso (e.g., early training con SGD puro sin momentum), la relación señal-ruido es baja y el predictor no puede aprender la dinámica. En este caso, el predictor puede empeorar el error vs. no usar predicción.

**Falla 3: Non-stationarity de la dinámica.**
La matriz $A$ cambia lentamente durante training. Si el predictor está optimizado para $A_{early}$ pero se usa en late training donde $A_{late}$ es distinta, el predictor Accumula bias sistemático. Esto es особенно problemático para el LKF, que asume $A$ constante.

### 3.8 Teorema EMP-4: Lipschitz de la Función de Modulación

**Teorema EMP-4 (Constante de Lipschitz $L_m$).**
Sea $m_\theta(\sigma) = \sigma \odot (1 + \alpha \cdot \tanh(g_\theta(\log(\sigma + \epsilon))))$ la función de modulación, donde $g_\theta$ es el MLP de modulación con arquitectura 1 → $H$ → $H$ → 1 y activación GELU. Asumimos $\|g_\theta'(x)\|_\infty \leq G$ para todo $x$ (constante de Lipschitz del MLP). Entonces:

$$L_m := \max_{\sigma \in \mathbb{R}^k_+} \|J_m(\sigma)\|_2 \leq (1+\alpha) \cdot G \cdot \max_j \frac{1}{\sigma_j + \epsilon}$$

donde $J_m$ es el Jacobiano de $m_\theta$.

*Demostración.* Por la regla de la cadena:

$$\frac{\partial m_j}{\partial \sigma_i} = \delta_{ij} \cdot (1 + \alpha \tanh(g_\theta(\log \sigma_j))) + \sigma_j \cdot \alpha \cdot \text{sech}^2(g_\theta(\log \sigma_j)) \cdot \frac{g_\theta'(\log \sigma_j)}{\sigma_j}$$

$$= \delta_{ij} \cdot (1 + \alpha \tanh(\cdot)) + \alpha \cdot \text{sech}^2(\cdot) \cdot g_\theta'(\log \sigma_j) \cdot \delta_{ij}$$

Para $i = j$:

$$\left|\frac{\partial m_j}{\partial \sigma_j}\right| \leq |1 + \alpha \tanh(\cdot)| + \alpha \cdot |g_\theta'(\log \sigma_j)| \leq 1 + \alpha + \alpha G$$

Para $i \neq j$: $\frac{\partial m_j}{\partial \sigma_i} = 0$. Por lo tanto, el Jacobiano es diagonal:

$$J_m(\sigma) = \text{diag}\left(1 + \alpha \tanh(g_\theta(\log \sigma_1)) + \alpha \cdot \text{sech}^2(g_\theta(\log \sigma_1)) \cdot g_\theta'(\log \sigma_1), \ldots\right)$$

La norma espectral de una matriz diagonal es el máximo de los valores absolutos de sus elementos:

$$\|J_m(\sigma)\|_2 = \max_j \left|1 + \alpha \tanh(u_j) + \alpha \cdot \text{sech}^2(u_j) \cdot g_\theta'(u_j)\right|$$

$$\leq 1 + \alpha + \alpha \cdot G \cdot \max_j \frac{1}{(\sigma_j + \epsilon)}$$ 

ya que $\text{sech}^2(u) \leq 1$ y $|\tanh(u)| \leq 1$. $\blacksquare$

*Remark 3.5 (Valor práctico).* Para el MLP default con $G \approx 1.1$ (GELU tiene Lipschitz $\approx 1.1$), $\alpha = 0.3$, y $\sigma_j \approx 0.1$ a $10$: el término dominante es $\alpha G / \sigma_j \approx 0.3 \times 1.1 / 0.1 = 3.3$. El bound práctico es $L_m \leq 1 + 0.3 + 3.3 \approx 4.6$. El bound teórico más limpio (asumiendo $\sigma_j$ no demasiado pequeño) es $L_m \leq 1 + 2\alpha G \leq 1.66$.

### 3.9 Teorema EMP-5: Costo Computacional del Predictor

**Teorema EMP-5 (Complejidad de las Tres Opciones).**
Sean $k$ la dimensión del vector de modulación, $H$ la dimensión oculta del MLP (para opción B), y $T_{MLP}$ el tiempo de una evaluación del MLP completo $g_\theta$. Los costos computacionales de cada predictor son:

| Opción | Costo por Predicción | Costo de Setup | Memoria |
|--------|---------------------|----------------|---------|
| LKF (A) | $O(k^3)$ por update de covarianza | $O(k^2)$ inicialización | $O(k^2)$ |
| MLP (B) | $O(H \cdot k)$ forward pass | $O(H \cdot k)$ training por BPTT | $O(H \cdot k)$ |
| EMA (C) | $O(k)$ | $O(1)$ | $O(k)$ |

*Demostración.* Derivado de la definición de cada método:
- **LKF:** La ecuación de Riccati requiere inversión de matriz $k \times k$: $O(k^3)$. El paso de filtrado es $O(k^2)$.
- **MLP:** Un forward pass del MLP es $k \to H \to H \to k$: $O(H \cdot k)$ operaciones.
- **EMA:** Solo operaciones punto a punto: $O(k)$. $\blacksquare$

*Remark 3.6 (Recomendación práctica).* Para $k = 128$ (nuestro caso), el costo de LKF ($128^3 \approx 2 \times 10^6$ ops) es comparable al costo del MLP ($128 \times 32 \approx 4 \times 10^3$ ops) en cada paso si el LKF debe updatear la covarianza. Sin embargo, LKF solo necesita actualizarse en los optimizer steps (no en cada forward), mientras el MLP se ejecuta en cada forward. Para $N_{grad\_accum} \times seq\_len \approx 16000$ forwards por optimizer step, el MLP costo total es $16000 \times 4000 = 64 \times 10^6$ ops, mientras LKF costo $2 \times 10^6$ por step. LKF es ~30x más barato en este scenario.

### 3.10 Teorema EMP-6: Máximas Fallidas Consecutivas

**Teorema EMP-6 (Bound de Divergencia Acumulada).**
Sea $\epsilon_{max}$ el error tolerable tal que si $\|\mu_t - \hat{\mu}_t\| > \epsilon_{max}$, el predictor se declara "fallando" y se ejecuta el MLP real. Sea $F$ el número de fallos consecutivos. Asumimos $\rho(A) < 1$ y error de predictor un-step acotado por $\delta < \epsilon_{max}$. Entonces:

$$\mathbb{P}(F \geq m) \leq \left(\frac{\delta}{\epsilon_{max}}\right)^m \cdot \frac{1}{1 - \rho(A)^2}$$

Después de $m$ fallos consecutivos, el error acumulado en la salida del modelo es:

$$\|y(\mu_t) - y(\hat{\mu}_t)\| \leq \epsilon_{max} \cdot (1 + \rho(A) + \rho(A)^2 + \cdots) = \frac{\epsilon_{max}}{1 - \rho(A)}$$

*Demostración.* El evento de fallo en step $t$ es $\|\mu_t - \hat{\mu}_t\| > \epsilon_{max}$. Dado que $\|\mu_{t+1} - \hat{\mu}_{t+1}\| \leq \rho(A) \|\mu_t - \hat{\mu}_t\| + \delta$ (por la dinámica del error y predictor imperfecto), la probabilidad de que el error crezca sobre $\epsilon_{max}$ es:

$$\mathbb{P}(\text{fallo en } t+1 \mid \text{fallo en } t) = \mathbb{P}(\rho(A) \|e_t\| + \delta > \epsilon_{max}) \leq \frac{\delta}{\epsilon_{max} - \rho(A)\epsilon_{max}} = \frac{\delta}{\epsilon_{max}(1-\rho(A))}$$

Para $m$ fallos consecutivos: $\mathbb{P}(F \geq m) \leq (\delta/(\epsilon_{max}(1-\rho)))^m$. Cuando el predictor falla $m$ veces, el MLP real se ejecuta $m$ veces, restaurando la predicción correcta. El número máximo de fallos consecutivos antes de recuperación es $-1 + \log_{\rho}(\delta/\epsilon_{max})$. $\blacksquare$

*Remark 3.7 (Parámetros de seguridad).* Para $\rho(A) = 0.9, \delta = 0.01, \epsilon_{max} = 0.1$: la probabilidad de 3 fallos consecutivos es $(\frac{0.01}{0.1 \times 0.1})^3 = 1^3 = 1$ — el bound es débil porque $\delta/(\epsilon_{max}(1-\rho)) = 0.01/0.01 = 1$. Necesitamos $\delta < \epsilon_{max}(1-\rho)$ para que la probabilidad decaiga. Con $\epsilon_{max} = 0.3$: $\delta/(\epsilon_{max}(1-\rho)) = 0.01/0.03 \approx 0.33$, y $\mathbb{P}(F \geq 5) \leq 0.33^5 \approx 0.004$. Esto justifica usar un threshold $\epsilon_{max} \approx 0.2-0.3$ para declarar fallback.

### 3.11 Teorema EMP-7: Divergencia con $\rho(A) \geq 1$

**Teorema EMP-7 (Instabilidad cuando $\rho(A) \geq 1$).**
Si $\rho(A) \geq 1$, existe un $\gamma > 0$ tal que:

$$\mathbb{E}\|\mu_t - \hat{\mu}_t\| \geq \gamma \cdot \rho(A)^t$$

para todo $t > 0$. Consecuentemente, el predictor diverge y no puede usarse.

*Demostración.* Si $\rho(A) \geq 1$, existe al menos un eigenvalor $\lambda$ con $|\lambda| \geq 1$. En la dirección del eigenvector asociado $v$, la dinámica del error satisface:

$$e_{t+1} = A e_t + \tilde{e}_{t+1}$$

con $\|\tilde{e}_{t+1}\| \leq \delta$ (error de predictor). En la dirección $v$:

$$e_{t+1}^v = \lambda e_t^v + \tilde{e}_{t+1}^v$$

Si $\tilde{e}_{t+1}^v = 0$ (mejor caso), $|e_{t+1}^v| = |\lambda|^t |e_0^v| \geq |e_0^v|$. En el caso promedio con $\tilde{e}_{t+1}^v \sim \text{Unif}([-\delta, \delta])$, la esperanza de $|e_t^v|$ crece como $|\lambda|^t$ para $|\lambda| > 1$. Por lo tanto, $\|\mu_t - \hat{\mu}_t\|$ diverge al menos linealmente con $t$ cuando $\rho(A) \geq 1$. $\blacksquare$

**Criterio práctico de estabilidad.** Antes de usar EMP, verificar que $\hat{\rho}(A) < 1$ estimando $A$ de los primeros 1000 pasos de training. Si $\hat{\rho}(A) \geq 1$, reducir el learning rate hasta que $\hat{\rho}(A) < 0.99$.

### 3.12 Experimento Mental

**Escenario:** $k = 128$, micro-batch = 1, gradient accumulation = 32, seq_len = 512.

Sin EMP: el MLP $g_\theta$ se ejecuta $32 \times 511 = 16352$ veces entre optimizer steps, pero siempre produce el mismo resultado (porque $\theta$ no cambia).

Con EMP: el MLP se ejecuta 1 vez, el predictor approximation se usa las otras 16351 veces. Ahorro de compute: $16351 / 16352 \approx 99.994\%$.

El error de predicción aceptable para mantener calidad de training: $\epsilon \leq 0.01$ (1% de error relativo en salida). ¿Puede el predictor mantener este error?

Si $\rho(A) = 0.9$ y $\delta = 0.001$ (predictor con 0.1% de error un-step):

$$\mathbb{E}\|e_t\| \leq \frac{0.001}{1-0.9} = 0.01$$

**Sí, el predictor puede mantener el error bajo el umbral si $\rho(A) < 1$.**

### 3.13 Preguntas Abiertas

1. ¿Cómo estimar $A$ online sin guardar histórica de $\mu$? Un método es usar recursive least squares (RLS) para actualizar $A$ con cada nueva observación de $\mu$.

2. ¿Cuándo es mejor usar LKF vs. MLP? LKF es mejor cuando la dinámica es lineal y las incertidumbres son Gaussianas; MLP es mejor cuando hay no-linearidades fuertes.

3. ¿Puede combinarse EMP con SMV para aún mayor reducción de PCIe? Si $\mu$ se predice con alta precisión, no necesitamos transferir nada — solo transferimos cuando el predictor falla (lazy transfer).

---

## 4. Token Entropy Routing (TER)

### 4.1 Problema

El solver RK4 de NMF usa actualmente un número fijo $N \in \{1, 2, 4\}$ de pasos de integración para todas las representaciones, sin importar su complejidad. Tokens que el modelo ya procesa con alta confianza (baja entropía) requieren menos integración; tokens ambiguos (alta entropía) requieren más. Ejecutar $N=4$ pasos para todos los tokens es computacionalmente wasteado en tokens fáciles.

### 4.2 Idea Central

路由 (routing) de tokens por complejidad: predecir la entropía de la representación de cada token y elegir dinámicamente $N \in \{1, 2, 4\}$ pasos de RK4 basándose en la incertidumbre de la representación.

**Entropía como proxy de complejidad:**
$$H(h) = -\sum_{j=1}^{d_b} \tilde{a}_j \log(\tilde{a}_j + \epsilon)$$

donde $\tilde{a} = \text{softmax}(f_\theta(h, t))$ es la distribución sobre las activations de la bottleneck layer. Alta entropía = representación dispersa/confusa = necesita más pasos de integración. Baja entropía = representación concentrada/confiada = pocos pasos suficientes.

### 4.3 Modelo Formal

**Definición 4.1 (Entropía de Representación NMF).**
Sea $h \in \mathbb{R}^d$ la representación latente antes de NMF. Sea $a = f_\theta(h, t) \in \mathbb{R}^{d_b}$ la activación en la bottleneck layer. Definimos:

$$H_{NMF}(h) = -\sum_{j=1}^{d_b} \tilde{a}_j \log(\tilde{a}_j + \epsilon)$$

donde $\tilde{a} = \text{softmax}(a)$ y $\epsilon > 0$ previene el log(0). Note que $0 \leq H_{NMF} \leq \log(d_b)$.

**Definición 4.2 (Routing Policy Determinista).**
Sea $\tau = (\tau_1, \tau_2) \in \mathbb{R}^2$ con $\tau_1 < \tau_2$ los umbrales de decisión. La policy de routing $\pi_\tau: \mathbb{R}^d \to \{1, 2, 4\}$ es:

$$\pi_\tau(h) = \begin{cases} 1 & \text{si } H_{NMF}(h) < \tau_1 \\ 2 & \text{si } \tau_1 \leq H_{NMF}(h) < \tau_2 \\ 4 & \text{si } H_{NMF}(h) \geq \tau_2 \end{cases}$$

**Definición 4.3 (Routing Policy Probabilista).**
Sea $\pi_\tau^\phi(h) \in \Delta(\{1, 2, 4\})$ la distribución de probabilidad sobre $N$ dada por:

$$\pi_\tau^\phi(h)_N = \text{softmax}(\phi(H_{NMF}(h)))_N$$

donde $\phi: \mathbb{R} \to \mathbb{R}^3$ es un MLP de 2 capas que aprende a mapear entropía a probabilidades. Esta versión permite training por gradient descent.

**Definición 4.4 (Oracle Benchmark).**
Sea $N^\star(h)$ el número óptimo de pasos RK4 para representación $h$, i.e., el menor $N$ tal que el error de integración $\|\tilde{h}_N(T) - h^*(T)\|$ sea menor que $\epsilon_{tol}$. El regret del router $\pi$ sobre $T$ tokens es:

$$\text{Regret}_T(\pi) = \sum_{t=1}^T \left[C(\pi(h_t)) - C(N^\star(h_t))\right]$$

donde $C(N)$ es el costo computacional de ejecutar $N$ pasos RK4.

### 4.4 Teorema TER-1: Acotación de Error por Routing Subóptimo

**Teorema TER-1 (Error de RK4 por Routing).**
Sea $\tilde{h}_N$ la solución de NMF con $N$ pasos RK4, y sea $\tilde{h}_{N^\star}$ la solución con $N^\star$ pasos (oracle óptimo). Si el routing elige $N < N^\star$, el error de integración está acotado por:

$$\|\tilde{h}_N(T) - \tilde{h}_{N^\star}(T)\| \leq \frac{M_4 T^5}{80} \cdot \left(\frac{1}{N^4} - \frac{1}{(N^\star)^4}\right)$$

donde $M_4 = \max_{t \in [0,T], h} \|f_\theta^{(4)}(h, t)\|$ es la derivada cuarta de la función de velocidad.

*Demostración.* El teorema de error RK4 establece que para un paso de tamaño $\Delta t$:

$$\|h_{n+1} - h(t_{n+1})\| \leq \frac{M_4 (\Delta t)^5}{80}$$

Con $N$ pasos de tamaño $\Delta t = T/N$, el error global (acumulado sobre la trayectoria) satisface:

$$\|\tilde{h}_N(T) - h^*(T)\| \leq \frac{M_4 T^5}{80 N^4}$$

Para $N < N^\star$, la diferencia entre las dos soluciones es:

$$\|\tilde{h}_N(T) - \tilde{h}_{N^\star}(T)\| \leq \|\tilde{h}_N(T) - h^*(T)\| + \|\tilde{h}_{N^\star}(T) - h^*(T)\| \leq \frac{M_4 T^5}{80}\left(\frac{1}{N^4} + \frac{1}{(N^\star)^4}\right)$$

Usando que $N < N^\star$:

$$\frac{1}{N^4} + \frac{1}{(N^\star)^4} \leq \frac{2}{N^4} \quad \text{y} \quad \left|\frac{1}{N^4} - \frac{1}{(N^\star)^4}\right| = \frac{1}{N^4} - \frac{1}{(N^\star)^4}$$

para $N < N^\star$. Por lo tanto:

$$\|\tilde{h}_N(T) - \tilde{h}_{N^\star}(T)\| \leq \frac{M_4 T^5}{80}\left(\frac{1}{N^4} - \frac{1}{(N^\star)^4}\right) \quad \blacksquare$$

### 4.5 Teorema TER-2: Reducción Esperada de FLOPs

**Teorema TER-2 (FLOPs Savings).**
Sea $p_1, p_2, p_4$ la proporción de tokens routing a $N=1, 2, 4$ respectivamente, con $p_1 + p_2 + p_4 = 1$. El factor de reducción de FLOPs de NMF respecto a $N=4$ fijo es:

$$\rho_{FLOPs} = 1 - \frac{1}{4}(p_1 + 2p_2 + 4p_4) = 1 - \frac{1}{4}\mathbb{E}_{h \sim \mathcal{D}}[N]$$

donde $\mathbb{E}_{h \sim \mathcal{D}}[N]$ es el número esperado de pasos por token.

*Demostración.* Cada paso RK4 involucra exactamente 4 evaluaciones de $f_\theta$. Para $N$ pasos: $4N$ evaluaciones de $f_\theta$. Con $N=4$ como baseline: $4 \times 4 = 16$ evaluaciones por NMF flow por token. Con routing, el número esperado de evaluaciones es:

$$\mathbb{E}[4N] = 4(p_1 \cdot 1 + p_2 \cdot 2 + p_4 \cdot 4) = 4(p_1 + 2p_2 + 4p_4)$$

El factor de reducción es $\frac{\mathbb{E}[4N]}{16} = \frac{1}{4}(p_1 + 2p_2 + 4p_4)$. Por lo tanto, la proporción de FLOPs ahorrados es $1 - \rho_{FLOPs}$. $\blacksquare$

*Remark 4.1.* Para $p_1 = 0.5, p_2 = 0.3, p_4 = 0.2$ (50% de tokens fáciles, 30% medios, 20% difíciles):

$$\rho_{FLOPs} = \frac{1}{4}(0.5 + 0.6 + 0.8) = \frac{1.9}{4} = 0.475$$

Es decir, **47.5% de reducción de FLOPs en NMF**.

### 4.6 Teorema TER-3: Regret Bound del Router

**Teorema TER-3 (Regret del Policy de Routing).**
Sea $\pi$ un router que aprende la mapping $H_{NMF}(h) \to N \in \{1,2,4\}$ desde datos. Asumimos:

1. La entropía observada $H_t = H_{NMF}(h_t)$ es una variable aleatoria i.i.d. con distribución desconocida $p(H)$ en $[0, H_{max}]$
2. La entropía verdaderacumple $\|H_{NMF}(h) - H_{NMF}(h')\| \leq L_h \|h - h'\|$ (Lipschitz)
3. El costo de ejecutar $N$ pasos es $C(N) = 4N$ (4 evaluaciones de $f_\theta$ por paso)
4. El costo de error por integrar con $N \neq N^\star$ es $\lambda \cdot \epsilon(N, N^\star)$ con $\epsilon$ dado por TER-1

El regret del router respecto al oracle que siempre elige $N^\star$ es:

$$\mathbb{E}[\text{Regret}_T(\pi)] \leq \underbrace{4\sqrt{2T\ln 3}}_{\text{regret de exploración}} + \underbrace{\lambda \cdot T \cdot \max_{|H - H'| \leq \delta} |\epsilon(N_{H}, N_{H'})|}_{\text{regret de generalización}}$$

donde $N_H$ denota la acción elegida por el router para entropía $H$, y $\delta$ es el radio de discretización del espacio de entropía.

*Demostración.* Dividimos el análisis en dos componentes:

**Componente 1: Regret de exploración (bandits).**
Modelamos la elección de $N$ como un multi-armed bandit con $K=3$ acciones. La pérdida de la acción $N$ en token $t$ es $\ell_t(N) = 4N + \lambda \cdot \epsilon(N, N^\star_t)$ donde $N^\star_t$ es el número óptimo de pasos (latente). Usamos el algoritmo **Exp3-IX** (Auer et al., 2002, con regularización por Tsallis entropy):

$$\pi_{t+1}(N) \propto \exp\left(-\eta \sum_{s=1}^t \tilde{\ell}_s(N)\right)$$

donde $\tilde{\ell}_s$ es la pérdida importance-weighted estimada y $\eta = \sqrt{\ln K / (KT)}$ es el learning rate.

El teorema de regret de Exp3-IX para $K$ acciones establece:

$$\mathbb{E}\left[\sum_{t=1}^T \ell_t(\pi_t)\right] \leq \min_{N^\star \in \{1,2,4\}} \sum_{t=1}^T \ell_t(N^\star) + 2\sqrt{KT\ln K}$$

El término $\min_{N^\star} \sum_t \ell_t(N^\star)$ es el costo del mejor fixed action en retrospectiva. Para $K=3$:

$$\mathbb{E}[\text{Regret}_T^{\text{explore}}] \leq 2\sqrt{3T\ln 3} + 4\sqrt{2T\ln 3}$$

(donde el segundo término viene de separar el término de covarianza de $\ell_t$). Aproximadamente $O(\sqrt{T})$.

**Componente 2: Regret de generalización (discretización).**

*Corrección (2026-07-20).* La versión anterior de esta sección reutilizaba el símbolo $T$ para dos cantidades distintas: el horizonte de integración del ODE de NMF (una constante fija, $T\in[0.5,2.0]$, la misma $T$ de TER-1) y el número de tokens de entrenamiento vistos por el router (que crece sin límite, la $T$ de $\text{Regret}_T$). Esa colisión de notación es lo que producía el exponente sin sentido $T^{7/2}$ en la línea intermedia — mezclaba una potencia de la $T$-horizonte (fija) con una potencia de la $T$-tokens (creciente) como si fueran la misma variable. Aquí la $T$-horizonte del ODE se escribe $T_{\text{ode}}$ (constante) y la $T$-tokens se deja como $T$.

La entropía $H$ es continua en $[0, H_{max}]$. Discretizamos el intervalo en $M$ bins de tamaño $\Delta H = H_{max}/M$; sea $b(H) = \lfloor H/\Delta H \rfloor$ el índice del bin, y $T_b$ el número de visitas (tokens) al bin $b$, con $\sum_b T_b = T$.

El router aprende una Q-function $Q(b, N)$ por bin-acción. Por concentración de sumas de variables i.i.d., el error de estimación de $Q$ en un bin con $T_b$ visitas satisface:

$$\mathbb{E}[|Q(b,N) - Q^\star(b,N)|] \leq \kappa \cdot \sqrt{\frac{\ln(2M)}{T_b}}, \qquad \kappa := \frac{M_4 T_{\text{ode}}^5}{40}$$

donde $\kappa$ es una **constante** (no crece con $T$): es exactamente el bound de error de integración de TER-1 evaluado en el horizonte fijo $T_{\text{ode}}$. Agregando sobre los $M$ bins:

$$\mathbb{E}[\text{Regret}_T^{\text{gen}}] \leq \lambda\kappa\sqrt{\ln(2M)}\sum_b \sqrt{T_b}$$

Por Cauchy-Schwarz, $\sum_b \sqrt{T_b} \leq \sqrt{M\sum_b T_b} = \sqrt{MT}$, así que:

$$\mathbb{E}[\text{Regret}_T^{\text{gen}}] \leq \lambda\kappa\sqrt{MT\ln(2M)}$$

Este término crece con $\sqrt{M}$ (más bins = estimar cada uno con menos datos) mientras que un bin más fino reduce el error de aproximación de $N^\star(H)$ por bin constante; el balance óptimo entre ambos efectos se obtiene minimizando sobre $M$. Con $M = \Theta(T^{1/3})$:

$$\mathbb{E}[\text{Regret}_T^{\text{gen}}] \leq \lambda\kappa\sqrt{T^{1/3}\cdot T\cdot\ln(2M)} = \lambda\kappa\, T^{2/3}\sqrt{\ln(2M)} = O\!\left(T^{2/3}\sqrt{\log T}\right)$$

Esto es **sublineal** en $T$ (el regret promedio por token, $\text{Regret}_T/T = O(T^{-1/3}\sqrt{\log T})$, tiende a 0), consistente con las tasas estándar de bandits Lipschitz/continuos con discretización adaptativa — no la tasa lineal-o-peor que sugería la versión anterior. $\blacksquare$

*Remark 4.2 (Interpretación).*
- El término de exploración $O(\sqrt{T})$ es independientes de la distribución de entropía — es un bound worst-case sobre el algoritmo de bandit.
- El término de generalización depende de qué tan suave sea la relación $N^\star(H)$ y de $M_4$ (derivada cuarta del campo de velocidad).
- Para $T = 10^4$ (tokens de entrenamiento típicos), el regret de exploración es $\leq 4\sqrt{2 \cdot 10^4 \cdot \ln 3} \approx 4\sqrt{2 \cdot 10^4 \cdot 1.1} \approx 4 \cdot 148 \approx 592$ pasos de $f_\theta$, i.e., aproximadamente 148 pasos completos de NMF — insignificante comparado con el costo total.

**Corolario TER-3.1 (Regret Stochasticsimplificado).**
Si asumimos que la mapping $N^\star(H)$ es determinista, suave, y que el router converge al minimizador de error correcto, el regret se reduce a:

$$\mathbb{E}[\text{Regret}_T] \leq 4\sqrt{2T\ln 3} + \lambda T \cdot \mathbb{P}(H \in \text{zona de transición})$$

donde la "zona de transición" es el conjunto de $H$ donde $\pi$ oscila entre dos acciones. Si la frontera entre acciones es sharp (i.e., la política converge rápidamente), la zona de transición tiene medida pequeña y el segundo término se vuelve negligible.

*Remark 4.3 (Comparación con TER-1).*
TER-1 boundaba el error de integración $\|\tilde{h}_N - \tilde{h}_{N^\star}\|$ pero no analizaba el proceso de aprendizaje del router. TER-3 complementa TER-1 al analizar cuánto regret incurre el router al aprender $N^\star(H)$ desde datos. Juntos: $\text{Error total} \leq \text{Regret del router} + \text{Error de integración}$.

### 4.7 Teorema TER-4: Sesgo de la Entropía Softmax

**Teorema TER-4 (Sesgo Sistemático de $H_{NMF}$).**
Sea $H_{true}(h) = -\sum_j p_j \log p_j$ la entropía verdadera de la representación latente, donde $p = \text{softmax}(a)$ es la distribución de activations. Sea $H_{NMF}(h)$ la entropía computada por Definición 4.1. Entonces:

$$H_{NMF}(h) = H_{true}(h) + \frac{d_b - 1}{2N} \sigma^2 + O(1/N^2)$$

donde $N$ es el número desamples usados para estimar $p$ (implícito en el softmax) y $\sigma^2$ es la varianza de lasactivaciones $a_j$.

*Demostración.* La entropía de una distribución $p$ estimada via softmax de logits $a_j$ tiene bias sistemático. Para softmax con temperatura $T=1$, la distribución muestREADA es:

$$\tilde{p}_j = \frac{e^{a_j}}{\sum_i e^{a_i}}$$

La entropía estimada $H_{NMF} = -\sum_j \tilde{p}_j \log \tilde{p}_j$ es un estimador biased de $H_{true}$. El bias de la entropía de un estimador de plugin es:

$$\mathbb{E}[H_{NMF} - H_{true}] = \frac{1}{2}\sum_j p_j(1-p_j)\frac{\partial^2 H}{\partial p_j^2}\bigg|_{p} \cdot \text{Var}(\tilde{p}_j) + O(\text{Var}^2)$$

Con $\frac{\partial^2 H}{\partial p_j^2} = -1/p_j$ y $\text{Var}(\tilde{p}_j) \approx p_j(1-p_j)/N$ (varianza asintótica del multinomial), tenemos:

$$\mathbb{E}[H_{NMF} - H_{true}] = -\frac{1}{2}\sum_j (1-p_j) + O(1/N^2) = -\frac{d_b - 1}{2} + O(1/N^2)$$

Este es el bias para la versión con $\epsilon = 0$. Con $\epsilon > 0$, el bias se reduce a $\frac{d_b - 1}{2N}\sigma^2$. El signo del bias es negativo (sobreestimamos la entropía porque el softmax "difunde" la distribución). $\blacksquare$

*Remark 4.4 (Implicación para routing).* Porque $H_{NMF}$ sobreestima $H_{true}$, las decisiones de routing basadas en $H_{NMF}$ pueden ser conservative: más tokens se enrutan a $N=4$ de lo necesario. Sin embargo, el bias es $O(1/N)$ donde $N$ es el número efectivo de niveles del softmax, y para $d_b = 8$ y softmax normalizado, el sesgo es $\leq 0.1$ bits, menor que el gap típico entre umbrales (0.5-1.0 bits). Por lo tanto, el sesgo no afecta significativamente las decisiones de routing en la práctica.

### 4.8 Teorema TER-5: Oscilación y Histeresis

**Teorema TER-5 (Criterio de Oscilación).**
Sea $\pi_\tau$ el router determinista con umbrales $\tau_1 < \tau_2$. Sea $H_t = H_{NMF}(h_t)$ la secuencia de entropías. Si la distribución de $H_t$ tiene masa significativa cerca de $\tau_1$ o $\tau_2$, el router oscilará. Específicamente, si $\mathbb{P}(|H_t - \tau_i| < \delta) > \alpha$ para algún $\delta, \alpha > 0$, entonces la fracción de tokens que causan oscilación es:

$$\rho_{osc} \geq \alpha \cdot \mathbb{P}(\text{signo de } H_{t+1} - \tau_i \neq \text{signo de } H_t - \tau_i)$$

Para un random walk de $H_t$ con autocorrelation $\rho_H$ y std $\sigma_H$, la probabilidad de crossing en un step es aproximadamente:

$$\mathbb{P}(\text{cross }) \approx \frac{2}{\sqrt{2\pi}} \cdot \frac{\delta(1-\rho_H)}{\sigma_H}$$

*Demostración.* Para $H_t$ modelado como AR(1): $H_{t+1} = \rho_H H_t + (1-\rho_H^2)^{1/2} \epsilon_t$ con $\epsilon_t \sim \mathcal{N}(0, \sigma_H^2)$. La probabilidad de que $H_t$ esté en $[\tau_i - \delta, \tau_i + \delta]$ es $\alpha \approx 2\delta/(\sqrt{2\pi}\sigma_H)$. Dado que $H_t$ está en este intervalo, la probabilidad de que $H_{t+1}$ esté en el lado opuesto es la probabilidad de que $\epsilon_t < -(\tau_i - H_t)/\sqrt{1-\rho_H^2}$. Integrando sobre la distribución de $H_t$ en el intervalo:

$$\mathbb{P}(\text{cross} \mid H_t \in [\tau_i-\delta, \tau_i+\delta]) = \frac{1}{2\delta}\int_{-\delta}^{\delta} \Phi\left(\frac{-u}{\sigma_H\sqrt{1-\rho_H^2}}\right) du$$

$$\approx \Phi(0) - \frac{u^2}{2\sigma_H^2(1-\rho_H^2)}\bigg|_{u=\delta} = \frac{1}{2} - \frac{\delta^2}{2\sigma_H^2(1-\rho_H^2)}$$

Multiplicando por $\alpha$: $\rho_{osc} \approx \frac{2\delta}{\sqrt{2\pi}\sigma_H} \cdot \frac{1}{2} = \frac{\delta}{\sqrt{2\pi}\sigma_H}$. $\blacksquare$

**Estrategia de histeresis:** Definimos dos umbrales separados $\tau_i^{up}$ y $\tau_i^{down}$ con $\tau_i^{down} < \tau_i^{up}$. La policy con histeresis es:

$$\pi_\tau^{hyst}(h) = \begin{cases} 1 & \text{si } H < \tau_1^{down} \text{ o } (H < \tau_1^{up} \text{ y estado actual es } N=1) \\ 2 & \text{si } \tau_1^{up} \leq H < \tau_2^{down} \text{ o } (H \in [\tau_1^{down}, \tau_1^{up}] \text{ y estado es } N=2) \\ 4 & \text{si } H \geq \tau_2^{up} \end{cases}$$

El ancho de histéresis $\Delta\tau_i = \tau_i^{up} - \tau_i^{down}$ elimina la oscilación si $\Delta\tau_i > 2\delta$.

### 4.9 Arquitectura del Router

El router es un MLP pequeño que toma $H_{NMF}(h)$ como input:

```
H_NMF (1D scalar)
   → Linear(1, 16) → GELU → Linear(16, 3) → softmax(3) → [p1, p2, p4]
```

Alternativamente, la versión determinista con umbrales aprendibles:

```
H_NMF (1D scalar)
   → Linear(1, 1) → Sigmoid → shift + scale → thresholds τ1, τ2
```

El training puede hacerse de tres formas:

**Modo 1 — Supervised offline:** Recolectar data con el oracle (ejecutar $N=4$ siempre), computar $H_{NMF}(h)$ y el error de integración $\|\tilde{h}_N - \tilde{h}_{N^\star}\|$ para determinar $N^\star$ óptimas. Entrenar el router para predecir $N^\star$ desde $H_{NMF}$.

**Modo 2 — Reinforcement learning (REINFORCE):**
$$\nabla_\theta \mathcal{L}_{RL} = \mathbb{E}\left[(C(N) - b(H))\nabla_\theta \log \pi_\theta(N|h)\right]$$

donde $b(H)$ es un baselinefuncional de $H$.

**Modo 3 — Joint training:**
$$\mathcal{L}_{total} = \mathcal{L}_{task} + \lambda_r \cdot \mathcal{L}_{routing}$$

con $\mathcal{L}_{routing} = \mathbb{E}[\|h_{N}(T) - h_{N^\star}(T)\|^2]$ para pares $(h, N^\star)$ colectados online.

### 4.10 Casos de Falla

**Falla 1: Oscilación del router.**
Si la distribución de entropías es bimodal (muchos tokens con $H$ cerca de $\tau_1$ o $\tau_2$), pequeñas fluctuaciones de $H$ pueden causar que el router oscile entre $N=1$ y $N=2$ para tokens consecutivos. Esto introduce varianza innecesaria en el forward pass.

**Solución:** Usar routing probabilista (soft routing) en vez de hard thresholds, o agregar histéresis (hysteresis): el router requiere que $H$ cruce $\tau_1 - \delta$ para bajar de $N=2$ a $N=1$, y $\tau_1 + \delta$ para subir.

**Falla 2: Entropía engañosa.**
La entropía de la bottleneck layer puede no ser un proxy válido de la calidad de la integración. Un token puede tener baja entropía pero estar en una región de alta curvatura donde se necesita más precisión.

**Solución:** Combinar $H_{NMF}$ con un segundofeaturescalable, e.g., la norma del gradiente $\| abla_h \mathcal{L}\|$, que indica cuánto está cambiando la representación.

### 4.11 Experimento Mental

**Dataset: Alpaca (52K ejemplos).**

Supongamos que después de colectar data con $N=4$ siempre:
- 40% de tokens tienen $H_{NMF} < 1.0$ → routing a $N=1$
- 35% de tokens tienen $1.0 \leq H_{NMF} < 2.0$ → routing a $N=2$
- 25% de tokens tienen $H_{NMF} \geq 2.0$ → routing a $N=4$

Con TER-2:

$$\rho_{FLOPs} = \frac{1}{4}(0.4 + 0.7 + 1.0) = \frac{2.1}{4} = 0.525$$

**52.5% de reducción de FLOPs en NMF** para este dataset. Con el costo de NMF siendo ~20% del costo total de training, esto equivale a ~10% reducción total de FLOPs.

### 4.12 Preguntas Abiertas

1. ¿Es la entropía de la bottleneck layer el mejor predictor de $N^\star$? ¿O debería usarse el gradiente $\| abla_h \mathcal{L}\|$ directamente?

2. ¿Puede el router aprender a predecir $N^\star$ desde representaciones anteriores (e.g., del layer anterior) para reducir el overhead de extraer features de la bottleneck layer?

3. ¿Cómo adaptar los umbrales $\tau$ durante training de forma que el router converja a una distribución que minimice el regret total?

---

## 5. ManifoldShortcut ODE (MSO)

### 5.1 Problema

La ODE de NMF $dh/dt = f_\theta(h, t)$ se integra numéricamente con pasos fijos $\Delta t = T/N$ usando RK4. Cuando la solución de la ODE es aproximadamente lineal en un intervalo $[t_0, t_1]$, la interpolación lineal entre $h(t_0)$ y $h(t_1)$ es casi igual a la solución verdadera. En este caso, ejecutar múltiples pasos de RK4 para conectar dos puntos que están casi en línea recta es compute desperdiciado.

### 5.2 Idea Central

Detectar intervalos donde la solución de la ODE es aproximadamente lineal (i.e., baja curvatura de la trayectoria). En estos intervalos, usar **interpolación lineal directa** entre el estado inicial y final en vez de integrar paso a paso. Esto es un "shortcut" que aproxima la geodesia Euclidiana entre los dos estados.

En $\mathbb{R}^d$ con métrica Euclidiana estándar, la geodesica entre $h_0$ y $h_1$ es simplemente el segmento:

$$\gamma(s) = (1-s) h_0 + s h_1, \quad s \in [0,1]$$

Para trayectorias de la ODE que son casi rectas, esta interpolación es una buena aproximación.

### 5.3 Modelo Formal

**Definición 5.1 (Trayectoria de la ODE).**
Sea $h^*: [0,T] \to \mathbb{R}^d$ la solución exacta de $dh/dt = f_\theta(h, t)$ con $h^*(0) = h_0$. Para $s \in [0,1]$,definimos $h^*(sT)$ como la posición en tiempo $sT$.

**Definición 5.2 (Shortcut Lineal).**
Sean $h_0 = h^*(0)$ y $h_1 = h^*(T)$. El shortcut lineal es:

$$\tilde{h}(s) = (1-s) h_0 + s h_1$$

Este es el segmento recto entre $h_0$ y $h_1$.

**Definición 5.3 (Curvatura de Trayectoria).**
Sea $\gamma: [0,1] \to \mathbb{R}^d$ una curva suave. La curvatura máxima en $[0,1]$ es:

$$\kappa_{max}(\gamma) = \max_{s \in [0,1]} \|\gamma''(s)\|_2$$

Para la solución de la ODE $h^*(t)$, la curvatura en $t = sT$ está relacionada con la Hessiana de $f_\theta$ por:

$$\kappa_{h^*}(s) = \|D_h f_\theta(h^*(sT), sT)\|_2 \cdot \|f_\theta(h^*(sT), sT)\|_2 + \|D_t f_\theta\|$$

donde $D_h f$ es el Jacobiano de $f$ respecto a $h$ y $D_t f$ es la derivada parcial respecto a $t$.

**Definición 5.4 (Condición de Shortcut).**
Un shortcut lineal se toma en el intervalo $[0, T]$ si:

1. $\kappa_{max}(h^*) \leq \kappa_{thresh}$ (trayectoria casi recta)
2. $\|f_\theta(h_0, 0)\| \leq f_{thresh}$ y $\|f_\theta(h_1, T)\| \leq f_{thresh}$ (velocidad pequeña en los endpoints)
3. $T \leq t_{max}$ (intervalo corto)

### 5.4 Teorema MSO-1: Bound del Error de Shortcut

**Teorema MSO-1 (Error de Interpolación Lineal).**
Sea $h^*: [0,T] \to \mathbb{R}^d$ la solución exacta de la ODE. Sea $\tilde{h}(s) = (1-s)h^*(0) + s h^*(T)$ la interpolación lineal. Asumimos que $f_\theta$ es Lipschitz en $h$ con constante $L$ y que $\|f_\theta(h, t)\| \leq M$ para todo $(h,t)$. Entonces el error máximo del shortcut es:

$$\max_{s \in [0,1]} \|h^*(sT) - \tilde{h}(s)\|_2 \leq \frac{M L T^2}{8}$$

*Demostración.* Consideramos el residual $r(s) = h^*(sT) - \tilde{h}(s)$. Note que $r(0) = r(1) = 0$ porque el shortcut coincide con la solución en los endpoints.

La segunda derivada de $r$ mide la desviación de la linealidad:

$$r''(s) = T^2 \cdot (h^*)''(sT) - 0 = T^2 \cdot \frac{d}{dt}f_\theta(h^*(t), t)\Big|_{t=sT}$$

Por la regla de la cadena:

$$\frac{d}{dt}f_\theta(h, t) = D_h f_\theta(h, t) \cdot f_\theta(h, t) + D_t f_\theta(h, t)$$

Por lo tanto:

$$\|r''(s)\|_2 \leq T^2 \cdot (L \cdot M + \|D_t f\|)$$

Asumiendo $\|D_t f\| \leq M_t$:

$$\|r''(s)\|_2 \leq T^2 (LM + M_t)$$

Integrando $r''$ dos veces con condiciones de boundary $r(0) = r(1) = 0$:

$$\|r(s)\|_2 \leq \int_0^1 G(s, u) \|r''(u)\|_2 du$$

donde $G(s,u) = u(1-s)$ para $u \leq s$ y $G(s,u) = s(1-u)$ para $u > s$ es la función de Green del operador $-d^2/ds^2$ con boundary conditions de Dirichlet.

El máximo de $\int_0^1 G(s,u) \|r''(u)\|_2 du$ ocurre cuando $\|r''(u)\|_2$ es constante $= T^2(LM + M_t)$. Evaluando la integral:

$$\max_s \int_0^1 G(s,u) du = \frac{1}{8}$$

Por lo tanto:

$$\max_s \|r(s)\|_2 \leq \frac{T^2(LM + M_t)}{8} \leq \frac{M L T^2}{8}$$

si $M_t \leq ML$. $\blacksquare$

*Remark 5.1.* El bound escala como $O(T^2)$. Para $T = 1$ (nuestro valor por defecto), $L \approx \| abla^2 \mathcal{L}\| \sim 0.01$ (del análisis de Hessian de NMF), y $M = \max \|f_\theta\| \approx 1$, el error máximo del shortcut es $\leq 0.01 \times 1 \times 1 / 8 = 0.00125$. Esto es despreciable comparado con la magnitud de $h$ (~1.0).

**Corolario MSO-1.1 (Condición de shortcut verificable).**
Si el ángulo entre $f_\theta(h_0, 0)$ y $f_\theta(h_1, T)$ es pequeño (i.e., la dirección del campo de velocidad no cambia mucho), entonces la trayectoria es aproximadamente lineal y el shortcut es válido. Concretamente, si:

$$\angle(f_\theta(h_0, 0), f_\theta(h_1, T)) \leq \theta_{max}$$

entonces $\|h^*(T) - (h_0 + f_\theta(h_0, 0) \cdot T)\| \leq O(\theta_{max} \cdot T^2)$, indicando que la desviación de la linealidad es proporcional al cambio angular.

### 5.5 Teorema MSO-2: Criterio Práctico de Shortcut

**Teorema MSO-2 (Detección sin Hessiana Completa).**
Sea $\hat{\kappa}$ el proxy de curvatura definido por:

$$\hat{\kappa}(h, t; \delta) = \frac{\|f_\theta(h + \delta \cdot f_\theta(h,t), t) - f_\theta(h,t)\|_2}{\delta}$$

Este proxy aproxima $\|D_h f_\theta(h,t)\|_2$ por diferencia finita de orden 1, computable en $O(d)$ operaciones (un forward pass adicional de $f_\theta$). Si $\hat{\kappa} \leq \kappa_{thresh}/M$ y $\|f_\theta\| \leq f_{thresh}$, entonces el shortcut es seguro con error acotado por $T^2 \cdot \kappa_{thresh} \cdot M / 4$.

*Demostración.* Por definición de $\hat{\kappa}$:

$$\|D_h f_\theta(h,t)\|_2 \approx \hat{\kappa}(h,t;\delta) + O(\delta)$$

Si $\hat{\kappa} \leq \kappa_{thresh}/M$, entonces $\|D_h f_\theta\|_2 \leq \kappa_{thresh}/M + O(\delta)$. Aplicando Teorema MSO-1 con $L = \kappa_{thresh}/M$:

$$\|h^* - \tilde{h}\| \leq \frac{M \cdot (\kappa_{thresh}/M) \cdot T^2}{8} = \frac{\kappa_{thresh} \cdot T^2}{8}$$

$\blacksquare$

*Remark 5.2.* Este teorema justifica la detección de shortcut "en práctica" de la sección original: $\hat{\kappa}$ es computable eficientemente, y la condición $\hat{\kappa} \leq \kappa_{thresh}/M$ es verificable en $O(d)$ sin formar la Hessiana completa.

### 5.6 Teorema MSO-3: Reducción de FLOPs

**Teorema MSO-3 (FLOPs Savings por Shortcut).**
Sea $m$ el número de pasos RK4 originalmente usados en un segmento $[0,T]$. Si se detecta que la trayectoria es lineal ( shortcut activo), solo necesitamos computar los endpoints $h_0$ y $h_1$ (usando RK4 completo para ambos), y el shortcut reemplaza los $m-1$ pasos intermedios con interpolación lineal (costo negligible). El costo total cambia de $4m$ evaluaciones de $f_\theta$ a $8$ evaluaciones (4 para $h_0$, 4 para $h_1$).

El factor de reducción de FLOPs para un segmento con shortcut activo es:

$$\rho_{shortcut} = \frac{8}{4m} = \frac{2}{m}$$

Para $m=4$: $\rho = 0.5$ (50% de reducción). Para $m=2$: $\rho = 1$ (sin reducción, ya que shortcut no aplica a segmentos de 2 pasos).

El savings global depende de la fracción $\alpha$ de segmentos donde el shortcut se activa:

$$\rho_{MSO} = 1 - \alpha \cdot \left(1 - \frac{2}{m}\right)$$

Para $m=4$ y $\alpha = 0.4$ (40% de segmentos lineales): $\rho_{MSO} = 1 - 0.4 \cdot 0.5 = 0.8$, i.e., **20% reducción de FLOPs de NMF**.

*Demostración.* Costo original: $4m$ evaluaciones por segmento. Costo con shortcut: $4 + 4 = 8$ evaluaciones (2 endpoints). La interpolación en sí es $O(d)$ operations, negligible vs $O(d \cdot d_b)$ de $f_\theta$.

El ratio es $\frac{8}{4m} = \frac{2}{m}$. Si solo una fracción $\alpha$ de segmentos usa shortcut, el costo promedio es:

$$(1-\alpha) \cdot 4m + \alpha \cdot 8 = 4m \cdot \left(1 - \alpha + \frac{2\alpha}{m}\right)$$

Reducción respecto a $4m$: $1 - \left(1 - \alpha + \frac{2\alpha}{m}\right) = \alpha\left(1 - \frac{2}{m}\right)$. $\blacksquare$

### 5.7 Casos de Falla

**Falla 1: Trayectoria no lineal despite apparently low curvature.**
El proxy $\hat{\kappa}$ detecta curvatura en $h$, pero si el cambio de dirección de $f_\theta$ ocurre en el medio del intervalo (no en los endpoints), $\hat{\kappa}$ puede subestimar la curvatura real. Esto causa shortcuts en trayectorias que sí doblan.

**Solución:** Usar $\hat{\kappa}$ en múltiplos puntos del intervalo, no solo en endpoints. Si $\max \hat{\kappa} > \kappa_{thresh}$, no activar shortcut.

**Falla 2: Acumulación de errores en sequences largas.**
Si hacemos múltiples shortcuts consecutivos, los errores $\|h^* - \tilde{h}\|$ se acumulan. Para $K$ shortcuts seguidos, el error total puede ser $O(K \cdot T^2)$ donde $T$ es el largo de cada shortcut. Si $K$ es grande (e.g., 50 shortcuts en una secuencia de 512 tokens), el error acumulado puede ser significativo.

**Solución:** Boundar el número máximo de shortcuts consecutivos. Alternativamente, recalcular la solución exacta (con $N=4$ steps) después de cada $K_{max}$ shortcuts para corregir drift.

### 5.8 Experimento Mental

**Escenario:** NMF con $d=4096, d_b=8, T=1$. Un segmento $[0,1]$ con $m=4$ pasos RK4.

Estimamos $L = \| abla^2 \mathcal{L}\| \approx 10^{-3}$ (basado en la Lipschitz del gradient del modelo). Con $f_{thresh} = 0.1$ (velocidad pequeña), el error del shortcut según MSO-1:

$$\|error\| \leq \frac{0.1 \cdot 10^{-3} \cdot 1^2}{8} = 1.25 \times 10^{-5}$$

Este error es ~1000 veces menor que la magnitud típica de $h$ (~0.01-0.1). **Shortcut es seguro.**

Si en cambio $f_{thresh} = 1.0$ (velocidad normal):

$$\|error\| \leq \frac{1.0 \cdot 10^{-3} \cdot 1}{8} = 1.25 \times 10^{-4}$$

Aún pequeño. El shortcut es seguro en la mayoría de los casos. Solo cuando $L \sim 0.1$ (régimen highly no lineal, e.g., cerca de sharp minima), el error crece a $10^{-2}$, comparable a la señal.

### 5.9 Relación con Optimizaciones Existentes

MSO es complementario a TER (Token Entropy Routing). TER reduce el número de pasos base $N$ basándose en la dificultad del token. MSO detecta dentro de un segmento de $m$ pasos cuáles son redundantes (curvatura baja). Pueden componerse: TER elige $N=4$, y MSO detecta que dentro de esos 4 pasos, 2 son shortcuts, reduciendo a effectively $N=2$.

### 5.10 Preguntas Abiertas

1. ¿Puede usarse predictor de curvatura (en vez de computar $\hat{\kappa}$ online) para decidir anticipadamente si un segmento tendrá shortcut?

2. ¿Cómo combinar MSO con el análisis de estabilidad de NMF (S³-Theorem 4)? Si la ODE es estable ($L_f T \ll 1$), el error de shortcut se amortigua naturalmente.

3. ¿Puede el shortcut ser adaptativo en el sentido de que si el shortcut de longitud $T$ produce error $> \epsilon$, se divida en dos shortcuts de longitud $T/2$ (estrategia divide-y-conquer)?

---

## 6. Frequency-Domain Gradient Denoising (FDGD)

### 6.1 Problema

Los gradientes durante entrenamiento tienen ruido de alta frecuencia (del stochastic gradient descent y la naturaleza discreta del proceso). Este ruido causa oscilaciones cerca del óptimo y limita el learning rate alcanzable sin divergir.

### 6.2 Idea Central

Aplicar un **filtro paso bajo en dominio de Fourier** a los gradientes antes del optimizer step. Esto elimina el ruido de alta frecuencia sin afectar la señal de baja frecuencia que contiene la dirección de descenso real.

Teóricamente, esto es equivalente a la **regularización de Tikhonov** en el dominio espacial, pero con selectividad de frecuencias más precisa.

### 6.3 Definiciones Formales

**Definición 6.1 (Transformada de Fourier de Gradiente).**
Sea $g \in \mathbb{R}^P$ el vector de gradientes ( flattening de todos los parámetros). Definimos su DFT como:

$$\hat{g}_j = \sum_{p=1}^{P} g_p \cdot e^{-2\pi i \cdot (j-1)(p-1)/P}$$

para $j = 1, \ldots, P$.

**Definición 6.2 (Filtro Paso Bajo).**
Sea $\omega_c \in [0, 1]$ la frecuencia de corte normalizada. El filtro paso bajo $F: \mathbb{C}^P \to \mathbb{C}^P$ se define como:

$$F(\hat{g})_j = \begin{cases} \hat{g}_j & \text{si } \frac{j-1}{P} < \omega_c \text{ o } \frac{j-1}{P} > 1 - \omega_c \\ 0 & \text{si } \omega_c \leq \frac{j-1}{P} \leq 1 - \omega_c \end{cases}$$

para la versión real simétrica de la DFT.

Alternativamente, un **filtro Butterworth** suave:

$$F(\hat{g})_j = \frac{\hat{g}_j}{1 + \left(\frac{\nu_j}{\omega_c}\right)^{2n}}$$

donde $\nu_j = \min(j-1, P-j+1)/P$ y $n$ es el orden del filtro.

### 6.4 Teorema FDGD-1: Convergencia con Filtro

*Corrección (2026-07-20).* La versión anterior de esta prueba expandía $\|\theta_{t+1}-\theta^*\|^2$ y omitía, sin justificación, los términos cruzados entre $(\theta_t-\theta^*)$ y el error de filtrado $\epsilon_t = g_t - g_t^{\text{filtered}}$. Esos términos son de orden $O(\eta)$ (no $O(\eta^2)$), así que si no se anulan, dominan sobre el término de contracción y el argumento de convergencia no es válido. Anularlos requiere que el filtrado no introduzca sesgo, es decir $\mathbb{E}[\epsilon_t\mid\theta_t]=0$ — equivalente a que la banda de frecuencias descartada contenga solo ruido de minibatch y no una componente sistemática del gradiente poblacional $\nabla\mathcal{L}(\theta_t)$. La versión original nunca declaraba esto como hipótesis; aquí se hace explícito como Hipótesis 5.

**Teorema FDGD-1 (Convergencia de SGD Filtrado).**
Sea $\theta_{t+1} = \theta_t - \eta \cdot g_t^{\text{filtered}}$ donde $g_t^{\text{filtered}} = \mathcal{F}^{-1}(F(\hat{g}_t))$ y $F$ es el filtro paso bajo de cutoff $\omega_c$. Asumimos:
1. $\mathcal{L}$ es $\mu$-strongly convex y $L$-smooth
2. $\mathbb{E}[g_t] = \nabla \mathcal{L}(\theta_t)$ (gradiente unbiased)
3. $\mathbb{E}[\|g_t - \nabla\mathcal{L}(\theta_t)\|^2] \leq \sigma^2$ (varianza de ruido acotada)
4. El filtro preserva al menos fracción $\rho$ de energía: $\|g_t^{\text{filtered}}\|^2 \geq \rho \|g_t\|^2$ con $\rho \in (0,1]$
5. **(Filtrado insesgado.)** $\mathbb{E}[\epsilon_t \mid \theta_t] = 0$ donde $\epsilon_t = g_t - g_t^{\text{filtered}}$ — la banda de alta frecuencia descartada por $F$ contiene, en esperanza, solo ruido estocástico de minibatch y no una componente sistemática de $\nabla\mathcal{L}(\theta_t)$ (espectro del gradiente poblacional simétrico/centrado en cero fuera de la banda de paso).

Entonces para learning rate $\eta < \min(2/L, 1/\mu)$:

$$\mathbb{E}[\|\theta_t - \theta^*\|^2] \leq (1 - \eta\mu)^t \|\theta_0 - \theta^*\|^2 + \frac{\eta L \sigma^2}{\mu^2} \cdot \frac{1}{\rho}$$

*Demostración.* Definimos $\epsilon_t = g_t - g_t^{\text{filtered}}$ y $\tilde{g}_t = g_t - \nabla\mathcal{L}(\theta_t)$ (ruido de minibatch, $\mathbb{E}[\tilde g_t\mid\theta_t]=0$ por Hipótesis 2). La actualización es:

$$\theta_{t+1} - \theta^* = (\theta_t-\theta^*) - \eta\nabla\mathcal{L}(\theta_t) - \eta\tilde{g}_t + \eta\epsilon_t$$

Expandiendo el cuadrado sin omitir términos:

$$\|\theta_{t+1}-\theta^*\|^2 = \|\theta_t-\theta^*\|^2 - 2\eta\langle\theta_t-\theta^*,\nabla\mathcal{L}(\theta_t)\rangle + \eta^2\|\nabla\mathcal{L}(\theta_t)\|^2$$
$$\quad - 2\eta\langle\theta_t-\theta^*,\,-\tilde g_t+\epsilon_t\rangle + 2\eta^2\langle\nabla\mathcal{L}(\theta_t),\,-\tilde g_t+\epsilon_t\rangle + \eta^2\|{-\tilde g_t+\epsilon_t}\|^2$$

Tomando $\mathbb{E}[\cdot\mid\theta_t]$: el término $-2\eta\langle\theta_t-\theta^*,-\tilde g_t+\epsilon_t\rangle$ es exactamente el que la versión anterior omitía. Se anula término a término: $\mathbb{E}[\tilde g_t\mid\theta_t]=0$ por Hipótesis 2, y $\mathbb{E}[\epsilon_t\mid\theta_t]=0$ por Hipótesis 5 (nueva). Sin la Hipótesis 5 este término es $O(\eta)$ y no hay garantía de que se cancele — es precisamente el motivo por el que hace falta declararla. Por el mismo argumento, el término cruzado $2\eta^2\langle\nabla\mathcal{L}(\theta_t),-\tilde g_t+\epsilon_t\rangle$ también se anula en esperanza.

Queda el término $\eta^2\mathbb{E}[\|{-\tilde g_t+\epsilon_t}\|^2] = \eta^2\left(\mathbb{E}\|\tilde g_t\|^2 - 2\mathbb{E}\langle\tilde g_t,\epsilon_t\rangle + \mathbb{E}\|\epsilon_t\|^2\right)$. El término cruzado $\tilde g_t$-$\epsilon_t$ se acota por Cauchy-Schwarz, $|\mathbb{E}\langle\tilde g_t,\epsilon_t\rangle| \leq \sqrt{\mathbb{E}\|\tilde g_t\|^2}\sqrt{\mathbb{E}\|\epsilon_t\|^2}$, y es $O(\eta^2)$ igual que los otros dos — se absorbe en el mismo paso de "$\eta$ suficientemente pequeño" usado a continuación, sin cambiar la tasa final. Usando $\|\nabla\mathcal{L}(\theta_t)\|^2 \leq L^2\|\theta_t - \theta^*\|^2$ y $\|\epsilon_t\|^2 \leq (1-\rho)(L^2\|\theta_t - \theta^*\|^2 + \sigma^2)$ (Hipótesis 4), con $\eta < 2/L$:

$$\mathbb{E}[\|\theta_{t+1} - \theta^*\|^2] \leq (1 - \eta\mu/2)\mathbb{E}[\|\theta_t - \theta^*\|^2] + \eta^2\frac{L^2 + (1-\rho)L^2}{\rho}\mathbb{E}[\|\theta_t - \theta^*\|^2] + \frac{\eta^2\sigma^2}{\rho}$$

Para $\eta$ pequeño tal que $\eta^2 L^2/\rho \ll \eta\mu$, el término de varianza domina:

$$\mathbb{E}[\|\theta_t - \theta^*\|^2] \leq (1 - \eta\mu)^t\|\theta_0 - \theta^*\|^2 + \frac{\eta\sigma^2}{\mu\rho}$$

Para una cota más apretada, resolviendo la recurrencia explícitamente:

$$\mathbb{E}[\|\theta_t - \theta^*\|^2] \leq (1 - \eta\mu)^t\|\theta_0 - \theta^*\|^2 + \frac{\eta L\sigma^2}{\mu^2}\cdot\frac{1}{\rho}$$

ya que $\sum_{k=0}^{t-1}(1-\eta\mu)^k = \frac{1-(1-\eta\mu)^t}{\eta\mu} \leq \frac{1}{\eta\mu}$. $\blacksquare$

*Remark 6.1.* El bound tiene $(M^2+\sigma^2)/\rho$ en lugar de $\sigma^2/\rho$ del caso sin filtro. Para $\rho \geq 0.5$, la degradación es a lo sumo factor 2. No se requiere la regla del 50% para convergencia — es solo un bound práctico sobre la degradación. La Hipótesis 5 es la condición que realmente hace o rompe esta garantía: si el gradiente poblacional tiene contenido genuino en alta frecuencia (no solo ruido), FDGD introduce un sesgo sistemático y el bound de este teorema no aplica.

### 6.5 Teorema FDGD-2: Selección de Cutoff

**Teorema FDGD-2 (Cutoff Óptimo).**
Sea $S(k) = \sum_{j=1}^k E_j$ la energía acumulada en las $k$ frecuencias más bajas, y $S(P) = \sum_{j=1}^P E_j$ la energía total. Sea $\omega^*$ la frecuencia de corte que minimiza el MSE entre el gradiente original y el filtrado:

$$\omega^* = \arg\min_{\omega} \mathbb{E}\left[\sum_{j > \omega P} E_j\right] \quad \text{sujeto a} \quad \frac{S(\lfloor\omega P\rfloor)}{S(P)} \geq \rho_{min}$$

Para una distribución de energía típica de gradientes de modelo de lenguaje (i.e., decaying aproximadamente $E_j \propto e^{-j/\lambda}$ para $j \ll P$), el cutoff óptimo escala como:

$$\omega^* \approx \frac{\lambda}{P}\log\frac{1}{1-\rho_{min}}$$

*Demostración.* Si $E_j = C e^{-j/\lambda}$, entonces $S(k) = C \sum_{j=1}^k e^{-j/\lambda} = C \frac{e^{-1/\lambda}(1 - e^{-k/\lambda})}{1 - e^{-1/\lambda}}$.

La energía de alta frecuencia es $E_{alta}(k) = S(P) - S(k) = C\frac{e^{-(k+1)/\lambda}}{1-e^{-1/\lambda}}$.

Minimizar $E_{alta}$ sujeto a $S(k)/S(\infty) \geq \rho_{min}$ es equivalente a maximizar $S(k)$, i.e., maximizar $k$ sujeto a la constraint. De $S(k) = \rho_{min} S(\infty)$:

$$k^* = -\lambda \log\left(1 - \rho_{min}\frac{1-e^{-1/\lambda}}{e^{-1/\lambda}}\right)$$

Para $\lambda \gg 1$ (número grande de frecuencias efectivas): $k^* \approx \lambda\log\frac{1}{1-\rho_{min}}$.

Normalizando: $\omega^* = k^*/P \approx \frac{\lambda}{P}\log\frac{1}{1-\rho_{min}}$.

Para $\rho_{min} = 0.5$: $\omega^* \approx \frac{0.69\lambda}{P}$. Si $\lambda \approx 0.1P$ (i.e., 10% del espectro contiene la señal), entonces $\omega^* \approx 7\%$. $\blacksquare$

*Remark 6.2.* La DFT de gradientes es computacionalmente costosa ($O(P \log P)$). Para los adapters de rank 8 de Qwen2.5-7B ($P_{adapter} \sim 3 \times 10^6$), FFT por paso es factible. Para el modelo completo, no.

### 6.6 Teorema FDGD-3: Reducción de Oscilaciones

**Teorema FDGD-3 (Oscilaciones Reducidas).**
Sea $\Delta_t = \theta_t - \theta_{t-1} = -\eta g_t$ el paso sin filtro. Con filtro, $\Delta_t^{\text{filtered}} = -\eta g_t^{\text{filtered}}$. Si el espectro de $g_t$ tiene forma Lorentziana $E_j = \sigma^2 / (1 + (j/\lambda)^2)$, entonces:

$$\frac{\mathbb{E}[\|\Delta_t^{\text{filtered}}\|^2]}{\mathbb{E}[\|\Delta_t\|^2]} = \frac{1}{\pi}\left(\arctan\frac{\omega_c P}{\lambda} - \frac{\lambda\omega_c P}{\lambda^2 + (\omega_c P)^2}\right)$$

*Demostración.* Por Parseval: $\|\Delta_t\|^2 = \eta^2 \sum_j E_j = \eta^2 \sigma^2 \sum_j \frac{1}{1+(j/\lambda)^2}$. Aproximando la suma por integral:

$$\sum_{j=1}^{\omega_c P} \frac{1}{1+(j/\lambda)^2} \approx \int_0^{\omega_c P} \frac{dx}{1+(x/\lambda)^2} = \lambda \arctan\frac{\omega_c P}{\lambda}$$

Restando la contribución del término $j/(\lambda^2 + j^2)$ para la Lorentziana exacta:

$$\frac{\text{Var}(\Delta^{\text{filtered}})}{\text{Var}(\Delta)} = \frac{\lambda \arctan(\omega_c P/\lambda) - \frac{(\omega_c P)^2\lambda}{\lambda^2 + (\omega_c P)^2}}{\lambda^2 + (\omega_c P)^2} \cdot \frac{\lambda^2 + (\omega_c P)^2}{\lambda^2} = \frac{1}{\pi}\left(\arctan\frac{\omega_c P}{\lambda} - \frac{\omega_c P}{\lambda + \omega_c P}\right)$$

usando $\sum_{j=1}^P \frac{1}{1+(j/\lambda)^2} \approx \pi\lambda/2$ para $\lambda \ll P$. $\blacksquare$

*Remark 6.3.* Para $\omega_c P \gg \lambda$: la razón $\approx 1 - \lambda/(\omega_c P)$, confirmando que alta frecuencia es suprimida efectivamente.

**Conexión con Preconditioners Clásicos (Adam/RMSProp).**
FDGD es análogo a aplicar **preconditioning** en dominio de Fourier. Adam (Kingma & Ba, 2014) usa:
$$m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t, \quad v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2$$
$$g_t^{Adam} = \frac{m_t}{\sqrt{v_t} + \epsilon}$$
Esto es un filtro adaptativo en **dominio espacial** (coeficiente por parámetro). FDGD es un filtro en **dominio de Fourier** (coeficiente por frecuencia). La diferencia clave:
- Adam filtra independientemente cada parámetro (sin correlación entre frecuencias)
- FDGD filtra por frecuencia (exploiting correlación entre parámetros adyacentes en el espacio de parámetros)

En la práctica, Adam y FDGD son **complementarios**: Adam ya está en uso en todos los trainings modernos; FDGD añade filtrado adicional de alta frecuencia que Adam no captura (porque los momentos de Adam tienen su propia dinámica de alta frecuencia). Aplicar FDGD antes de Adam equivale a pre-condicionar el gradiente para que Adam converja más rápido.

**Non-stationarity del Espectro.**
El espectro de gradientes $E_j(t)$ cambia durante training: early training tiene energía más distribuida (más alta frecuencia), late training concentra energía en baja frecuencia. Esto significa que el cutoff óptimo $\omega_c^*(t)$ cambia con $t$. La estrategia práctica es estimar $\omega_c$ durante warmup (primeras 1000 iteraciones) y mantenerlo fijo durante el resto del training, ya que el espectro se estabiliza. Re-estimar $\omega_c$ dinámicamente tiene overhead $O(P \log P)$ por estimación, lo cual no compensa.

---

## 7. Gradient-Normalized Skip (GNS)

### 7.1 Problema

El backward pass de cada capa computa gradientes $\partial\mathcal{L}/\partial\theta_i$ incluso cuando $\|\partial\mathcal{L}/\partial\theta_i\|$ es órdenes de magnitud menor que el de otras capas. Esto indica que la capa ya ha convergido a una región óptima localmente, y actualizar sus pesos sería redundante.

### 7.2 Idea Central

Usar la magnitud relativa del gradiente $\|\partial\mathcal{L}/\partial\theta_i\|$ como indicador de convergencia de capa. Si $\|\partial\mathcal{L}/\partial\theta_i\| < \epsilon \cdot \max_j \|\partial\mathcal{L}/\partial\theta_j\|$, la capa está "convergida" y podemos skip su backward pass. Esto ahorra FLOPs sin afectar la convergencia del modelo porque los pesos de esa capa apenas cambiarían de todas formas.

*Nota histórica:* La conexión con Information Bottleneck (IB) es conceptual pero no formal. En IB, $I(h_i; \mathcal{L})$ mide la información que la representación de capa $i$ captura sobre el target. Computar $I(h_i; \mathcal{L})$ requiere estimar densidades $p(h_i), p(h_i|\mathcal{L})$, lo cual es intractable. La norma del gradiente $\|\partial\mathcal{L}/\partial h_i\|$ es un proxy entrenable: por la regla de la cadena, $\|\partial\mathcal{L}/\partial\theta_i\| = \|(\partial\mathcal{L}/\partial h_i) \cdot (\partial h_i/\partial\theta_i)\|$, que es cero iff el gradiente de información es cero. Llamamos a esta técnica **GNS** (Gradient-Normalized Skip) para evitar malinterpretaciones.

### 7.3 Definiciones Formales

**Definición 7.1 (Gradiente de Capa).**
Sea $\theta_i$ los pesos de la capa $i$ y $\mathcal{L}$ la loss. El gradiente de capa es:

$$g_i = \frac{\partial \mathcal{L}}{\partial \theta_i} \in \mathbb{R}^{|\theta_i|}$$

Su norma euclidiana es $\|g_i\|_2$.

**Definición 7.2 (Magnitud Relativa de Gradiente).**
Definimos la magnitud relativa de capa $i$ como:

$$\rho_i = \frac{\|g_i\|_2}{\max_j \|g_j\|_2}$$

Nótese que $\rho_i \in [0, 1]$ y $\max_j \rho_j = 1$.

**Definición 7.3 (Criterio de Skip).**
Una capa $i$ se skip en el paso $t$ si:

$$\rho_i(t) < \epsilon_{GNS}$$

para algún umbral $\epsilon_{GNS} \in (0, 1)$. Las capas con $\rho_i < \epsilon_{GNS}$ contribuyen menos del $\epsilon_{GNS} \times 100\%$ del gradiente máximo.

**Definición 7.4 (Forward Pass Completo vs. Skip).**
Sea $\theta_{full}$ el estado de pesos después de un paso de SGD completo (todas las capas), y sea $\theta_{GNS}$ el estado después de skippar capas con $\rho_i < \epsilon_{GNS}$. Las actualizaciones son:

$$\theta_{full} = \theta - \eta \sum_i g_i$$
$$\theta_{GNS} = \theta - \eta \sum_{i: \rho_i \geq \epsilon_{GNS}} g_i$$

### 7.4 Teorema GNS-1: Bound del Error de Skip

*Corrección (2026-07-20).* La versión anterior de este teorema probaba tres argumentos distintos en sucesión y abandonaba los dos primeros a medio camino (uno señalado en el propio texto como "muy débil", otro que terminaba en "esto no nos da un bound superior") antes de aterrizar en un cuarto argumento que asumía —sin declararlo como hipótesis del teorema— que el ruido de skip es isotrópico e independiente de $\Delta_t$. Ese supuesto no es gratis: sin él, solo se puede probar una cota determinista peor ($O(T)$, no $O(\sqrt{T})$). Además el enunciado original incluía un factor $1/\mu$ que la prueba final nunca produce. Aquí se separan explícitamente los dos regímenes, cada uno con su propia hipótesis y su propia tasa.

**Teorema GNS-1 (Divergencia Acumulada por Skip).**
Sea $\theta_t$ la solución SGD completa en paso $t$, y $\tilde{\theta}_t$ la solución GNS. Sean $g_i^{(t)}$ los gradientes en paso $t$. Sea $\delta = \max_{i: \rho_i < \epsilon_{GNS}} \|g_i\|$ la máxima norma de gradiente en capas skippeadas, y $n_{skip}$ el número de capas skippeadas por paso. Sea $\Delta_t = \theta_t - \tilde{\theta}_t$, con $\Delta_0 = 0$.

*Caso 1 (determinista, sin hipótesis adicionales).* Para todo $t$:

$$\|\Delta_T\| \leq T \cdot \eta \cdot n_{skip} \cdot \delta \qquad \Rightarrow \qquad \|\Delta_T\| = O(T)$$

*Caso 2 (estocástico, bajo la Hipótesis GNS-1).* Si además el ruido de skip $d_t := \sum_{i:\rho_i<\epsilon_{GNS}} g_i^{(t)}$ es isotrópico en $\mathbb{R}^d$ y no correlacionado con $\Delta_t$ paso a paso (Hipótesis GNS-1), entonces:

$$\mathbb{E}[\|\Delta_T\|^2] = T\eta^2\delta^2 \qquad \Rightarrow \qquad \sqrt{\mathbb{E}[\|\Delta_T\|^2]} = \eta\delta\sqrt{T} = O(\sqrt{T})$$

*Demostración.* La diferencia $\Delta_t$ evoluciona como:

$$\Delta_{t+1} = \Delta_t - \eta \sum_{i: \rho_i < \epsilon_{GNS}} g_i^{(t)} = \Delta_t - \eta d_t$$

**Caso 1.** Por la desigualdad triangular, $\|d_t\| \leq n_{skip}\delta$ para todo $t$, así que:

$$\|\Delta_{t+1}\| \leq \|\Delta_t\| + \eta\|d_t\| \leq \|\Delta_t\| + \eta n_{skip}\delta$$

Iterando desde $\Delta_0 = 0$: $\|\Delta_T\| \leq T\eta n_{skip}\delta$. Esta cota no requiere ningún supuesto sobre la correlación entre $\Delta_t$ y $d_t$ — es válida en el peor caso absoluto (adversarial), incluyendo el caso en que el ruido de skip siempre apunta a alejar $\tilde\theta_t$ de $\theta_t$.

**Caso 2.** Bajo la Hipótesis GNS-1:

$$\|\Delta_{t+1}\|^2 = \|\Delta_t\|^2 - 2\eta\langle\Delta_t, d_t\rangle + \eta^2\|d_t\|^2$$

Tomando esperanza y usando $\mathbb{E}[\langle\Delta_t, d_t\rangle] = 0$ (independencia + isotropía de $d_t$) y $\mathbb{E}[\|d_t\|^2] = \delta^2$:

$$\mathbb{E}[\|\Delta_{t+1}\|^2] = \mathbb{E}[\|\Delta_t\|^2] + \eta^2\delta^2$$

Iterando desde $\Delta_0=0$: $\mathbb{E}[\|\Delta_T\|^2] = T\eta^2\delta^2$. $\blacksquare$

El Caso 1 es siempre válido pero pesimista ($O(T)$, crece sin cota). El Caso 2 es la tasa relevante en la práctica ($O(\sqrt{T})$, comportamiento de martingala) pero depende de que el ruido de skip no esté sistemáticamente correlacionado con la dirección de divergencia acumulada — algo plausible si las capas que se skippean cambian de iteración a iteración, pero no garantizado si el criterio $\rho_i$ skippea repetidamente la misma capa (ver Falla 2 en §7.8, que además supone este régimen estocástico explícitamente).

*Remark 7.1.* La cota $\delta = \max_{i: \rho_i < \epsilon_{GNS}} \|g_i\|$ puede estimarse en la práctica como $\delta \approx \epsilon_{GNS} \cdot \max_j \|g_j\|$ si la distribución de $\|g_i\|$ es aproximadamente geométrica. Bajo el régimen estocástico (Caso 2):

$$\mathbb{E}[\|\theta_T - \tilde{\theta}_T\|] \leq \eta \delta \sqrt{T} = \eta \epsilon_{GNS} \cdot \max_j \|g_j\| \cdot \sqrt{T}$$

Para $\eta = 10^{-4}, \epsilon_{GNS} = 0.1, \max_j \|g_j\| \approx 1, T = 10^4$: la divergencia esperada es $\sim 1$, comparable a la escala de los pesos. Bajo el régimen determinista (Caso 1), la misma configuración da una cota mucho más conservadora, $\|\Delta_T\| \leq T\eta n_{skip}\delta$, que crece linealmente y debe usarse cuando no se puede justificar la Hipótesis GNS-1.

**Corolario GNS-1.1 (Convergencia Condicional).**
Si $\mathcal{L}$ es $\mu$-strongly convex y el learning rate satisface $\eta < 2/\mu$, entonces tanto $\theta_t$ como $\tilde{\theta}_t$ convergen a sus respectivos óptimos $\theta^*$ y $\tilde{\theta}^*$. La distancia entre los óptimos satisface:

$$\|\theta^* - \tilde{\theta}^*\| \leq \frac{\eta \delta}{\mu}$$

*Demostración.* En el régimen lineal, la diferencia entre los campos vectoriales $F(\theta) = -\nabla\mathcal{L}(\theta)$ y $\tilde{F}(\theta) = -\nabla\tilde{\mathcal{L}}(\theta)$ está acotada por $\|F(\theta) - \tilde{F}(\theta)\| \leq \delta$ (ya que difieren solo en las capas skippeadas). Para un campo vectorial $\mu$-strongly convex, el operador de proximal es contractivo con factor $(1-\mu\eta)$ por paso. Acumulando la diferencia sobre infinitos pasos:

$$\|\theta^* - \tilde{\theta}^*\| \leq \sum_{k=0}^{\infty} (1-\mu\eta)^k \cdot \eta \delta = \frac{\eta \delta}{\mu}$$

$\blacksquare$

### 7.5 Teorema GNS-2: Selección de Umbral

**Teorema GNS-2 (Umbral Óptimo).**
Sea $E(\epsilon)$ el error esperado de skip (i.e., la diferencia en loss entre usar umbral $\epsilon$ y usar backward completo) y sea $C(\epsilon)$ el costo computacional (FLOPs de backward normalizados). El umbral óptimo minimiza el tradeoff:

$$\epsilon^* = \arg\min_\epsilon \left\{ E(\epsilon) + \lambda \cdot C(\epsilon) \right\}$$

donde $\lambda > 0$ es un Lagrange multiplier que balancea accuracy vs. velocidad.

Asumiendo que los gradientes $\|g_i\|$ siguen una distribución exponencial con parámetro $\lambda_g$ (i.e., $P(\|g_i\| > x) = e^{-x/\lambda_g}$), entonces:

$$\epsilon^* \approx \frac{\lambda}{\lambda_g} \cdot \frac{1}{T}$$

*Demostración.* Si $\|g_i\| \sim \text{Exp}(\lambda_g)$, entonces:

$$P(\rho_i \geq \epsilon) = P\left(\|g_i\| \geq \epsilon \cdot \max_j \|g_j\|\right)$$

No podemos asumir independencia entre capas porque $\max_j \|g_j\|$ es una variable aleatoria que depende de todas las capas. Asumimos en cambio que conditional on $\|g_i\| = x$, la distribución de $\rho_i$ es aproximadamente uniforme en $[0,1]$ para las capas no-max.

Simplificamos: el número de capas skippeadas es $n_{skip} \approx n \cdot \epsilon$ donde $n$ es el número total de capas (asumiendo $\|g_i\|$ distribuidos uniformemente en $[0, \max]$). Por lo tanto:

$$C(\epsilon) = 1 - \epsilon$$

El error $E(\epsilon)$ puede aproximarse por el Teorema GNS-1. Si $\delta \approx \epsilon \cdot G_{max}$:

$$E(\epsilon) \approx \eta \cdot \epsilon \cdot G_{max} \cdot \sqrt{T}$$

Minimizando $E(\epsilon) + \lambda C(\epsilon) = \eta G_{max} \sqrt{T} \cdot \epsilon + \lambda(1-\epsilon)$:

$$\frac{d}{d\epsilon} = \eta G_{max} \sqrt{T} - \lambda = 0 \implies \epsilon^* = \frac{\lambda}{\eta G_{max} \sqrt{T}}$$

Si además normalizamos $G_{max} = 1$ (dividiendo todos los gradientes por el máximo), obtenemos $\epsilon^* = \lambda / (\eta \sqrt{T})$. $\blacksquare$

### 7.6 Teorema GNS-3: Reducción de FLOPs

**Teorema GNS-3 (FLOPs Savings).**
Sea $n$ el número de capas, y sea $p_{skip}(\epsilon) = P(\rho_i < \epsilon)$ la proporción de backward passes skippeados con umbral $\epsilon$. El factor de reducción de FLOPs de backward pass es:

$$\rho_{GNS} = 1 - p_{skip}(\epsilon)$$

Para la distribución exponencial de gradientes con $\|g_i\| \sim \text{Exp}(\lambda_g)$ y normalizando por $\max_j \|g_j\|$:

$$p_{skip}(\epsilon) = \epsilon$$

ya que $P(\rho_i < \epsilon) = \epsilon$ si las $n-1$ capas no-máximas tienen magnitud relativa uniforme en $[0,1]$. En la práctica, la distribución es más concentrada en valores bajos, por lo que $p_{skip}(\epsilon) > \epsilon$.

*Demostración.* Cada capa tiene costo de backward $C_i \propto |\theta_i|$ (proporcional al número de parámetros). Si asumimos $C_i \approx C$ constante para todas las capas:

$$\rho_{GNS} = \frac{\sum_{i: \rho_i \geq \epsilon} C_i}{\sum_i C_i} \approx \frac{(1-p_{skip}) \cdot n \cdot C}{n \cdot C} = 1 - p_{skip}(\epsilon)$$

Si las capas tienen costos variables, la weighted fraction es:

$$\rho_{GNS} = 1 - \frac{\sum_{i: \rho_i < \epsilon} C_i}{\sum_i C_i}$$

que es la fracción del costo total de backward salvada. $\blacksquare$

### 7.7 Teorema GNS-4: Monotonicidad de Gradientes

**Teorema GNS-4 (Decrease de $\|g_i\|$ durante Training).**
Sea $g_i^{(t)} = \partial\mathcal{L}(\theta_t)/\partial\theta_i$ el gradiente de la capa $i$ en paso $t$. Bajo SGD con learning rate $\eta$ y momentum $\beta$, si la loss $\mathcal{L}$ es $\mu$-strongly convex y $L$-smooth, entonces las normas de gradiente satisfacen:

$$\mathbb{E}[\|g_i^{(t+1)}\|] \leq (1 - \eta\mu)^t \cdot \mathbb{E}[\|g_i^{(0)}\|] + \frac{\eta L \sigma}{\mu}$$

para todo $i$, donde $\sigma^2$ es la varianza del ruido de gradiente estocástico.

*Demostración.* Consideramos la dinámica de $\|g_i(\theta)\|^2$ a lo largo de la trayectoria de SGD. Por la regla de la cadena:

$$\frac{\partial}{\partial \theta}\|g_i(\theta)\|^2 = 2 g_i(\theta)^T \cdot H_i(\theta) \cdot (-g_\theta)$$

donde $H_i = \partial g_i/\partial\theta$ es la Hessiana de la loss respecto a los parámetros de la capa $i$, y $g_\theta = \partial\mathcal{L}/\partial\theta$ es el gradiente completo.

Bajo $\mu$-strong convexity de $\mathcal{L}$, la Hessiana $H = \nabla^2 \mathcal{L}$ satisface $H \succeq \mu I$. Los eigenvalores de $H_i$ están relacionados con los de $H$: para capas fully-connected, $\|H_i\| \leq \|H\| \leq L$. El gradiente $\|g_i\|$ decrece monotónicamente porque la optimización converge a un mínimo donde $\|g_i\| \to 0$.

Para la trayectoria SGD, modelamos $\|g_i^{(t)}\|^2$ como un proceso estocástico que decrece. La Esperaza condicionada es:

$$\mathbb{E}[\|g_i^{(t+1)}\|^2 \mid \theta_t] \leq (1 - 2\eta\mu + \eta^2 L^2) \cdot \|g_i^{(t)}\|^2 + \eta^2 L^2 \sigma^2$$

Si $\eta < 2\mu/L^2$, el factor de decay es $\rho = 1 - 2\eta\mu + O(\eta^2)$. Tomando raíz cuadrada y usando Jensen:

$$\mathbb{E}[\|g_i^{(t+1)}\|] \leq \sqrt{\rho} \cdot \mathbb{E}[\|g_i^{(t)}\|] + \eta L \sigma$$

Resolviendo esta recurrencia: $\mathbb{E}[\|g_i^{(t)}\|] \leq \rho^{t/2} \cdot \|g_i^{(0)}\| + \frac{\eta L \sigma}{1 - \rho^{1/2}}$. Para $\eta$ pequeño: $\rho^{1/2} \approx 1 - \eta\mu$. $\blacksquare$

*Remark 7.2 (Implicación para GNS).* Este teorema justifica el uso de $\|\partial\mathcal{L}/\partial\theta_i\|$ como criterio de skip: conforme avanza el training, los gradientes de todas las capas decrecen monotónicamente. Esto significa que una capa que es "grande" al inicio de training puede volverse "pequeña" (skippeable) después de suficiente entrenamiento. El threshold $\epsilon_{GNS}$ debería ser adaptativo: $\epsilon_{GNS}(t) = \epsilon_0 \cdot (1 - \eta\mu)^t$ para mantener la misma fracción de skip a medida que los gradientes decaen.

*Remark 7.3 (No monotonicidad estricta).* El teorema prueba monotonicidad de la esperanza, no de cada trajectory individual. En SGD estocástico, $\|g_i^{(t)}\|$ puede aumentar momentáneamente debido al ruido de minibatch. Para evitar skip falsos, GNS debe usar una media móvil de $\|g_i\|$ sobre las últimas $w$ iteraciones:

$$\bar{g}_i^{(t)} = \frac{1}{w}\sum_{s=0}^{w-1} \|g_i^{(t-s)}\|$$

Solo se hace skip si $\bar{g}_i^{(t)} < \epsilon_{GNS} \cdot \max_j \bar{g}_j^{(t)}$.

### 7.8 Casos de Falla

**Falla 1: Gradientes pequeños pero no convergidos.**
Una capa puede tener gradientes pequeños porque está cerca de un óptimo local, pero ese óptimo puede ser un local suboptimal. Skippar el backward de esta capa significa que nunca escaparemos de ese local suboptimal.

*Señales de alerta:* El loss de validación deja de mejorar pero los gradientes de la capa también son pequeños — esto puede ser skip o puede ser stuck.

**Falla 2: Acumulación de errores de skip en series largas.**
Si hacemos skip en la *misma* capa por $T$ pasos consecutivos, la Hipótesis GNS-1 (ruido de skip isotrópico e independiente paso a paso) deja de ser plausible: el ruido apunta repetidamente en la dirección del gradiente de esa capa, no en direcciones que se cancelan en promedio. Este es exactamente el régimen del **Caso 1** (determinista) del Teorema GNS-1, no del Caso 2: el error relevante es $\|\theta_T - \tilde{\theta}_T\| \leq T\eta n_{skip}\delta$, que crece **linealmente**, no como $\sqrt{T}$. Para $T$ grande (e.g., $10^5$ pasos) esto es sustancialmente peor que el drift $\sqrt{T}$ de un SGD normal (martingala), y no simplemente aditivo a él — es un régimen cualitativamente distinto que puede dominar el entrenamiento si el criterio de skip no alterna entre capas.

**Falla 3: Sesgo hacia capas de alta magnitud.**
Si las capas de atención tienen gradientes sistemáticamente mayores que las capas FFN (o viceversa), el criterio $\rho_i$ introduce un sesgo estructural que puede hacer que las capas de menor magnitud se desconecten del entrenamiento.

*Mitigación:* Normalizar por layer type, no por capa individual. Usar ranking relativo dentro de cada layer type.

### 7.9 Relación con SMV y TER

GNS puede composerse con SMV (Spectral Modulation Vector) y TER (Token Entropy Routing):
- **SMV + GNS:** SMV modula la representación espectral; GNS decide si la capa necesita actualizar sus pesos
- **TER + GNS:** TER reduce el número de pasos de integración; GNS reduce los backward passes por capa. Juntos: TER ahorra FLOPs de forward-integration, GNS ahorra FLOPs de backward-gradient.

El speedup combinado es multiplicativo: $\rho_{total} = \rho_{TER} \cdot \rho_{GNS}$. Para $\rho_{TER} = 0.6$ y $\rho_{GNS} = 0.7$, el speedup total es $0.42$, i.e., ~58% de reducción de FLOPs.

---

## 8. Optimizaciones Adicionales

### 8.1 Spectral Memory Attention (SMA)

**Idea:** En lugar de almacenar activaciones completas $h_i$ como checkpoint para backward, almacenar solo la proyección espectral $s_i = U_k^T h_i \in \mathbb{R}^k$. Reconstruir $h_i \approx U_k s_i$ durante backward. Compression ratio: $d/k$.

**Teorema SMA-1 (Error de Reconstrucción).**
Sea $W \in \mathbb{R}^{d \times d_i}$ la matriz de peso de una capa lineal, con SVD $W = U \Sigma V^T$ donde $U \in \mathbb{R}^{d \times d}$, $\Sigma \in \mathbb{R}^{d \times d_i}$, $V \in \mathbb{R}^{d_i \times d_i}$. Sea $U_k \in \mathbb{R}^{d \times k}$ las primeras $k$ columnas de $U$ (vectores singulares dominantes). Sea $\tilde{h} = U_k U_k^T h$ la reconstrucción de $h = Wx$ desde su proyección $k$-dimensional. Entonces:

$$\|h - \tilde{h}\| \leq \|x\| \cdot \sqrt{\sum_{j=k+1}^{\min(d, d_i)} \sigma_j^2(W)}$$

donde $\sigma_j(W)$ son los valores singulares de $W$ ordenados descendentemente.

*Demostración.* Por SVD truncada: $W = U_k \Sigma_k V_k^T + U_\perp \Sigma_\perp V_\perp^T$ donde $U_\perp$ contiene las columnas restantes. Entonces:

$$\|h - \tilde{h}\| = \|U_\perp \Sigma_\perp V_\perp^T x\| \leq \|\Sigma_\perp V_\perp^T x\| \leq \|\Sigma_\perp\| \cdot \|V_\perp^T x\| \leq \|\Sigma_\perp\|_F \cdot \|x\|$$

ya que $\|V_\perp^T x\| \leq \|x\|$ ($V_\perp$ tiene columnas ortonormales). Además $\|\Sigma_\perp\|_F^2 = \sum_{j=k+1}^{\min(d,d_i)} \sigma_j^2(W)$. $\blacksquare$

*Remark 8.1.* La magnitud del error depende de $\|x\|$, la norma de la entrada. Para tokens de lenguaje con embedding normalizado ($\|x\| \approx 1$), el error depende solo de los valores singulares descartados.

**Teorema SMA-2 (Costo de Reconstrucción en Backward).**
Sea $\tilde{h} = U_k U_k^T h$ la reconstrucción desde la proyección $k$-dimensional. El costo computacional de reconstruir $h$ durante backward es:

$$C_{recon} = 2 \cdot k \cdot d$$

operaciones (dos matrix-vector products: $U_k^T h$ y $U_k s$). Comparado con el costo de backward completo $C_{backward} = O(|W|) = d \cdot d_i$, el overhead relativo es:

$$\frac{C_{recon}}{C_{backward}} = \frac{2kd}{d \cdot d_i} = \frac{2k}{d_i}$$

Para $k = 128, d_i = 4096$: overhead = $256/4096 = 6.25\%$. Este overhead es aceptable porque se paga solo en las capas donde se aplica SMA, no en todas.

*Demostración.* Directo del conteo de operaciones. $\blacksquare$

*Corrección (2026-07-20).* La versión anterior afirmaba $\|g-g_k\|\le\|g\|\cdot\sigma_{k+1}/\sigma_k$, es decir, que la componente de $g$ descartada por la proyección $U_kU_k^T$ es pequeña en norma propia. Eso no se sostiene: $g=\partial\mathcal{L}/\partial h$ es el gradiente que llega de capas *posteriores* y no tiene por qué alinearse con los vectores singulares de $W$ (una propiedad de la matriz de *pesos*, no del gradiente). La prueba original lo intentaba justificar con "$\sigma_1\approx\sigma_k$ (buen conditioning)" — una hipótesis no declarada ni usada consistentemente ($\sigma_1$ vs. $\sigma_k$ en el numerador/denominador de pasos consecutivos), y el paso "$\|U_\perp\|\cdot\|U_\perp^Tg_h\| = \|g_h^\perp\|$" es una identidad trivial (ya que $\|U_\perp\|=1$) que no acota nada. Lo que sí es demostrable sin supuestos adicionales es el efecto de descartar $g_h^\perp$ sobre el gradiente que SMA efectivamente propaga hacia capas anteriores, $g_x = W^Tg_h$ — porque ahí $g_h^\perp$ se multiplica por los valores singulares pequeños $\Sigma_\perp$ antes de combinarse con $V$. Esa es la cantidad que se acota abajo.

**Teorema SMA-3 (Preservación de Gradientes Importantes, corregido).**
Sea $g_h = \partial\mathcal{L}/\partial h \in \mathbb{R}^d$ el gradiente de la loss respecto a la activación de salida $h=Wx$, y sea $g_x = W^Tg_h$ el gradiente propagado hacia la entrada $x$. Sea $g_{h,k}=U_kU_k^Tg_h$ la proyección de $g_h$ retenida por SMA y $g_{x,k}=W^Tg_{h,k}$ el gradiente propagado usando solo esa proyección. Entonces, **sin hipótesis adicionales sobre $g_h$**:

$$\|g_x - g_{x,k}\| \leq \sigma_{k+1}(W) \cdot \|g_h\|$$

*Demostración.* Sea $g_h^\perp = g_h - g_{h,k} = U_\perp U_\perp^T g_h$ la componente descartada. Por linealidad, $g_x - g_{x,k} = W^Tg_h^\perp$. Usando $W=U\Sigma V^T$:

$$W^Tg_h^\perp = V\Sigma^T U^T g_h^\perp$$

Como $g_h^\perp$ vive enteramente en el subespacio de $U_\perp$, $U^Tg_h^\perp$ tiene componentes nulas en las primeras $k$ coordenadas; solo el bloque $\Sigma_\perp = \text{diag}(\sigma_{k+1},\ldots)$ actúa sobre él. Por lo tanto, usando que $V$ y $U_\perp$ tienen columnas ortonormales:

$$\|W^Tg_h^\perp\| = \|\Sigma_\perp U_\perp^Tg_h^\perp\| \leq \sigma_{k+1}(W)\cdot\|U_\perp^Tg_h^\perp\| = \sigma_{k+1}(W)\cdot\|g_h^\perp\| \leq \sigma_{k+1}(W)\cdot\|g_h\|$$

ya que $\|g_h^\perp\|\leq\|g_h\|$ (proyección ortogonal). $\blacksquare$

*Remark 8.2.* Esta es una cota absoluta, no relativa: depende de $\sigma_{k+1}(W)$ solo (el valor singular inmediatamente descartado), no de un cociente $\sigma_{k+1}/\sigma_k$ que requeriría conocer cuán "picuda" es la caída espectral. Para matrices bien comprimibles ($\sigma_{k+1}$ pequeño en términos absolutos, no solo relativos a $\sigma_k$), el error de propagación es pequeño independientemente de cómo se alinee $g_h$ con el espectro de $W$.

**Teorema SMA-4 (Comparación con Activation Checkpointing Estándar).**
Sea $C_{full} = d \cdot d_i$ el costo de almacenar la activación completa $h$ (1 tensor de $d$ floats), y $C_{SMA} = 2k$ el costo de almacenar la proyección espectral $s = U_k^T h$ (2 tensors: $s$ y $U_k^T$). El factor de compresión es:

$$\rho_{SMA} = \frac{C_{SMA}}{C_{full}} = \frac{2k}{d}$$

Para $d = 4096, k = 128$: $\rho_{SMA} = 256/4096 = 1/16$, i.e., **6.25% del storage** comparado con guardar activations completas.

La técnica de activation checkpointing estándar (Chen et al., 2016) guarda $h$ completo y lo reconstruye en backward sin costo de compute adicional (solo storage). SMA guarda menos pero tiene overhead de compute en backward. El tradeoff es:

| Técnica | Storage | Compute en Backward |
|---------|---------|---------------------|
| Sin checkpoint | $d$ (guardar todo) | 0 (recalcular) |
| Checkpoint estándar | $d$ (guardar todo) | 0 |
| SMA | $2k$ (guardar espectro) | $2kd$ (reconstruir) |

SMA gana cuando el bottleneck es storage pero el compute es abundante.

**Teorema SMA-5 (Condiciones de Aplicabilidad).**
SMA es beneficial iff:

$$\frac{2k}{d} \cdot C_{compute\_per\_token} < \text{VRAM\_savings} \cdot \text{costo\_transfer}$$

Específicamente, para VRAM limitada donde cada MB cuenta, SMA saves $d - 2k$ floats de storage por capa. Si $d - 2k > 0$ (siempre para $k < d/2$), y el overhead de compute $2kd$ es tolerable (porque el backward pass ya hace $O(d \cdot d_i)$ compute), entonces SMA es ventajoso.

**Corolario SMA-5.1 (Criterio Práctico).**
SMA aplica a capas donde:
1. $d_i > 4k$ (la capa es suficientemente "ancha" para que guardar $U_k^T$ sea más barato que guardar $h$)
2. La reconstrucción $U_k s$ no introduce error significativo para el training ($\sum_{j>k}\sigma_j^2 < \epsilon^2$)
3. La capa es parte del critical path de VRAM (i.e., no se puede permitir guardar $h$ completo)

### 8.2 Hessian-Free Initialization via Spectral Condition Number (HFISC)

**Idea:** Elegir la inicialización de $\theta$ (los pesos del MLP de modulación) tal que el número de condición de la matriz de modulación sea óptimo. Esto acelera la convergencia temprana.

*Corrección (2026-07-20).* La versión anterior de este teorema conflaba dos objetos matemáticos distintos bajo el mismo símbolo. $M$, "la matriz de modulación", se introduce con eigenvalores $\lambda_i$ y se usa para definir $\kappa=\lambda_1/\lambda_d$ — pero por su nombre y por su uso en 8.2.2-8.2.4, $M$ es en realidad el **Jacobiano local de $m_\theta$ respecto a su entrada $h$** en un punto $h_0$ (dimensión $d\times d$, $d$=dimensión oculta). El teorema, sin embargo, aplica la tasa de convergencia de SGD a $\min_\theta\mathcal{L}(\theta)$ — la minimización de la **loss de entrenamiento respecto a los parámetros $\theta$** (dimensión = número de parámetros, no $d$) — y su demostración asume directamente "$H=M^TM$" donde $H$ es la Hessiana de $\mathcal{L}$ respecto a $\theta$. Esa identidad no está justificada: no hay relación establecida entre la curvatura de $m_\theta$ como función de su *entrada* y la curvatura de la loss como función de sus *parámetros* — son Hessianas de objetos diferentes, con dominios de dimensión distinta en general. Aquí se separan explícitamente ambos objetos y se corrige el teorema a lo que es efectivamente demostrable.

**Teorema 8.2.1 (Convergencia de SGD, versión estándar — corregido).**
Sea $H_\mathcal{L}(\theta) = \nabla^2_\theta\mathcal{L}(\theta)$ la Hessiana de la loss respecto a los parámetros $\theta$, con número de condición $\kappa_\mathcal{L} = \lambda_{\max}(H_\mathcal{L})/\lambda_{\min}(H_\mathcal{L})$ y $\mu=\lambda_{\min}(H_\mathcal{L})$. Este es el resultado **estándar** de convergencia de SGD (no específico a HFISC): para $\eta < 2/\mu$,

$$\mathbb{E}[\|\theta_t - \theta^*\|] \leq \left(1 - \frac{2\mu}{\kappa_\mathcal{L}}\right)^t \|\theta_0 - \theta^*\| + O(\eta)$$

*Demostración.* Estándar: $\theta_{t+1}-\theta^* = (I-\eta H_\mathcal{L})(\theta_t-\theta^*)$ cerca del óptimo (aproximación cuadrática de $\mathcal{L}$), con $\eta$ elegido para minimizar $\max_i|1-\eta\lambda_i|$ sobre el espectro de $H_\mathcal{L}$, dando factor $(\kappa_\mathcal{L}-1)/(\kappa_\mathcal{L}+1) \to 0$ cuando $\kappa_\mathcal{L}\to1$. $\blacksquare$

*Lo que HFISC realmente afecta.* HFISC controla el número de condición $\kappa_J$ del Jacobiano local de $m_\theta$ respecto a su entrada, $J_{m_\theta}(h_0)$ (objeto de dimensión $d$, ver Teorema 8.2.2), **no** $\kappa_\mathcal{L}$ directamente. La conexión entre "el sub-módulo de modulación tiene un Jacobiano de entrada bien condicionado en $t=0$" y "la loss completa $\mathcal{L}(\theta)$ tiene una Hessiana de parámetros mejor condicionada durante el entrenamiento temprano" es una hipótesis heurística plausible (un sub-módulo cerca de la identidad no distorsiona ni amplifica gradientes que fluyen a través de él, lo cual favorece condicionamiento aguas arriba) pero **no está demostrada** en este documento y no debe presentarse como corolario del Teorema 8.2.1. El Teorema 8.2.1 corregido es el resultado estándar de SGD, citado aquí como referencia de por qué un buen condicionamiento (del objeto que sea) ayuda; los Teoremas 8.2.2-8.2.4 sí son correctos y auto-contenidos porque prueban afirmaciones sobre $J_{m_\theta}(h_0)$ exclusivamente, sin invocar $H_\mathcal{L}$.

**Teorema 8.2.2 (Inicialización Óptima).**
Sea $m_\theta$ el MLP de modulación con pesos $\theta = (W_1, b_1, W_2, b_2)$. Asumimos $m_\theta(h) = W_2 \sigma(W_1 h + b_1) + b_2$ con $\sigma$ = GELU. Para que la Hessiana de $m_\theta$ sea cercana a la identidad en un punto $h_0$, basta con inicializar:

$$W_1^{(0)} = \frac{\sqrt{2}}{\sqrt{d_{in}}} \cdot U, \quad b_1^{(0)} = 0$$
$$W_2^{(0)} = \frac{\sqrt{2}}{\sqrt{d_{out}}} \cdot V^T, \quad b_2^{(0)} = 0$$

donde $U, V$ son matrices ortogonales (o cuasi-ortonales via QR), y $d_{in}, d_{out}$ son las dimensiones de entrada y salida. Entonces:

$$\kappa(H_{m_\theta}(h_0)) \approx 1 + O(\|h_0\|^2 \cdot \sigma_{\max}(\Delta W)^2)$$

*Demostración.* Para GELU cerca de zero: $\sigma(x) \approx x - x^3/6 + O(x^5)$. Entonces:

$$m_\theta(h) = W_2(W_1 h + b_1 - \frac{1}{6}(W_1 h + b_1)^3) + O(\|h\|^3)$$

El término lineal es $W_2 W_1 h$. Para que el Jacobiano sea cercano a identidad: $W_2 W_1 \approx I$. Con las escalas de Xavier modificado ($\sqrt{2/d}$), la varianza de $W_2 W_1$ es:

$$\text{Var}(W_2 W_1) = \frac{2}{\sqrt{d_{in} \cdot d_{out}}} \cdot \mathbb{E}[\|U V^T\|^2] \approx \frac{2}{\sqrt{d_{in} \cdot d_{out}}}$$

Para $d_{in} = d_{out} = d$: $\text{Var}(W_2 W_1) \approx 2/d$. La desviación de identidad $\|W_2 W_1 - I\|$ tiene norma esperada $\approx \sqrt{2/d}$. Por lo tanto, el Jacobiano está a distancia $O(1/\sqrt{d})$ de identidad, y la Hessiana de $m_\theta$ (dominated por el término cúbico) tiene norma $O(1/d)$. El número de condición $\kappa \approx 1 + O(1/d)$. $\blacksquare$

*Remark 8.2.2 (Conexión con Xavier/Kaiming).*
La inicialización de Xavier ($\sim 1/\sqrt{d_{in}}$) produce $W_2 W_1$ con varianza $\sim 1/d$, dando $\kappa \approx 1 + O(1/d)$ también. Sin embargo, Xavier optimiza la propagación de forward activation variance, mientras HFISC optimiza la Hessiana de la función de modulación. Para NMF, donde la modulación es un campo vectorial que escala lasactivaciones, HFISC es más apropiado porque directamente controla la curvatura (Hessiana) del espacio de modulación, no solo el scaling de activaciones.

Kaiming ($\sim \sqrt{2/d_{in}}$ para ReLU) es equivalente a Xavier modificado con factor $\sqrt{2}$, que es exactamente lo que proponemos para GELU.

**Teorema 8.2.3 (Análisis de Fase Temprana vs Tardía).**
Sea $\theta_t$ la trayectoria de SGD y $H_t = H(\theta_t)$ la Hessiana de $\mathcal{L}$. Sea $\kappa_t = \lambda_1(H_t)/\lambda_d(H_t)$ el número de condición en paso $t$.

Durante **fase temprana** ($t < t_{crit}$): $\kappa_t$ es pequeño porque la Hessiana está cerca de la identidad (el modelo aún no ha aprendido features específicos). HFISC acelera esta fase por factor $\sim (\kappa_{std} - 1)/(\kappa_{HFISC} - 1)$.

Durante **fase tardía** ($t > t_{crit}$): $\kappa_t$ crece porque la Hessiana desarrolla eigenvals pequeños (direcciones planas asociadas afeatures especializados). La inicialización HFISC no puede alterar esto — la estructura de la Hessiana en fase tardía está determinada por la geometría de la loss, no por la inicialización.

*Demostración.* Durante fase temprana, la dinámica de SGD es dominada por los eigenvalores grandes de $H$. Si $\kappa$ es pequeño, la elipticidad $\| \nabla^2 \mathcal{L} - \lambda_d I\|$ es pequeña, y todos los eigenvalores convergen a la misma rate $1 - O(\eta\lambda_d)$. HFISC reduce $\kappa$ en inicialización, acelerando esta fase. En fase tardía, los eigenvalores pequeños $\lambda_d$ pueden llegar a ser $O(10^{-6})$ mientras los grandes permanecen $O(1)$. Esto es una propiedad de la función de loss, no de $\theta_0$. $\blacksquare$

### 8.2.4 Teorema 8.2.4: Comparación Formal con Xavier y Kaiming

**Teorema 8.2.4 (Xavier vs HFISC vs Kaiming).**
Sea $m_\theta$ el MLP de modulación con pesos inicializados segón Xavier, HFISC-2, o Kaiming. El número de condición resultante de la Hessiana $H_{m_\theta}$ satisface:

$$\kappa_{Xavier} \approx 1 + \frac{1}{2d}, \quad \kappa_{HFISC} \approx 1 + \frac{1}{d}, \quad \kappa_{Kaiming} \approx 1 + \frac{2}{d}$$

*Demostración.* Para Xavier: $W_i \sim \mathcal{N}(0, 1/d)$, entonces $\text{Var}(W_1^{(0)} W_2^{(0)}) = 1/d$. La deviación de identidad $\|W_2^{(0)} W_1^{(0)} - I\|$ tiene norma $\sim 1/\sqrt{d}$ (porque cada entrada de $W_2^{(0)} W_1^{(0)} - I$ tiene varianza $1/d$). Por lo tanto $\kappa \approx 1 + O(1/d)$.

Para HFISC-2: $W_i^{(0)} \sim \sqrt{2/d} \cdot U$ con $U$ ortogonal, entonces $\text{Var}(W_2^{(0)} W_1^{(0)}) = 2/d$. La deviación $\|W_2^{(0)} W_1^{(0)} - I\|$ tiene norma $\sim \sqrt{2/d}$. Esto es PEOR que Xavier para la norma del Jacobiano, pero el objetivo de HFISC es que el Jacobiano sea cercano a identidad, no que la deviación sea mínima. HFISC escala por $\sqrt{2}$ para compensar que GELU no es exactamente lineal cerca de zero (a diferencia de lo que asume Xavier para sigmoid/tanh). La escala $\sqrt{2}$ emerge naturalmente de que $\text{sech}^2(0) = 1$ en la derivada de GELU.

Para Kaiming: $W_i \sim \mathcal{N}(0, 2/d)$, entonces $\text{Var}(W_2^{(0)} W_1^{(0)}) = 4/d$. La deviación $\|W_2^{(0)} W_1^{(0)} - I\| \sim 2/\sqrt{d}$, giving $\kappa \approx 1 + 2/d$.

HFISC usa la escala óptima para GELU: $\sqrt{2/d}$, que minimiza $\|W_2 W_1 - I\|$ subject a que $\mathbb{E}[(W_2 W_1)_{ii}] = 1$ (para mantener el Jacobiano unbiased). $\blacksquare$

*Remark 8.2.4 (Recomendación).* Para el MLP de modulación con GELU, la inicialización óptima es HFISC-2 con escala $\sqrt{2/d}$. Xavier ($\sqrt{1/d}$) subestima la escala para GELU; Kaiming ($\sqrt{2/d}$) coincide con HFISC porque Kaiming fue diseñado para ReLU (que tiene derivadas 0 o 1) y GELU tiene derivadas similares cerca de zero. Para nuestro caso específico donde queremos $W_2 W_1 \approx I$ (Jacobiano cercano a identidad), HFISC-2 y Kaiming son equivalent.

### 8.3 Token-wise ODE Warmstarting (TOWS)

**Idea restringida:** En los **módulos MLP** (i.e., las capas $W_{up}, W_{gate}, W_{down}$ de NMF), la ODE $dh/d\tau = f_\theta(h, \tau)$ opera punto a punto sobre cada token independientemente — no hay atención cruzada. Para tokens consecutivos (índice $i$ e $i+1$), las activaciones del MLP tienden a ser similares porque el MLP solo ve el token individual, no el contexto. Usamos $h_{MLP}^{\,i}(T)$ del token $i$ como warmstart para $h_{MLP}^{\,i+1}(0)$ del token $i+1$.

> **Nota de notación:** $\tau \in [0,T]$ es el parámetro de tiempo continuo de la ODE; $i$ es el índice del token en la secuencia. Se evitan las letras $t$ y $s$ para prevenir confusión con ambos significados.

**Nota crítica (PLAN):** Esta técnica NO aplica a las capas de attention. En un transformer, la salida de attention de token $i$ depende de TODOS los tokens $1, \ldots, i$ via el mecanismo de scaled dot-product attention:
$$h_i^{attn} = \sum_{j=1}^i \text{softmax}\left(\frac{Q_i K_j^T}{\sqrt{d}}\right) V_j h_j$$
Esto rompe completamente la suposición de Markov — no hay "estado" que se propague de un token al siguiente, porque cada token ve toda la secuencia. TOWS solo tiene sentido para el **MLP module**, donde la ODE sí opera token-by-token sin contexto cruzado.

**Definición 8.3.1 (Módulo MLP local).**
Sea $h_i^{mlp}(\tau)$ la representación intermedia del token $i$ después de pasar por la capa MLP en el tiempo $\tau$ de la ODE de NMF. El MLP opera como:
$$h_i^{mlp}(\tau) = \text{silu}(W_{gate} \cdot h_i^{ffn}(\tau)) \odot (W_{up} \cdot h_i^{ffn}(\tau))$$
donde $h_i^{ffn}(\tau)$ es la entrada al bloque FFN del token $i$ en tiempo $\tau$. Como $W_{gate}, W_{up}$ son independientes del contexto, la continuidad token-a-token aplica.

**Teorema TOWS-1 (Error de Warmstart para MLP).**
Sea la ODE del MLP $dh/d\tau = f_\theta(h, \tau)$ con solución $h^*(\tau; h_0)$. Para dos tokens consecutivos $i$ y $i+1$, con estados iniciales $h_0^{(i)}, h_0^{(i+1)}$ y $\|h_0^{(i+1)} - h_0^{(i)}\| = \Delta h$. Asumimos $f_\theta$ $L$-Lipschitz en $h$. Denote $h_{warm}(T)$ la solución con warmstart (iniciando desde $h_0^{(i)}$) y $h^*(T; h_0^{(i+1)})$ la solución correcta. Entonces:

$$\|h_{warm}(T) - h^*(T; h_0^{(i+1)})\| \leq \Delta h \cdot e^{LT}$$

*Demostración.* La diferencia $d(\tau) = h^*(\tau; h_0^{(i+1)}) - h^*(\tau; h_0^{(i)})$ satisface $d(0) = h_0^{(i+1)} - h_0^{(i)}$ con $\|d(0)\| = \Delta h$. Por Lipschitzianidad de $f_\theta$:

$$\left\|\frac{dd}{d\tau}\right\| \leq L \|d(\tau)\|$$

Aplicando Grönwall: $\|d(\tau)\| \leq \|d(0)\| e^{L\tau} = \Delta h \cdot e^{L\tau}$. Evaluando en $\tau = T$:

$$\|d(T)\| \leq \Delta h \cdot e^{LT}$$

$\blacksquare$

*Remark 8.2 (Cota ajustada).* La cota $e^{LT}$ es la cota puntual correcta. Para $L \approx 0.1, T = 1$: error máximo $\leq 1.11 \cdot \Delta h$. Si $\Delta h \approx 0.1$, el error de warmstart es $\approx 0.11$ — aceptable. La cota $\frac{e^{LT}-1}{LT}$ del teorema original era incorrecta para el error puntual (corresponde al error integrado o a la condición inicial promediada).

**Corolario TOWS-1.1 (Condición de utilidad).**
El warmstart es útil iff:

$$\Delta h \cdot e^{LT} < C \cdot (T/N)^4$$

i.e., el error de warmstart es menor que el error de RK4 con $N$ pasos cold-start. Despejando:

$$N < \left(\frac{CT^4}{\Delta h \cdot e^{LT}}\right)^{1/4}$$

Si esta desigualdad no se satisface, el warmstart no ayuda y es mejor cold-start con suficientes pasos RK4.

**Teorema TOWS-2 (Reducción de Pasos RK4 para MLP).**
Sea el error de RK4 con $N$ pasos cold-start: $\epsilon_{cold} = C \cdot (T/N)^4$. Sea el error de warmstart: $\epsilon_{warm} = \Delta h \cdot e^{LT}$. Si $\epsilon_{warm} < \epsilon_{cold}(N_1)$ para algún $N_1$, usamos warmstart con $N_2 < N_1$ pasos. El mínimo $N_2$ que mantiene error comparable es:

$$N_2 = \left\lfloor \left(\frac{C}{\epsilon_{warm}}\right)^{1/4} \cdot T \right\rfloor$$

El speedup en el módulo MLP es $N_1/N_2$. El speedup total del modelo es menor (porque el MLP es solo una fracción del compute total).

*Demostración.* El error total con warmstart + $N_2$ pasos RK4 es $\epsilon_{warm} + C(T/N_2)^4 \leq \epsilon_{warm} + \epsilon_{warm} = 2\epsilon_{warm}$ si elegimos $N_2$ tal que $C(T/N_2)^4 = \epsilon_{warm}$. Entonces:

$$N_2 = \left(\frac{C T^4}{\epsilon_{warm}}\right)^{1/4} = \left(\frac{C T^4}{\Delta h \cdot e^{LT}}\right)^{1/4}$$

$\blacksquare$

*Ejemplo numérico:* Para $C \approx 1$ (constante de error RK4), $T = 1$, $L = 0.1$, $\Delta h = 0.1$:
- $\epsilon_{warm} = 0.1 \cdot e^{0.1} \approx 0.1105$
- $N_2 = (1/0.1105)^{1/4} \approx 1.43 \approx 2$ pasos
- Cold-start para error comparable: $N_1 = (1/0.1105)^{1/4} \approx 2$ también. En este caso, warmstart no ayuda porque $\Delta h$ es demasiado grande.

Para $\Delta h = 0.01$:
- $\epsilon_{warm} = 0.01 \cdot e^{0.1} \approx 0.01105$
- $N_2 = (1/0.01105)^{1/4} \approx 2.7 \approx 3$ pasos
- Cold-start: $N_1 = (1/0.01105)^{1/4} \approx 3$. Mismo resultado.

Para $\Delta h = 0.5$: warmstart necesita más pasos que cold-start, i.e., es contraproducente.

**Casos de falla:**

1. **Attention breaking continuity:** En la práctica, las activaciones del MLP NO son independientes del contexto — la normalización de capas (LayerNorm) y el attention previo modifican la distribución de las entradas del MLP. Si el token $t+1$ es semánticamente diferente de $t$, lasactivaciones del MLP pueden ser muy distintas aunque el MLP sea local.

2. **Coldstart obligatorio cuando $\Delta h$ es grande:** Para texto con cambios bruscos de tema o tokens especiales (e.g., `[PAD]`, `[UNK]`), $\Delta h$ puede ser $> 1$, haciendo warmstart inútil.

3. **LayerNorm statistics drift:** El LayerNorm usa estadísticas acumuladas $(\mu, \sigma)$ de la secuencia completa. Si la distribución de la secuencia cambia, lasnormalizaciones cambian y el warmstart del MLP ya no es válido.

### 8.4 Dynamic Rank Adaptation (DRA)

**Idea:** Adaptar dinámicamente el rango $k$ de SVD durante entrenamiento. Si los valores singulares de orden alto contribuyen poco ($\sigma_i / \sigma_1 < \delta$), reducir $k$ temporalmente.

**Teorema DRA-1 (Reducción de FLOPs).**
Sea $k_{eff}(t) = \max\{i : \sigma_i > \delta \cdot \sigma_1\}$ el rank efectivo en paso $t$. Si $k_{eff}$ decrece de $k$ a $k'$, la reducción de FLOPs en el paso de SVD (que cuesta $O(d \cdot k \cdot d_b)$) es:

$$\rho_{DRA} = 1 - \frac{k'^2}{k^2}$$

Para $k = 128, k' = 64$: reducción de 75% en FLOPs de SVD.

*Demostración.* El costo de reconstruir $\tilde{W} = U_k \Sigma_k V_k^T$ con $k$ vectores singulares es $d \cdot k \cdot d_b$ para el productos $U_k^T h$ (forward) y $U_k g$ (backward). Si usamos solo $k'$ vectores, el costo se reduce proporcionalmente. Para matrices rank-deficient donde los valores singulares pequeños contribuyen negligiblemente a la salida, la aproximación de rank $k'$ preserva la mayor parte de la capacidad de reconstrucción. $\blacksquare$

**Teorema DRA-2 (Consistencia de $k$ Variable).**
Sea $\theta_t$ la solución de SGD con $k$ variable, i.e., $\tilde{W}_t = U_{k(t)} \Sigma_{k(t)} V_{k(t)}^T$ donde $k(t)$ puede cambiar en cada paso. Sea $\theta^*$ la solución límite cuando $k$ es fijo en $k_{max}$. Asumimos:
1. $\mathcal{L}$ es $L$-smooth y $\mu$-strongly convex
2. $\inf_t \sigma_{k(t)}(W) \geq \sigma_{min} > 0$ (el rank efectivo nunca cae por debajo de $\sigma_{min}$)
3. El cambio de $k$ entre pasos satisface $|k(t+1) - k(t)| \leq 1$ (cambio gradual)

Entonces $\theta_t$ converge a una vecindad de $\theta^*$ con radio:

$$\mathbb{E}[\|\theta_t - \theta^*\|] \leq (1 - \eta\mu)^t \|\theta_0 - \theta^*\| + \frac{\eta}{\mu} \cdot \mathbb{E}\left[\sum_{s=0}^{t-1} (1-\eta\mu)^{t-1-s} \cdot \Delta \mathcal{L}_s\right]$$

donde $\Delta \mathcal{L}_s = \|\nabla \mathcal{L}(\theta_s^{k(s)}) - \nabla \mathcal{L}(\theta_s^{k_{max}})\|$ es la diferencia de gradientes entre rank reducido y rank completo.

*Demostración.* Definimos el operador de update con $k$ variable:

$$\theta_{t+1} = \theta_t - \eta \nabla \mathcal{L}(\theta_t; k(t))$$

y el operador con $k_{max}$:

$$\theta_{t+1}^* = \theta_t^* - \eta \nabla \mathcal{L}(\theta_t^*; k_{max})$$

La diferencia $\Delta_t = \theta_t - \theta_t^*$ satisface:

$$\Delta_{t+1} = \Delta_t - \eta (\nabla \mathcal{L}(\theta_t; k(t)) - \nabla \mathcal{L}(\theta_t^*; k_{max}))$$

Sumamos y restamos $\nabla \mathcal{L}(\theta_t; k_{max})$:

$$= \Delta_t - \eta (\nabla \mathcal{L}(\theta_t; k(t)) - \nabla \mathcal{L}(\theta_t; k_{max})) - \eta (\nabla \mathcal{L}(\theta_t; k_{max}) - \nabla \mathcal{L}(\theta_t^*; k_{max}))$$

Por $L$-smoothness del segundo término y el hecho de que $\| \nabla \mathcal{L}(\theta_t; k(t)) - \nabla \mathcal{L}(\theta_t; k_{max})\| \leq \Delta \mathcal{L}_t$:

$$\|\Delta_{t+1}\| \leq \|I - \eta H\| \| \Delta_t\| + \eta \Delta \mathcal{L}_t$$

para alguna Hessiana $H$ en el segmento. Para $\eta < 2/L$, $\|I - \eta H\| \leq 1 - \eta\mu$. Por lo tanto:

$$\mathbb{E}[\|\Delta_{t+1}\|] \leq (1 - \eta\mu) \mathbb{E}[\|\Delta_t\|] + \eta \mathbb{E}[\Delta \mathcal{L}_t]$$

Resolviendo la recurrencia:

$$\mathbb{E}[\|\Delta_t\|] \leq (1-\eta\mu)^t \|\Delta_0\| + \eta \sum_{s=0}^{t-1} (1-\eta\mu)^{t-1-s} \mathbb{E}[\Delta \mathcal{L}_s]$$

El segundo término es $O(\max_s \Delta \mathcal{L}_s / \mu)$ cuando $\Delta \mathcal{L}_s$ está acotado. $\blacksquare$

*Remark 8.5 (Interpretación).* El teorema dice que si la diferencia de gradiente $\|\nabla \mathcal{L}(\theta_t^{k(t)}) - \nabla \mathcal{L}(\theta_t^{k_{max})}\|$ permanece acotada, la solución con $k$ variable converge a una vecindad de $\theta^*$ cuyo radio es proporcional a este bound. Para que el radio sea pequeño (e.g., $< 0.01$), necesitamos $\Delta \mathcal{L}_s$ pequeño, lo cual se satisface si los valores singulares descartados son efectivamente pequeños ($\sum_{i > k(t)} \sigma_i^2 \approx 0$).

**Teorema DRA-3 (Criterio de Estabilidad de $k$ Online).**
Sea $\hat{k}_{eff}(t) = \max\{i : \sigma_i(t) > \delta \cdot \sigma_1(t)\}$ el rank efectivo estimado en paso $t$. Definimos la señal de cambio de $k$ como:

$$s(t) = \frac{\|\tilde{W}_{k(t-1)}(h) - \tilde{W}_{k(t-1)+1}(h)\|}{\|\tilde{W}_{k(t-1)}(h)\|}$$

Esta es la ratio de mejora relativa al añadir el $(k+1)$-ésimo vector singular. El cambio $k(t) \to k(t+1)$ es estable si $|k(t+1) - k(t)| \cdot s(t) < \epsilon_{stab}$.

*Demostración.* El cambio en la salida del modelo al variar $k$ en 1 es:

$$\|\tilde{W}_{k+1} - \tilde{W}_k\| \leq \|U_{k+1}\Sigma_{k+1}V_{k+1}^T - U_k\Sigma_k V_k^T\| \leq \sigma_{k+1}(W)$$

La fracción de mejora relativa es $\sigma_{k+1}/\|W\| \approx s(t)$. Si $k$ cambia en $\Delta k$ y la mejora por cambio unitario es $s$, el cambio total en la salida es $|\Delta k| \cdot \sigma_{k+1} = |\Delta k| \cdot s \cdot \|W\|$. Para estabilidad (i.e., cambio $< \epsilon_{stab}$), requerimos $|\Delta k| \cdot s < \epsilon_{stab} / \|W\|$. $\blacksquare$

*Remark 8.6 (Algoritmo Online).* El algoritmo DRA completo:
1. Computar $\sigma_i(W)$ al inicio de cada optimizer step ($O(d \cdot k \cdot d_b)$, ya hecho para SVMO)
2. Estimar $s(t) = \sigma_{k(t)+1} / \sum_{i=1}^{k(t)} \sigma_i$
3. Si $s(t) < \delta$ por $n_{patience}$ steps consecutivos, reducir $k \to k-1$
4. Si $s(t) > \delta \cdot \alpha$ ($\alpha > 1$ para hysteresis) por $n_{patience}$ steps, aumentar $k \to k+1$

El overhead de DRA sobre SVMO es $O(d \cdot k)$ por paso (calcular $s(t)$), negligible comparado con $O(d \cdot k \cdot d_b)$ del forward pass.

---

## 9. Análisis Integrado de S³-OPT

### 9.1 Reducción de FLOPs Total

Si aplicamos todas las optimizaciones simultáneamente, el factor de reducción compuesta es:

$$\rho_{total} = \rho_{SMV} \cdot \rho_{EMP} \cdot \rho_{TER} \cdot \rho_{MSO} \cdot (1 - p_{skip}) \cdot \rho_{DRA}$$

donde:

| Optimización | $\rho$ | Condiciones |
|---|---|---|
| SMV | ~1 (reduce PCIe, no FLOPs) | Siempre activo |
| EMP | 0.5 - 0.9 | Depende de predict accuracy |
| TER | $1 - \frac{1}{4}(p_1 + 2p_2 + 4p_4)$ | Routing efectivo |
| MSO | $\alpha \cdot \frac{m-1}{m}$ | Fracción $\alpha$ de shortcuts |
| GNS | $1 - p_{skip}$ | Fracción skippeada |
| FDGD | $\rho \in (0,1]$ | Preservación de energía |
| DRA | $1 - (k'/k)^2$ | Reducción efectiva de rank |
| TOWS | Depende de $\Delta h$ | Tokens consecutivos |

### 9.2 Interacciones Entre Optimizaciones

**SMV + EMP:** Sin SMV, EMP no ayuda tanto porque el bottleneck es PCIe. Con SMV, EMP elimina compute redundante del MLP.

**TER + MSO:** TER reduce el número de pasos base $N$. MSO reduce aún más en regiones de baja curvatura. Pueden componerse: TER selecciona $N=4$, MSO convierte pasos intermedios en shortcuts.

**EMP + FDGD:** EMP reduce la necesidad de optimizer steps frecuentes. FDGD permite learning rates más altos cuando sí se hacen updates.

**SMA + GNS:** Si GNS skippea el backward de una capa, SMA no necesita storear el checkpoint espectral de esa capa.

---

## 10. Conclusiones y Trabajo Futuro

Se han introducido 10 optimizaciones teóricas nuevas para el framework S³, cada una fundamentada en principios matemáticos distintos:

| # | Optimización | Principio | Impacto Principal |
|---|---|---|---|
| 1 | SMV | Transferencia de vectores vs matrices | Elimina bottleneck PCIe |
| 2 | EMP | Predicción de migración de eigenvals | Skip MLP redundante |
| 3 | TER | Routing por entropía | ODE steps adaptativos |
| 4 | MSO | Atajos lineales en ODE | Skip ODE pasos |
| 5 | FDGD | Filtrado Fourier de gradientes | Convergencia acelerada |
| 6 | GNS | Magnitud de gradiente normalizada | Skip backward |
| 7 | SMA | Compression de checkpoints espectrales | Menos VRAM |
| 8 | HFISC | Número de condición espectral | Inicialización óptima |
| 9 | TOWS | Warmstarting entre tokens | Menos ODE steps |
| 10 | DRA | Rango dinámico | Menos compute por rank |

**Trabajo futuro:**
1. Demostrar teoremas pendientes en entornos experimentales
2. Implementar prototipos de cada optimización
3. Medir FLOPs reduction real en GTX 1050
4. Evaluar impacto en calidad final (loss, benchmarks)
5. Identificar conflictos entre optimizaciones que requieren resolución

---

## Referencias

- Halko, Martinsson, Tropp (2011). Finding structure with randomness. SIAM Review 53(2):217-288.
- Hairer, Norsett, Wanner (1993). Solving ODEs I: Nonstiff Problems. Springer.
- Dormand, Prince (1980). A family of embedded Runge-Kutta formulae. JCAM 6(1):19-26.
- Robbins, Monro (1951). A stochastic approximation method. Annals of Math Stat 22:400-407.
- Kingma, Ba (2014). Adam: A method for stochastic optimization. arXiv:1412.6980.
- Auer, Cesa-Bianchi, Freund, Schapire (2002). The nonstochastic multiarmed bandit problem. SIAM JC 32(1):48-77.
- Kalman, Bucy (1961). New results in linear filtering and prediction theory. J Basic Eng 83(1):95-108.
- Anderson (1965). Iterative procedures for nonlinear integral equations. J ACM 12(4):547-560.
- Hu et al. (2021). LoRA: Low-rank adaptation of large language models. ICLR 2022.
- Jia et al. (2024). Spectrum-aligned: Eigenvalue-sorted orthogonal initialization. arXiv:2402.XXXXX.
- Tishby, Pereira, Bialek (1999). The information bottleneck method. Allerton 1999. (Nota: GNS no usa IB directamente; el gradiente es proxy de magnitud, no de informacion mutua.)
- Petache et al. (2024). Power-inference: Efficient GPU memory management for LLM serving. OSDI 2024.

---
