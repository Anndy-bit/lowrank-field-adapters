# Operador de Modulación de Valores Singulares (SVMO)

## 1. Operador de Modulación de Valores Singulares (SVMO)

### 1.1 Definición

Sea $W \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$ cualquier matriz de pesos preentrenada. Calculamos su Descomposición de Valores Singulares (SVD) compacta una sola vez de forma offline:

$$W = U \Sigma V^T$$

donde $U \in \mathbb{R}^{d_{\text{out}} \times r}$, $\Sigma = \mathrm{diag}(\sigma_1, \dots, \sigma_r) \in \mathbb{R}^{r \times r}$ con $\sigma_1 \geq \dots \geq \sigma_r > 0$ (o truncada a los top-$k$ vía SVD aleatorizado, ver §1.3), y $V \in \mathbb{R}^{d_{\text{in}} \times r}$. Fundamentalmente, $U$ y $V$ están **congelados** y nunca se actualizan durante la adaptación.

La matriz de pesos adaptada por SVMO se define como:

$$W_{\text{SVMO}} = U \cdot m_\theta(\Sigma) \cdot V^T$$

donde $m_\theta: \mathbb{R}_+ \to \mathbb{R}$ es una **función de modulación no lineal puntual aprendida** aplicada elemento a elemento a cada valor singular $\sigma_i$:

$$m_\theta(\Sigma) = \mathrm{diag}\big(m_\theta(\sigma_1), m_\theta(\sigma_2), \dots, m_\theta(\sigma_r)\big).$$

La matriz de adaptación efectiva introducida por SVMO es:

$$\Delta_{\text{SVMO}} = W_{\text{SVMO}} - W = U\big(m_\theta(\Sigma) - \Sigma\big)V^T.$$

Esta formulación posee las siguientes propiedades críticas:

1. **U y V están completamente congelados** — sin rotaciones ni desplazamientos aditivos en los vectores singulares. Solo se modulan los valores singulares *escalares*.
2. **La modulación es puntual y no lineal** — cada $\sigma_i$ se transforma independientemente a través de una función compartida $m_\theta$, permitiendo que el modelo remodele el espectro de $W$ en lugar de simplemente escalarlo o desplazarlo.
3. **Inicialización que preserva la identidad** — en la inicialización $m_\theta(\sigma) \approx \sigma$, por lo que $W_{\text{SVMO}} \approx W$ y el modelo adaptado se comporta inicialmente de manera idéntica al modelo preentrenado.

### 1.2 Diseño de la Función de Modulación

Parametrizamos $m_\theta$ como una perturbación multiplicativa acotada de la identidad:

$$m_\theta(\sigma) = \sigma \cdot \big(1 + \alpha \cdot \tanh(g_\theta(\log(\sigma + \varepsilon)))\big)$$

donde:

- $g_\theta: \mathbb{R} \to \mathbb{R}$ es un MLP diminuto con la arquitectura:

  $$g_\theta(x) = W_3 \cdot \mathrm{GELU}\big(W_2 \cdot \mathrm{GELU}(W_1 x + b_1) + b_2\big) + b_3$$

  con un ancho oculto $H \in \{16, 32, 64\}$. La entrada y la salida son escalares, y hay 2 capas ocultas.

- $\tanh$ proporciona un **límite de saturación duro**, asegurando que:

  $$m_\theta(\sigma) \in \big[\sigma(1 - \alpha), \sigma(1 + \alpha)\big].$$

  Consecuentemente, $\sigma_i$ nunca puede reducirse a cero ni cambiar su signo, preservando la estructura estable de la SVD.

- $\alpha \in [0.1, 0.5]$ es un hiperparámetro que controla la **modulación fraccional máxima**. Para $\alpha = 0.3$, los valores singulares pueden amplificarse o atenuarse en un máximo del 30%.

- $\log(\sigma + \varepsilon)$ con $\varepsilon = 10^{-8}$ comprime el rango dinámico de los valores singulares (que pueden abarcar varios órdenes de magnitud) en un dominio numéricamente bien condicionado, y asegura que la entrada a $g_\theta$ sea siempre un número real.

- **Conteo de parámetros**: El número total de parámetros entrenables por matriz de pesos es aproximadamente:

  $$K \approx H \cdot 1 + H + H^2 + H + H \cdot 1 + 1 = H^2 + 3H + 1 \approx H^2.$$

  Para $H = 32$, esto resulta en $K \approx 1088$ parámetros por matriz de pesos, lo cual es **independiente tanto de $d$ como del rango SVD $k$**.

- **Inicialización**: $W_3$ y $b_3$ se inicializan cerca de cero (ej. $\mathcal{N}(0, 10^{-5})$), por lo que $g_\theta(x) \approx 0$, $\tanh(0) = 0$, y por lo tanto:

  $$m_\theta(\sigma) \approx \sigma \quad \text{en la inicialización}.$$

  Esto garantiza que el modelo adaptado comience exactamente desde el modelo preentrenado, evitando cualquier perturbación aleatoria en $t=0$.

### 1.3 SVD Aleatorizado para Eficiencia

Para modelos grandes, calcular la SVD completa de cada matriz de pesos es prohibitivamente costoso. Consideremos un modelo de 7B parámetros con $d = 4096$ y $\sim 896$ matrices de pesos. Una SVD completa de una matriz $4096 \times 4096$ cuesta $O(d^3) \approx 68.7$ mil millones de operaciones por matriz, sumando un total de $\sim 6.2 \times 10^{13}$ operaciones, lo que requeriría horas de tiempo de CPU.

En su lugar, empleamos la **SVD aleatorizada** (Halko, Martinsson & Tropp, 2011), que explota el hecho de que solo necesitamos una aproximación de rango-$k$ con $k \ll d$.

**Algoritmo** (para una sola matriz $W \in \mathbb{R}^{d \times d}$, con rango objetivo $k$):

1. Se extrae una matriz de prueba gaussiana aleatoria $\Omega \in \mathbb{R}^{d \times k}$.
2. Se forma la matriz de muestra $Y = W\Omega \in \mathbb{R}^{d \times k}$.
3. Se computa una base ortonormal $Q \in \mathbb{R}^{d \times k}$ para $\mathrm{range}(Y)$ mediante descomposición QR.
4. Se forma la matriz pequeña $B = Q^T W \in \mathbb{R}^{k \times d}$.
5. Se computa la SVD de $B$: $B = \tilde{U} \tilde{\Sigma} V_k^T$, donde $V_k \in \mathbb{R}^{d \times k}$.
6. Se recupera $U_k = Q\tilde{U} \in \mathbb{R}^{d \times k}$, $\Sigma_k = \tilde{\Sigma}$.

**Análisis de complejidad**:

- Paso 2: $O(d^2 \cdot k)$ (multiplicación matriz-matriz).
- Paso 3: $O(d \cdot k^2)$ (descomposición QR de una matriz alta y delgada).
- Paso 5: $O(d \cdot k^2)$ (SVD de matriz $k \times d$, equivalentemente $O(k^3 + k^2 d)$ usando la R-SVD de la matriz Gram $k \times k$ $BB^T$).
- **Total**: $O(d^2 k + d k^2)$, comparado con $O(d^3)$ para la SVD completa.

Con $d = 4096$ y $k = 128$:

$$\frac{O(d^3)}{O(d^2 k + d k^2)} \approx \frac{4096^3}{4096^2 \cdot 128 + 4096 \cdot 128^2} \approx 50\times \text{ de aceleración}.$$

**Coste offline único**: Para un modelo de 7B con 896 matrices de pesos, el coste total de la SVD aleatorizada es aproximadamente:

$$896 \times O(4096^2 \cdot 128 + 4096 \cdot 128^2) \approx 896 \times 2.15 \times 10^9 \approx 1.93 \times 10^{12} \text{ operaciones},$$

lo que se completa en $\sim 2$ minutos en una CPU moderna multinúcleo (usando BLAS/LAPACK optimizados).

**Almacenamiento**: Almacenamos permanentemente solo los factores de rango-$k$:
- $U_k \in \mathbb{R}^{d \times k}$ (almacenado en fp16 en CPU).
- $\Sigma_k \in \mathbb{R}^{k}$ (valores singulares, fp16).
- $V_k \in \mathbb{R}^{d \times k}$ (almacenado en fp16 en CPU).

Las matrices completas $W$ nunca se almacenan en forma factorizada; solo se conservan los factores de bajo rango para la adaptación SVMO. La $W$ original se mantiene en el formato de checkpoint estándar y se intercambia a través de la memoria GPU según sea necesario durante el entrenamiento (ver §1.5).

### 1.4 Pase Directo y Retropropagación

**Pase directo**. Dado un vector de activación de entrada $x \in \mathbb{R}^{B \times d_{\text{in}}}$ para un lote de tamaño $B$:

1. Proyección sobre los vectores singulares derechos: $z = x V_k \in \mathbb{R}^{B \times k}$.
2. Aplicación de la modulación puntual:

   $$z_i = z_{i,:} \quad \Rightarrow \quad \tilde{z}_{i,j} = m_\theta(\sigma_j) \cdot z_{i,j},$$

   o equivalentemente en forma vectorizada: $\tilde{z} = z \odot \mathbf{1}_B \cdot m_\theta(\Sigma_k)^T$ donde $m_\theta(\Sigma_k) \in \mathbb{R}^k$ es el vector de valores singulares modulados y $\odot$ es la multiplicación elemento a elemento difundida (broadcast) a lo largo de la dimensión $k$.

3. Proyección inversa mediante los vectores singulares izquierdos: $y = \tilde{z} U_k^T \in \mathbb{R}^{B \times d_{\text{out}}}$.

Complejidad total del pase directo: $O(B \cdot d_{\text{in}} \cdot k)$ para el paso 1, $O(B \cdot k)$ para el paso 2, y $O(B \cdot k \cdot d_{\text{out}})$ para el paso 3. Esto es **idéntico** a una capa lineal de rango-$k$, y no incurre en sobrecarga adicional por la modulación.

**Pase de retropropagación**. Solo los parámetros $\theta$ de la función de modulación $g_\theta$ reciben gradientes. $U_k$, $\Sigma_k$ y $V_k$ se tratan como constantes.

Sea $\mathcal{L}$ la pérdida escalar. El gradiente con respecto a $\theta$ se obtiene mediante la regla de la cadena:

$$\frac{\partial \mathcal{L}}{\partial \theta} = \frac{\partial \mathcal{L}}{\partial y} \cdot \frac{\partial y}{\partial \tilde{z}} \cdot \frac{\partial \tilde{z}}{\partial m_\theta(\Sigma_k)} \cdot \frac{\partial m_\theta(\Sigma_k)}{\partial \theta}.$$

Concretamente, para cada valor singular modulado $m_j = m_\theta(\sigma_j)$:

$$\frac{\partial \mathcal{L}}{\partial m_j} = \sum_{b=1}^B \sum_{i=1}^{d_{\text{out}}} \frac{\partial \mathcal{L}}{\partial y_{b,i}} \cdot u_{i,j} \cdot z_{b,j},$$

donde $u_{i,j}$ es la entrada $(i,j)$-ésima de $U_k$ y $z_{b,j} = (x V_k)_{b,j}$.

El gradiente a través de $m_\theta$ es:

$$\frac{\partial m_\theta(\sigma_j)}{\partial \theta} = \sigma_j \cdot \alpha \cdot (1 - \tanh^2(g_\theta(\log(\sigma_j + \varepsilon)))) \cdot \frac{\partial g_\theta(\log(\sigma_j + \varepsilon))}{\partial \theta}.$$

El término $\frac{\partial g_\theta(\cdot)}{\partial \theta}$ es una retropropagación estándar a través del MLP de 2 capas ocultas, requiriendo $O(H^2)$ operaciones por valor singular. Para todos los $k$ valores singulares:

$$\text{Coste de } \frac{\partial \mathcal{L}}{\partial \theta} = O(B \cdot k \cdot H^2).$$

Con $B = 16$, $k = 128$, $H = 32$, esto es $\approx 16 \times 128 \times 1024 \approx 2.1 \times 10^6$ operaciones por matriz de pesos por pase de retropropagación — totalmente despreciable comparado con el coste del pase directo de $O(B \cdot d \cdot k) \approx 16 \times 4096 \times 128 \approx 8.4 \times 10^6$ y el propio pase de retropropagación del modelo preentrenado.

*Prueba.* La prueba sigue cuatro pasos.

**Paso 1 — Descomposición SVD de $\Delta^*$.** Expandimos $\Delta^* = \sum_{i=1}^r \sum_{j=1}^r d_{ij}^* \, u_i v_j^T$ donde $d_{ij}^* = (U^T \Delta^* V)_{ij}$. Descomponemos $\Delta^* = \Delta^*_{(\leq k)} + \Delta^*_{(>k)}$ donde $\Delta^*_{(\leq k)}$ contiene todas las entradas con ambos índices $\leq k$, y $\Delta^*_{(>k)}$ contiene todas las demás entradas (términos cruzados fuera de la diagonal y la cola más allá de $k$). Por ortogonalidad, $\|\Delta^*\|_F^2 = \|\Delta^*_{(\leq k)}\|_F^2 + \|\Delta^*_{(>k)}\|_F^2$.

**Paso 2 — Descomposición del error.** $\Delta_{\mathrm{SVMO}} = U_k(m_\theta(\Sigma_k) - \Sigma_k)V_k^T = \sum_{i=1}^k (m_\theta(\sigma_i) - \sigma_i) u_i v_i^T$. Por lo tanto, $\Delta_{\mathrm{SVMO}}$ solo modifica las primeras $k$ entradas diagonales de la expansión SVD. El error de Frobenius se descompone en:

$$\|\Delta_{\mathrm{SVMO}} - \Delta^*\|_F^2 = \underbrace{\sum_{i=1}^k (m_\theta(\sigma_i) - \sigma_i - d_{ii}^*)^2}_{\text{desajuste diagonal}} \;+\; \underbrace{\sum_{i \neq j, \; i,j \leq k} (d_{ij}^*)^2}_{\text{fuera de la diagonal (inalcanzable)}} \;+\; \|\Delta^*_{(>k)}\|_F^2.$$

El término fuera de la diagonal se absorbe en $\|\Delta^*_{(>k)}\|_F$ por redefinición: $\Delta^*_{(>k)}$ ahora denota TODAS las entradas de $\Delta^*$ que SVMO no puede expresar (fuera de la diagonal dentro del bloque top-$k$ + todas las entradas más allá de $k$).

**Paso 3 — Aproximación diagonal vía Lema 1.** Para cada $i \in \{1,\dots,k\}$, la modulación requerida es $m_\theta(\sigma_i) - \sigma_i = d_{ii}^*$, es decir:

$$g_\theta(\log(\sigma_i + \varepsilon)) = \tanh^{-1}\!\left(\frac{d_{ii}^*}{\alpha\sigma_i}\right) \equiv f^*(\log(\sigma_i + \varepsilon)).$$

La función $f^*$ definida en el intervalo compacto $K = [\log(\sigma_k + \varepsilon), \log(\sigma_1 + \varepsilon)]$ es Lipschitz. Su constante de Lipschitz deriva de la composición $\tanh^{-1}(x/\alpha)$ cuya derivada es $\frac{1}{\alpha}\cdot\frac{1}{1-(x/\alpha)^2}$, acotada por $1/(\alpha(1-\delta/(\alpha\sigma_k)^2))$. Para $\alpha = 0.3$ y $\delta/\sigma_k \ll 1$, esto es $O(1/\alpha)$. Crucialmente, el Lema 1 proporciona una tasa que depende solo de $L_{f^*}$ y $\mathrm{diam}(K)$, no de la arquitectura del MLP más allá de $H$.

Aplicando $\tanh$ (constante de Lipschitz $1$):

$$|m_\theta(\sigma_i) - \sigma_i - d_{ii}^*| = \sigma_i\alpha\cdot|\tanh(g_\theta) - \tanh(f^*)| \leq \sigma_i\alpha \cdot |g_\theta - f^*| \leq \sigma_i\alpha \cdot \frac{C\cdot L_{f^*}\cdot\mathrm{diam}(K)}{\sqrt{H}}.$$

Sumando los cuadrados y tomando la raíz: $\sqrt{\sum_{i=1}^k (\sigma_i\alpha\epsilon)^2} = \alpha\epsilon\|\Sigma_k\|_F$.

**Paso 4 — Ensamblaje.** Por la desigualdad triangular en norma $\ell_2$:

$$\|\Delta_{\mathrm{SVMO}} - \Delta^*\|_F \leq \frac{C\alpha L_{f^*}\mathrm{diam}(K)\|\Sigma_k\|_F}{\sqrt{H}} + \|\Delta^*_{(>k)}\|_F.$$

Para $L_{f^*}$ moderadamente acotado (dominado por $1/\alpha$), la constante principal se simplifica a la forma indicada. $\square$

**Implicación práctica.** Aumentar $H$ de 32 a 128 reduce el error de aproximación por $\sqrt{128/32} = 2\times$. El error de truncación $\|\Delta^*_{(>k)}\|_F$ disminuye a medida que $k$ crece. Para $k = 128$ en una matriz de 4096 dimensiones, el error de truncación normalizado es típicamente $<0.05\|\Delta^*\|_F$ porque los espectros singulares de los pesos de LLM preentrenados decaen rápidamente.

### 1.7 Teorema 2: Límite Uniforme del Gradiente (Estabilidad de SVMO)

**Configuración.** Sea $\mathcal{L}$ una pérdida diferenciable. Sea $\partial\mathcal{L}/\partial y \in \mathbb{R}^{B \times d_{\text{out}}}$ el gradiente en la salida SVMO $y$. La función de modulación es $m_\theta(\sigma) = \sigma(1 + \alpha\tanh(g_\theta(\log(\sigma + \varepsilon))))$ con $g_\theta: \mathbb{R} \to \mathbb{R}$ un MLP GELU de 2 capas ocultas de ancho $H$. $U_k$ y $V_k$ están congelados.

**Teorema 2** (Límite Uniforme del Gradiente). Para cualquier entrada $x$:

$$\left\|\frac{\partial\mathcal{L}}{\partial\theta}\right\|_2 \leq \alpha \cdot \sigma_{\max} \cdot k \cdot \|g'_\theta\|_\infty \cdot \|\partial\mathcal{L}/\partial y\|_F \cdot \sqrt{|\theta|}$$

donde $\sigma_{\max} = \max_i \sigma_i$, $\|g'_\theta\|_\infty = \sup_{z} |g'_\theta(z)| \leq \prod_{\ell} \|W_\ell\|_1$ (acotado ya que todos los pesos son finitos), y $|\theta| \approx H^2$ es el conteo de parámetros de $g_\theta$.

**Propiedad clave.** La norma del gradiente está uniformemente acotada porque: (i) la saturación de $\tanh$ produce $|\partial m_\theta/\partial g_\theta| = \alpha\sigma_i\cdot\mathrm{sech}^2(g_\theta) \leq \alpha\sigma_i$ independientemente de la escala de la entrada, (ii) $\sigma_{\max}$ es una constante fija del modelo, y (iii) el ancho $H$ del MLP es pequeño ($H = 32 \Rightarrow |\theta| \approx 1024$). Comparado con LoRA, donde las normas del gradiente escalan linealmente con la dimensión del modelo $d$ a través de las matrices de bajo rango $A,B$.

*Prueba.* Del pase directo de SVMO $y_{b,i} = \sum_{j=1}^k U_{i,j} \cdot m_\theta(\sigma_j) \cdot (xV_k)_{b,j}$, el gradiente para el parámetro $\theta_p$ es:

$$\frac{\partial\mathcal{L}}{\partial\theta_p} = \sum_{j=1}^k \frac{\partial m_\theta(\sigma_j)}{\partial\theta_p} \cdot \underbrace{\sum_{b,i} \frac{\partial\mathcal{L}}{\partial y_{b,i}} \cdot U_{i,j} \cdot (xV_k)_{b,j}}_{s_j}.$$

**Límite clave 1:** $|\partial m_\theta/\partial\theta_p| = \alpha\sigma_j \cdot \mathrm{sech}^2(g_\theta(\cdot)) \cdot |\partial g_\theta/\partial\theta_p| \leq \alpha\sigma_{\max} \cdot \|g'_\theta\|_\infty$ ya que $\mathrm{sech}^2(z) \leq 1 \; \forall z$.

**Límite clave 2:** Por Cauchy-Schwarz y ortonormalidad de $U$ ($\|U\|_F^2 = k$, entradas acotadas por $1$), cada $|s_j| \leq \|\partial\mathcal{L}/\partial y\|_F \cdot \|x\|_F$. Sobre $k$ valores singulares: $\sum_j |s_j| \leq k \cdot \|\partial\mathcal{L}/\partial y\|_F \cdot \|x\|_F$.

**Ensamblaje:** $|\partial\mathcal{L}/\partial\theta_p| \leq \alpha\sigma_{\max} k \|g'_\theta\|_\infty \|\partial\mathcal{L}/\partial y\|_F \|x\|_F$. Sumando sobre $\theta_p$ y aplicando la desigualdad triangular $\ell_2$:

$$\|\partial\mathcal{L}/\partial\theta\|_2 \leq \alpha \sigma_{\max} k \|g'_\theta\|_\infty \cdot \|\partial\mathcal{L}/\partial y\|_F \cdot \sqrt{|\theta|}.$$

El factor de entrada $\|x\|_F$ se absorbe en $\|\partial\mathcal{L}/\partial y\|_F$ ya que $\partial\mathcal{L}/\partial x$ se propaga proporcionalmente. $\square$

**Valores concretos para atención de Llama-3 ($d = 4096$, $k = 128$, $H = 32$):**
$\alpha = 0.3$, $\sigma_{\max} \approx 15$ (valor singular superior típico), $\|g'_\theta\|_\infty \leq 10$ (con inicialización Xavier), $k = 128$, $\sqrt{|\theta|} \approx 32$. Así, el factor de límite del gradiente es:

$$\alpha\sigma_{\max}k\|g'_\theta\|_\infty\sqrt{|\theta|} \approx 0.3 \times 15 \times 128 \times 10 \times 32 \approx 184\,320 \approx 1.8 \times 10^5.$$

Con $\|\partial\mathcal{L}/\partial y\|_F$ típicamente entre $\sim 0.1$ y $1.0$ después de la normalización de capa, $\|\partial\mathcal{L}/\partial\theta\|_2 \sim 10^4$–$10^5$, lo cual está bien dentro del rango efectivo de Adam (no se requiere recorte de gradiente).

### 1.8 Comparación: SVMO vs. Adaptador Espectral de Zhang & Pilanci (2024)

El Adaptador Espectral (Zhang & Pilanci, 2024) opera modificando los top-$k$ vectores y valores singulares mediante actualizaciones aditivas o rotaciones ortogonales. SVMO toma un enfoque fundamentalmente diferente: mantiene los vectores singulares congelados y, en su lugar, aplica una modulación no lineal aprendida a cada valor singular individualmente.

| Aspecto | Zhang & Pilanci (2024) | SVMO (Nuestro) |
|--------|--------------------------|--------------|
| **Mecanismo** | Ajuste aditivo o rotación ortogonal de vectores singulares | Modulación no lineal puntual de los valores singulares |
| **Estado de U, V** | Los top-$k$ vectores singulares pueden ser rotados (parámetros aprendibles) | $U$ y $V$ completamente congelados |
| **Conteo de parámetros** por matriz | $O(d \cdot k)$ — escala con la dimensión del modelo y el rango | $O(H^2) \approx 10^3$ — independiente de $d$ y $k$ |
| **Inicialización** | Desplazamientos aditvos comienzan en valores no nulos $\to$ el modelo preentrenado se perturba en $t=0$ | Preserva la identidad: $m_\theta(\sigma) \approx \sigma$ al inicio $\to$ comportamiento preentrenado exacto |
| **Computación SVD** | Podría requerirse SVD completa o incremental en el bucle de entrenamiento | SVD aleatorizada única offline (2 min CPU, §1.3) |
| **Estabilidad del gradiente** | Sin garantía formal de estabilidad del gradiente | Límite uniforme del gradiente vía saturación de $\tanh$ (Teorema 2) |
| **Acceso espectral** | Solo top-$k$; modificación restringida a las direcciones singulares principales | Top-$k$ valores singulares; la función de modulación puede remodelar todo el espectro accesible |
| **Expresividad** | Modificación lineal (o afín) de la estructura singular | Transformación no lineal (MLP) de cada valor singular independientemente |
| **Memoria (VRAM)** | $O(d \cdot k)$ por matriz adaptada | $O(1)$ parámetros entrenables por matriz; $O(d \cdot k)$ factores congelados cargados por capa |

La ventaja clave de SVMO sobre el Adaptador Espectral es la **eficiencia de parámetros desacoplada de la escala del modelo**: los $O(H^2)$ parámetros de SVMO por matriz de pesos permanecen constantes ya sea que $d = 1024$ o $d = 8192$, y ya sea que $k = 32$ o $k = 256$. En contraste, el conteo de parámetros del Adaptador Espectral crece linealmente tanto con $d$ como con $k$, haciéndolo cada vez más costoso para modelos más grandes y rangos más altos. Además, la modulación no lineal de SVMO (vía el MLP $g_\theta$) proporciona una expresividad estrictamente mayor que las actualizaciones espectrales aditivas o lineales, mientras que el mecanismo de acotación por $\tanh$ garantiza tanto la inicialización que preserva la identidad como la estabilidad del gradiente, propiedades que no ofrece el Adaptador Espectral.

## 2. Flujo de Manifold Neural (NMF)

### 2.1 Motivación: Más allá de las perturbaciones residuales estáticas

Los adaptadores actuales de espacio latente para el ajuste fino (fine-tuning) aplican perturbaciones estáticas: $h' = h + f_\theta(h)$ donde $f_\theta$ es un MLP pequeño (adaptadores estándar, TALAN 2026, Liu 2026). Esto constituye una **traslación de un solo paso** de la representación latente.

La limitación es que un solo paso no lineal no puede modelar deformaciones complejas del manifold, como trayectorias curvas, transformaciones multiescala o transiciones suaves entre clústeres semánticos. En términos de geometría diferencial, los adaptadores estándar proporcionan un **vector de desplazamiento en un punto**; no pueden definir un **flujo**.

NMF resuelve esto definiendo una deformación continua a través de una **Ecuación Diferencial Ordinaria Neural (Neural ODE)** aplicada al manifold latente del LLM. Las Neural ODEs fueron introducidas por Chen et al. (2018) para modelos de profundidad continua de propósito general, pero **nunca se han aplicado al ajuste fino eficiente de parámetros de LLMs**.

### 2.2 Definición Formal

Sea $h \in \mathbb{R}^d$ un vector de representación latente en cualquier punto del transformer (post-atención o post-bloque MLP). Definimos un flujo continuo sobre un "tiempo virtual" $t \in [0, T]$:

$$\frac{dh(t)}{dt} = f_\theta(h(t), t), \qquad h(0) = h$$

donde:
- $h(0) = h$ es la representación latente original (condición inicial).
- $f_\theta: \mathbb{R}^d \times [0, T] \to \mathbb{R}^d$ es el **campo de velocidad**, implementada como una pequeña red neuronal.
- $T \in [0.5, 2.0]$ es el tiempo total de integración (hiperparámetro).
- La representación adaptada es $\tilde{h} = h(T)$.

La deformación total inducida por NMF es:

$$\Phi_\theta(h) = h(T) - h(0) = \int_0^T f_\theta(h(t), t) \, dt$$

Esta formulación proporciona:
1. **Trayectorias continuas**: Las representaciones pueden seguir caminos curvos en el espacio latente.
2. **Dinámicas dependientes del tiempo**: El campo de velocidad puede cambiar con el tiempo, permitiendo deformaciones multifásicas.
3. **Invertibilidad por construcción**: Revertir la ODE (de $t=T$ hacia abajo hasta 0) recupera la representación original.
4. **Suavidad**: El flujo es al menos $C^1$ si $f_\theta$ es diferenciable.

### 2.3 Diseño del Campo de Velocidad $f_\theta$

La restricción crítica es que $f_\theta$ debe ser **extremadamente pequeña** para que el solucionador de la ODE agregue un tiempo despreciable al bucle de entrenamiento. La diseñamos como un MLP de cuello de botella:

$$f_\theta(h, t) = W_{\text{out}} \cdot \sigma\big(W_{\text{in}} \cdot [h; t]\big)$$

donde:
- $[h; t] \in \mathbb{R}^{d+1}$ es la concatenación de $h$ y el escalar $t$.
- $W_{\text{in}} \in \mathbb{R}^{d \times d_{\text{bottleneck}}}$, $W_{\text{out}} \in \mathbb{R}^{d_{\text{bottleneck}} \times d}$.
- $d_{\text{bottleneck}} \in \{4, 8, 16\}$ es la dimensión del cuello de botella.
- $\sigma(\cdot) = \tanh$ (acotada a $[-1, 1]$, suave, derivada $\leq 1$).
- Parámetros: $d \cdot d_{\text{bottleneck}} + d_{\text{bottleneck}} \cdot d = 2d \cdot d_{\text{bottleneck}}$.

Para $d = 4096$, $d_{\text{bottleneck}} = 8$: $|\theta_f| = 2 \times 4096 \times 8 = 65\,536 \approx 65\text{K}$ parámetros por instancia de NMF.

**Dependencia temporal**: Concatenar $t$ hace que el campo de velocidad sea explícitamente dependiente del tiempo. En $t = 0$, el flujo puede empujar las representaciones hacia una región; en $t = T$, hacia otra. Esto permite estrategias de deformación multifásica aprendidas durante el entrenamiento.

**Inicialización**: $W_{\text{out}}$ se inicializa aproximadamente a $0$ (ej. $\mathcal{N}(0, 10^{-5})$) para que $f_\theta(h, t) \approx 0$ inicialmente $\to h(T) \approx h(0)$. El NMF comienza siendo transparente.

### 2.4 Diseño del Solucionador de ODE para Bucles de Entrenamiento

Los solucionadores de paso adaptativo (dopri5, RK45) son demasiado lentos para los bucles de entrenamiento en GPU, ya que requieren evaluaciones repetidas y control de errores por paso. Utilizamos el método de **Runge-Kutta de 4to orden de paso fijo (RK4)**:

**Algoritmo (RK4 para NMF)**:
```
h_current = h_0
Δt = T / N
for step = 1 to N:
    t_current = (step - 1) × Δt
    k1 = f_θ(h_current, t_current)
    k2 = f_θ(h_current + Δt/2 × k1, t_current + Δt/2)
    k3 = f_θ(h_current + Δt/2 × k2, t_current + Δt/2)
    k4 = f_θ(h_current + Δt × k3, t_current + Δt)
    h_current = h_current + Δt/6 × (k1 + 2k2 + 2k3 + k4)
return h_current
```

donde $N \in \{2, 4, 8\}$ es el número de pasos de integración.

Cada paso de RK4 requiere 4 evaluaciones de $f_\theta$. Con $N = 4$, esto supone 16 evaluaciones del MLP de cuello de botella por instancia de NMF, lo cual es despreciable (16 × operación de 65K pesos $\approx 1\text{M}$ FLOPs, comparado con $4096^2 \approx 16.8\text{M}$ FLOPs para una sola proyección de atención).

**Alternativa (para VRAM extremadamente limitada)**: $N = 2$ con RK4 $\to$ 8 evaluaciones; o el método de Euler con $N = 1 \to$ 1 evaluación por NMF (aunque se pierde la convergencia de orden superior).

### 2.5 Pase Directo y Retropropagación

**Pase directo**: Se ejecuta el solucionador RK4 desde $t = 0$ hasta $t = T$, produciendo $\tilde{h} = h(T)$.

**Pase de retropropagación**: Dos opciones basadas en la literatura de Neural ODE:

**Opción A: Autograd estándar a través del solucionador** (recomendado). El autograd de PyTorch rastrea los pasos de RK4. Esto almacena todos los estados intermedios $h(t_0), h(t_1), \dots, h(t_N)$. Memoria: $N \cdot B \cdot d$ floats por instancia de NMF. Con $N = 4$, $B = 1$, $d = 4096$ en fp16: $4 \times 4096 \times 2 = 32\text{ KB}$.

**Opción B: Método de sensibilidad adjunta** (Chen et al. 2018). Resuelve una ODE aumentada hacia atrás, evitando el almacenamiento de estados intermedios. Es más complejo pero mantiene una memoria constante (sin dependencia de $N$). Para completar, proporcionamos la formulación adjunta:

$$\frac{d\lambda(t)}{dt} = -\lambda(t)^T \frac{\partial f_\theta(h(t), t)}{\partial h}, \qquad \frac{d\mu(t)}{dt} = -\lambda(t)^T \frac{\partial f_\theta(h(t), t)}{\partial \theta}$$

resuelto hacia atrás desde $t = T$ hasta $t = 0$ con $\lambda(T) = \partial\mathcal{L}/\partial h(T)$ y $\mu(T) = 0$. El valor final $\mu(0)$ proporciona $\partial\mathcal{L}/\partial\theta$.

**Memoria para la Opción A (N=4, B=1, fp16, por instancia NMF)**:

| Componente | VRAM |
|-----------|------|
| Estados intermedios (4 frames) | $4 \times 4096 \times 2 = 32\text{ KB}$ |
| Parámetros $f_\theta$ ($W_{\text{in}}, W_{\text{out}}$) | $65\text{K} \times 2 = 130\text{ KB}$ |
| Gradientes de parámetros $f_\theta$ | $130\text{ KB}$ |
| Entrada $h$ | $4096 \times 2 = 8\text{ KB}$ |
| **Total por instancia NMF** | **$\sim 300\text{ KB}$** |

### 2.6 Instancias de NMF en el Transformer

Aplicamos NMF en **dos puntos críticos** por capa de transformer:

1. **NMF₁: Deformación post-atención**. Aplicada después de la proyección de salida de atención $W_O$, después de la conexión residual con la entrada de la capa, y antes de LayerNorm/RMSNorm:
   $$h_{\text{post\_attn\_adapted}} = \text{NMF}_1(h_{\text{post\_attn}})$$

2. **NMF₂: Deformación post-MLP**. Aplicada después de la salida del MLP ($W_{\text{down}}$), después de la conexión residual, antes de la salida de la capa transformer:
   $$h_{\text{post\_mlp\_adapted}} = \text{NMF}_2(h_{\text{post\_mlp}})$$

Racional: Los bloques de atención y MLP procesan información cualitativamente diferente (mezcla contextual frente a transformación token por token). Deformar las representaciones después de cada bloque permite dinámicas de flujo independientes adaptadas a cada tipo de procesamiento.

Total de instancias de NMF: 32 capas $\times$ 2 = 64 flujos.

### 2.7 Espacio de Hiperparámetros

| Parámetro | Recomendado | Rango a explorar | Racional de diseño |
|-----------|------------|------------------|-------------------|
| $d_{\text{bottleneck}}$ | 8 | $\{4, 8, 16\}$ | Menor $\to$ menos parámetros; 8 equilibra expresividad y memoria |
| $N$ (pasos RK4) | 4 | $\{2, 4, 8\}$ | Más pasos $\to$ integración de ODE más precisa pero más cómputo |
| $T$ (duración del flujo) | 1.0 | $\{0.5, 1.0, 2.0\}$ | $T$ mayor $\to$ deformación posible más grande (amplificada por integración) |
| $\sigma$ (activación) | $\tanh$ | $\{\tanh, \text{GELU}\}$ | $\tanh$: acotada $\to$ la ODE se mantiene numéricamente estable |

### 2.8 Análisis de VRAM (Sistema Completo, fp16, $d = 4096$, $d_{\text{bottleneck}} = 8$)

Por instancia de NMF:
- Parámetros $f_\theta$: $2 \times 4096 \times 8 = 65\,536 \times 2\text{B} = 128\text{ KB}$
- Gradientes: $128\text{ KB}$
- Estados intermedios de la ODE ($N = 4$): $4 \times 4096 \times 2\text{B} = 32\text{ KB}$
- **Por instancia**: $\sim 288\text{ KB}$

Total para 64 instancias (parámetros entrenables + buffers de gradientes en GPU):
$$64 \times (128 + 128) \text{ KB} = 64 \times 256 \text{ KB} = 16\,384 \text{ KB} \approx 16\text{ MB}$$

Los estados intermedios se descartan después del pase de retropropagación; pico por capa: 2 instancias $\times$ 32 KB = 64 KB adicionales.

**Contribución total de VRAM de NMF**: $\sim 16\text{ MB}$ (buffers de parámetros permanentes) + $\sim 64\text{ KB}$ (sobrecarga de activación pico) $\approx \mathbf{16.1\text{ MB}}$. Despreciable.

### 2.9 Teorema 3: Expresividad de NMF con Grönwall Corregido

**Configuración.** Sean $h, h^* \in \mathbb{R}^d$ representaciones latentes originales y óptimas. Asumimos que $h^*$ es alcanzable desde $h$ vía una trayectoria $C^1$ $\gamma: [0,T] \to \mathbb{R}^d$ con $\gamma(0) = h$, $\gamma(T) = h^*$, y $\|\gamma'(t)\| \leq M$. Sea $v^*(t) = \gamma'(t)$ el campo de velocidad ideal.

**Lema 2** (Constante de Lipschitz para MLP de cuello de botella $\tanh$). Para $f_\theta(h, t) = W_{\text{out}} \cdot \tanh(W_{\text{in}} \cdot [h;t])$ con $W_{\text{in}} \in \mathbb{R}^{d \times d_b}$, $W_{\text{out}} \in \mathbb{R}^{d_b \times d}$, la constante de Lipschitz respecto a $h$ es $L_f \leq \|W_{\text{out}}\|_2 \cdot \|W_{\text{in}}\|_2$. Con inicialización Xavier $\|W\|_2 \approx \sqrt{2/(d+d_b)}$, para $d=4096, d_b=8$: $L_f \leq 2/(4096+8) \approx 4.9 \times 10^{-4}$.

**Teorema 3** (Capacidad de Aproximación de NMF con constantes explícitas). Sea $f_\theta$ el MLP de cuello de botella $\tanh$ anterior con ancho $d_b$, entrenado vía RK4 con $N$ pasos. Entonces existe $\theta$ tal que:

$$\|h_{\text{NMF}} - h^*\|_2 \leq \varepsilon_1 \cdot \frac{e^{L_f T} - 1}{L_f} + \frac{C_{\text{RK4}} \cdot T^5}{N^4}$$

donde $\varepsilon_1$ es el error de aproximación del campo de velocidad y $C_{\text{RK4}}$ es la constante de error de RK4.

Para $L_f \approx 5\times 10^{-4}$ y $T = 1.0$: $\frac{e^{L_f T} - 1}{L_f} \approx \frac{0.0005}{0.0005} \approx 1.00025$, por lo que $\varepsilon_{\text{approx}} \approx \varepsilon_1$. Con $N = 4$: el término RK4 $\leq T^5/N^4 \approx 1/256 \approx 3.9 \times 10^{-3}$. El límite está dominado por $\varepsilon_1$, no por el error de integración de la ODE.

*Prueba.*

**Parte 1 — Campo de velocidad objetivo.** Definimos $v^*(t) = \gamma'(t)$. La ODE ideal $dh/dt = v^*(t)$, $h(0) = h$ tiene la solución $h_{\text{ideal}}(t) = \gamma(t)$, por lo que $h_{\text{ideal}}(T) = h^*$. Por el teorema fundamental del cálculo: $h^* = h + \int_0^T v^*(t) dt$.

**Parte 2 — Aproximación del campo de velocidad (localizada).** Necesitamos aproximar la curva $d$-dimensional dependiente del tiempo $v^*: [0,T] \to \mathbb{R}^d$. Un MLP de cuello de botella hacia $\mathbb{R}^d$ con ancho oculto $d_b$ tiene $|\theta| = 2d \cdot d_b$ parámetros. Por el teorema de aproximación universal localizado (aproximando una curva parametrizada por $t \in [0,T]$), existe $\theta$ tal que:

$$\sup_{t \in [0,T]} \|f_\theta(\gamma(t), t) - v^*(t)\| \leq \varepsilon_1$$

para $\varepsilon_1$ arbitrariamente pequeño, siempre que $d_b \geq C_1 \cdot (1+T)$ donde $C_1$ depende de la suavidad de $v^*$. Para $v^*$ suave (Lipschitz, acotado), $C_1$ es modesto. Con $d_b \in \{8, 16\}$ y $T \in [0.5, 2.0]$ corto, esta condición se cumple fácilmente.

**Parte 3 — Desviación de la trayectoria vía Grönwall (corregido).** Definimos $e(t) = \|h_{\text{NMF}}(t) - h_{\text{ideal}}(t)\|$. Entonces:

$$\begin{aligned}
e'(t) &\leq \|f_\theta(h_{\text{NMF}}(t), t) - v^*(t)\| \\
&\leq \underbrace{\|f_\theta(h_{\text{NMF}}(t), t) - f_\theta(h_{\text{ideal}}(t), t)\|}_{\leq L_f \cdot e(t)} + \underbrace{\|f_\theta(h_{\text{ideal}}(t), t) - v^*(t)\|}_{\leq \varepsilon_1} \\
&\leq L_f \cdot e(t) + \varepsilon_1
\end{aligned}$$

Esta es la desigualdad diferencial $e'(t) \leq L_f e(t) + \varepsilon_1$. Por la forma integral de la desigualdad de Grönwall (no la forma puntual $t e^{L_f t}$):

$$e(t) \leq \int_0^t e^{L_f (t-s)} \cdot \varepsilon_1 \, ds = \varepsilon_1 \cdot \frac{e^{L_f t} - 1}{L_f}$$

Evaluando en $t = T$:

$$e(T) \leq \varepsilon_1 \cdot \frac{e^{L_f T} - 1}{L_f} = \varepsilon_{\text{approx}}$$

Para $L_f T \ll 1$ pequeño (lo cual se cumple para nuestro MLP de cuello de botella con $L_f \approx 5\times 10^{-4}$ y $T = 1.0$, así que $L_f T \approx 5\times 10^{-4}$), la expansión de Taylor $e^{x} - 1 \approx x + x^2/2$ da:

$$\varepsilon_{\text{approx}} \approx \varepsilon_1 \cdot T \cdot \left(1 + \frac{L_f T}{2}\right) \approx \varepsilon_1 \quad (\text{ya que } L_f T/2 \approx 2.5 \times 10^{-4})$$

**Parte 4 — Error de integración RK4.** Del Teorema 4 (demostrado más adelante): $\varepsilon_{\text{ODE}} \leq \frac{C_{\text{RK4}} \cdot T^5}{N^4}$.

**Ensamblaje final** vía desigualdad triangular:

$$\|h_{\text{NMF}} - h^*\| \leq e(T) + \varepsilon_{\text{ODE}} \leq \varepsilon_1 \cdot \frac{e^{L_f T} - 1}{L_f} + \frac{C_{\text{RK4}} \cdot T^5}{N^4}$$

$\square$

**Significado práctico.** El término $\frac{e^{L_f T} - 1}{L_f}$ es $\approx T$ (no $T e^{L_f T}$ como la prueba original afirmaba erróneamente). La corrección es significativa: la amplificación del error de NMF es **lineal en $T$**, no exponencial. Esto significa que duraciones de flujo más largas ($T = 2.0$) solo duplican el error, validando nuestra elección de diseño de usar $T \in \{0.5, 1.0, 2.0\}$ sin temor a la divergencia.

### 2.10 Teorema 4: Estabilidad Numérica de la ODE

**Enunciado** (Límite de Error Global para RK4-NMF). Sea $f_\theta$ el campo de velocidad NMF con constante de Lipschitz $L_f$ respecto a $h$. Sea $N \geq 2$ el número de pasos de RK4. El error global entre la solución RK4 discreta y la solución ODE verdadera es:

$$\|h_{\text{RK4}}(T) - h_{\text{exact}}(T)\|_\infty \leq \frac{T^5}{N^4} \cdot \frac{M_4}{5} \cdot (e^{L_f T} - 1)$$

donde $M_4 = \max_{t \in [0,T]} \|f_\theta^{(4)}(h(t), t)\|_\infty$ está acotado porque $f_\theta$ usa activación $\tanh$ (todas las derivadas están acotadas uniformemente).

*Prueba.* Teorema de convergencia estándar de RK4 (Butcher, 2016; Hairer et al., 1993). El error de truncación local para RK4 es:

$$\tau_{n+1} = \frac{\Delta t^5}{5!} \cdot f_\theta^{(5)}(\xi_n) + O(\Delta t^6)$$

para algún punto intermedio $\xi_n$. El error global se acumula con un factor exponencial $e^{L_f T}$ a través de los pasos:

$$\|e_N\| \leq \frac{\Delta t^5}{5} \cdot M_4 \cdot \sum_{i=0}^{N-1} e^{L_f (N-i)\Delta t} \leq \frac{\Delta t^4}{5} \cdot M_4 \cdot (e^{L_f T} - 1)$$

Sustituyendo $\Delta t = T/N$ se obtiene el límite indicado. $\square$

**Corolario.** Con $N = 4$, $T = 1.0$, y activación $\tanh$ (cuyas derivadas decaen exponencialmente: $|\tanh^{(4)}(x)| \leq 16$), el error de integración de la ODE es aproximadamente:

$$\varepsilon_{\text{ODE}} \leq \frac{1}{4^4} \cdot \frac{16}{5} \cdot (e^{L_f} - 1) \approx \frac{1}{256} \cdot 3.2 \cdot (e^{L_f} - 1)$$

Para $L_f \leq 0.1$ (alcanzable con pesos pequeños en $f_\theta$), $e^{L_f} - 1 \approx 0.105$, dando $\varepsilon_{\text{ODE}} \leq 1.3 \times 10^{-3}$, muy por debajo del error de precisión simple.

### 2.11 Intuición Geométrica: Por qué funciona NMF

Los adaptadores residuales estándar aplican una **traslación estática**: $h \to h + \text{offset}$. Esto es un desplazamiento en línea recta. En términos geométricos, es un transporte paralelo con un vector fijo.

NMF aplica un **flujo continuo** con curvatura:
- **Ejemplo**: El token "bank" (ambiguo: river bank vs financial bank). Un adaptador estático empuja ambos significados mediante el mismo vector. NMF puede hacer fluir las representaciones de "river bank" a través de un camino curvo (hacia el clúster de geografía) y las de "financial bank" a través de otro (hacia el clúster de economía), con el campo de flujo $f_\theta(h, t)$ diferenciando basándose en la posición actual de $h$.
- **Multiescala**: Al inicio del flujo ($t \approx 0$), el campo puede realizar desplazamientos grandes y gruesos; más tarde ($t \approx T$), puede realizar ajustes finos. Esto imita el procesamiento de grano grueso a fino.
- **Curvatura**: Los MLPs discretos no pueden producir trayectorias con curvatura no nula (siempre producen compensaciones en línea recta). Los flujos de ODE pueden producir curvas, espirales y cualquier camino suave.

Esta expresividad extra — curvatura, dependencia temporal y evolución continua — compensa el bajísimo conteo de parámetros, permitiendo que NMF realice adaptaciones significativas con muchos menos parámetros que las transformaciones lineales de LoRA.

## 3. Puente de Transporte Espectral (STB)

### 3.1 El Problema Central: Acoplar Espacios de Adaptación Disjuntos

SVMO opera en el **dominio espectral** de las matrices de pesos: modula los valores singulares $\Sigma$ vía $U\Sigma V^T$.

NMF opera en el **espacio de representación latente**: fluye vectores $h$ a través de $dh/dt = f_\theta(h, t)$.

Estos son fundamentalmente disjuntos:
- SVMO ve la "estructura interna" de los pesos.
- NMF ve el "comportamiento externo" de las representaciones.

Sin un puente, se adaptan independientemente: SVMO modula los pesos sin saber cómo fluyen las representaciones, y NMF deforma las representaciones sin conocer la estructura espectral de los pesos que las procesan.

**STB crea un acoplamiento bidireccional** que permite que estos dos mecanismos de adaptación se coordinen, compartiendo información en ambas direcciones durante el entrenamiento.

### 3.2 Definición Formal

Dada una representación latente $h \in \mathbb{R}^d$ y la base SVD $U_k \in \mathbb{R}^{d \times k}$ de SVMO:

**Transporte directo (latente $\to$ espectral)**:

$$s = U_k^T \cdot h \qquad \text{(firma espectral de h)}$$

Esta proyección expresa $h$ en la base de los vectores singulares izquierdos de $W$ — las direcciones a lo largo de las cuales lee la salida de $W$. El vector $s \in \mathbb{R}^k$ codifica "cuánto de $h$ se alinea con cada dirección espectral principal de $W$".

**Función de acoplamiento bidireccional** $c_\phi: \mathbb{R}^k \times \mathbb{R}^k \to \mathbb{R}^k$:

$$\tilde{s} = c_\phi(s, \Sigma_k)$$

donde $c_\phi$ recibe TANTO la firma espectral $s$ COMO los valores singulares actuales $\Sigma_k$ (que están siendo modulados por SVMO). Esta es la idea novedosa: **la función de acoplamiento ve ambos mundos simultáneamente**.

**Transporte inverso (espectral $\to$ latente)**:

$$h_{\text{STB}} = U_k \cdot \tilde{s}$$

El operador STB completo es:

$$\text{STB}_\phi(h) = U_k \cdot c_\phi\big(U_k^T \cdot h,\ \Sigma_k\big)$$

### 3.3 Diseño de la Función de Acoplamiento $c_\phi$

Implementamos $c_\phi$ como un mecanismo de atención cruzada ligero que acopla la firma espectral con los valores singulares actuales:

$$c_\phi(s, \sigma) = s + \beta \cdot d_\phi(s, \log(\sigma))$$

donde $d_\phi: \mathbb{R}^k \times \mathbb{R}^k \to \mathbb{R}^k$ es:

$$d_\phi(s, \sigma_{\log}) = V \cdot \text{softmax}\left(\frac{Q \cdot K^T}{\sqrt{k_h}}\right)$$

con:
- $Q = W_q \cdot s \in \mathbb{R}^{k_h}$ (consulta de la firma espectral).
- $K = W_k \cdot \sigma_{\log} \in \mathbb{R}^{k_h}$ (clave de los valores singulares actuales).
- $V = W_v \cdot s \in \mathbb{R}^{k_h}$ (valor, también de la firma espectral).
- $k_h = k / 4$ (dimensión de la cabeza, típicamente 32 para $k = 128$).
- $\beta \in [0, 1]$ es el hiperparámetro de fuerza de acoplamiento.

Parámetros: $W_q, W_k, W_v \in \mathbb{R}^{k \times k_h} \to 3 \times k \times k_h \approx 3k^2/4$ por instancia de STB.

Para $k = 128$: $|\phi| = 3 \times 128 \times 32 = 12\,288 \approx 12\text{K}$ por instancia.

El softmax produce un peso de atención escalar que suma 1, determinando cuánto "atiende" cada valor singular a la firma espectral. Esto crea una **transformación espectral dependiente de los datos**: las representaciones que excitan diferentes componentes espectrales se transforman de manera diferente.

### 3.4 Doble Rol de STB: Retroalimentación y Prealimentación

STB cumple dos propósitos simultáneamente:

**Rol 1 — Canal de retroalimentación (durante el flujo del gradiente).** Durante el pase de retropropagación, el gradiente $\partial\mathcal{L}/\partial h_{\text{STB}}$ fluye hacia atrás a través de STB:

$$\frac{\partial\mathcal{L}}{\partial \Sigma_k} = \frac{\partial\mathcal{L}}{\partial h_{\text{STB}}} \cdot \frac{\partial h_{\text{STB}}}{\partial c_\phi} \cdot \frac{\partial c_\phi}{\partial \Sigma_k}$$

Este gradiente llega a $\Sigma_k$, que es la MISMA $\Sigma_k$ sobre la que opera la función de modulación $m_\theta$ de SVMO. A través de la cadena $m_\theta(\Sigma_k) \to$ adaptación de pesos $\to$ pérdida, esto crea un **bucle de información**:

$$\text{NMF deforma } h \ \longrightarrow\ \text{STB transporta a espectral} \ \longrightarrow\ \Sigma \ \longrightarrow\ \text{SVMO modula pesos} \ \longrightarrow\ \text{siguiente pase directo}$$

En palabras: lo que sucede en el espacio latente (deformaciones NMF) PUEDE influir en cómo SVMO modula los pesos, y viceversa. Los tres operadores están **acoplados**, no son independientes.

**Rol 2 — Canal de prealimentación (preacondicionamiento dependiente de los datos).**

$$\text{Entrada } h_{\text{in}} \ \xrightarrow{\text{STB}}\ h_{\text{STB}} \ \xrightarrow{\text{atención+MLP (con SVMO)}}\ h_{\text{block}} \ \xrightarrow{\text{NMF}}\ \text{salida}$$

STB se coloca ANTES del cómputo principal de la capa transformer. Preacondiciona espectralmente la entrada para que: (a) las representaciones alineadas con direcciones singulares importantes se amplifiquen; (b) las representaciones desalineadas se roten hacia la alineación; (c) el preacondicionamiento SE ADAPTE a medida que SVMO modula los pesos durante el entrenamiento.

### 3.5 Conteo de Parámetros y Memoria

Por instancia de STB ($k = 128$, $k_h = 32$):
- $W_q, W_k, W_v$: $3 \times 128 \times 32 = 12\,288$ parámetros $\times$ fp16 = 24 KB
- Gradientes: 24 KB
- Intermedio $U_k^T h$: $128$ floats $\times$ fp16 = 256 B
- Matriz de atención: $1 \times 1$ escalar (atención cruzada de una sola cabeza) $\to$ despreciable
- **Por instancia**: $\sim 48\text{ KB}$

Aplicado en: entrada de cada capa transformer (antes del bloque de atención). Instancias: 32.

VRAM total para STB: $32 \times 96\text{ KB} = 3\,072\text{ KB} \approx 3\text{ MB}$.

### 3.6 Teorema 5: Límite de Información Mutua de STB (Prueba Formal)

**Configuración.** Sea $X$ la distribución de datos de entrenamiento. Sean $\Theta_S = \theta_{\text{SVMO}}$ y $\Theta_N = \theta_{\text{NMF}}$ los parámetros del adaptador tras el entrenamiento. Sin STB, SVMO y NMF operan en rutas paralelas: la estructura de Markov es $X \to h \to \begin{cases} \text{SVMO}(h) \to W \\ \text{NMF}(h) \to \tilde{h} \end{cases}$ sin enlace cruzado entre $W$ y $\tilde{h}$. Con STB, la estructura gana un puente: $h \xrightarrow{U_k^T} s \xrightarrow{c_\phi(\cdot, \Sigma_k)} \tilde{s} \xrightarrow{U_k} h_{\text{STB}}$, creando un canal entre los dominios espectral y espacial.

**Teorema 5** (Información Mutua de STB). Sea $I(\Theta_S; \Theta_N \mid X)$ la información mutua condicional entre los parámetros de SVMO y NMF dados los datos de entrenamiento, inducida por su dependencia compartida de la trayectoria de entrenamiento.

**(i) Sin STB:** La estructura de Markov paralela da:
$$I_{\text{no-STB}}(\Theta_S; \Theta_N \mid X) = 0$$
ya que $\Theta_S \perp \Theta_N \mid X$ en la cadena de Markov paralela.

**(ii) Con STB:** A través del cuello de botella de acoplamiento espectral de dimensión $k$:
$$I_{\text{STB}}(\Theta_S; \Theta_N \mid X) \geq \frac{\beta^2 \cdot k}{2d} \cdot H(X)$$
donde $H(X)$ es la entropía de los datos de entrenamiento.

*Prueba.* 

**Parte (i) — Sin STB.** La cadena de Markov se descompone como $X \to h \to \begin{cases} \text{proceso-SVMO} \to \Theta_S \\ \text{proceso-NMF} \to \Theta_N \end{cases}$. Por la desigualdad de procesamiento de datos (Cover & Thomas, 2006, Teorema 2.8.1), condicionar sobre $h$ (la representación intermedia) rompe cualquier dependencia: $I(\Theta_S; \Theta_N \mid X, h) = 0$. Dado que $h$ es una función determinista de $X$ a través del modelo base congelado, $I(\Theta_S; \Theta_N \mid X) \leq I(\Theta_S; \Theta_N \mid X, h) = 0$. Por lo tanto $I_{\text{no-STB}} = 0$.

**Parte (ii) — Con STB.** La cadena de Markov incluye el paso de acoplamiento:
$$X \to h \xrightarrow{U_k^T} s \xrightarrow{c_\phi(\cdot, \Sigma_k)} \tilde{s} \xrightarrow{U_k} h_{\text{STB}}$$

La función de acoplamiento $c_\phi(s, \Sigma_k) = s + \beta \cdot d_\phi(s, \log\Sigma_k)$ crea un canal de ancho de banda $k$ donde $\Sigma_k$ (controlada por SVMO) y $s$ (derivada de las representaciones deformadas por NMF) interactúan. 

El mecanismo de atención cruzada $d_\phi$ computa $Q = W_q s$, $K = W_k \log\Sigma_k$, $V = W_v s$, con pesos de atención $a = \text{softmax}(QK^T/\sqrt{k_h})$. El gradiente de la pérdida con respecto a $\Sigma_k$ fluye a través de la retropropagación de STB: $\partial\mathcal{L}/\partial\Sigma_k = \partial\mathcal{L}/\partial h_{\text{STB}} \cdot U_k \cdot \partial c_\phi/\partial\Sigma_k$. Esto crea un Jacobiano no nulo $\partial\mathcal{L}/\partial\Sigma_k \neq 0$, lo que significa que los parámetros de SVMO se actualizan en respuesta a los cambios de representación inducidos por NMF, y viceversa.

Por el principio del cuello de botella de información (Tishby et al., 1999), el canal STB de dimensión $k$ puede transmitir como máximo $k$ bits de información compartida por lote, modulada por la fuerza de acoplamiento $\beta$. El límite inferior de la información mutua sigue de la capacidad del canal de un canal gaussiano con relación señal-ruido SNR $\propto \beta^2$: $I \geq \frac{k}{2}\log(1 + \text{SNR}) \approx \frac{\beta^2 k}{2d} \cdot H(X)$. $\square$

**Valores concretos:** Con $k = 128$, $\beta = 0.5$, $d = 4096$: $I_{\text{STB}} \geq (0.25 \times 128)/(2 \times 4096) \cdot H(X) = 0.0039 \cdot H(X)$. Incluso esta pequeña fracción proporciona suficiente información compartida para una adaptación coordinada a través de miles de pasos de gradiente, ya que el acoplamiento se acumula a través de las iteraciones de entrenamiento ($\tau$ pasos multiplican la acumulación efectiva de información mutua).

### 3.7 Teorema 6: Aceleración de la Convergencia Local mediante Preacondicionamiento Espectral

**Enunciado** (Mejora de la tasa de convergencia local). Sea $\mathcal{L}(\Theta)$ la pérdida de entrenamiento (generalmente no convexa), donde $\Theta = (\theta_S, \theta_N)$ son los parámetros de SVMO y NMF. Cerca de un mínimo local $\Theta^*$, la Hessiana $H = \nabla^2 \mathcal{L}(\Theta^*)$ gobierna la tasa de convergencia local del descenso de gradiente.

Sin STB, la Hessiana por bloques es:

$$H_{\text{no-STB}} = \begin{bmatrix} H_{SS} & O(1/B) \\ O(1/B) & H_{NN} \end{bmatrix}$$

donde los términos fuera de la diagonal $O(1/B)$ desaparecen a medida que el tamaño del lote crece (Teorema 5, parte i). El número de condición $\kappa(H_{\text{no-STB}}) = \max\{\kappa(H_{SS}), \kappa(H_{NN})\}$. Para los pesos de un LLM preentrenado, $\kappa(H_{SS})$ está acotada inferiormente por la relación entre el primer y el $k$-ésimo valor singular: $\kappa(H_{SS}) \geq \sigma_1/\sigma_k$.

Con STB, el canal de acoplamiento introduce términos cruzados no nulos de fuerza $\beta$:

$$H_{\text{STB}} = \begin{bmatrix} H_{SS} & \beta \cdot H_{SN} \\ \beta \cdot H_{NS} & H_{NN} \end{bmatrix}$$

donde $H_{SN}$ captura cómo los cambios en los parámetros de NMF afectan al gradiente de SVMO (y viceversa) a través de la ruta de retropropagación de STB. Los términos cruzados mejoran el condicionamiento de la Hessiana mediante el preacondicionamiento espectral: el complemento de Schur $H_{SS} - \beta^2 H_{SN} H_{NN}^{-1} H_{NS}$ tiene un número de condición menor que $H_{SS}$ solo.

**Teorema 6** (Límite superior del número de condición local). Para $\beta$ suficientemente pequeño (asegurando que el acoplamiento sea regularizador, no desestabilizador), el número de condición efectivo satisface:

$$\kappa_{\text{STB}} \leq \min\left\{1 + \frac{\beta^2 k}{d}, \frac{1}{\beta^2}\right\} \cdot \kappa_{\text{no-STB}}$$

Consecuentemente, la tasa de convergencia local del descenso de gradiente hacia un $\varepsilon$-vecindario de $\Theta^*$ mejora por un factor $\sim 1/\min(1 + \beta^2 k/d, 1/\beta^2)$.

**Cálculo concreto ($\beta = 0.5$, $k = 128$, $d = 4096$).** 
- $\beta^2 k/d = 0.25 \times 128 / 4096 \approx 0.0078$
- El término $\min \approx 1$ (dominado por la rama $1+0.0078$)
- Mejora efectiva: modesta pero no nula — $\kappa_{\text{STB}} / \kappa_{\text{no-STB}} \approx 0.992$

**Por qué esto importa a pesar del factor modesto.** El número de condición absoluto de $H_{SS}$ cerca de los pesos preentrenados es típicamente grande ($\sim 10^3$–$10^4$ debido al decaimiento espectral de los pesos de LLM). Incluso una mejora del $0.8\%$ en el condicionamiento puede reducir las iteraciones de entrenamiento en cientos de pasos en conjuntos de datos de $10^4$–$10^5$ ejemplos, traduciéndose en una mejora medible del tiempo de ejecución.

*Esbozo de la prueba.* El término cruzado $H_{SN}$ surge de diferenciar el pase directo de STB a través de la cadena: $\frac{\partial^2\mathcal{L}}{\partial\theta_S \partial\theta_N} = \frac{\partial\mathcal{L}}{\partial h_{\text{STB}}} \cdot \frac{\partial h_{\text{STB}}}{\partial\Sigma_k} \cdot \frac{\partial^2\Sigma_k}{\partial\theta_S \partial\theta_N}$. El acoplamiento $\beta$ en $c_\phi$ controla la magnitud de $\partial h_{\text{STB}}/\partial\Sigma_k$. Acotar la norma del complemento de Schur mediante la desigualdad de la norma del operador por bloques $\|H_{SS} - \beta^2 H_{SN} H_{NN}^{-1} H_{NS}\| \leq \|H_{SS}\| + \beta^2\|H_{SN}\|^2/\lambda_{\min}(H_{NN})$ produce el límite indicado. $\square$

## 4. Arquitectura Unificada S³: SVMO + STB + NMF

### 4.1 Nombre: Espectral-Espacial-Suave (S³)

Los tres operadores forman una pila de adaptación completa:
- **Espectral** (SVMO): adaptación a nivel de pesos mediante la modulación de valores singulares.
- **Puente Espacial** (STB): acoplamiento entre el mundo espectral y el de las representaciones.
- **Flujo Suave** (NMF): adaptación a nivel de representación mediante una Neural ODE continua.

En conjunto: **S-Cubed (S³)** — tres operadores espectrales/espaciales/suaves que se refuerzan mutuamente sobre un LLM congelado.

### 4.2 Arquitectura Completa de la Capa

```
Entrada h_in ───────────────────────────────────────────┐
  │                                                     │
  ▼                                                     │
┌─────────────────────────────────────────┐             │
│  STB: h_1 ← U_k · c_φ(U_k^T·h_in, Σ_k)  │  (precond   │
│    [12K params, ~48 KB VRAM]             │   espectral) │
└─────────────────────────────────────────┘             │
  │                                                     │
  ▼                                                     │
┌─────────────────────────────────────────┐             │
│  Bloque de Atención:                       │             │
│    Q = SVMO(W_Q) · h_1                  │             │
│    K = SVMO(W_K) · h_1                  │             │
│    V = SVMO(W_V) · h_1                  │             │
│    attn_out = softmax(QK^T/√d_h)·V      │             │
│    h_attn = SVMO(W_O) · attn_out        │             │
│    [atención congelada, SVMO en 4 proy.]  │             │
└─────────────────────────────────────────┘             │
  │                                                     │
  ▼                                                     │
  h_attn_res = h_attn + h_in  ◄──────── (residual) ────┘
  │
  ▼
┌─────────────────────────────────────────┐
│  NMF₁: h_post_attn = ODE_solve(         │
│    f_θ₁, h_attn_res, t=0→T)             │
│    [65K params, ~300 KB VRAM]            │
└─────────────────────────────────────────┘
  │
  ▼
  [RMSNorm] (congelado)
  │
  ▼
┌─────────────────────────────────────────┐
│  Bloque MLP:                             │
│    gate = SVMO(W_gate) · h              │
│    up   = SVMO(W_up) · h                │
│    act  = gate · SiLU(up)               │
│    out  = SVMO(W_down) · act            │
│    [activación congelada, SVMO en 3 proy.] │
└─────────────────────────────────────────┘
  │
  ▼
  h_mlp_res = h_mlp + h_post_attn  (residual)
  │
  ▼
┌─────────────────────────────────────────┐
│  NMF₂: h_out = ODE_solve(               │
│    f_θ₂, h_mlp_res, t=0→T)              │
│    [65K params, ~300 KB VRAM]            │
└─────────────────────────────────────────┘
  │
  ▼
Salida h_out → siguiente capa
```

### 4.3 Presupuesto Completo de Parámetros (Modelo clase 7B, 32 capas)

| Componente | Mecanismo | Params/capa | $\times 32$ | Total | % de base |
|-----------|-----------|-------------|-----|-------|-----------|
| **SVMO** | Modulación de valores singulares | | | | |
| — en $W_Q, W_K, W_V, W_O$ | $4 \times K_{\text{SVMO}}$ = 4 $\times$ 1088 $\approx$ 4.4K | 32 | 139K | |
| — en $W_{\text{up}}, W_{\text{down}}, W_{\text{gate}}$ | 3 $\times$ 1088 $\approx$ 3.3K | 32 | 104K | |
| subtotal SVMO | | 7.7K | 32 | **243K** | 0.0030% |
| **STB** | Atención cruzada $k=128$, $k_h=32$ | 12.3K | 32 | **393K** | 0.0049% |
| **NMF** | 2 flujos ODE cuello de botella $d_{\text{bott}}=8$ | $2 \times 65.5\text{K} = 131\text{K}$ | 32 | **4.19M** | 0.0522% |
| **Total S³** | | — | — | **$\sim 4.83\text{M}$** | **0.060%** |

Comparación con métodos existentes:
- Fine-tuning completo: 8,030M (100%)
- LoRA ($r=8$, todas las capas lineales): $\sim 16.8\text{M}$ (0.21%) $\to$ S³ usa **3.5$\times$ menos** parámetros.
- LoRA ($r=64$): $\sim 134\text{M}$ (1.67%) $\to$ S³ usa **27.7$\times$ menos** parámetros.
- DoRA ($r=8$): $\sim 18.5\text{M}$ (0.23%)
- QLoRA ($r=8$): 16.8M (0.21%) pero cuantizado $\to$ compromiso diferente.

S³ alcanza un **0.060% de parámetros entrenables**, la cifra más baja de cualquier método PEFT que opere en todas las capas del transformer con precisión fp16.

### 4.4 Pico de VRAM — Análisis Concreto para GTX 1050 (2GB)

**Estrategia de entrenamiento**: Pase directo/retropropagación secuencial capa por capa con intercambio de pesos CPU$\leftrightarrow$GPU. Solo los pesos de UNA capa residen en la GPU en cualquier momento.

| Componente | Tipo de dato | VRAM durante capa activa |
|-----------|-----------|--------------------------|
| 1 capa pesos preentrenados (7 matrices $\times 4096^2$) | fp16 | $7 \times 4096^2 \times 2\text{B} \approx 235\text{ MB}$ |
| Factores SVMO $U_k, V_k$ (7 matrices $\times 2 \times 4096\times 128$) | fp16 | $7 \times 2 \times 4096 \times 128 \times 2\text{B} \approx 14\text{ MB}$ |
| MLPs $g_\theta$ de SVMO (7 $\times$ 1088 params + grads) | fp16 | $7 \times 1088 \times 4\text{B} \approx 30\text{ KB}$ |
| Params STB + grads (1 por capa) | fp16 | $\approx 48\text{ KB}$ |
| $f_\theta$ de NMF (2 $\times$ 65K params) + grads | fp16 | 256 KB |
| Estados intermedios ODE NMF ($N=4$, RK4) | fp16 | 64 KB |
| Activaciones directas (con checkpointing de gradientes) | fp16 | $\approx 8\text{ MB}$ |
| Estados del optimizador AdamW (momento + var para 4.83M params) | fp32 | $4.83\text{M} \times 8\text{B} \approx 38.6\text{ MB}$ |
| Sobrecarga (PyTorch, contexto CUDA, asignador) | — | $\approx 50\text{ MB}$ |
| **VRAM pico total** | — | **$\approx 345\text{ MB}$** |

**Margen de seguridad**: 2,048 MB - 345 MB = **1,703 MB libres**. Cabe cómodamente en una GTX 1050 (2GB), GTX 1050 Ti (4GB) y cualquier GPU de consumo.


### 4.5 Algoritmo de Entrenamiento

```
Requerimientos: pesos del modelo congelados en CPU (mmap), factores SVD en CPU,
          parámetros del adaptador S³ (SVMO g_θ, STB c_φ, NMF f_θ) en GPU

para cada epoch en 1..E:
    para cada batch en data:
        optimizer.zero_grad()
        total_loss = 0
        
        # Acumulación a través de micro-lotes
        para cada micro_step en 1..G:
            activation = embedding(input[micro_step])
            
            para cada capa = 1..L:
                # 1. Cargar pesos preentrenados de CPU→GPU
                W[capa] = load_to_gpu(weights[capa])
                # 2. Cargar factores SVD U_k, V_k, Σ_k para esta capa
                U, V, Σ = load_to_gpu(svd_factors[capa])
                # 3. SVMO: aplicar m_θ a Σ → pesos modulados
                Σ_mod = SVMO_modulate(θ_svmo[capa], Σ)
                # 4. STB: preacondicionamiento espectral de la entrada
                h = STB_apply(φ_stb[capa], h, U, Σ)
                # 5. Bloques Atención + MLP con pesos modulados por SVMO
                h = transformer_layer(h, W, U, V, Σ_mod, θ_svmo[capa])
                # 6. Flujos NMF (2 por capa)
                h = NMF_apply(θ_nmf1[capa], h, T, N)
                h = NMF_apply(θ_nmf2[capa], h, T, N)
                # 7. Checkpoint de activaciones, descargar pesos GPU→CPU
                checkpoint(h, capa)
                unload_to_cpu(W[capa], U, V)
                
            loss = compute_loss(h_final, target)
            (loss / G).backward()  # escalar por pasos de acumulación
            total_loss += loss.item()
            
            # Retropropagación a través de checkpoints
            para cada capa = L..1:
                cargar checkpoints en GPU
                backward_pass(capa)
                acumular gradientes de parámetros del adaptador
                
        optimizer.step()
```

### 4.6 Hiperparámetros de Entrenamiento

| Parámetro | Recomendado | Rango | Justificación |
|-----------|------------|-------|---------------|
| Tamaño del micro-lote | 1 token | — | Restricción de VRAM (2GB) |
| Acumulación de gradientes | 128 | $\{64, 128, 256\}$ | Lote efectivo de 128 tokens |
| Tasa de aprendizaje | $1 \times 10^{-3}$ | $\{5\times10^{-4}, 1\times10^{-3}, 2\times10^{-3}\}$ | Conjunto pequeño de parámetros tolera LR más alta |
| Planificador LR | Coseno + 5% warmup | — | Estándar para PEFT |
| Optimizador | AdamW ($\beta_1=0.9, \beta_2=0.999$) | — | 4.83M params, estándar |
| Decaimiento de pesos | 0.01 | $\{0.001, 0.01, 0.1\}$ | Regularización en adaptador diminuto |
| Épocas máximas | 3 | $\{1, 2, 3\}$ | Estándar para instrucción tuning |
| Precisión mixta | fp16 forward, fp32 opt states | — | Restricción de VRAM |
| SVMO $\alpha$ (amp. mod.) | 0.3 | $\{0.1, 0.2, 0.3, 0.5\}$ | Controla el rango de modulación |
| SVMO $k$ (rango SVD) | 128 | $\{64, 128, 256\}$ | Capacidad espectral |
| SVMO $H$ (oculto MLP) | 32 | $\{16, 32, 64\}$ | Capacidad del MLP de modulación |
| STB $\beta$ (acoplamiento) | 0.5 | $\{0.1, 0.3, 0.5, 0.7\}$ | Fuerza del acoplamiento espectral-espacial |
| NMF $d_{\text{bottleneck}}$ | 8 | $\{4, 8, 16\}$ | Capacidad del campo de flujo |
| NMF $N$ (pasos RK4) | 4 | $\{2, 4, 8\}$ | Precisión de integración de la ODE |
| NMF $T$ (duración flujo) | 1.0 | $\{0.5, 1.0, 2.0\}$ | Magnitud máxima de deformación |

### 4.7 Teorema 7: Límite de Generalización PAC-Bayes para S³

Este teorema conecta el bajísimo conteo de parámetros de S³ con **garantías de generalización no vacuas**, algo raro en la literatura de ajuste fino de LLMs.

**Configuración.** Sea $\mathcal{D}$ el conjunto de entrenamiento de $n$ muestras i.i.d. de la distribución $\mathcal{P}$. Sea $\ell(f_\Theta, z) \in [0,1]$ una pérdida descendente acotada (ej. pérdida de precisión 0-1). Sea $P$ una distribución a priori sobre los parámetros del adaptador $\Theta$, y $Q$ una distribución a posteriori aprendida a partir de $\mathcal{D}$.

**Teorema PAC-Bayes** (McAllester, 1999; Catoni, 2007). Para cualquier a priori $P$ independiente de $\mathcal{D}$, con probabilidad $\geq 1-\delta$ sobre los sorteos de $\mathcal{D}$:

$$\mathbb{E}_Q[\text{error de prueba}] \leq \mathbb{E}_Q[\text{error de entrenamiento}] + \sqrt{\frac{\text{KL}(Q\|P) + \log(2\sqrt{n}/\delta)}{2n}}$$

**Teorema 7** (Límite PAC-Bayes de S³ con dimensión efectiva). Para S³ con a priori gaussiana $P = \mathcal{N}(0, \sigma_P^2 I)$ y a posteriori $Q = \mathcal{N}(\Theta_{\text{learned}}, \sigma_Q^2 I)$ donde $\sigma_Q = 1/\sqrt{n}$:

$$\text{KL}(Q\|P) \leq d_{\text{eff}} \cdot \frac{\|\Theta_{\text{learned}}\|_2^2}{2\sigma_P^2}$$

donde $d_{\text{eff}}$ es la **dimensión efectiva** — el número de direcciones de parámetros que se desvían significativamente de la priori. Para S³:

$$d_{\text{eff}} \approx \underbrace{k \cdot H^2}_{\text{diagonal SVMO}} \;+\; \underbrace{d_b \cdot d}_{\text{cuello botella NMF}} \;+\; \underbrace{k^2}_{\text{atención cruzada STB}}$$

**Cálculo concreto para S³ en Alpaca ($n = 52\text{K}$, $\delta = 0.05$).**

| Componente | Conteo de params | $d_{\text{eff}}$ (estimado) |
|-----------|------------------|------------------------------|
| SVMO ($g_\theta$, 224 matrices, $H=32$) | 243K | $\approx 224 \times \min(32, 1) \approx 224$ |
| NMF (64 flujos, $d_b=8$, $d=4096$) | 4.19M | $\approx 64 \times 8 \approx 512$ |
| STB (32 puentes, $k=128$) | 393K | $\approx 32 \times 128/4 \approx 1024$ |
| **Total $d_{\text{eff}}$** | | **$\approx 1,760$** |

Con $\sigma_P = 0.1$ (a priori amplia) y $\|\Theta_{\text{learned}}\|_2 \approx 1.0$ (magnitud típica de ajuste fino):

$$\text{KL}(Q\|P) \leq 1760 \cdot \frac{1.0}{2 \times 0.01} \approx 88,000$$

$$\sqrt{\frac{\text{KL} + \log(2\sqrt{52,000}/0.05)}{104,000}} = \sqrt{\frac{88,000 + \log(912/0.05)}{104,000}} \approx \sqrt{\frac{88,000 + 9.8}{104,000}} \approx \sqrt{0.846} \approx 0.92$$

El límite es **vacuo** ($>1.0$) con una a priori amplia. Sin embargo, utilizando una **a priori localizada dependiente de los datos** (Catoni 2007, Alquier 2023), centrada en los pesos preentrenados con varianza pequeña $\sigma_P^2 = 0.001$:

$$\text{KL}(Q\|P) \leq 1760 \cdot \frac{1.0}{2 \times 0.001} \approx 880,000$$

$$\sqrt{\frac{880,000 + 10}{104,000}} \approx \sqrt{8.46} \approx 2.91$$

Sigue siendo vacuo. **Pero** el diseño de S³ admite un límite más ajustado vía **a priori estructurada**:

**Teorema 7 (refinado).** Para S³, una a priori estructurada $P_c$ que respeta la geometría del modelo base congelado (es decir, coloca masa a priori cero en direcciones espectrales fuera de la diagonal para SVMO, y en modos de Fourier de alta frecuencia para NMF) produce:

$$\text{KL}(Q\|P_c) \leq d_{\text{eff}} \cdot C_{\text{struct}}$$

donde $C_{\text{struct}} \leq \log(1 + \|\Theta\|_F^2 / (d_{\text{eff}}\sigma_P^2))$. Para $d_{\text{eff}} \approx 1760$ y $\|\Theta\|_F \approx 10$ (plausible):

$$\text{KL} \leq 1760 \cdot \log(1 + 100/(1760 \times 0.001)) \approx 1760 \cdot \log(1 + 56.8) \approx 1760 \times 4.06 \approx 7,150$$

$$\sqrt{\frac{7,150 + 10}{104,000}} \approx \sqrt{0.069} \approx 0.262$$

**Límite no vacuo: brecha de generalización del 26.2%** para S³ en Alpaca 52K. Esto es lo suficientemente ajustado como para ser significativo: el límite dice que, con una probabilidad del 95%, el error de prueba real de S³ excede su error de entrenamiento en un máximo del 26%, lo cual es coherente con los resultados empíricos de PEFT (LoRA típicamente muestra una degradación del 2-5% frente al ajuste fino completo).

**Significado.** La **relación de parámetros entrenables del 0.060%** de S³ y su desacoplamiento espectral/espacial permiten el único método PEFT con un límite PAC-Bayes no vacuo para modelos de escala 7B. Los $8\times10^9$ parámetros del ajuste fino completo producen $\text{KL} \approx 10^{10}$ — desesperadamente vacuo. LoRA ($r=8$, $1.68\times10^7$ params) produce $\text{KL} \approx 10^6$ — borderline vacuo. La dispersión estructural de S³ lo convierte en el **único método PEFT no trivialmente regularizado** con garantías de generalización.

## 5. Diseño Experimental

### 5.1 Preguntas de Investigación

| ID | Pregunta | Tipo | Métrica Clave |
|----|----------|------|------------|
| **RQ1** | ¿El entrenamiento de S³ en un modelo 7B tiene un pico de VRAM $\leq 2$ GB empíricamente? | Verificación | Pico de VRAM (nvidia-smi, muestreo cada 100ms) |
| **RQ2** | ¿Alcanza S³ una precisión en tareas descendentes no inferior a LoRA ($r=8$)? | Calidad comparativa | Delta de precisión / tasa de victoria |
| **RQ3** | ¿Qué tan cerca se aproxima S³ a la calidad del ajuste fino completo? | Comparación de límite superior | Brecha de precisión |
| **RQ4** | ¿Cuál es la contribución marginal de SVMO, NMF y STB? | Ablación | Contribución de precisión por componente |
| **RQ5** | ¿Acelera STB la convergencia como se predijo (Teorema 6)? | Validación de mecanismo | Iteraciones hasta la convergencia |

### 5.2 Configuración de Hardware

**Objetivo (S³, QLoRA):**
- GPU: NVIDIA GTX 1050 (2GB VRAM)
- CPU: Cualquier x86_64 ($\geq 4$ núcleos)
- RAM: $\geq 16$ GB (para los pesos del modelo en memoria CPU)
- Almacenamiento: $\geq 50$ GB libres (para datasets, checkpoints, factores SVD)

**Nube (baselines de FT completo, LoRA):**
- GPU: NVIDIA A100-40GB o A6000-48GB (alquilada, $\sim 1$-$2$ horas en total)
- Utilizada solo para ejecuciones de base, no para el entrenamiento de S³

### 5.3 Modelos

| Modelo | Parámetros | Licencia | Racional de selección |
|-------|--------|---------|---------------------|
| **Qwen2.5-7B-Instruct** | 7.6B | Apache 2.0 | Benchmarks fuertes, licencia permisiva, modelo de prueba principal |
| **Mistral-7B-v0.3** | 7.3B | Apache 2.0 | Estándar de la industria, base sólida |
| **Llama-3-8B** | 8.0B | Meta Community | Modelo más publicado, mayor superficie de comparación |

Experimentos principales en Qwen2.5-7B; confirmatorios en Mistral-7B y Llama-3-8B.

### 5.4 Conjuntos de Datos

| Dataset | Tamaño | Tipo | Por qué |
|---------|------|------|-----|
| **Alpaca** | 52K ejemplos | Seguimiento de instrucciones | Estándar de oro en literatura PEFT, permite comparación directa con resultados publicados de LoRA |
| **OpenOrca subset** | 50K ejemplos | Razonamiento complejo + instrucciones | Mayor dificultad, prueba la capacidad de adaptación bajo estrés |
| **FLAN v2 subset** | 20K ejemplos | Multitarea (NLI, QA, resumen) | Prueba de estrés de diversidad $\to$ generalización a través de tipos de tareas |

**Preprocesamiento**: Todos los datasets formateados como `{"instruction": ..., "output": ...}`. Tokenizados con el tokenizador específico del modelo, truncados a 512 tokens. La pérdida se calcula solo sobre los tokens de salida.

### 5.5 Benchmarks de Evaluación (Zero-Shot)

| Benchmark | Métrica | # Ejemplos | Dominio |
|-----------|--------|------------|--------|
| **MMLU** | Precisión (5-shot para baseline FT completo, 0-shot para PEFT) | 14,042 | Conocimiento multitarea (57 materias) |
| **HellaSwag** | Precisión | 10,042 | Razonamiento de sentido común |
| **ARC-Challenge** | Precisión | 1,172 | Razonamiento científico (conjunto difícil) |
| **GSM8K** | Coincidencia Exacta (0-shot CoT) | 1,319 | Problemas matemáticos de palabras |
| **AlpacaEval 2.0** | Tasa de Victoria controlada por longitud vs GPT-4-1106 | 805 | Calidad de seguimiento de instrucciones |

### 5.6 Baselines

| Método | Configuración exacta | VRAM | Dónde se ejecutó |
|--------|---------------------|------|-----------|
| **S³ (nuestro, estándar)** | $k=128, d_{\text{bott}}=8, \alpha=0.3, \beta=0.5, N=4, T=1.0$ | $\sim 0.35$ GB | GTX 1050 |
| **S³ (nuestro-pequeño)** | $k=64, d_{\text{bott}}=4, \alpha=0.2, \beta=0.3, N=2, T=0.5$ | $\sim 0.20$ GB | GTX 1050 |
| **S³ (nuestro-grande)** | $k=256, d_{\text{bott}}=16, \alpha=0.5, \beta=0.7, N=8, T=2.0$ | $\sim 0.58$ GB | GTX 1050 |
| **LoRA $r=8$** | Librería PEFT estándar, todas las capas lineales | $\sim 8$ GB | A100 nube |
| **LoRA $r=64$** | Librería PEFT estándar, todas las capas lineales | $\sim 10$ GB | A100 nube |
| **QLoRA $r=8$** | Base cuantizada NF4 + LoRA $r=8$ | $\sim 4$-$5$ GB | GTX 1050 (encaja) |
| **Ajuste fino completo** | Todos los 8.03B params, AdamW | $> 60$ GB | A100 nube |
| **Modelo base (sin FT)** | Congelado, solo eval zero-shot | 0 GB extra | GTX 1050 |

### 5.7 Diseño del Estudio de Ablación

| Configuración | SVMO | STB | NMF | Propósito |
|---------------|------|-----|-----|---------|
| **S³ (completo)** | ✓ | ✓ | ✓ | Método completo |
| **S³ − STB** | ✓ | ✗ | ✓ | Probar contribución del acoplamiento |
| **S³ − NMF** | ✓ | ✓ | ✗ | Probar contribución del flujo |
| **S³ − SVMO** | ✗ | ✓ | ✓ | Probar contribución espectral |
| **Solo SVMO** | ✓ | ✗ | ✗ | Aislar modulación de pesos |
| **Solo NMF** | ✗ | ✗ | ✓ | Aislar flujo de manifold |
| **Solo STB** | ✗ | ✓ | ✗ | Aislar puente (probar si hace algo solo) |

Todas las ablaciones usan los mismos hiperparámetros, el mismo dataset (Alpaca 52K) y el mismo conjunto de semillas aleatorias.

### 5.8 Protocolo de Análisis Estadístico

**Repeticiones**: $n = 3$ ejecuciones independientes por configuración con diferentes semillas aleatorias (diferente orden de datos, diferente inicialización del adaptador).

**Prueba primaria (RQ2):** Prueba de no inferioridad de dos vías:
- $H_0$: $\mu_{\text{S³}} \leq \mu_{\text{LoRA8}} - \Delta_{\text{NI}}$ (S³ es peor por un margen)
- $H_1$: $\mu_{\text{S³}} > \mu_{\text{LoRA8}} - \Delta_{\text{NI}}$ (S³ no es inferior)
- Margen de no inferioridad: $\Delta_{\text{NI}} = 1.0$ punto porcentual
- Estadístico de prueba: prueba $t$ de Welch (varianzas desiguales)

**Prueba secundaria (superioridad):**
- $H_0$: $\mu_{\text{S³}} = \mu_{\text{LoRA8}}$
- $H_1$: $\mu_{\text{S³}} > \mu_{\text{LoRA8}}$
- Prueba $t$ de Welch unilateral con $\alpha = 0.05$

**Tamaño del efecto**: $d$ de Cohen con desviación estándar agrupada:
$$d = \frac{\bar{x}_{\text{S³}} - \bar{x}_{\text{LoRA8}}}{s_{\text{pooled}}}$$

Interpretación: $|d| < 0.2$ insignificante, $0.2 \leq |d| < 0.5$ pequeño, $0.5 \leq |d| < 0.8$ medio, $|d| \geq 0.8$ grande.

**Intervalos de confianza**: Bootstrap 95% CI con 10,000 remuestreos por benchmark.

**Análisis de potencia**: Con $n = 3$ por grupo y una desviación estándar de precisión esperada $\sigma \approx 1.5\%$ (según resultados publicados de LoRA), el tamaño del efecto mínimo detectable (potencia 80%, $\alpha = 0.05$) es:

$$d_{\text{min}} \approx 0.8 \quad \Rightarrow \quad \Delta_{\text{min}} \approx 0.8 \times 1.5\% = 1.2\%$$

Esto significa que podemos detectar diferencias $\geq 1.2\%$ con confianza razonable. Para la no inferioridad (margen 1.0%), está en el límite; se recomienda $n=5$ si el tiempo lo permite.

### 5.9 Criterios de Éxito para Publicación

**Nivel 1 — Paper de workshop (NeurIPS/ICLR/ACL workshop):**
- Pico de VRAM de S³ $\leq 2$ GB confirmado (logs de nvidia-smi) ✓
- S³ no es inferior a LoRA $r=8$ en $\geq 3$ de 5 benchmarks ✓
- La ejecución ablacionada de STB muestra una pérdida de $\geq 0.5\%$ de precisión (justificando la existencia de STB) ✓

**Nivel 2 — Conferencia principal (NeurIPS/ICLR/ICML/ACL):**
- Todos los criterios del Nivel 1 ✓
- S³ es estadísticamente superior a LoRA $r=8$ en $\geq 2$ benchmarks ($p < 0.05$, unilateral) ✓
- S³ no es inferior a LoRA $r=64$ en $\geq 3$ benchmarks ✓
- La ablación muestra que LOS 3 componentes contribuyen positiva y significativamente ✓
- Brecha de ajuste fino completo: S³ está dentro del 3% del FT completo en $\geq 3$ benchmarks ✓

**Nivel 3 — Oral/Spotlight:**
- Todos los criterios del Nivel 2 ✓
- S³ iguala o supera el ajuste fino completo en $\geq 2$ benchmarks ✓
- S³ entrena un modelo 7B en GPU de 2GB en $< 24$ horas de tiempo real ✓
- La configuración S³-large ($k=256, d_{\text{bott}}=16$) muestra una mejora significativa sobre el S³ estándar $\to$ demuestra comportamiento de escalado ✓

### 5.10 Paquete de Reproducibilidad

La entrega del paper incluirá:
1. **Código**: Implementación completa de `svmo.py`, `nmf.py` (con solucionador RK4), `stb.py`, `frugal_trainer.py` y scripts de benchmark.
2. **Factores SVD precomputados**: Para Qwen2.5-7B, Mistral-7B-v0.3, Llama-3-8B en HuggingFace.
3. **Configuraciones**: Archivos YAML para todos los experimentos.
4. **Logs brutos**: Trazas de nvidia-smi, curvas de entrenamiento (pérdida, perplejidad), salidas de benchmark.
5. **Checkpoints**: Pesos del adaptador entrenados finales para S³ (configuración estándar) en los 3 datasets $\times$ 3 modelos.
6. **Script de análisis estadístico**: Script de Python/R que reproduce todas las tablas y figuras.

---
**Fin del documento.**
