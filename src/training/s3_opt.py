"""
S3-OPT: Optimizaciones Teóricas de Fine-Tuning para VRAM Limitada

Este módulo implementa las 10 optimizaciones teóricas de formalismo_optimizacion.md:

1. SMV  — Spectral Modulation Vector (transferencia vector vs matriz)
2. EMP  — Eigenvalue Migration Predictor (predicción de μ dinámica)
3. TER  — Token Entropy Routing (pasos ODE adaptativos por entropía)
4. MSO  — Manifold Shortcut ODE (atajos lineales en ODE)
5. FDGD — Frequency-Domain Gradient Denoising (filtrado Fourier)
6. GNS  — Gradient-Normalized Skip (skip backward por magnitud)
7. SMA  — Spectral Memory Attention (checkpoints espectrales)
8. HFISC — Hessian-Free Initialization (inicialización por condición)
9. TOWS — Token-wise ODE Warmstarting (warmstart entre tokens)
10. DRA  — Dynamic Rank Adaptation (rank k adaptativo)

Referencias: formalismo_optimizacion.md (S³-OPT, Julio 2026)
"""

import torch
import torch.nn as nn
from typing import Optional, List, Tuple, Callable
from dataclasses import dataclass, field
from abc import ABC, abstractmethod


# =============================================================================
# SMV — Spectral Modulation Vector (Secciones 2.1-2.13)
# =============================================================================

@dataclass
class SMVConfig:
    """Configuración para SMV (Spectral Modulation Vector)."""
    k: int = 128
    dtype_transfer: str = "fp16"  # fp16 o fp8
    correlation_threshold: float = 0.5  # ρ > 0.5 para delta-modulación
    keyframe_interval: int = 32  # resincronización cada N tokens


class SMVTransfer:
    """SMV: Transferencia de vectores en vez de matrices por PCIe.

    Idea central: U_k y V_k^T son invariantes durante training (frozen).
    Solo μ_θ = m_θ(σ) cambia por forward pass. Transferir solo μ
    en vez de las matrices completas reduce PCIe ~99.97%.

    Key insight de SMV-7: si ρ > 0.5 (correlación temporal alta),
    se puede usar delta-modulación (Δμ = μ_t - μ_{t-1}) para
    comprimir aún más: 4-8 bits por componente en vez de 16.

    Usage:
        smv = SMVTransfer(U_k, V_k, config)
        for token in sequence:
            mu = smv.compute_mu(sigma, theta)
            if smv.needs_keyframe():
                smv.send_full_mu(mu)  # keyframe
            else:
                smv.send_delta_mu(mu)  # delta modulation
    """

    def __init__(
        self,
        U_k: torch.Tensor,
        V_k: torch.Tensor,
        config: Optional[SMVConfig] = None,
    ):
        self.U_k = U_k.clone()  # Frozen, GPU resident
        self.V_k = V_k.clone()  # Frozen, GPU resident
        self.config = config or SMVConfig()
        self.k = self.config.k

        # State for delta modulation
        self._mu_prev: Optional[torch.Tensor] = None
        self._token_count = 0
        self._bits_sent = 0

    def compute_mu(
        self,
        sigma: torch.Tensor,
        theta_mlp: nn.Module,
    ) -> torch.Tensor:
        """Computa μ_θ(σ) = σ ⊙ (1 + α·tanh(g_θ(log(σ + ε)))."""
        #σ_j = sigma[j] es el j-ésimo valor singular
        # g_θ: 1→H→H→1 MLP compartido por todos los componentes
        # Esta es la misma computación que SVMO original
        # Retorna μ ∈ R^k
        log_sigma = torch.log(sigma + 1e-8)
        g_out = theta_mlp(log_sigma)  # [k] → [k] (g_θ aplicado componente a componente)
        alpha = 0.3  # hyperparameter SVMO
        mu = sigma * (1 + alpha * torch.tanh(g_out))
        return mu

    def needs_keyframe(self) -> bool:
        """Indica si el próximo frame debe ser keyframe (resincronización)."""
        self._token_count += 1
        if self._token_count >= self.config.keyframe_interval:
            self._token_count = 0
            return True
        return False

    def send_mu(self, mu: torch.Tensor, transport_fn: Callable) -> None:
        """Envía μ via PCIe usando delta-modulación si ρ > threshold."""
        if self.needs_keyframe() or self._mu_prev is None:
            # Keyframe: transferir μ completo
            transport_fn(mu.to(self.U_k.device))
            self._mu_prev = mu.detach().clone()
            self._bits_sent += mu.numel() * 16  # fp16
        else:
            # Delta-modulación: transferir Δμ = μ - μ_prev
            delta_mu = mu - self._mu_prev
            # Cuantizar a 4-8 bits por componente
            delta_quantized = self._quantize_delta(delta_mu)
            transport_fn(delta_quantized.to(self.U_k.device))
            self._bits_sent += delta_quantized.numel() * 8  # ~8 bits
            self._mu_prev = mu.detach().clone()

    def _quantize_delta(self, delta_mu: torch.Tensor) -> torch.Tensor:
        """Cuantiza Δμ a 8 bits (rango dinámico adaptativo)."""
        # Escala por el máximo absoluto para aprovechar rango
        scale = torch.max(torch.abs(delta_mu)) + 1e-8
        delta_normalized = delta_mu / scale
        # Cuantizar a [-127, 127] (int8)
        delta_int = torch.round(delta_normalized * 127).to(torch.int8)
        # Empaquetar con scale para reconstrucción
        return delta_int  # Solo los valores cuantizados; scale se transfiere por separado

    def reconstruction_error(self, mu: torch.Tensor, mu_prev: torch.Tensor) -> float:
        """Calcula error de reconstrucción de delta-modulación."""
        delta = mu - mu_prev
        return torch.norm(delta).item() / (torch.norm(mu).item() + 1e-8)

    @property
    def compression_ratio(self) -> float:
        """Ratio de compresión efectivo (bits usados / bits fp16)."""
        return self._bits_sent / (self._token_count * self.k * 16 + 1e-8)


# =============================================================================
# EMP — Eigenvalue Migration Predictor (Secciones 3.1-3.12)
# =============================================================================

@dataclass
class EMPConfig:
    """Configuración para EMP (Eigenvalue Migration Predictor)."""
    k: int = 128
    predictor_type: str = "kalman"  # "kalman" | "ema" | "mlp"
    use_adam_moments: bool = True  # Usar v_t de Adam como proxy de H
    prediction_horizon: int = 32  # Pasos a predecir
    stability_threshold: float = 0.99  # ρ(A) debe ser < esto


class EMPPredictor:
    """EMP: Predice μ_{t+1} desde μ_t sin ejecutar MLP.

    Idea central: μ cambia suavemente bajo SGD con momentum. Modelamos
    la dinámica como sistema LTI: μ_{t+1} = A·μ_t + b + w_t.

    Si ρ(A) < 1, el predictor converge y podemos usar μ_predicho
    en vez de μ_MLP para la mayoría de los forward passes.

    EMP-2 (key insight): Adam storea v_t ≈ diag(H). Si A ≈ I - η·diag(v_t),
    podemos actualizar A sin costo O(k³) usando los momentos de Adam.

    Usage:
        emp = EMPPredictor(k=128, use_adam_moments=True)
        for step in training:
            # Durante optimizer step, actualizar A con Adam moments
            emp.update_A_from_adam(v_t)
            # Durante forward, predecir en vez de ejecutar MLP
            if emp.is_stable():
                mu_pred = emp.predict(mu_t)
            else:
                mu_pred = execute_mlp(sigma, theta)
    """

    def __init__(self, config: Optional[EMPConfig] = None):
        self.config = config or EMPConfig()
        self.k = self.config.k

        # Modelo de transición: μ_{t+1} = A·μ_t + b
        self.A = torch.eye(self.k)  # A ≈ I - η·diag(H)
        self.b = torch.zeros(self.k)
        self.rho_A = 0.0  # Radio espectral estimado

        # Kalman filter state para predicción
        self.P = torch.eye(self.k)  # Covarianza del error
        self.Q = torch.eye(self.k) * 0.01  # Ruido de proceso (innovation)
        self.R = torch.eye(self.k) * 0.1  # Ruido de medición

        # Historial de μ para tracking
        self._mu_history: List[torch.Tensor] = []

        # Estimación online de ρ(A)
        self._stability_ok = False

    def update_A_from_adam(
        self,
        v_t: torch.Tensor,
        eta: float = 1e-3,
    ) -> None:
        """Actualiza A usando momentos de segundo orden de Adam como proxy de H.

        v_t ≈ diag(F) donde F es Fisher Information Matrix.
        Para losses cuadráticas, F = H. Para cross-entropy, F ≈ H solo
        cerca del óptimo. El régimen de validez es asintótico.

        A_diag ≈ I - η·v_t
        """
        if not self.config.use_adam_moments:
            return

        if v_t.shape[0] != self.k:
            return  # Dimensionality mismatch, skip

        # A ≈ I - η·diag(v_t)
        self.A = torch.eye(self.k) - eta * torch.diag(v_t)

        # Estimar radio espectral (approximación: máximo eigenvalor)
        eigenvalues = torch.linalg.eigvalsh(self.A)
        self.rho_A = torch.max(torch.abs(eigenvalues)).item()

        # Estabilidad: ρ(A) < 1 para convergencia
        self._stability_ok = self.rho_A < self.config.stability_threshold

    def predict(self, mu_t: torch.Tensor) -> torch.Tensor:
        """Predice μ_{t+1} = A·μ_t + b."""
        return self.A @ mu_t + self.b

    def update(
        self,
        mu_observed: torch.Tensor,
        mu_predicted: Optional[torch.Tensor] = None,
    ) -> float:
        """Actualiza el modelo con la observación real del predictor.

        Si mu_predicted es None, usamos la predicción del modelo.
        Retorna el error de predicción normalizado.
        """
        if mu_predicted is None:
            mu_predicted = self.predict(mu_observed)

        error = mu_observed - mu_predicted
        error_norm = torch.norm(error).item() / (torch.norm(mu_observed).item() + 1e-8)

        # Kalman update
        S = self.A @ self.P @ self.A.T + self.Q
        K = self.P @ S.inverse()  # Ganancia de Kalman
        self.b = self.b + K @ error
        self.A = self.A + torch.outer(K, error)  # Actualización rango-1
        self.P = (torch.eye(self.k) - K @ self.A) @ self.P

        # Guardar historial
        self._mu_history.append(mu_observed.detach().clone())
        if len(self._mu_history) > 100:
            self._mu_history.pop(0)

        return error_norm

    def is_stable(self) -> bool:
        """Retorna True si el sistema es contractivo (ρ(A) < 1)."""
        return self._stability_ok

    def skip_probability(self) -> float:
        """Probabilidad de skip del MLP basada en estabilidad."""
        if not self._stability_ok:
            return 0.0
        # Mientras más pequeño sea ρ(A), más seguro es skippar
        margin = self.config.stability_threshold - self.rho_A
        return float(max(0.0, min(1.0, margin * 10)))


# =============================================================================
# TER — Token Entropy Routing (Secciones 4.1-4.11)
# =============================================================================

@dataclass
class TERConfig:
    """Configuración para TER (Token Entropy Routing)."""
    d_bottleneck: int = 8  # Dimensión de bottleneck del NMF
    threshold_low: float = 1.0  # H < τ1 → N=1
    threshold_high: float = 2.0  # H > τ2 → N=4
    use_hysteresis: bool = True
    hysteresis_margin: float = 0.2


class TERRouter:
    """TER: Routing adaptativo del número de pasos ODE por entropía del token.

    Idea central: tokens con baja entropía (alta confianza) necesitan
    menos pasos de integración ODE. Tokens con alta entropía (ambiguos)
    necesitan más pasos.

    H_NMF(h) = -Σ a_j·log(a_j + ε) donde a = softmax(f_θ(h, t)).

    TER-4: H_NMF sesga sistemáticamente hacia arriba por O(1/N).
    Esto es MEDIBLE, mientras M_4 (derivada 4ta de f_θ) no lo es.
    Por eso el sesgo es útil en la práctica.

    Usage:
        ter = TERRouter(config)
        for token in sequence:
            H = ter.compute_entropy(nmf_bottleneck_activation)
            N = ter.route(H)
            nmf.set_N_steps(N)  # 1, 2, o 4
    """

    def __init__(self, config: Optional[TERConfig] = None):
        self.config = config or TERConfig()
        self.d_b = self.config.d_bottleneck
        self.eps = 1e-8

        # Hysteresis state
        self._current_N = 4
        self._history: List[float] = []

    def compute_entropy(self, bottleneck_activation: torch.Tensor) -> float:
        """Computa H_NMF(h) = -Σ softmax(a)_j · log(softmax(a)_j + ε).

        La activación de la bottleneck layer (pre-softmax) es a ∈ R^{d_b}.
        """
        a = bottleneck_activation
        # Softmax
        a_exp = torch.exp(a - torch.max(a))  # Estabilidad numérica
        a_softmax = a_exp / (torch.sum(a_exp) + self.eps)

        # Entropy: -Σ p_j · log(p_j)
        H = -torch.sum(a_softmax * torch.log(a_softmax + self.eps))
        return H.item()

    def route(self, H: float) -> int:
        """Decide N ∈ {1, 2, 4} basado en entropía H.

        Usa histéresis para evitar oscilación: necesita cruzar threshold
        por margen para cambiar de decisión.
        """
        if not self.config.use_hysteresis:
            return self._route_simple(H)

        self._history.append(H)
        if len(self._history) > 10:
            self._history.pop(0)

        prev_N = self._current_N

        if self._current_N == 1:
            # Solo sube si supera threshold_high + margin
            if H > self.config.threshold_high + self.config.hysteresis_margin:
                self._current_N = 2
        elif self._current_N == 2:
            if H < self.config.threshold_low - self.config.hysteresis_margin:
                self._current_N = 1
            elif H > self.config.threshold_high + self.config.hysteresis_margin:
                self._current_N = 4
        else:  # N == 4
            # Solo bajo si supera threshold_low + margin
            if H < self.config.threshold_low - self.config.hysteresis_margin:
                self._current_N = 2

        return self._current_N

    def _route_simple(self, H: float) -> int:
        """Routing sin histéresis (para comparación)."""
        if H < self.config.threshold_low:
            return 1
        elif H < self.config.threshold_high:
            return 2
        return 4

    @property
    def expected_N(self) -> float:
        """N esperado basado en distribución de entropías (estimado)."""
        return float(self._current_N)


# =============================================================================
# MSO — Manifold Shortcut ODE (Secciones 5.1-5.10)
# =============================================================================

@dataclass
class MSOConfig:
    """Configuración para MSO (Manifold Shortcut ODE)."""
    T: float = 1.0
    m: int = 4  # Pasos RK4 por segmento
    curvature_threshold: float = 0.01  # κ < κ_thresh → shortcut
    velocity_threshold: float = 0.1  # ||f_θ|| < v_thresh → shortcut


class MSOController:
    """MSO: Detecta segmentos de trayectoria lineal para shortcuts.

    Idea central: si la ODE tiene curvatura baja entre t0 y t1,
    la interpolación lineal es buena aproximación y podemos
    skippar los pasos RK4 intermedios.

    MSO-1: error de shortcut ≤ M·L·T²/8 donde M = max||f_θ||, L = Lipschitz de f_θ.

    MSO-2: proxy de curvatura computable online sin Hessian completa:
    κ_hat = ||f_θ(h + δ·f_θ) - f_θ(h)|| / δ

    Usage:
        mso = MSOController(config)
        for segment in trajectory:
            if mso.is_linear(segment):
                h_end = mso.shortcut(h_start, h_end)  # Interpolación lineal
            else:
                h_end = mso.integrate(h_start)  # RK4 completo
    """

    def __init__(self, config: Optional[MSOConfig] = None):
        self.config = config or MSOConfig()
        self.T = self.config.T
        self.m = self.config.m
        self.dt = self.T / self.m

    def detect_shortcut(
        self,
        f_theta: Callable,
        h: torch.Tensor,
        t: float = 0.0,
        delta: float = 1e-3,
    ) -> bool:
        """Detecta si la trayectoria es aproximadamente lineal en [t, t+T].

        Usa proxy de curvatura: κ_hat = ||f_θ(h + δ·f_θ) - f_θ(h)|| / δ

        Si κ_hat < κ_thresh Y ||f_θ|| < v_thresh, shortcut es válido.
        """
        with torch.no_grad():
            f_h = f_theta(h, t)
            f_h_norm = torch.norm(f_h).item()

            if f_h_norm < self.config.velocity_threshold:
                # Velocidad pequeña → trayectoria plana
                return True

            h_plus = h + delta * f_h
            f_h_plus = f_theta(h_plus, t)
            kappa_hat = torch.norm(f_h_plus - f_h).item() / delta

            return kappa_hat < self.config.curvature_threshold

    def shortcut_interpolate(
        self,
        h_start: torch.Tensor,
        h_end: torch.Tensor,
        s: float = 0.5,
    ) -> torch.Tensor:
        """Interpolación lineal entre estados: h(s) = (1-s)·h_start + s·h_end.

        Para shortcut completo (s=1), retorna h_end.
        Para midpoint (s=0.5), retorna el punto medio.
        """
        return (1 - s) * h_start + s * h_end

    def flops_fraction(self, alpha: float = 0.4) -> float:
        """Fracción de FLOPs ahorrados por MSO.

        α = fracción de segmentos con shortcut activo.
        Con m=4: shortcut usa 8 evals vs 16 para RK4 completo.
        """
        if alpha == 0:
            return 0.0
        # Costo original: 4m evals. Costo con shortcut: 8 evals.
        # Ahorro por segmento: (4m - 8) / 4m = 1 - 2/m
        return alpha * (1 - 2 / self.m)


# =============================================================================
# FDGD — Frequency-Domain Gradient Denoising (Secciones 6.1-6.6)
# =============================================================================

@dataclass
class FDGDConfig:
    """Configuración para FDGD (Frequency-Domain Gradient Denoising)."""
    omega_cutoff: float = 0.1  # Frecuencia de corte normalizada
    filter_order: int = 4  # Orden del filtro Butterworth
    P: int = 1000000  # Número de parámetros (para FFT)
    use_adaptive_q: bool = True


class FDGDFilter:
    """FDGD: Filtra gradientes en dominio de Fourier para convergencia acelerada.

    Idea central: eliminar componentes de alta frecuencia del gradiente
    antes del optimizer step reduce oscilaciones.

    FDGD-1: con filtro paso-bajo y energía preservada ≥ ρ, la convergencia
    tiene factor adicional 1/ρ en el bound de ruido.

    FDGD-2: cutoff óptimo para espectro tipo Lorentziana escala como
    ω* ≈ (λ/P)·log(1/(1-ρ_min)).

    Connection con EMP: el ruido de alta frecuencia filtrado puede
    alimentar Q (covarianza de innovación) si Q es adaptativa.

    Usage:
        fdgd = FDGDFilter(config)
        for step in training:
            g_raw = compute_gradient()
            g_filtered = fdgd.filter(g_raw)
            optimizer.step(g_filtered)
    """

    def __init__(self, config: Optional[FDGDConfig] = None):
        self.config = config or FDGDConfig()
        self.omega_c = self.config.omega_cutoff
        self.n = self.config.filter_order

        # Adaptivity state
        self._Q_adaptive: Optional[torch.Tensor] = None
        self._energy_history: List[float] = []

    def filter(self, g: torch.Tensor) -> torch.Tensor:
        """Aplica filtro paso-bajo Butterworth en dominio de Fourier.

        Filtro: F(ω) = 1 / (1 + (ω/ω_c)^{2n})

        El gradiente g es flatten a R^P y transformado con FFT.
        """
        P = g.numel()
        g_flat = g.flatten()

        # FFT
        g_fft = torch.fft.fft(g_flat)
        freqs = torch.fft.fftfreq(P)

        # Filter response (simétrico)
        omega_abs = torch.abs(freqs)
        filter_response = 1.0 / (1.0 + (omega_abs / self.omega_c) ** (2 * self.n))

        # Aplicar filtro
        g_fft_filtered = g_fft * filter_response

        # IFFT
        g_filtered = torch.fft.ifft(g_fft_filtered).real

        # Guardar energía para adaptividad
        energy = torch.norm(g_filtered).item() / (torch.norm(g_flat).item() + 1e-8)
        self._energy_history.append(energy)
        if len(self._energy_history) > 100:
            self._energy_history.pop(0)

        return g_filtered.reshape(g.shape)

    def energy_preservation(self) -> float:
        """Fracción de energía preservada después del filtro."""
        if not self._energy_history:
            return 1.0
        return sum(self._energy_history) / len(self._energy_history)

    def update_Q_adaptive(self, g_high_freq: torch.Tensor) -> torch.Tensor:
        """Actualiza Q (matriz de covarianza) con energía de alta frecuencia.

        Q debe ser adaptativa: en fase temprana hay más energía de alta
        frecuencia, en fase tardía menos. Q_t = α(t)·diag(g_high²).

        Esto conecta con EMP: el ruido filtrado por FDGD alimenta la
        covarianza de innovación del predictor.
        """
        if not self.config.use_adaptive_q:
            return self._Q_adaptive or torch.eye(128) * 0.01

        # g_high = componentes filtradas (alta frecuencia)
        g_squared = g_high_freq ** 2
        Q_new = torch.diag(g_squared.mean(dim=0) if g_high_freq.dim() > 1 else g_squared)

        # Suavizado exponencial
        if self._Q_adaptive is None:
            self._Q_adaptive = Q_new
        else:
            beta = 0.9
            self._Q_adaptive = beta * self._Q_adaptive + (1 - beta) * Q_new

        return self._Q_adaptive


# =============================================================================
# GNS — Gradient-Normalized Skip (Secciones 7.1-7.9)
# =============================================================================

@dataclass
class GNSConfig:
    """Configuración para GNS (Gradient-Normalized Skip)."""
    epsilon_skip: float = 0.1  # ρ_i < ε → skip
    window_size: int = 10  # Media móvil para suavizado
    use_adaptive_threshold: bool = True


class GNSController:
    """GNS: Skip del backward pass cuando el gradiente de la capa es pequeño.

    Idea central: si ||∂L/∂θ_i|| << max_j ||∂L/∂θ_j||, la capa está
    cerca de convergencia local y actualizar sus pesos es redundante.

    GNS-1: el error acumulado por skip escala como η·δ·√T, donde δ es
    la norma máxima de gradientes skippeados.

    GNS-4: bajo SGD con momentum y loss strongly convex, ||g_i|| decrece
    monotónicamente en esperanza: E[||g_i^{(t)}||] ≤ (1-ημ)^t·||g_i^0|| + ηLσ/μ.

    Usage:
        gns = GNSController(config)
        for layer in model.layers:
            grad_norm = gns.compute_grad_norm(layer)
            if gns.should_skip(grad_norm, layer_id):
                gns.skip_backward(layer)  # No computing gradients
            else:
                compute_backward(layer)
    """

    def __init__(self, config: Optional[GNSConfig] = None):
        self.config = config or GNSConfig()
        self.eps = self.config.epsilon_skip

        # State
        self._grad_norms: dict = {}  # layer_id → history of norms
        self._skip_counts: dict = {}
        self._total_counts: dict = {}
        self._current_skipped: set = set()

    def compute_grad_norm(self, grad: torch.Tensor) -> float:
        """Computa ||∂L/∂θ|| para una capa."""
        return torch.norm(grad).item()

    def should_skip(self, grad_norm: float, layer_id: str) -> bool:
        """Determina si el backward de esta capa debe ser skippeado.

        Usa media móvil de normas de gradiente para evitar skip falsos
        debido a ruido de minibatch.
        """
        if layer_id not in self._grad_norms:
            self._grad_norms[layer_id] = []
            self._skip_counts[layer_id] = 0
            self._total_counts[layer_id] = 0

        history = self._grad_norms[layer_id]
        history.append(grad_norm)
        if len(history) > self.config.window_size:
            history.pop(0)

        self._total_counts[layer_id] += 1

        # Media móvil
        avg_norm = sum(history) / len(history)

        # Comparar con máximo global (para normalizar)
        all_norms = [max(self._grad_norms.get(l, [0])) for l in self._grad_norms]
        max_norm = max(all_norms) if all_norms else 1.0

        rho = avg_norm / (max_norm + 1e-8)

        skip = rho < self.eps
        if skip:
            self._skip_counts[layer_id] += 1
            self._current_skipped.add(layer_id)
        else:
            self._current_skipped.discard(layer_id)

        return skip

    def get_skip_fraction(self, layer_id: str) -> float:
        """Fracción de backward passes saltados para esta capa."""
        if self._total_counts.get(layer_id, 0) == 0:
            return 0.0
        return self._skip_counts.get(layer_id, 0) / self._total_counts[layer_id]

    def flops_saved(self) -> float:
        """Fracción total de FLOPs de backward salvados."""
        total_skip = sum(self._skip_counts.values())
        total_all = sum(self._total_counts.values())
        if total_all == 0:
            return 0.0
        return total_skip / total_all

    def reset(self) -> None:
        """Reset counters for new epoch."""
        self._grad_norms.clear()
        self._skip_counts.clear()
        self._total_counts.clear()
        self._current_skipped.clear()


# =============================================================================
# SMA — Spectral Memory Attention (Secciones 8.1-8.1)
# =============================================================================

@dataclass
class SMAConfig:
    """Configuración para SMA (Spectral Memory Attention)."""
    k: int = 128  # Rank de la proyección espectral
    d: int = 4096  # Dimensión completa


class SMACheckpointManager:
    """SMA: Almacena proyecciones espectrales U_k^T·h en vez de h completo.

    Idea central: el checkpoint para backward es h ∈ R^d. Pero podemos
    guardar solo s = U_k^T·h ∈ R^k (proyección espectral) y reconstruir
    h_approx = U_k·s durante backward.

    SMA-1: error de reconstrucción ≤ ||x|| · √(Σ_{j>k} σ_j²)

    SMA-2: overhead de reconstrucción es 2kd operaciones (~6.25% para k=128, d=4096).

    SMA-4: factor de compresión = 2k/d (6.25% del storage para k=128, d=4096).

    Usage:
        sma = SMACheckpointManager(U_k, config)
        # Forward: save spectral projection
        s = sma.save_checkpoint(h)
        # Backward: reconstruct from projection
        h_reconstructed = sma.reconstruct(s)
    """

    def __init__(
        self,
        U_k: torch.Tensor,
        config: Optional[SMAConfig] = None,
    ):
        self.U_k = U_k.clone()
        self.U_k_t = U_k.t().clone()
        self.config = config or SMAConfig()
        self.k = self.config.k
        self.d = self.config.d

        # Checkpoint storage
        self._checkpoints: List[torch.Tensor] = []

    def save_checkpoint(self, h: torch.Tensor) -> torch.Tensor:
        """Guarda proyección espectral s = U_k^T·h ∈ R^k."""
        s = self.U_k_t @ h  # [d] → [k]
        self._checkpoints.append(s.detach())
        return s

    def reconstruct(self, s: torch.Tensor) -> torch.Tensor:
        """Reconstruye h_approx = U_k·s ∈ R^d desde proyección."""
        return self.U_k @ s

    def reconstruct_batch(self, s_batch: torch.Tensor) -> torch.Tensor:
        """Reconstruye para batch: s_batch ∈ R^{B×k} → h_batch ∈ R^{B×d}."""
        return s_batch @ self.U_k_t

    def compression_ratio(self) -> float:
        """Fracción de storage usado vs guardar h completo."""
        return (2 * self.k) / self.d  # 2k floats vs d floats

    @property
    def overhead_fraction(self) -> float:
        """Overhead de compute en backward como fracción de backward completo."""
        return (2 * self.k * self.d) / (self.d * self.d)  # 2kd / d² = 2k/d


# =============================================================================
# HFISC — Hessian-Free Initialization via Spectral Condition (Secciones 8.2.1-8.2.4)
# =============================================================================

@dataclass
class HFISCConfig:
    """Configuración para HFISC (Hessian-Free Initialization)."""
    scale_mode: str = "hfisc"  # "xavier" | "kaiming" | "hfisc"
    d_in: int = 1
    d_out: int = 1


class HFISCInitializer:
    """HFISC: Inicialización óptima para MLP de modulación basada en condición.

    Idea central: inicializar los pesos del MLP para que la Hessiana de
    m_θ sea cercana a identidad → número de condición κ ≈ 1.

    HFISC-2: W_1^{(0)} = √(2/d_in) · U, W_2^{(0)} = √(2/d_out) · V^T
    donde U, V son ortogonales.

    HFISC-4: para GELU, la escala óptima es √(2/d), no 1/√d (Xavier)
    ni 2/d (Kaiming para ReLU).

    HFISC-1: convergencia con κ ≈ 1 es ~2μ/κ más rápida.

    Usage:
        hfisc = HFISCInitializer(config)
        mlp = build_modulation_mlp()
        hfisc.initialize(mlp)  # Aplica init óptima
    """

    def __init__(self, config: Optional[HFISCConfig] = None):
        self.config = config or HFISCConfig()

    def initialize(self, mlp: nn.Module) -> None:
        """Aplica inicialización óptima a las capas del MLP.

        El MLP de modulación es 1→H→H→1 (por cada componente σ_j).
        HFISC aplica a W_1 y W_2.
        """
        for name, param in mlp.named_parameters():
            if "weight" in name:
                d_out, d_in = param.shape
                if self.config.scale_mode == "hfisc":
                    # HFISC: escala √(2/d) con ortogonalidad
                    scale = (2.0 / d_in) ** 0.5
                elif self.config.scale_mode == "xavier":
                    scale = (1.0 / d_in) ** 0.5
                elif self.config.scale_mode == "kaiming":
                    scale = (2.0 / d_in) ** 0.5
                else:
                    scale = 1.0

                # Inicializar con ortogonalidad parcial
                with torch.no_grad():
                    param.copy_((torch.randn_like(param) * scale))
                    # Hacer filas aproximadamente ortonormales
                    Q, _ = torch.linalg.qr(param.T)
                    param.copy_(Q.T * scale)


# =============================================================================
# TOWS — Token-wise ODE Warmstarting (Secciones 8.3-8.3)
# =============================================================================

@dataclass
class TOWSConfig:
    """Configuración para TOWS (Token-wise ODE Warmstarting)."""
    T: float = 1.0
    L_estimate: float = 0.1  # Lipschitz estimada de f_θ
    coherence_threshold: float = 0.11  # Δh·e^{LT} > threshold → no warmstart


class TOWSWarmstarter:
    """TOWS: Usa warmstart entre tokens consecutivos para MLP modules.

    Idea central: para tokens consecutivos, las activaciones del MLP
    tienden a ser similares porque el MLP opera token-by-token sin
    atención cruzada. Usamos h(t)_MLP del token t como warmstart
    para el token t+1.

    TOWS-1: error de warmstart ≤ Δh·e^{LT} donde Δh = ||h_0^{(t+1)} - h_0^{(t)}||.

    TOWS-1 CORREGIDO: la cota puntual correcta es e^{LT}, no (e^{LT}-1)/(LT).
    La cota (e^{LT}-1)/(LT) aplica al error integrado, no al puntual.

    Connection con TER: si el error de warmstart explota, es señal de
    cambio de régimen semántico → activar TER con más pasos RK4.

    Usage:
        tows = TOWSWarmstarter(config)
        for token in sequence:
            if tows.should_warmstart(h_prev, h_current):
                nmf.set_warmstart(tows.get_warmstart_state())
            else:
                nmf.set_coldstart()
    """

    def __init__(self, config: Optional[TOWSConfig] = None):
        self.config = config or TOWSConfig()
        self.T = self.config.T
        self.L = self.config.L_estimate

        # Warmstart state from previous token
        self._warmstart_h: Optional[torch.Tensor] = None
        self._warmstart_tau: float = 0.0

    def should_use_warmstart(
        self,
        h_prev: torch.Tensor,
        h_current: torch.Tensor,
    ) -> bool:
        """Determina si usar warmstart o cold-start.

        Warmstart es válido si Δh·e^{LT} < threshold.

        Si el error explota (tokens muy diferentes), cold-start.
        """
        delta_h = torch.norm(h_current - h_prev).item()
        error_bound = delta_h * torch.exp(torch.tensor(self.L * self.T)).item()

        return error_bound < self.config.coherence_threshold

    def get_warmstart_state(self) -> Optional[torch.Tensor]:
        """Retorna estado de warmstart para ODE."""
        return self._warmstart_h

    def set_warmstart(self, h_final: torch.Tensor) -> None:
        """Guarda estado final como warmstart para próximo token."""
        self._warmstart_h = h_final.detach().clone()

    def update_coherence_signal(
        self,
        h_prev: torch.Tensor,
        h_current: torch.Tensor,
    ) -> float:
        """Retorna señal de coherencia = Δh·e^{LT}.

        Si esta señal explota, indica cambio de tema/idioma/etc.
        Usado por TER para activar más pasos RK4.
        """
        delta_h = torch.norm(h_current - h_prev).item()
        coherence = delta_h * torch.exp(torch.tensor(self.L * self.T)).item()
        return coherence


# =============================================================================
# DRA — Dynamic Rank Adaptation (Secciones 8.4-8.4)
# =============================================================================

@dataclass
class DRAConfig:
    """Configuración para DRA (Dynamic Rank Adaptation)."""
    k_initial: int = 128
    k_min: int = 32
    k_max: int = 256
    s_threshold: float = 0.05  # s(t) = σ_{k+1}/Σσ_i < δ → reducir k
    n_patience: int = 10  # Pasos consecutivos para cambiar k


class DRAController:
    """DRA: Adapta dinámicamente el rank k de SVD durante training.

    Idea central: si los valores singulares de orden alto contribuyen poco
    (σ_{k+1} << Σσ_i), reducir k temporalmente para ahorrar compute.
    Si hay más energía en componentes de alto orden, aumentar k.

    DRA-1: reducción de FLOPs = 1 - (k'/k)² cuando k' < k.

    DRA-3: s(t) < δ por n_patience pasos → cambio de k estable.

    NOTA: s(t) depende de W (fija), NO de θ (entrenable). La estabilidad
    del rank no implica convergencia de la capa.

    Connection con GNS: DRA identifica capas de alta complejidad
    (s(t) alto), GNS identifica capas de bajo gradiente. Juntos
    permiten redistribución dinámica de FLOPs.

    Usage:
        dra = DRAController(config)
        for step in training:
            k_eff = dra.compute_effective_rank(sigma)
            if dra.should_adjust(k_eff):
                new_k = dra.adjust_k(k_eff)
                svmo.set_rank(new_k)
    """

    def __init__(self, config: Optional[DRAConfig] = None):
        self.config = config or DRAConfig()
        self.k_current = self.config.k_initial

        # Tracking
        self._s_history: List[float] = []
        self._consecutive_below_threshold = 0
        self._k_history: List[int] = [self.k_current]

    def compute_s(self, sigma: torch.Tensor) -> float:
        """Computa s(t) = σ_{k+1} / Σσ_i.

        Mide la fracción de energía en el (k+1)-ésimo valor singular.
        """
        k = self.k_current
        if k >= len(sigma):
            return 1.0  # Edge case

        sigma_k1 = sigma[k].item()  # (k+1)-ésimo
        sigma_sum = sigma[:k].sum().item() + sigma[k:].sum().item()
        if sigma_sum < 1e-10:
            return 1.0

        return sigma_k1 / sigma_sum

    def should_adjust(self, s: float) -> bool:
        """Determina si el rank debe ajustarse basado en s(t)."""
        self._s_history.append(s)
        if len(self._s_history) > self.config.n_patience:
            self._s_history.pop(0)

        if s < self.config.s_threshold:
            self._consecutive_below_threshold += 1
        else:
            self._consecutive_below_threshold = 0

        # Adjust si below threshold por n_patience pasos consecutivos
        return self._consecutive_below_threshold >= self.config.n_patience

    def adjust_k(self, s: float) -> int:
        """Ajusta k basándose en la señal s(t)."""
        if s < self.config.s_threshold and self.k_current > self.config.k_min:
            # Reducir k (componentes de alto orden contribuyen poco)
            self.k_current = max(self.k_current // 2, self.config.k_min)
        elif s > self.config.s_threshold * 2 and self.k_current < self.config.k_max:
            # Aumentar k (hay más energía en componentes de alto orden)
            self.k_current = min(self.k_current * 2, self.config.k_max)

        self._k_history.append(self.k_current)
        return self.k_current

    def flops_fraction(self) -> float:
        """Fracción de FLOPs salvados por reducción de k."""
        k_ratio = self.k_current / self.config.k_initial
        return 1.0 - k_ratio ** 2


# =============================================================================
# Unified S3OPT Interface
# =============================================================================

class S3OPTOptimizer:
    """Interfaz unificada para todas las optimizaciones S3-OPT.

    Combina las 10 optimizaciones en un sistema orquestado donde:
    - SMV reduce transferencia PCIe
    - EMP predice μ para skip de MLP
    - TER rotea pasos ODE por entropía
    - GNS skippea backward por magnitud
    - TOWS detecta coherencia semántica
    - DRA adapta rank dinámicamente
    - SMA comprime checkpoints
    - FDGD filtra gradientes
    - MSO detecta shortcuts lineales
    - HFISC inicializa óptimamente

    Usage:
        opt = S3OPTOptimizer(model, config)
        for epoch in range(epochs):
            opt.set_epoch(epoch)
            for batch in data:
                output = opt.forward(input)
                loss.backward()
                opt.backward()
                opt.step()
    """

    def __init__(
        self,
        model: nn.Module,
        smv_config: Optional[SMVConfig] = None,
        emp_config: Optional[EMPConfig] = None,
        ter_config: Optional[TERConfig] = None,
        gns_config: Optional[GNSConfig] = None,
        dra_config: Optional[DRAConfig] = None,
        tows_config: Optional[TOWSConfig] = None,
        sma_config: Optional[SMAConfig] = None,
        fdgd_config: Optional[FDGDConfig] = None,
        mso_config: Optional[MSOConfig] = None,
        hfisc_config: Optional[HFISCConfig] = None,
    ):
        self.model = model

        # Initialize all optimizers
        self.smv = SMVTransfer(
            model.U_k, model.V_k, smv_config or SMVConfig()
        )
        self.emp = EMPPredictor(emp_config or EMPConfig())
        self.ter = TERRouter(ter_config or TERConfig())
        self.gns = GNSController(gns_config or GNSConfig())
        self.dra = DRAController(dra_config or DRAConfig())
        self.tows = TOWSWarmstarter(tows_config or TOWSConfig())
        self.sma = SMACheckpointManager(
            model.U_k, sma_config or SMAConfig()
        )
        self.fdgd = FDGDFilter(fdgd_config or FDGDConfig())
        self.mso = MSOController(mso_config or MSOConfig())
        self.hfisc = HFISCInitializer(hfisc_config or HFISCConfig())

    def set_epoch(self, epoch: int) -> None:
        """Actualiza configuraciones dependientes de epoch."""
        pass

    def flops_summary(self) -> dict:
        """Resumen de ahorros de FLOPs teóricos."""
        return {
            "smv": {"description": "Δ-modulación de μ", "savings": "~75% PCIe"},
            "emp": {"description": "Predicción de μ desde Adam", "savings": "~99% MLP skip"},
            "ter": {"description": "Routing por entropía", "savings": "~50% ODE steps"},
            "gns": {"description": "Skip por gradiente normalizado", "savings": f"{self.gns.flops_saved():.1%}"},
            "dra": {"description": "Rank adaptativo", "savings": f"{self.dra.flops_fraction():.1%}"},
            "tows": {"description": "Warmstart entre tokens", "savings": "~25% ODE compute"},
            "sma": {"description": "Checkpoints espectrales", "savings": "~93.75% storage"},
            "fdgd": {"description": "Filtrado Fourier", "savings": "Estabilización"},
            "mso": {"description": "Shortcuts lineales", "savings": "~20% ODE"},
            "hfisc": {"description": "Inicialización óptima", "savings": "Convergencia más rápida"},
        }