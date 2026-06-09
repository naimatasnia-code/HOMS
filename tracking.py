"""
homs_tracking.py
================
HOMS Layer 2 — Parameter Estimation & Real-Time Tracking Pipeline
(Math Engines & Digital Twin)

Implements the two-stage parameter estimation pipeline (Section 5.2):

    Stage 1 — SINDy Structural Identification (pysindy)
        θ* = argmin ‖B̂ − Θ·ξ‖₂² + λ‖ξ‖₁
        Identifies the sparse ODE structure from ≥14 days of wearable data.
        Run once at system initialisation.

    Stage 2 — Extended Kalman Filter (filterpy)
        θ̂(t+Δt) = θ̂(t) + K(t) [y(t) − h(θ̂(t))]
        Tracks parameter drift θ(t) in real time as biomarkers stream in.
        Continuous adaptation: HRV drop → increase rest load;
        circadian misalignment → shift sleep onset.

Integration with the rest of HOMS:
    - Identified θ* from SINDy seeds the EKF prior.
    - EKF-tracked θ̂(t) feeds back into engine.py's Params at each epoch.
    - Enables the Digital Twin to simulate 'X months ahead' scenarios.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Tuple

import pysindy as ps
from filterpy.kalman import ExtendedKalmanFilter


# ─────────────────────────────────────────────────────────────────────────────
# Stage 1 — SINDy Structural Identification
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SINDyResult:
    """Output from Stage 1 SINDy identification."""
    model: ps.SINDy
    coefficients: np.ndarray      # Sparse coefficient matrix ξ
    feature_names: list[str]      # Library function names
    sigma_est: float              # Estimated σ from identified model
    rho_est: float                # Estimated ρ from identified model
    beta_est: float               # Estimated β from identified model

    def summary(self) -> str:
        lines = ["SINDy Identification Result:"]
        lines.append(f"  Estimated σ: {self.sigma_est:.3f}  (ref: 10.0)")
        lines.append(f"  Estimated ρ: {self.rho_est:.3f}  (ref: 28.0)")
        lines.append(f"  Estimated β: {self.beta_est:.3f}  (ref: 2.667)")
        lines.append("  Non-zero library terms identified:")
        for i, name in enumerate(self.feature_names):
            row = self.coefficients[:, i] if self.coefficients.ndim > 1 else []
            if np.any(np.abs(row) > 1e-4):
                lines.append(f"    {name}: {row}")
        return "\n".join(lines)


def run_sindy_identification(
    trajectory: np.ndarray,
    t_eval: np.ndarray,
    threshold: float = 0.1,
    alpha: float = 0.05,
) -> SINDyResult:
    """
    Run SINDy sparse regression to identify ODE structure from biomarker data.

    Architecture document (Section 5.2, Stage 1):
        θ* = argmin ‖B̂ − Θ·ξ‖₂² + λ‖ξ‖₁

    Requires ≥14 days of wearable data (≥1000 time steps at dt=0.02).

    Parameters
    ----------
    trajectory : np.ndarray, shape (N, 3)
        Observed state trajectory [x, y, z]. In production this comes
        from diffusion-map embedding of biomarker observations.
    t_eval : np.ndarray, shape (N,)
        Time axis corresponding to trajectory rows.
    threshold : float
        STLSQ sparsity threshold (default 0.1 — removes terms with
        smaller coefficients from the library).
    alpha : float
        L2 regularisation strength.

    Returns
    -------
    SINDyResult with identified sparse model and extracted parameter estimates.
    """
    if len(trajectory) < 1000:
        raise ValueError(
            f"SINDy requires ≥1000 time steps (≥14 days at dt=0.02); "
            f"got {len(trajectory)}. Accumulate more wearable data."
        )

    # Build polynomial + trig library (captures Lorenz bilinear terms x*y, x*z)
    library = ps.PolynomialLibrary(degree=2, include_interaction=True)

    optimizer = ps.STLSQ(threshold=threshold, alpha=alpha)

    model = ps.SINDy(
        feature_library=library,
        optimizer=optimizer,
    )

    dt = float(t_eval[1] - t_eval[0]) if len(t_eval) > 1 else 0.02
    model.fit(trajectory, t=dt, feature_names=["x", "y", "z"])

    coeff = model.coefficients()           # shape (n_states, n_features)
    feat  = model.get_feature_names()

    # ── Extract σ, ρ, β from identified coefficients ──────────────────────
    # dx/dt ≈ σ(y−x) → coefficient of 'y' in x-equation ≈ σ
    sigma_est = _extract_param(coeff, feat, state_idx=0, term="y")

    # dy/dt ≈ ρx − xz − y → coefficient of 'x' in y-equation ≈ ρ
    rho_est   = _extract_param(coeff, feat, state_idx=1, term="x")

    # dz/dt ≈ xy − βz → coefficient of 'z' in z-equation ≈ −β (flip sign)
    beta_est  = -_extract_param(coeff, feat, state_idx=2, term="z")

    return SINDyResult(
        model=model,
        coefficients=coeff,
        feature_names=feat,
        sigma_est=abs(sigma_est),
        rho_est=abs(rho_est),
        beta_est=abs(beta_est),
    )


def _extract_param(coeff: np.ndarray, feat: list[str], state_idx: int, term: str) -> float:
    """Extract coefficient of a specific feature term for a given state equation."""
    for j, name in enumerate(feat):
        if name == term:
            return float(coeff[state_idx, j])
    return 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Stage 2 — Extended Kalman Filter (EKF) Online Tracking
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class EKFParams:
    """EKF configuration."""
    dim_x: int = 3          # State dimension: [σ, ρ, β] (parameter tracking)
    dim_z: int = 1          # Observation dimension (e.g. x(t) observed via HRV proxy)
    dt: float = 0.02        # Integration timestep
    Q_scale: float = 1e-4   # Process noise covariance scale (parameter drift rate)
    R_scale: float = 0.5    # Measurement noise variance (sensor noise)


class HOMSParameterTracker:
    """
    Extended Kalman Filter for real-time HOMS parameter tracking.

    Tracks the drift of ODE parameters θ = [σ, ρ, β] as biomarkers
    stream in. Each biomarker observation updates the EKF, enabling
    continuous real-time adaptation.

    Architecture document (Section 5.2, Stage 2):
        θ̂(t+Δt) = θ̂(t) + K(t) [y(t) − h(θ̂(t))]

    Practical adaptations:
        - HRV drop detected → EKF infers increased rho (stress load) → signals GA to
          increase recovery weight in next chromosome generation.
        - Circadian misalignment (irregular sleep timestamps) → EKF shifts β estimate
          → triggers recommendation to advance sleep onset.
    """

    def __init__(
        self,
        theta_init: np.ndarray,
        ekf_params: Optional[EKFParams] = None,
    ) -> None:
        """
        Parameters
        ----------
        theta_init : np.ndarray, shape (3,)
            Initial parameter estimate [σ₀, ρ₀, β₀].
            Typically seeded from SINDy identification result.
        ekf_params : EKFParams, optional
        """
        cfg = ekf_params or EKFParams()
        self.cfg = cfg
        self.dt  = cfg.dt

        self.ekf = ExtendedKalmanFilter(dim_x=cfg.dim_x, dim_z=cfg.dim_z)

        # State vector: θ = [σ, ρ, β]
        self.ekf.x = theta_init.copy().reshape(cfg.dim_x, 1).astype(float)

        # Initial uncertainty (parameter estimates have moderate uncertainty)
        self.ekf.P = np.eye(cfg.dim_x) * 1.0

        # Process noise: parameters drift slowly
        self.ekf.Q = np.eye(cfg.dim_x) * cfg.Q_scale

        # Measurement noise: sensor noise on observed proxy signal
        self.ekf.R = np.array([[cfg.R_scale]])

        # State transition: parameters modelled as random walk (I + ε)
        self.ekf.F = np.eye(cfg.dim_x)

        # History storage
        self.theta_history: list[np.ndarray] = [theta_init.copy()]
        self.P_history: list[np.ndarray] = [self.ekf.P.copy()]

    def _h(self, x: np.ndarray) -> np.ndarray:
        """
        Observation function h(θ): maps parameter state to observable.
        Here: proxy = σ (changes in HRV map to changes in sigma).
        Override for multi-observable deployments.
        """
        return x[0:1]   # Observe σ component

    def _H_jacobian(self, x: np.ndarray) -> np.ndarray:
        """Jacobian of h(θ) w.r.t. θ."""
        H = np.zeros((1, self.cfg.dim_x))
        H[0, 0] = 1.0   # ∂h/∂σ = 1
        return H

    def update(self, observation: float) -> np.ndarray:
        """
        Ingest one biomarker observation and update parameter estimates.

        Parameters
        ----------
        observation : float
            Observed proxy value (e.g. normalised HRV reading, glucose level).

        Returns
        -------
        np.ndarray, shape (3,) : Updated parameter estimates [σ̂, ρ̂, β̂]
        """
        # Predict step (parameters follow random walk)
        self.ekf.predict()

        # Update step with new observation
        z = np.array([[float(observation)]])
        self.ekf.update(
            z,
            HJacobian=lambda x: self._H_jacobian(x),
            Hx=lambda x: self._h(x),
        )

        theta_est = self.ekf.x.flatten()
        self.theta_history.append(theta_est.copy())
        self.P_history.append(self.ekf.P.copy())

        return theta_est

    def get_estimates(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return full history arrays: (theta_history, uncertainty_history)."""
        return (
            np.array(self.theta_history),
            np.array([P.diagonal() for P in self.P_history]),
        )

    def current_params(self) -> dict:
        """Return current best-estimate parameters as a dict for engine.py."""
        theta = self.ekf.x.flatten()
        return {
            "sigma": float(abs(theta[0])),
            "rho":   float(abs(theta[1])),
            "beta":  float(abs(theta[2])),
        }

    def drift_alert(self, sigma_threshold: float = 2.0, rho_threshold: float = 5.0) -> list[str]:
        """
        Check for significant parameter drift relative to baseline.

        Returns list of alert messages; empty list if all within normal range.
        Used by the Digital Twin's early-warning detection (λ₁ shift monitor).
        """
        if len(self.theta_history) < 2:
            return []

        baseline = self.theta_history[0]
        current  = self.theta_history[-1]
        alerts   = []

        delta_sigma = abs(current[0] - baseline[0])
        delta_rho   = abs(current[1] - baseline[1])

        if delta_sigma > sigma_threshold:
            alerts.append(
                f"⚠ σ drift detected: Δσ={delta_sigma:.2f} (threshold {sigma_threshold}). "
                "Possible autonomic stress — recommend recovery audit."
            )
        if delta_rho > rho_threshold:
            alerts.append(
                f"⚠ ρ drift detected: Δρ={delta_rho:.2f} (threshold {rho_threshold}). "
                "Inflammatory load increasing — flag for biomarker review."
            )

        return alerts


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline runner — combines Stage 1 + Stage 2
# ─────────────────────────────────────────────────────────────────────────────

def run_full_pipeline(
    trajectory: np.ndarray,
    t_eval: np.ndarray,
    biomarker_stream: Optional[np.ndarray] = None,
    sindy_threshold: float = 0.1,
    ekf_params: Optional[EKFParams] = None,
) -> Tuple[SINDyResult, HOMSParameterTracker]:
    """
    Run the full two-stage parameter estimation pipeline.

    Parameters
    ----------
    trajectory       : np.ndarray (N, 3)  — initialisation trajectory (≥14 days)
    t_eval           : np.ndarray (N,)    — time axis
    biomarker_stream : np.ndarray (M,), optional
        Live biomarker proxy observations for EKF updating.
        If None, tracker is returned ready for streaming updates.
    sindy_threshold  : float — STLSQ sparsity threshold
    ekf_params       : EKFParams, optional

    Returns
    -------
    (SINDyResult, HOMSParameterTracker)
    """
    # Stage 1: SINDy
    print("Stage 1: Running SINDy structural identification...")
    sindy_result = run_sindy_identification(trajectory, t_eval, threshold=sindy_threshold)
    print(sindy_result.summary())

    # Seed EKF from SINDy estimates
    theta_init = np.array([
        sindy_result.sigma_est,
        sindy_result.rho_est,
        sindy_result.beta_est,
    ])
    print(f"\nStage 2: Initialising EKF with θ₀ = {theta_init}")

    tracker = HOMSParameterTracker(theta_init=theta_init, ekf_params=ekf_params)

    # Stage 2: Ingest live stream if provided
    if biomarker_stream is not None:
        print(f"  Processing {len(biomarker_stream)} live observations...")
        for obs in biomarker_stream:
            tracker.update(float(obs))

        alerts = tracker.drift_alert()
        if alerts:
            print("\n" + "\n".join(alerts))
        else:
            print("  No significant parameter drift detected.")

    return sindy_result, tracker


# ─────────────────────────────────────────────────────────────────────────────
# Quick self-test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from scipy.integrate import solve_ivp

    print("=== HOMS Tracking Pipeline Self-Test ===\n")

    # Generate Lorenz trajectory (proxy for 14-day wearable initialisation data)
    def lorenz(t, s, sigma=10, rho=28, beta=8/3):
        x, y, z = s
        return [sigma*(y-x), x*(rho-z)-y, x*y-beta*z]

    dt = 0.02
    T  = 40.0    # Covers > 1000 steps
    t_span = (0, T)
    t_eval = np.arange(0, T, dt)

    sol = solve_ivp(lorenz, t_span, [1.0, 1.0, 1.0], method='RK45',
                    t_eval=t_eval, dense_output=False)
    traj = sol.y.T

    # Simulate biomarker stream (normalised HRV proxy observations)
    rng = np.random.default_rng(0)
    hrv_stream = traj[:200, 0] + rng.normal(0, 0.5, 200)   # 200 live readings

    sindy_res, tracker = run_full_pipeline(
        trajectory=traj,
        t_eval=t_eval,
        biomarker_stream=hrv_stream,
    )

    print("\nFinal EKF parameter estimates:")
    params = tracker.current_params()
    for k, v in params.items():
        print(f"  {k}: {v:.3f}")

    theta_hist, uncert_hist = tracker.get_estimates()
    print(f"\nTracked {len(theta_hist)} EKF updates.")
    print("✓ Tracking pipeline operational.")
