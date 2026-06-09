"""
homs_health_state.py
====================
HOMS Layer 3 — Health State Function H(x)
(Math Engines & Digital Twin)

Computes the composite health score H(x) from a trajectory matrix
as defined in Andrea's corrected architecture document (Section 5.3, Eq. 21):

    H(x) = w1 * 1/(1 + lambda_1) + w2 * 1/(1 + VA) + w3 * 1/(1 + D2) - w4 * Phi(x)

where:
    lambda_1 : Maximal Lyapunov Exponent  (nolds.lyap_r)
    VA        : Attractor volume           (convex hull of 3D samples)
    D2        : Correlation dimension       (nolds.corr_dim)
    Phi(x)   : Biomarker penalty (see homs_constraints.py)

This module is the primary fitness component imported by the GA optimizer.
"""

from __future__ import annotations

import numpy as np
import nolds
from dataclasses import dataclass, field
from typing import Optional
from scipy.spatial import ConvexHull


# ─────────────────────────────────────────────────────────────────────────────
# Weight container (three-tier calibration: prior → Bayesian → 90-day update)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class HealthWeights:
    """
    Weights w1..w4 for the H(x) formula.

    Tier 1 (prior): equal weights 0.25 each — used at system initialisation.
    Tier 2 (Bayesian): updated per individual once ≥14 days of biomarker data
                       are available via update_bayesian().
    Tier 3 (longitudinal): re-calibrated every 90 days via online variational Bayes.

    Constraint: w1 + w2 + w3 + w4 == 1 at all times.
    """
    w1: float = 0.25   # Lyapunov stability term weight
    w2: float = 0.25   # Attractor volume term weight
    w3: float = 0.25   # Correlation dimension term weight
    w4: float = 0.25   # Biomarker penalty weight

    def validate(self) -> None:
        total = self.w1 + self.w2 + self.w3 + self.w4
        if not np.isclose(total, 1.0, atol=1e-6):
            raise ValueError(f"Weights must sum to 1.0; got {total:.6f}")

    def update_bayesian(self, posterior_w: np.ndarray) -> None:
        """
        Update weights from Bayesian posterior P(w|B).
        posterior_w : array of shape (4,), must sum to 1.
        """
        posterior_w = np.asarray(posterior_w, dtype=float)
        if not np.isclose(posterior_w.sum(), 1.0, atol=1e-6):
            raise ValueError("Posterior weights must sum to 1.0")
        self.w1, self.w2, self.w3, self.w4 = posterior_w
        self.validate()


# ─────────────────────────────────────────────────────────────────────────────
# Dynamical invariant computation
# ─────────────────────────────────────────────────────────────────────────────

def compute_lyapunov(trajectory: np.ndarray, emb_dim: int = 3) -> float:
    """
    Compute Maximal Lyapunov Exponent (lambda_1) from trajectory x-component.

    Parameters
    ----------
    trajectory : np.ndarray, shape (N, 3)
        Simulated state-space trajectory [x, y, z].
    emb_dim : int
        Embedding dimension for nolds.lyap_r (default 3 for Lorenz-Rössler).

    Returns
    -------
    float : lambda_1 (positive → chaotic; decreasing → more stable)
    """
    # Use x-component; nolds expects 1-D time series
    x_series = trajectory[:, 0]
    # Require minimum length for reliable estimation
    if len(x_series) < 500:
        raise ValueError(f"Trajectory too short ({len(x_series)} steps); need ≥500.")
    try:
        lam1 = nolds.lyap_r(x_series, emb_dim=emb_dim)
        return float(lam1)
    except Exception as exc:
        raise RuntimeError(f"Lyapunov computation failed: {exc}") from exc


def compute_attractor_volume(trajectory: np.ndarray, n_sample: int = 2000) -> float:
    """
    Compute attractor volume VA via convex hull of 3D trajectory samples.

    A smaller VA indicates a tighter, more constrained attractor —
    associated with healthier physiological regulation.

    Parameters
    ----------
    trajectory : np.ndarray, shape (N, 3)
    n_sample   : int — number of points subsampled for hull computation.

    Returns
    -------
    float : convex-hull volume in state-space units³
    """
    # Discard transient (first 20% of trajectory)
    transient_cut = len(trajectory) // 5
    traj_ss = trajectory[transient_cut:]

    if len(traj_ss) < 10:
        raise ValueError("Insufficient post-transient trajectory for volume estimate.")

    # Subsample for efficiency
    idx = np.random.default_rng(42).choice(len(traj_ss), size=min(n_sample, len(traj_ss)), replace=False)
    pts = traj_ss[idx]

    try:
        hull = ConvexHull(pts)
        return float(hull.volume)
    except Exception as exc:
        raise RuntimeError(f"Convex hull computation failed: {exc}") from exc


def compute_correlation_dimension(trajectory: np.ndarray, emb_dim: int = 3) -> float:
    """
    Compute correlation dimension D2 via Grassberger-Procaccia algorithm (nolds.corr_dim).

    D2 quantifies the fractal geometry of the attractor.
    Healthy trajectories exhibit moderate D2; very high values indicate
    dysregulation, very low values indicate loss of physiological variability.

    Parameters
    ----------
    trajectory : np.ndarray, shape (N, 3)
    emb_dim    : int — embedding dimension.

    Returns
    -------
    float : D2 correlation dimension
    """
    x_series = trajectory[:, 0]
    if len(x_series) < 500:
        raise ValueError(f"Trajectory too short ({len(x_series)} steps); need ≥500.")
    try:
        d2 = nolds.corr_dim(x_series, emb_dim=emb_dim)
        return float(d2)
    except Exception as exc:
        raise RuntimeError(f"Correlation dimension computation failed: {exc}") from exc


# ─────────────────────────────────────────────────────────────────────────────
# Health State Function  H(x)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class HealthMetrics:
    """Container for all computed health metrics."""
    lambda_1: float       # Maximal Lyapunov Exponent
    VA: float             # Attractor volume
    D2: float             # Correlation dimension
    phi: float            # Biomarker penalty
    H: float              # Composite health score (higher = healthier)
    weights: HealthWeights = field(default_factory=HealthWeights)

    def summary(self) -> str:
        return (
            f"  λ₁ (MLE):          {self.lambda_1:.4f}\n"
            f"  VA (Attractor vol): {self.VA:.4f}\n"
            f"  D₂ (Corr. dim):    {self.D2:.4f}\n"
            f"  Φ(x) (Penalty):    {self.phi:.4f}\n"
            f"  H(x) (Score):      {self.H:.6f}  ← higher is healthier"
        )


def compute_health_score(
    trajectory: np.ndarray,
    phi: float = 0.0,
    weights: Optional[HealthWeights] = None,
    emb_dim: int = 3,
) -> HealthMetrics:
    """
    Compute the composite Health State Function H(x).

    H(x) = w1/(1+λ₁) + w2/(1+VA) + w3/(1+D2) - w4·Φ(x)

    This is the primary function imported by the GA (ga.py) as its
    fitness evaluator. The GA maximises H(x) over the 200-day horizon.

    Parameters
    ----------
    trajectory : np.ndarray, shape (N, 3)
        Simulated state-space trajectory from engine.py.
    phi        : float
        Pre-computed biomarker penalty Φ(x) from homs_constraints.py.
        Pass 0.0 if no biomarker data available (uses dynamical terms only).
    weights    : HealthWeights, optional
        Weight set w1..w4. Defaults to equal-weight prior (0.25 each).
    emb_dim    : int
        Embedding dimension for Lyapunov and D2 calculations.

    Returns
    -------
    HealthMetrics : all computed invariants and the composite H score.
    """
    if weights is None:
        weights = HealthWeights()
    weights.validate()

    lam1 = compute_lyapunov(trajectory, emb_dim=emb_dim)
    VA   = compute_attractor_volume(trajectory)
    D2   = compute_correlation_dimension(trajectory, emb_dim=emb_dim)

    H = (
        weights.w1 * (1.0 / (1.0 + lam1))
        + weights.w2 * (1.0 / (1.0 + VA))
        + weights.w3 * (1.0 / (1.0 + D2))
        - weights.w4 * phi
    )

    return HealthMetrics(
        lambda_1=lam1,
        VA=VA,
        D2=D2,
        phi=phi,
        H=H,
        weights=weights,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Quick self-test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Generate a sample Lorenz trajectory for testing
    from scipy.integrate import solve_ivp

    def lorenz(t, state, sigma=10, rho=28, beta=8/3):
        x, y, z = state
        return [sigma*(y-x), x*(rho-z)-y, x*y - beta*z]

    print("Running H(x) self-test with synthetic Lorenz trajectory...")
    sol = solve_ivp(lorenz, [0, 100], [1, 1, 1], method='RK45',
                    max_step=0.02, dense_output=False)
    traj = sol.y.T  # shape (N, 3)
    print(f"Trajectory shape: {traj.shape}")

    metrics = compute_health_score(traj, phi=0.05)
    print("\nHealth Metrics:")
    print(metrics.summary())
