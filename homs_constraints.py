"""
homs_constraints.py
===================
HOMS Layer 4 — Physiological Feasibility Constraints & Penalty Function Φ(x)
Math Engines & Digital Twin)

Implements the penalty logic Φ(x) used in:
    H(x) = w1/(1+λ1) + w2/(1+VA) + w3/(1+D2) - w4·Φ(x)

And the penalty-augmented GA fitness:
    Fc(c) = F(c) - λp · max(0, g(c))

as specified in Andrea's corrected architecture document (Section 5.4).

Hard physiological bounds (the Ω feasibility set):
    - Caloric intake  ≥ 1200 kcal/day
    - Sleep           7–9 h/night
    - HI exercise recovery gap ≥ 48 h
    - Fasting glucose 70–100 mg/dL

Per-biomarker penalty weights vi must satisfy Σvi = 1.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Optional


# ─────────────────────────────────────────────────────────────────────────────
# Biomarker target ranges and per-biomarker penalty weights
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class BiomarkerSpec:
    """Target range and penalty weight for a single biomarker."""
    name: str
    target_min: float     # Lower bound of healthy range
    target_max: float     # Upper bound of healthy range
    weight: float         # vi (contribution weight in Φ)

    @property
    def target_mid(self) -> float:
        return (self.target_min + self.target_max) / 2.0


# Default biomarker library — weights must sum to 1.0
DEFAULT_BIOMARKERS = [
    BiomarkerSpec("fasting_glucose_mgdl",  70.0,  100.0,  0.30),
    BiomarkerSpec("sleep_hours",            7.0,    9.0,   0.25),
    BiomarkerSpec("calories_kcal",       1200.0, 2500.0,  0.20),
    BiomarkerSpec("hi_recovery_hours",     48.0,  120.0,  0.15),
    BiomarkerSpec("hrv_ms",               30.0,   80.0,   0.10),
]


def validate_biomarker_weights(specs: list[BiomarkerSpec]) -> None:
    total = sum(s.weight for s in specs)
    if not np.isclose(total, 1.0, atol=1e-6):
        raise ValueError(
            f"Biomarker penalty weights must sum to 1.0; got {total:.6f}. "
            f"Adjust the 'weight' fields in your BiomarkerSpec list."
        )


# ─────────────────────────────────────────────────────────────────────────────
# Per-biomarker penalty function φi(bi)
# ─────────────────────────────────────────────────────────────────────────────

def phi_i(value: float, spec: BiomarkerSpec) -> float:
    """
    Compute individual biomarker penalty φi(bi).

    Returns 0 if value is within [target_min, target_max].
    Returns normalised L2 deviation otherwise.

    φi(bi) = 0                           if target_min ≤ bi ≤ target_max
           = ((bi - mid) / range)²       otherwise
    """
    lo, hi = spec.target_min, spec.target_max
    mid = spec.target_mid
    half_range = (hi - lo) / 2.0

    if lo <= value <= hi:
        return 0.0

    deviation = (value - mid) / half_range
    return float(deviation ** 2)


# ─────────────────────────────────────────────────────────────────────────────
# Composite penalty Φ(x) = Σ vi · φi(bi)
# ─────────────────────────────────────────────────────────────────────────────

def compute_phi(
    biomarker_values: Dict[str, float],
    biomarker_specs: Optional[list[BiomarkerSpec]] = None,
) -> float:
    """
    Compute composite biomarker penalty Φ(x) = Σᵢ vᵢ · φᵢ(bᵢ).

    Parameters
    ----------
    biomarker_values : dict  { biomarker_name : measured_value }
        Only biomarkers present in both the dict AND specs are penalised.
    biomarker_specs  : list[BiomarkerSpec], optional
        Defaults to DEFAULT_BIOMARKERS.

    Returns
    -------
    float : Φ(x) ∈ [0, ∞)   — 0 means all biomarkers within target ranges.
    """
    if biomarker_specs is None:
        biomarker_specs = DEFAULT_BIOMARKERS

    validate_biomarker_weights(biomarker_specs)

    phi_total = 0.0
    for spec in biomarker_specs:
        value = biomarker_values.get(spec.name)
        if value is None:
            # Missing biomarker: skip (handled upstream by EKF uncertainty propagation)
            continue
        phi_total += spec.weight * phi_i(value, spec)

    return float(phi_total)


# ─────────────────────────────────────────────────────────────────────────────
# Hard constraint violation function g(c)
# Used in penalty-augmented GA fitness: Fc(c) = F(c) - λp · max(0, g(c))
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RMNSSchedule:
    """
    Decoded RMNS schedule values for constraint checking.
    These are decoded from the GA chromosome c for a given time window.
    """
    calories_kcal_per_day: float    # Must be ≥ 1200
    sleep_hours_per_night: float    # Must be 7–9
    hi_recovery_gap_hours: float    # Time since last high-intensity session (must be ≥ 48)
    fasting_glucose_mgdl: float     # Must be 70–100


def constraint_violation(schedule: RMNSSchedule) -> float:
    """
    Compute g(c): total constraint violation magnitude.

    Returns 0 if all constraints satisfied (schedule is in Ω).
    Returns positive value proportional to violation severity.

    GA augmented fitness: Fc(c) = F(c) - λp · max(0, g(c))
    """
    violations = []

    # 1. Minimum caloric intake (≥1200 kcal/day)
    if schedule.calories_kcal_per_day < 1200.0:
        violations.append(1200.0 - schedule.calories_kcal_per_day)

    # 2. Sleep window (7–9 hours)
    if schedule.sleep_hours_per_night < 7.0:
        violations.append(7.0 - schedule.sleep_hours_per_night)
    elif schedule.sleep_hours_per_night > 9.0:
        violations.append(schedule.sleep_hours_per_night - 9.0)

    # 3. HI exercise recovery (≥48 hours between sessions)
    if schedule.hi_recovery_gap_hours < 48.0:
        violations.append(48.0 - schedule.hi_recovery_gap_hours)

    # 4. Fasting glucose (70–100 mg/dL)
    if schedule.fasting_glucose_mgdl < 70.0:
        violations.append(70.0 - schedule.fasting_glucose_mgdl)
    elif schedule.fasting_glucose_mgdl > 100.0:
        violations.append(schedule.fasting_glucose_mgdl - 100.0)

    return float(sum(violations))


def penalty_augmented_fitness(
    fitness: float,
    schedule: RMNSSchedule,
    lambda_p: float = 0.1,
) -> float:
    """
    Penalty-augmented GA fitness: Fc(c) = F(c) - λp · max(0, g(c))

    Parameters
    ----------
    fitness    : float — raw fitness F(c) from H(x) integration
    schedule   : RMNSSchedule — decoded chromosome values
    lambda_p   : float — penalty strength (default 0.1)

    Returns
    -------
    float : Fc(c) — penalised fitness used by GA selection
    """
    g = constraint_violation(schedule)
    return fitness - lambda_p * max(0.0, g)


# ─────────────────────────────────────────────────────────────────────────────
# Quick self-test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== HOMS Constraints Module Self-Test ===\n")

    # --- Test 1: All within healthy range (phi = 0) ---
    healthy = {
        "fasting_glucose_mgdl": 85.0,
        "sleep_hours": 8.0,
        "calories_kcal": 1800.0,
        "hi_recovery_hours": 72.0,
        "hrv_ms": 55.0,
    }
    phi_healthy = compute_phi(healthy)
    print(f"Healthy biomarkers  → Φ(x) = {phi_healthy:.4f}  (expected: 0.0)")

    # --- Test 2: Violations ---
    unhealthy = {
        "fasting_glucose_mgdl": 130.0,   # Too high
        "sleep_hours": 5.5,               # Too low
        "calories_kcal": 900.0,           # Below minimum
        "hi_recovery_hours": 24.0,        # Insufficient recovery
        "hrv_ms": 20.0,                   # Suppressed HRV
    }
    phi_unhealthy = compute_phi(unhealthy)
    print(f"Unhealthy biomarkers → Φ(x) = {phi_unhealthy:.4f}  (expected: > 0)")

    # --- Test 3: Constraint violation and penalty ---
    sched_ok = RMNSSchedule(
        calories_kcal_per_day=1800,
        sleep_hours_per_night=8.0,
        hi_recovery_gap_hours=72.0,
        fasting_glucose_mgdl=85.0,
    )
    sched_bad = RMNSSchedule(
        calories_kcal_per_day=900,         # Violates 1200 min
        sleep_hours_per_night=5.0,         # Violates 7h min
        hi_recovery_gap_hours=24.0,        # Violates 48h min
        fasting_glucose_mgdl=120.0,        # Violates 100 max
    )

    g_ok  = constraint_violation(sched_ok)
    g_bad = constraint_violation(sched_bad)
    print(f"\nFeasible schedule   → g(c) = {g_ok:.2f}   (expected: 0.0)")
    print(f"Infeasible schedule → g(c) = {g_bad:.2f}  (expected: large)")

    raw_fitness = 0.45
    fc = penalty_augmented_fitness(raw_fitness, sched_bad, lambda_p=0.1)
    print(f"\nRaw fitness F(c)  = {raw_fitness:.4f}")
    print(f"Penalised Fc(c)   = {fc:.4f}")
    print("\n✓ Constraints module operational.")
