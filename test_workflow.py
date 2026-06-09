"""
HOMS Integration Test — runs the full workflow end-to-end.
Inputs → engine → health state → constraints → tracking → GA fitness
"""
import numpy as np
from engine import Params, simulate
from homs_health_state import compute_health_score, HealthWeights
from homs_constraints import compute_phi, RMNSSchedule, penalty_augmented_fitness
from homs_tracking import run_full_pipeline, EKFParams

# ── Step 1: Run the ODE engine (partner's code) ───────────────────────────────
print("\n" + "="*60)
print("STEP 1: Lorenz-Rössler ODE simulation (engine.py)")
print("="*60)

params = Params(sigma=10.0, rho=28.0, beta=8/3)
t, traj = simulate(
    params=params,
    scenario="no-stress",
    t_max=100.0,
    dt=0.02,
    x0=(1.0, 1.0, 1.0),
    seed=42,
)
print(f"  Trajectory shape : {traj.shape}   (timesteps × [x,y,z])")
print(f"  Time range       : {t[0]:.1f} → {t[-1]:.1f}")

# ── Step 2: Compute H(x) — your health state module ──────────────────────────
print("\n" + "="*60)
print("STEP 2: Health State Function H(x)  (homs_health_state.py)")
print("="*60)

# Simulate a small biomarker penalty (e.g. glucose slightly elevated)
biomarkers = {
    "fasting_glucose_mgdl": 108.0,   # slightly above 100 range
    "sleep_hours": 7.5,
    "calories_kcal": 1900.0,
    "hi_recovery_hours": 60.0,
    "hrv_ms": 48.0,
}
from homs_constraints import compute_phi
phi = compute_phi(biomarkers)

metrics = compute_health_score(traj, phi=phi)
print(metrics.summary())

# ── Step 3: Constraint penalty — your constraints module ──────────────────────
print("\n" + "="*60)
print("STEP 3: Feasibility Constraints  (homs_constraints.py)")
print("="*60)

healthy_sched = RMNSSchedule(
    calories_kcal_per_day=1900,
    sleep_hours_per_night=7.5,
    hi_recovery_gap_hours=60.0,
    fasting_glucose_mgdl=85.0,
)
stressed_sched = RMNSSchedule(
    calories_kcal_per_day=950,    # violation
    sleep_hours_per_night=5.0,    # violation
    hi_recovery_gap_hours=20.0,   # violation
    fasting_glucose_mgdl=115.0,   # violation
)

from homs_constraints import constraint_violation
g_ok  = constraint_violation(healthy_sched)
g_bad = constraint_violation(stressed_sched)
fc_ok  = penalty_augmented_fitness(metrics.H, healthy_sched)
fc_bad = penalty_augmented_fitness(metrics.H, stressed_sched)

print(f"  Healthy schedule  → g(c)={g_ok:.1f},  Fc={fc_ok:.4f}")
print(f"  Stressed schedule → g(c)={g_bad:.1f}, Fc={fc_bad:.4f}")

# ── Step 4: SINDy + EKF tracking — your tracking module ──────────────────────
print("\n" + "="*60)
print("STEP 4: SINDy + EKF parameter tracking  (homs_tracking.py)")
print("="*60)

# Simulate a live biomarker stream (HRV proxy — noisy x-component)
rng = np.random.default_rng(1)
hrv_stream = traj[1000:1200, 0] + rng.normal(0, 0.4, 200)

sindy_result, tracker = run_full_pipeline(
    trajectory=traj,
    t_eval=t,
    biomarker_stream=hrv_stream,
)
print("\nEKF current parameter estimates:")
for k, v in tracker.current_params().items():
    print(f"  {k}: {v:.3f}")

# ── Step 5: GA fitness using H(x) — connects your code to partner's GA ────────
print("\n" + "="*60)
print("STEP 5: GA fitness via H(x)  (ga.py + homs_health_state.py)")
print("="*60)

from ga import optimize_rmns

best_weights, spread = optimize_rmns(
    params=params,
    scenario="no-stress",
    t_max=60.0,
    dt=0.02,
    seed=42,
    spike_time=30.0,
    spike_amp=6.0,
    spike_width=3.0,
    generations=8,
    pop_size=10,
)
print(f"  Optimal RMNS weights : {[round(w,3) for w in best_weights]}")
print(f"  Post-spike spread    : {spread:.4f}  (lower = more stable attractor)")

# Run the winning weights through H(x) to get the full health score
_, traj_opt = simulate(
    params=params, scenario="no-stress", t_max=60.0, dt=0.02,
    seed=42, spike_time=30.0, spike_amp=6.0, spike_width=3.0,
    rmns_weights=np.array(best_weights),
)
metrics_opt = compute_health_score(traj_opt, phi=phi)
print(f"\n  H(x) for optimised schedule:")
print(metrics_opt.summary())

print("\n" + "="*60)
print("✓ Full workflow complete.")
print("="*60)
