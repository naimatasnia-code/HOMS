"""
HOMS Visualisation — plots attractor, H(x) components, and GA fitness
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from engine import Params, simulate
from homs_health_state import compute_health_score
from homs_constraints import compute_phi, RMNSSchedule, penalty_augmented_fitness
from homs_tracking import run_full_pipeline

plt.style.use("dark_background")
fig = plt.figure(figsize=(18, 12))
fig.suptitle("HOMS — Health Optimisation & Modelling System", fontsize=16, color="cyan", fontweight="bold")

params = Params()

# ── Run two scenarios ─────────────────────────────────────────────────────────
t, traj_healthy = simulate(params, "no-stress", t_max=100.0, dt=0.02, seed=42)
_, traj_stress  = simulate(params, "stress",    t_max=100.0, dt=0.02, seed=42)

biomarkers_ok  = {"fasting_glucose_mgdl": 85.0,  "sleep_hours": 8.0,  "calories_kcal": 1800.0, "hi_recovery_hours": 72.0, "hrv_ms": 55.0}
biomarkers_bad = {"fasting_glucose_mgdl": 120.0, "sleep_hours": 5.5,  "calories_kcal": 900.0,  "hi_recovery_hours": 20.0, "hrv_ms": 18.0}

phi_ok  = compute_phi(biomarkers_ok)
phi_bad = compute_phi(biomarkers_bad)

m_healthy = compute_health_score(traj_healthy, phi=phi_ok)
m_stress  = compute_health_score(traj_stress,  phi=phi_bad)

# ── Plot 1: 3D Attractor — Healthy ───────────────────────────────────────────
ax1 = fig.add_subplot(3, 4, 1, projection="3d")
skip = 20   # plot every 20th point so it's not too slow
ax1.plot(traj_healthy[::skip,0], traj_healthy[::skip,1], traj_healthy[::skip,2],
         lw=0.5, color="cyan", alpha=0.7)
ax1.set_title("Attractor — Healthy", color="cyan", fontsize=9)
ax1.set_xlabel("x", fontsize=7, color="white"); ax1.set_ylabel("y", fontsize=7, color="white"); ax1.set_zlabel("z", fontsize=7, color="white")
ax1.tick_params(labelsize=6)

# ── Plot 2: 3D Attractor — Stress ────────────────────────────────────────────
ax2 = fig.add_subplot(3, 4, 2, projection="3d")
ax2.plot(traj_stress[::skip,0], traj_stress[::skip,1], traj_stress[::skip,2],
         lw=0.5, color="tomato", alpha=0.7)
ax2.set_title("Attractor — Stress", color="tomato", fontsize=9)
ax2.set_xlabel("x", fontsize=7, color="white"); ax2.set_ylabel("y", fontsize=7, color="white"); ax2.set_zlabel("z", fontsize=7, color="white")
ax2.tick_params(labelsize=6)

# ── Plot 3: H(x) Component Bar Chart ─────────────────────────────────────────
ax3 = fig.add_subplot(3, 4, 3)
labels  = ["1/(1+λ₁)", "1/(1+VA)", "1/(1+D₂)", "−w₄·Φ(x)", "H(x)"]
w = 0.25; m = m_healthy

h_vals = [m.weights.w1/(1+m.lambda_1), m.weights.w2/(1+m.VA),
          m.weights.w3/(1+m.D2), -m.weights.w4*m.phi, m.H]
s_vals = [m_stress.weights.w1/(1+m_stress.lambda_1), m_stress.weights.w2/(1+m_stress.VA),
          m_stress.weights.w3/(1+m_stress.D2), -m_stress.weights.w4*m_stress.phi, m_stress.H]

x_pos = np.arange(len(labels))
bars1 = ax3.bar(x_pos - w/2, h_vals, w, label="Healthy", color="cyan",    alpha=0.8)
bars2 = ax3.bar(x_pos + w/2, s_vals, w, label="Stress",  color="tomato",  alpha=0.8)
ax3.set_xticks(x_pos); ax3.set_xticklabels(labels, fontsize=7, rotation=15)
ax3.set_title("H(x) Component Breakdown", color="white", fontsize=9)
ax3.legend(fontsize=7); ax3.tick_params(labelsize=7)
ax3.axhline(0, color="white", lw=0.5)

# ── Plot 4: H(x) Score Comparison ────────────────────────────────────────────
ax4 = fig.add_subplot(3, 4, 4)
scenarios = ["Healthy\nschedule", "Stress\nschedule"]
scores    = [m_healthy.H, m_stress.H]
colors    = ["cyan", "tomato"]
bars = ax4.bar(scenarios, scores, color=colors, alpha=0.85, width=0.45)
for bar, score in zip(bars, scores):
    ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
             f"{score:.4f}", ha="center", va="bottom", fontsize=9, color="white", fontweight="bold")
ax4.set_title("Composite H(x) Score", color="white", fontsize=9)
ax4.set_ylabel("H(x)", color="white", fontsize=8)
ax4.tick_params(labelsize=8)

# ── Plot 5: x(t) time series comparison ──────────────────────────────────────
ax5 = fig.add_subplot(3, 4, (5, 6))
ax5.plot(t, traj_healthy[:, 0], color="cyan",   lw=0.8, alpha=0.9, label="Healthy x(t)")
ax5.plot(t, traj_stress[:,  0], color="tomato", lw=0.8, alpha=0.9, label="Stress x(t)")
ax5.set_title("State Variable x(t) — Healthy vs Stress", color="white", fontsize=9)
ax5.set_xlabel("Time (days)", fontsize=8, color="white")
ax5.set_ylabel("x(t)", fontsize=8, color="white")
ax5.legend(fontsize=8); ax5.tick_params(labelsize=7)

# ── Plot 6: Constraint penalty bars ──────────────────────────────────────────
ax6 = fig.add_subplot(3, 4, 7)
from homs_constraints import DEFAULT_BIOMARKERS, phi_i
names   = [b.name.replace("_", "\n") for b in DEFAULT_BIOMARKERS]
phi_ok_vals  = [b.weight * phi_i(biomarkers_ok[b.name],  b) for b in DEFAULT_BIOMARKERS]
phi_bad_vals = [b.weight * phi_i(biomarkers_bad[b.name], b) for b in DEFAULT_BIOMARKERS]
x_pos = np.arange(len(names))
ax6.bar(x_pos - 0.2, phi_ok_vals,  0.38, label="Healthy",  color="cyan",   alpha=0.8)
ax6.bar(x_pos + 0.2, phi_bad_vals, 0.38, label="Unhealthy", color="tomato", alpha=0.8)
ax6.set_xticks(x_pos); ax6.set_xticklabels(names, fontsize=6)
ax6.set_title("Per-Biomarker Penalty  vᵢ·φᵢ(bᵢ)", color="white", fontsize=9)
ax6.legend(fontsize=7); ax6.tick_params(labelsize=7)

# ── Plot 7: SINDy + EKF parameter tracking ───────────────────────────────────
ax7 = fig.add_subplot(3, 4, 8)
rng = np.random.default_rng(5)
stream = traj_healthy[500:1500, 0] + rng.normal(0, 0.5, 1000)
_, tracker = run_full_pipeline(traj_healthy, t, biomarker_stream=stream)
theta_hist, uncert = tracker.get_estimates()
steps = np.arange(len(theta_hist))
ax7.plot(steps, theta_hist[:, 0], color="cyan",    lw=1.2, label="σ̂")
ax7.plot(steps, theta_hist[:, 1], color="gold",    lw=1.2, label="ρ̂")
ax7.plot(steps, theta_hist[:, 2], color="magenta", lw=1.2, label="β̂")
ax7.axhline(10.0,    color="cyan",    lw=0.6, ls="--", alpha=0.5)
ax7.axhline(28.0,    color="gold",    lw=0.6, ls="--", alpha=0.5)
ax7.axhline(8/3,     color="magenta", lw=0.6, ls="--", alpha=0.5)
ax7.set_title("EKF Parameter Tracking θ̂(t)", color="white", fontsize=9)
ax7.set_xlabel("EKF update steps", fontsize=8, color="white")
ax7.legend(fontsize=8); ax7.tick_params(labelsize=7)
ax7.text(len(steps)*0.6, 29.5, "dashed = true values", fontsize=6.5, color="white", alpha=0.6)

# ── Plot 8: GA fitness evolution ──────────────────────────────────────────────
ax8 = fig.add_subplot(3, 4, (9, 10))
from ga import optimize_rmns, fitness, Params as GaParams
import random

ga_params = GaParams()
pop_size, gens = 10, 15
population = [[random.uniform(0.0, 3.5) for _ in range(4)] for _ in range(pop_size)]
best_per_gen, mean_per_gen = [], []

for gen in range(gens):
    scores = [fitness(ind, ga_params, "no-stress", 60.0, 0.02, 42, 30.0, 6.0, 3.0) for ind in population]
    best_per_gen.append(max(scores))
    mean_per_gen.append(np.mean(scores))
    scored = sorted(zip(scores, population), reverse=True)
    elites = [ind for _, ind in scored[:3]]
    nxt = elites[:]
    from ga import mutate, crossover
    while len(nxt) < pop_size:
        nxt.append(mutate(crossover(random.choice(elites), random.choice(elites))))
    population = nxt

ax8.plot(best_per_gen, color="lime",  lw=2,   marker="o", markersize=4, label="Best fitness")
ax8.plot(mean_per_gen, color="white", lw=1.2, ls="--",    alpha=0.6,    label="Mean fitness")
ax8.set_title("GA Fitness Evolution", color="white", fontsize=9)
ax8.set_xlabel("Generation", fontsize=8, color="white")
ax8.set_ylabel("Fitness", fontsize=8, color="white")
ax8.legend(fontsize=8); ax8.tick_params(labelsize=7)

# ── Plot 9: Poincaré section (z=27 plane) ─────────────────────────────────────
ax9 = fig.add_subplot(3, 4, 11)
z_target = 27.0
tol = 0.3
crossings_h = [(traj_healthy[i,0], traj_healthy[i,1])
               for i in range(1, len(traj_healthy))
               if abs(traj_healthy[i,2] - z_target) < tol and traj_healthy[i-1,2] < z_target]
crossings_s = [(traj_stress[i,0], traj_stress[i,1])
               for i in range(1, len(traj_stress))
               if abs(traj_stress[i,2] - z_target) < tol and traj_stress[i-1,2] < z_target]
if crossings_h:
    xh, yh = zip(*crossings_h)
    ax9.scatter(xh, yh, s=6, color="cyan",   alpha=0.7, label="Healthy")
if crossings_s:
    xs2, ys2 = zip(*crossings_s)
    ax9.scatter(xs2, ys2, s=6, color="tomato", alpha=0.7, label="Stress")
ax9.set_title(f"Poincaré Section  z={z_target}", color="white", fontsize=9)
ax9.set_xlabel("x", fontsize=8, color="white"); ax9.set_ylabel("y", fontsize=8, color="white")
ax9.legend(fontsize=7); ax9.tick_params(labelsize=7)

# ── Plot 10: Lyapunov & H(x) summary table ───────────────────────────────────
ax10 = fig.add_subplot(3, 4, 12)
ax10.axis("off")
rows = [
    ["Metric",       "Healthy",             "Stress"],
    ["λ₁ (MLE)",     f"{m_healthy.lambda_1:.4f}", f"{m_stress.lambda_1:.4f}"],
    ["VA (vol)",     f"{m_healthy.VA:.1f}",       f"{m_stress.VA:.1f}"],
    ["D₂",          f"{m_healthy.D2:.4f}",        f"{m_stress.D2:.4f}"],
    ["Φ(x)",        f"{m_healthy.phi:.4f}",        f"{m_stress.phi:.4f}"],
    ["H(x)  ↑",     f"{m_healthy.H:.4f}",         f"{m_stress.H:.4f}"],
]
tbl = ax10.table(cellText=rows[1:], colLabels=rows[0],
                 loc="center", cellLoc="center")
tbl.auto_set_font_size(False); tbl.set_fontsize(9)
tbl.scale(1, 1.6)
for (r, c), cell in tbl.get_celld().items():
    cell.set_facecolor("#0D2135" if r % 2 == 0 else "#0A1520")
    cell.set_text_props(color="white" if c != 0 or r == 0 else "cyan")
    cell.set_edgecolor("#1C3A5F")
    if r == 0:
        cell.set_facecolor("#0D9488")
ax10.set_title("Summary Metrics", color="white", fontsize=9, pad=12)

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig("homs_output.png", dpi=150, bbox_inches="tight", facecolor="#0F172A")
plt.show()
print("\n✓ Plot saved → homs_output.png")
