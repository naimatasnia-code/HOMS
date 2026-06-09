from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional
from scipy.optimize import differential_evolution

@dataclass
class Params:
    sigma: float = 10.0
    rho: float = 28.0
    beta: float = 8.0 / 3.0

    alpha_u: float = 3.0   # RMNS forcing strength into y_dot
    gamma_b: float = 2.0   # biomarker forcing strength into z_dot

    stress_noise_std: float = 0.8
    nostress_noise_std: float = 0.05

def rmns_controls(t: float, scenario: str) -> np.ndarray:
    if scenario == "stress":
        R = 0.4 + 0.2 * np.sin(0.3 * t) + 0.2 * np.sin(2.3 * t)
        M = 0.2 + 0.6 * (np.sin(0.15 * t) > 0.8)
        N = 0.5 + 0.3 * np.sin(0.5 * t + 1.0)
        S = 0.9
    else:
        R = 0.8 + 0.1 * np.sin(0.2 * t)
        M = 0.4 + 0.3 * (np.sin(0.12 * t) > 0.6)
        N = 0.7 + 0.1 * np.sin(0.35 * t)
        S = 0.2
    return np.array([R, M, N, S], dtype=float)

def f_rmns(u: np.ndarray, rmns_weights: Optional[np.ndarray] = None) -> float:
    R, M, N, S = u
    if rmns_weights is None:
        return float((0.9 * R + 0.6 * N + 0.4 * M) - (1.2 * S))
    wR, wM, wN, wS = map(float, rmns_weights)
    return float((wR * R + wN * N + wM * M) - (wS * S))

def biomarker_inflammation_spike(t: float, spike_time: float, amp: float, width: float) -> float:
    return float(amp * np.exp(-0.5 * ((t - spike_time) / width) ** 2))

def g_bio(z: float, b_inflammation: float) -> float:
    return float(b_inflammation + 0.05 * z)

def simulate(
    params: Params,
    scenario: str,
    t_max: float = 80.0,
    dt: float = 0.02,
    x0: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    seed: int = 7,
    spike_time: float = 35.0,
    spike_amp: float = 8.0,
    spike_width: float = 3.0,
    rmns_weights: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    t_eval = np.arange(0.0, t_max, dt, dtype=float)
    n = len(t_eval)

    rng = np.random.default_rng(seed)
    noise_std = params.stress_noise_std if scenario == "stress" else params.nostress_noise_std
    noise = rng.normal(0.0, noise_std, size=(n, 3)).astype(float)

    traj = np.zeros((n, 3), dtype=float)
    traj[0] = x0
    x, y, z = x0

    # FAST FIXED-STEP EULER LOOP
    for i in range(1, n):
        t = t_eval[i-1]

        u = rmns_controls(t, scenario)
        u_term = params.alpha_u * f_rmns(u, rmns_weights=rmns_weights)

        b = biomarker_inflammation_spike(t, spike_time=spike_time, amp=spike_amp, width=spike_width)
        b_term = params.gamma_b * g_bio(z=z, b_inflammation=b)

        dx = params.sigma * (y - x) + noise[i-1, 0]
        dy = x * (params.rho - z) - y + u_term + noise[i-1, 1]
        dz = x * y - params.beta * z + b_term + noise[i-1, 2]

        x += dx * dt
        y += dy * dt
        z += dz * dt
        traj[i] = [x, y, z]

    return t_eval, traj


def coupled_simulate(
    params: Params,
    t_max: float = 100.0,
    dt: float = 0.02,
    seed: int = 42,
    K: float = 0.0,              
    msg_amp: float = 1.0,        
    msg_freq: float = 0.1,       
    noise_std: float = 0.0,      
    lambda_val: float = 0.8,
    omega_val: float = 0.7
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Coupled Master-Slave chaotic synchronization engine.
    Updated for Demo 3 (Skin Cancer Risk & Depression).
    """
    t_eval = np.arange(0.0, t_max, dt, dtype=float)
    n = len(t_eval)

    rng = np.random.default_rng(seed)
    channel_noise = rng.normal(0.0, noise_std, size=n).astype(float)

    traj_master = np.zeros((n, 3), dtype=float)
    traj_slave = np.zeros((n, 3), dtype=float)
    
    s_signal = np.zeros(n, dtype=float)       
    m_recovered = np.zeros(n, dtype=float)    
    sync_error = np.zeros(n, dtype=float)     

    xm, ym, zm = 1.0, 1.0, 1.0
    xs, ys, zs = 1.0, 1.0, 1.0
    traj_master[0] = (xm, ym, zm)
    traj_slave[0] = (xs, ys, zs)

    # DEMO 3 CONSTANTS
    G_BIO = 10.0 

    # Master is HEALTHY (lambda = 0, omega = 0)
    rho_m = 15.0
    alpha_m = 1.0
    gamma_m = 0.0

    # Slave is LEO (Stressed Patient with Mole Risk)
    rho_s = 15.0 + (13.0 * lambda_val)
    alpha_s = 1.0 + (4.0 * lambda_val)
    gamma_s = (omega_val * lambda_val) * G_BIO

    for i in range(1, n):
        t = t_eval[i-1]

        # Biological RMNS Inputs 
        rmns_m = rmns_controls(t, "no-stress")
        um = f_rmns(rmns_m)

        rmns_s = rmns_controls(t, "stress")
        us = f_rmns(rmns_s)

        # MASTER SYSTEM (Healthy Baseline)
        dx_m = params.sigma * (ym - xm)
        dy_m = xm * (rho_m - zm) - ym + alpha_m * um
        dz_m = xm * ym - params.beta * zm + gamma_m
        
        # TASK 1 & 2: Secret Message + Channel Noise
        m_t = msg_amp * np.sin(2.0 * np.pi * msg_freq * t)
        s_t = xm + m_t + channel_noise[i-1]
        s_signal[i-1] = s_t

        # SLAVE SYSTEM (Leo's Body) - Chaos Control (K) + "Two-Hit" Math
        dx_s = params.sigma * (ys - xs) + K * (s_t - xs)
        dy_s = xs * (rho_s - zs) - ys + alpha_s * us
        dz_s = xs * ys - params.beta * zs + gamma_s

        # Euler Integration 
        xm += dx_m * dt
        ym += dy_m * dt
        zm += dz_m * dt
        
        xs += dx_s * dt
        ys += dy_s * dt
        zs += dz_s * dt

        # Save states
        traj_master[i] = (xm, ym, zm)
        traj_slave[i] = (xs, ys, zs)
        
        # Decrypt message and calculate error
        m_recovered[i] = s_t - xs
        sync_error[i] = xm - xs

    # Fill the last index for signals
    s_signal[-1] = xm + (msg_amp * np.sin(2.0 * np.pi * msg_freq * t_eval[-1])) + channel_noise[-1]
    m_recovered[-1] = s_signal[-1] - xs
    sync_error[-1] = xm - xs

    return t_eval, traj_master, traj_slave, s_signal, m_recovered, sync_error



# TASK 4 - ALTERNATIVE 1: ADAPTIVE CONTROL LAW

def adaptive_coupled_simulate(
    params: Params,
    t_max: float = 100.0,
    dt: float = 0.02,
    seed: int = 42,
    lambda_adapt: float = 0.1,   # Learning rate for the medication dosage
    k_max: float = 50.0,         # Maximum allowable dosage (Safety Ceiling)
    noise_std: float = 0.0,      
    lambda_val: float = 0.8,     # Stress Input
    omega_val: float = 0.7       # CT Scan Input
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Alternative 1: Dosage (K) is dynamic. It automatically increases 
    based on the real-time illness of the patient (sync error).
    """
    t_eval = np.arange(0.0, t_max, dt, dtype=float)
    n = len(t_eval)
    
    traj_master = np.zeros((n, 3), dtype=float)
    traj_slave = np.zeros((n, 3), dtype=float)
    k_array = np.zeros(n, dtype=float) # Tracks the dosage over time
    sync_error = np.zeros(n, dtype=float) 

    xm, ym, zm = 1.0, 1.0, 1.0
    xs, ys, zs = 1.0, 1.0, 1.0
    k = 0.0 # Initial dosage is zero

    traj_master[0] = (xm, ym, zm)
    traj_slave[0] = (xs, ys, zs)
    k_array[0] = k

    # DEMO 3 CONSTANTS (Preserving your Two-Hit Math)
    G_BIO = 10.0  
    rho_m, alpha_m, gamma_m = 15.0, 1.0, 0.0
    rho_s = 15.0 + (13.0 * lambda_val)
    alpha_s = 1.0 + (4.0 * lambda_val)
    gamma_s = (omega_val * lambda_val) * G_BIO

    # FAST FIXED-STEP EULER LOOP
    for i in range(1, n):
        t = t_eval[i-1]

        rmns_m = rmns_controls(t, "no-stress")
        um = f_rmns(rmns_m)
        rmns_s = rmns_controls(t, "stress")
        us = f_rmns(rmns_s)

        # Master (Healthy)
        dx_m = params.sigma * (ym - xm)
        dy_m = xm * (rho_m - zm) - ym + alpha_m * um
        dz_m = xm * ym - params.beta * zm + gamma_m
        
        # Slave (Patient) - Notice 'k' is used instead of fixed 'K'
        dx_s = params.sigma * (ys - xs) + k * (xm - xs)
        dy_s = xs * (rho_s - zs) - ys + alpha_s * us
        dz_s = xs * ys - params.beta * zs + gamma_s

        # Adaptive Law Math: dk/dt = lambda_adapt * error^2
        error_x = xm - xs
        dk = lambda_adapt * (error_x ** 2)
        
        # Safety Ceiling
        if k >= k_max and dk > 0:
            dk = 0.0

        # Euler Integration
        xm += dx_m * dt
        ym += dy_m * dt
        zm += dz_m * dt
        xs += dx_s * dt
        ys += dy_s * dt
        zs += dz_s * dt
        k += dk * dt

        traj_master[i] = (xm, ym, zm)
        traj_slave[i] = (xs, ys, zs)
        k_array[i] = k
        sync_error[i] = xm - xs

    return t_eval, traj_master, traj_slave, k_array, sync_error

# TASK 4 - ALTERNATIVE 2: GENETIC ALGORITHM OPTIMIZATION

def _calculate_cost_for_ga(k_input, params, t_max, dt, lambda_val, omega_val, w1_error, w2_dosage):
    """Hidden helper function to calculate the biological cost of a given dosage."""
    test_k = k_input[0]
    
    # Run existing manual simulation with the test dosage
    _, _, _, _, _, sync_error = coupled_simulate(
        params=params, t_max=t_max, dt=dt, K=test_k, 
        lambda_val=lambda_val, omega_val=omega_val
    )
    
    # Cost = Integral of (W1 * error^2 + W2 * K^2) dt
    cost = np.sum(w1_error * (sync_error**2) + w2_dosage * (test_k**2)) * dt
    return cost

def run_ga_optimization(
    params: Params,
    lambda_val: float,
    omega_val: float,
    w1_error: float = 1.0,   # Weight given to patient health
    w2_dosage: float = 0.1,  # Weight given to medication side effects
    t_max: float = 50.0,
    dt: float = 0.02
) -> Tuple[float, float]:
    """
    Alternative 2: Uses a Genetic Algorithm to find the single perfect 
    fixed dosage (K) before the simulation even starts.
    """
    bounds = [(0.0, 50.0)] # Search between K=0 and K=50
    
    result = differential_evolution(
        _calculate_cost_for_ga, 
        bounds, 
        args=(params, t_max, dt, lambda_val, omega_val, w1_error, w2_dosage),
        strategy='best1bin', 
        popsize=15, 
        maxiter=30  # Kept at 30 to ensure the API responds quickly
    )
    
    optimal_k = result.x[0]
    minimum_cost = result.fun
    
    return float(optimal_k), float(minimum_cost)
