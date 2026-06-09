import random
import numpy as np
import concurrent.futures
from engine import simulate, Params

def spread_proxy(traj: np.ndarray) -> float:
    mu = traj.mean(axis=0)
    return float(np.mean(np.linalg.norm(traj - mu, axis=1)))

def fitness(weights4, params: Params, scenario: str, t_max: float, dt: float, seed: int,
            spike_time: float, spike_amp: float, spike_width: float) -> float:
    t, traj = simulate(
        params=params, scenario=scenario, t_max=t_max, dt=dt, seed=seed,
        spike_time=spike_time, spike_amp=spike_amp, spike_width=spike_width,
        rmns_weights=np.array(weights4, dtype=float),
    )
    idx = int(np.searchsorted(t, spike_time))
    post = traj[idx:] if idx < len(traj) else traj
    return 1.0 / (1.0 + spread_proxy(post))

def mutate(chromosome, mutation_rate=0.2):
    out = chromosome.copy()
    for i in range(4):
        if random.random() < mutation_rate:
            out[i] += random.uniform(-0.3, 0.3)
            out[i] = max(0.0, min(3.5, out[i]))
    return out

def crossover(p1, p2):
    return [(a + b) / 2.0 for a, b in zip(p1, p2)]

def optimize_rmns(
    params: Params, scenario: str, t_max: float, dt: float, seed: int,
    spike_time: float, spike_amp: float, spike_width: float,
    generations: int = 12, pop_size: int = 14,
):
    population = [[random.uniform(0.0, 3.5) for _ in range(4)] for _ in range(pop_size)]
    best_fit = -1e9
    best_w = None
    fitness_cache = {}

    def evaluate(ind):
        ind_key = tuple(round(x, 4) for x in ind)
        if ind_key not in fitness_cache:
            fitness_cache[ind_key] = fitness(ind, params, scenario, t_max, dt, seed, spike_time, spike_amp, spike_width)
        return fitness_cache[ind_key], ind

    for gen in range(generations):
        scored = []
        with concurrent.futures.ThreadPoolExecutor() as executor:
            results = executor.map(evaluate, population)
            for score, ind in results:
                scored.append((score, ind))

        scored.sort(reverse=True, key=lambda x: x[0])
        current_best_fit = scored[0][0]

        if current_best_fit > best_fit: 
            best_fit = current_best_fit
            best_w = scored[0][1]

        elites = [ind for _, ind in scored[:max(2, int(pop_size * 0.25))]]
        nxt = elites[:]
        while len(nxt) < pop_size:
            nxt.append(mutate(crossover(random.choice(elites), random.choice(elites))))
        population = nxt

    return [float(v) for v in best_w], float((1.0 / best_fit) - 1.0)
