"""Experiment building blocks shared by the scripts (all picklable for multiprocessing)."""

from __future__ import annotations

import gc
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor

# Worker processes inherit the environment: one BLAS thread per worker (the pool already
# uses all cores; many multi-threaded BLAS instances exhaust memory).
for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import numpy as np

from .agents import make_agent
from .envs import Environment
from .fitting import fit_model
from .likelihood import ChoiceData, Likelihood
from .simulate import aligned_curve, reversal_metrics, run_session

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RESULTS = os.path.join(ROOT, "results")
FIGURES = os.path.join(ROOT, "figures")
DATA = os.path.join(ROOT, "data")


def n_workers():
    """Worker processes: HRL_WORKERS if set, else (cores - 2) capped at 6 (each worker
    holds its own copy of NumPy/SciPy; more workers mainly cost memory)."""
    if os.environ.get("HRL_WORKERS"):
        return max(1, int(os.environ["HRL_WORKERS"]))
    return max(1, min(6, (os.cpu_count() or 2) - 2))


def _call_with_retry(args):
    """Run one task; if the machine is momentarily out of memory, wait and try again."""
    func, task = args
    for attempt in range(8):
        try:
            return func(task)
        except MemoryError:
            gc.collect()
            time.sleep(20 * (attempt + 1))
    return func(task)


def pmap(func, tasks, workers=None, desc=None):
    """Parallel map with a process pool (order preserved)."""
    tasks = list(tasks)
    workers = n_workers() if workers is None else workers
    if workers <= 1 or len(tasks) <= 1:
        return [func(t) for t in tasks]
    out = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        jobs = [(func, t) for t in tasks]
        for i, res in enumerate(pool.map(_call_with_retry, jobs, chunksize=1)):
            out.append(res)
            if desc and ((i + 1) % max(1, len(tasks) // 10) == 0 or i + 1 == len(tasks)):
                print(f"  {desc}: {i + 1}/{len(tasks)}", flush=True)
    return out


# --------------------------------------------------------------------------------------
# Simulation of one parameter configuration over many seeds
# --------------------------------------------------------------------------------------
PRE, POST = 10, 30


def simulate_config(task):
    """task = dict(model, params, n_sessions, seed, [agent_kwargs], [n_trials]).

    Runs ``n_sessions`` independent 1000-trial reversal sessions and returns the mean
    and s.e.m. (across sessions) of every behavioural measure plus the mean
    reversal-aligned choice curve.
    """
    model, params = task["model"], task["params"]
    kwargs = task.get("agent_kwargs", {})
    n_trials = task.get("n_trials", 1000)
    seeds = np.random.SeedSequence(task["seed"]).spawn(task["n_sessions"])
    metrics, curves = [], []
    for i, ss in enumerate(seeds):
        env_seed, agent_seed, mem_seed = ss.spawn(3)
        env = Environment(n_trials=n_trials, seed=env_seed)
        kw = dict(kwargs)
        if model in ("HOP", "HYB", "HOPC", "HOPK", "HYBK"):
            kw["seed"] = int(mem_seed.generate_state(1)[0])
        agent = make_agent(model, params, **kw)
        sim = run_session(agent, env, seed=agent_seed)
        metrics.append(reversal_metrics(sim["choices"], sim["rewards"], sim["better_arm"],
                                        sim["reversals"]))
        offsets, curve, _ = aligned_curve(sim["choices"], sim["better_arm"], sim["reversals"],
                                          pre=PRE, post=POST)
        curves.append(curve)
    out = {"model": model, "params": params, "agent_kwargs": kwargs,
           "offsets": offsets.tolist(),
           "curve": np.nanmean(np.array(curves), axis=0).tolist()}
    for key in metrics[0]:
        vals = np.array([m[key] for m in metrics], dtype=float)
        out[key] = float(np.nanmean(vals))
        out[key + "_sem"] = float(np.nanstd(vals, ddof=1) / np.sqrt(np.sum(~np.isnan(vals))))
    return out


# --------------------------------------------------------------------------------------
# Recovery: simulate from one model, fit all models
# --------------------------------------------------------------------------------------
def sample_params(model, rng, L_values=(1, 2, 3), M_values=(4, 8, 16, 32, 64)):
    """Ground-truth parameters for the recovery analysis (ranges in which the agents
    perform the task and the parameters have a visible effect on behaviour)."""
    p = {}
    if model in ("Q", "QK", "HYB"):
        p["alpha"] = float(rng.uniform(0.1, 0.9))
    p["theta"] = float(rng.uniform(2.0, 10.0))
    if model == "QK":
        p["kappa"] = float(rng.uniform(0.5, 2.0))
    if model in ("HOP", "HYB"):
        p["beta"] = float(rng.uniform(0.5, 4.0))
        p["L"] = int(rng.choice(L_values))
        p["M"] = int(rng.choice(M_values))
    if model == "HYB":
        p["lam"] = float(rng.uniform(0.2, 0.8))
    return p


def recovery_task(task):
    """task = dict(gen_model, seed, fit_models, n_starts, n_starts_grid, [n_trials])."""
    cache = task.get("cache")                     # finished earlier: resume
    if cache and os.path.exists(cache):
        with open(cache) as f:
            return json.load(f)
    ss = np.random.SeedSequence(task["seed"])
    s_par, s_env, s_agent, s_fit = ss.spawn(4)
    true = sample_params(task["gen_model"], np.random.default_rng(s_par))
    env = Environment(n_trials=task.get("n_trials", 1000), seed=s_env)
    sim = run_session(make_agent(task["gen_model"], true), env, seed=s_agent)
    lik = Likelihood(ChoiceData(sim["choices"], sim["rewards"]))
    rng = np.random.default_rng(s_fit)
    fits = {}
    for m in task["fit_models"]:
        # per (L, M) cell: Hopfield gets n_starts_grid random starts; the hybrid gets
        # 3 starts derived from its nested models plus n_starts_grid // 2 random ones
        n_starts = {"HOP": task["n_starts_grid"], "HYB": max(1, task["n_starts_grid"] // 2)}.get(
            m, task["n_starts"])
        fit = fit_model(lik, m, n_starts=n_starts, rng=rng)
        fits[m] = {"params": fit["params"], "nll": fit["nll"], "aic": fit["aic"],
                   "bic": fit["bic"], "k": fit["k"],
                   "start_spread": float(np.max(fit["start_nlls"]) - np.min(fit["start_nlls"]))}
    metrics = reversal_metrics(sim["choices"], sim["rewards"], sim["better_arm"], sim["reversals"])
    out = {"gen_model": task["gen_model"], "seed": task["seed"], "true": true, "fits": fits,
           "reward_rate": metrics["reward_rate"], "n": int(lik.T)}
    if cache:
        os.makedirs(os.path.dirname(cache), exist_ok=True)
        with open(cache + ".tmp", "w") as f:
            json.dump(out, f)
        os.replace(cache + ".tmp", cache)
    return out
