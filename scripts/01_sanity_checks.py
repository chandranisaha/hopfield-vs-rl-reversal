"""Step 1 of the proposal: check each building block on its own.

(a) Q-learning converges to the true arm values on a fixed-probability task.
(b) The classical Hopfield network retrieves stored patterns from corrupted cues and
    starts failing near P = 0.14 N; (c) cue-noise tolerance; (d) the modern Hopfield
    network stores far more patterns, with retrieval sharpness set by beta.

Outputs: figures/fig1_sanity.(pdf|png), results/sanity.json
"""

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from hopfield_rl.agents import QLearningAgent
from hopfield_rl.envs import Environment
from hopfield_rl.experiments import FIGURES, RESULTS, pmap
from hopfield_rl.hopfield import ClassicalHopfield, ModernHopfield, corrupt, overlap, random_patterns
from hopfield_rl.plotting import ACCENT, INK2, MODEL_COLORS, panel_label, plt, save, seq_colors, set_style
from hopfield_rl.simulate import run_session

QUICK = "--quick" in sys.argv
LOADS = np.round(np.arange(0.02, 0.3001, 0.02), 2)
NOISES = np.round(np.arange(0.0, 0.5001, 0.05), 2)


# ---------------------------------------------------------------- (a) Q-learning
def q_convergence(n_seeds=200, T=600, alpha=0.1, theta=1.0):
    traces = np.empty((n_seeds, T, 2))
    for s in range(n_seeds):
        env = Environment(n_trials=T, block_range=None, seed=[1, s], first_better=0)
        agent = QLearningAgent(alpha, theta)
        rng = np.random.default_rng([2, s])
        agent.reset()
        for t in range(T):
            traces[s, t] = agent.Q
            c = agent.choose(rng)
            agent.update(c, env.step(c))
    return traces


# ---------------------------------------------------------------- (b) capacity
def capacity_task(task):
    N, load, noise, rep = task
    rng = np.random.default_rng([3, N, int(round(load * 1000)), int(round(noise * 1000)), rep])
    P = max(1, int(round(load * N)))
    X = random_patterns(P, N, rng)
    out = ClassicalHopfield(N).store(X).retrieve(corrupt(X, noise, rng))
    m = overlap(out, X)
    return float(np.mean(m)), float(np.mean(m > 0.95))


def modern_task(task):
    N, load, beta, noise, rep = task
    rng = np.random.default_rng([4, N, int(round(load * 100)), int(round(beta * 1000)),
                                 int(round(noise * 1000)), rep])
    P = max(1, int(round(load * N)))
    X = random_patterns(P, N, rng)
    probe = X[:min(P, 200)]
    out = np.sign(ModernHopfield(beta).store(X).retrieve(corrupt(probe, noise, rng)))
    m = overlap(out, probe)
    return float(np.mean(m)), float(np.mean(m > 0.95))


def main():
    set_style()
    results = {}

    # (a)
    alpha, theta = 0.1, 1.0
    traces = q_convergence(n_seeds=50 if QUICK else 200, alpha=alpha, theta=theta)
    q_mean, q_sd = traces.mean(axis=0), traces.std(axis=0)
    late = traces[:, -100:, :].mean(axis=(0, 1))
    results["q_convergence"] = {
        "alpha": alpha, "theta": theta, "true_values": [0.8, 0.2],
        "mean_Q_last_100_trials": late.tolist(),
        "sd_across_runs_last_trial": q_sd[-1].tolist(),
        "theoretical_stationary_sd": float(np.sqrt(alpha / (2 - alpha) * 0.8 * 0.2)),
    }

    # (b) capacity curve vs N
    Ns = [100, 200, 400] if QUICK else [100, 200, 400, 800]
    reps = {100: 40, 200: 20, 400: 10, 800: 5}
    cap = {}
    for N in Ns:
        tasks = [(N, float(l), 0.1, r) for l in LOADS for r in range(reps[N])]
        res = np.array(pmap(capacity_task, tasks)).reshape(len(LOADS), reps[N], 2).mean(axis=1)
        cap[N] = res
        print(f"capacity N={N}: overlap at loads .10/.14/.20 = "
              f"{res[4, 0]:.3f}/{res[6, 0]:.3f}/{res[9, 0]:.3f}", flush=True)
    results["classical_capacity"] = {
        "loads": LOADS.tolist(), "cue_noise": 0.1,
        "mean_overlap": {str(N): cap[N][:, 0].tolist() for N in Ns},
        "fraction_retrieved": {str(N): cap[N][:, 1].tolist() for N in Ns},
    }

    # (c) cue noise, classical N = 400 at three loads, modern at a load of 2
    N_noise = 400
    noise_loads = [0.05, 0.10, 0.14]
    noise_res = {}
    for load in noise_loads:
        tasks = [(N_noise, load, float(z), r) for z in NOISES for r in range(10)]
        noise_res[load] = np.array(pmap(capacity_task, tasks)).reshape(len(NOISES), 10, 2).mean(axis=1)
    tasks = [(N_noise, 2.0, 1.0, float(z), r) for z in NOISES for r in range(5)]
    noise_modern = np.array(pmap(modern_task, tasks)).reshape(len(NOISES), 5, 2).mean(axis=1)
    results["cue_noise"] = {
        "N": N_noise, "noise": NOISES.tolist(),
        "classical_mean_overlap": {str(l): noise_res[l][:, 0].tolist() for l in noise_loads},
        "modern_load2_beta1_mean_overlap": noise_modern[:, 0].tolist(),
    }

    # (d) modern network: load far beyond 0.14 N, effect of beta
    N_mod = 64
    mod_loads = [0.14, 0.5, 1, 2, 5, 10, 20, 50]
    betas = [1.0, 0.25, 0.1, 0.05]
    mod = {}
    for beta in betas:
        tasks = [(N_mod, float(l), beta, 0.1, r) for l in mod_loads for r in range(5)]
        mod[beta] = np.array(pmap(modern_task, tasks)).reshape(len(mod_loads), 5, 2).mean(axis=1)
    tasks = [(N_mod, float(l), 0.1, r) for l in mod_loads[:4] for r in range(40)]
    cls_small = np.array(pmap(capacity_task, tasks)).reshape(4, 40, 2).mean(axis=1)
    results["modern_capacity"] = {
        "N": N_mod, "loads": mod_loads, "cue_noise": 0.1,
        "mean_overlap": {str(b): mod[b][:, 0].tolist() for b in betas},
        "classical_same_N_mean_overlap": dict(zip(map(str, mod_loads[:4]), cls_small[:, 0].tolist())),
    }

    os.makedirs(RESULTS, exist_ok=True)
    with open(os.path.join(RESULTS, "sanity.json"), "w") as f:
        json.dump(results, f, indent=1)

    # ------------------------------------------------------------ figure
    fig, axes = plt.subplots(1, 4, figsize=(7.2, 2.0), constrained_layout=True)
    ax = axes[0]
    t = np.arange(q_mean.shape[0])
    for a, (lab, col) in enumerate([("better arm", MODEL_COLORS["Q"]), ("worse arm", INK2)]):
        ax.fill_between(t, q_mean[:, a] - q_sd[:, a], q_mean[:, a] + q_sd[:, a], color=col,
                        alpha=0.15, lw=0)
        ax.plot(t, q_mean[:, a], color=col, label=f"Q({lab})")
    for v in (0.8, 0.2):
        ax.axhline(v, color=ACCENT, ls="--", lw=0.8)
    ax.text(t[-1], 0.83, "true values", color=ACCENT, ha="right", fontsize=6.5)
    ax.set(xlabel="trial", ylabel="Q-value (mean ± s.d.)", ylim=(0, 1),
           title="Q-learning, fixed arms")
    ax.legend(loc="center right")
    panel_label(ax, "A")

    ax = axes[1]
    for N, col in zip(Ns, seq_colors(len(Ns))):
        ax.plot(LOADS, cap[N][:, 0], "-o", color=col, ms=2.5, label=f"N = {N}")
    ax.axvline(0.138, color=ACCENT, ls="--", lw=0.8)
    ax.text(0.145, 0.97, "0.138 N", color=ACCENT, fontsize=6.5)
    ax.set(xlabel="load  P / N", ylabel="retrieval overlap", ylim=(0.3, 1.02),
           title="Classical: capacity")
    ax.legend(loc="lower left")
    panel_label(ax, "B")

    ax = axes[2]
    for load, col in zip(noise_loads, seq_colors(3)):
        ax.plot(NOISES, noise_res[load][:, 0], "-o", color=col, ms=2.5, label=f"classical, {load}")
    ax.plot(NOISES, noise_modern[:, 0], "-s", color=MODEL_COLORS["HOP"], ms=2.5,
            label="modern, 2")
    ax.set(xlabel="fraction of cue units flipped", ylabel="retrieval overlap", ylim=(-0.05, 1.02),
           title=f"Cue noise (N = {N_noise})")
    ax.legend(loc="lower left", fontsize=6, title="network, load P/N", title_fontsize=6)
    panel_label(ax, "C")

    ax = axes[3]
    greens = ["#0b5c40", "#1baf7a", "#6fd0ad", "#b5e6d3"]
    for beta, col in zip(betas, greens):
        ax.plot(mod_loads, mod[beta][:, 0], "-o", color=col, ms=2.5, label=f"modern, β = {beta:g}")
    ax.plot(mod_loads[:4], cls_small[:, 0], "-o", color=INK2, ms=2.5, label="classical")
    ax.axvline(0.138, color=ACCENT, ls="--", lw=0.8)
    ax.set(xscale="log", xlabel="load  P / N", ylabel="retrieval overlap", ylim=(-0.05, 1.02),
           title=f"Modern (N = {N_mod})")
    ax.legend(loc="center right", fontsize=5.5, handlelength=1.2)
    panel_label(ax, "D")
    save(fig, os.path.join(FIGURES, "fig1_sanity"))
    print(json.dumps(results["q_convergence"], indent=1))
    print("modern overlaps:", {b: np.round(mod[b][:, 0], 3).tolist() for b in betas})
    print("classical N=64:", np.round(cls_small[:, 0], 3).tolist())


if __name__ == "__main__":
    main()
