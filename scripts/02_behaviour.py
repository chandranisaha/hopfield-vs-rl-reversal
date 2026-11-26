"""Step 2 of the proposal: simulate every model on the reversal task over parameter
grids, plot choices aligned to reversals, and measure the behavioural signatures
(trials to recover, perseveration, reward rate).  Also the behavioural part of the
capacity-and-noise evaluation (classical-Hopfield agent vs. memory load, cue noise).

Outputs: results/behaviour.json, figures/fig2_behaviour.*, figures/fig3_capacity_behaviour.*
Usage:   python scripts/02_behaviour.py [--quick] [--plot-only]
"""

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from hopfield_rl.experiments import FIGURES, RESULTS, pmap, simulate_config
from hopfield_rl.plotting import (ACCENT, INK2, MODEL_COLORS, MUTED, panel_label, plt, save,
                                  seq_colors, set_style)

QUICK = "--quick" in sys.argv
PLOT_ONLY = "--plot-only" in sys.argv
ONLY = sys.argv[sys.argv.index("--only") + 1].split(",") if "--only" in sys.argv else None
OUT = os.path.join(RESULTS, "behaviour.json")

THETA = 6.0
MS = [2, 4, 8, 16, 32, 64, 128]
BETAS = [0.0, 0.5, 1.0, 2.0, 4.0, 8.0]
LS = [1, 2, 3]
ALPHAS = [0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0]
THETAS = [3.0, 6.0, 10.0]
KAPPAS = [-1.0, 0.0, 0.5, 1.0, 2.0, 3.0]
LAMS = [0.0, 0.25, 0.5, 0.75, 1.0]
CODE_BITS = [2, 4, 16, 64]             # classical network size N = (L + 2) * code_bits
NOISES = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
L_CLASSICAL = 2
REPRESENTATIVE = {
    "Q": {"alpha": 0.4, "theta": THETA},
    "QK": {"alpha": 0.4, "theta": THETA, "kappa": 1.0},
    "HOP": {"beta": 4.0, "M": 64, "L": 2, "theta": THETA},
    "HYB": {"alpha": 0.4, "theta": THETA, "beta": 4.0, "M": 64, "L": 2, "lam": 0.5},
}


def build_tasks():
    n = 20 if QUICK else 100
    n_cls = 8 if QUICK else 30
    tasks = []

    def add(group, model, params, n_sessions=n, **kw):
        tasks.append({"group": group, "model": model, "params": params, "n_sessions": n_sessions,
                      "seed": 1000 + len(tasks), **kw})

    for m, p in REPRESENTATIVE.items():
        add("representative", m, p)
    for L in LS:
        for M in MS:
            for beta in BETAS:
                add("hop_grid", "HOP", {"beta": beta, "M": M, "L": L, "theta": THETA})
    for alpha in ALPHAS:
        for theta in THETAS:
            add("q_grid", "Q", {"alpha": alpha, "theta": theta})
    for kappa in KAPPAS:
        add("kappa_grid", "QK", {"alpha": 0.4, "theta": THETA, "kappa": kappa})
    for lam in LAMS:
        add("lambda_grid", "HYB", {"alpha": 0.1, "theta": THETA, "beta": 2.0, "M": 8, "L": 2,
                                   "lam": lam})
    for cb in CODE_BITS:
        for M in MS:
            add("classical_grid", "HOPC", {"M": M, "L": L_CLASSICAL, "theta": THETA}, n_cls,
                agent_kwargs={"code_bits": cb, "value_bits": cb})
    for noise in NOISES:
        for beta in (2.0, 8.0):
            add("noise", "HOP", {"beta": beta, "M": 16, "L": 2, "theta": THETA},
                agent_kwargs={"cue_noise": noise})
        add("noise", "HOPC", {"M": 16, "L": 2, "theta": THETA}, n_cls,
            agent_kwargs={"code_bits": 32, "value_bits": 32, "cue_noise": noise})
    return tasks


def run():
    tasks = build_tasks()
    if ONLY:                      # re-run some groups, keep the stored results of the others
        with open(OUT) as f:
            old = [r for r in json.load(f) if r["group"] not in ONLY]
        tasks = [t for t in tasks if t["group"] in ONLY]
    print(f"{len(tasks)} parameter configurations", flush=True)
    res = pmap(simulate_config, tasks, desc="simulating")
    for t, r in zip(tasks, res):
        r["group"] = t["group"]
    if ONLY:
        res = old + res
    os.makedirs(RESULTS, exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(res, f)
    return res


def pick(res, group, **match):
    out = []
    for r in res:
        if r["group"] != group:
            continue
        merged = {**r["params"], **r.get("agent_kwargs", {})}
        if all(merged.get(k) == v for k, v in match.items()):
            out.append(r)
    return out


def one(res, group, **match):
    hits = pick(res, group, **match)
    assert len(hits) == 1, (group, match, len(hits))
    return hits[0]


def curve_axis(ax, title):
    ax.axvline(-0.5, color=MUTED, lw=0.8, ls=":")
    ax.axhline(0.5, color=MUTED, lw=0.5)
    ax.set(xlabel="trials from reversal", ylim=(0, 1), xlim=(-10, 29), title=title)


def figure_behaviour(res):
    fig, axes = plt.subplots(2, 4, figsize=(7.2, 3.9), constrained_layout=True)
    labels = {"Q": "Q-learning", "QK": "Q + stickiness", "HOP": "Hopfield", "HYB": "Hybrid"}

    ax = axes[0, 0]
    for m in REPRESENTATIVE:
        r = [x for x in pick(res, "representative") if x["model"] == m][0]
        ax.plot(r["offsets"], r["curve"], color=MODEL_COLORS[m], label=labels[m])
    curve_axis(ax, "Four models")
    ax.set_ylabel("P(choose new better arm)")
    ax.legend(loc="lower right", fontsize=5.5, handlelength=1.2, borderaxespad=0.1)
    panel_label(ax, "A")

    ax = axes[0, 1]
    for M, col in zip(MS, seq_colors(len(MS))):
        r = one(res, "hop_grid", M=M, beta=2.0, L=2)
        ax.plot(r["offsets"], r["curve"], color=col, lw=1.2, label=f"M = {M}")
    curve_axis(ax, "Hopfield: window M")
    ax.text(0.03, 0.97, "β = 2, L = 2"+chr(10)+"light → dark:"+chr(10)+"M = 2, 4, …, 128", transform=ax.transAxes,
            ha="left", va="top", fontsize=5.5, color=INK2)
    panel_label(ax, "B")

    ax = axes[0, 2]
    for beta, col in zip(BETAS, seq_colors(len(BETAS))):
        r = one(res, "hop_grid", M=64, beta=beta, L=2)
        ax.plot(r["offsets"], r["curve"], color=col, lw=1.2, label=f"β = {beta:g}")
    curve_axis(ax, "Hopfield: sharpness β")
    ax.text(0.03, 0.97, "M = 64, L = 2"+chr(10)+"light → dark:"+chr(10)+"β = 0, 0.5, 1, 2, 4, 8", transform=ax.transAxes,
            ha="left", va="top", fontsize=5.5, color=INK2)
    panel_label(ax, "C")

    ax = axes[0, 3]
    sel = [0.05, 0.1, 0.2, 0.4, 0.8]
    for alpha, col in zip(sel, seq_colors(len(sel))):
        r = one(res, "q_grid", alpha=alpha, theta=THETA)
        ax.plot(r["offsets"], r["curve"], color=col, lw=1.2, label=f"α = {alpha:g}")
    curve_axis(ax, "Q-learning: rate α")
    ax.text(0.03, 0.97, "θ = 6"+chr(10)+"light → dark:"+chr(10)+"α = 0.05, 0.1, 0.2, 0.4, 0.8", transform=ax.transAxes,
            ha="left", va="top", fontsize=5.5, color=INK2)
    panel_label(ax, "D")

    q_best = max(pick(res, "q_grid", theta=THETA), key=lambda r: r["reward_rate"])
    specs = [("trials_to_recover", "trials to recover", "E"),
             ("perseveration", "perseveration\n(old arm, first 10 trials)", "F"),
             ("reward_rate", "reward rate", "G")]
    for ax, (key, ylabel, letter) in zip(axes[1, :3], specs):
        for beta, col in zip(BETAS, seq_colors(len(BETAS))):
            rs = [one(res, "hop_grid", M=M, beta=beta, L=2) for M in MS]
            ax.errorbar(MS, [r[key] for r in rs], yerr=[r[key + "_sem"] for r in rs], color=col,
                        marker="o", ms=2.5, lw=1.2, elinewidth=0.6, label=f"β = {beta:g}")
        ax.axhline(q_best[key], color=MODEL_COLORS["Q"], ls="--", lw=1.0)
        ax.set_xscale("log", base=2)
        ax.set_xticks(MS)
        ax.set_xticklabels(MS)
        ax.set(xlabel="memory window M", ylabel=ylabel)
        panel_label(ax, letter)
    axes[1, 0].set_title("Hopfield (L = 2)")
    axes[1, 0].legend(fontsize=5, ncol=2, loc="upper left", handlelength=1.2, columnspacing=0.8)
    axes[1, 1].set_title("dashed: best Q-learning", fontsize=6.5, fontweight="normal", color=INK2)
    axes[1, 2].set_title(f"(α = {q_best['params']['alpha']:g}, θ = 6)", fontsize=6.5,
                         fontweight="normal", color=INK2)

    ax = axes[1, 3]
    for L, col in zip(LS, seq_colors(len(LS))):
        rs = [one(res, "hop_grid", M=M, beta=4.0, L=L) for M in MS]
        ax.errorbar(MS, [r["reward_rate"] for r in rs], yerr=[r["reward_rate_sem"] for r in rs],
                    color=col, marker="o", ms=2.5, lw=1.2, elinewidth=0.6, label=f"L = {L}")
    ax.set_xscale("log", base=2)
    ax.set_xticks(MS)
    ax.set_xticklabels(MS)
    ax.set(xlabel="memory window M", ylabel="reward rate", title="Key length L (β = 4)")
    ax.legend(loc="lower left", fontsize=6)
    panel_label(ax, "H")
    save(fig, os.path.join(FIGURES, "fig2_behaviour"))


def figure_capacity(res):
    fig, axes = plt.subplots(1, 4, figsize=(7.2, 2.0), constrained_layout=True)
    Ns = [(L_CLASSICAL + 2) * cb for cb in CODE_BITS]
    cols = seq_colors(len(CODE_BITS))

    ax = axes[0]
    for cb, N, col in zip(CODE_BITS, Ns, cols):
        rs = [one(res, "classical_grid", M=M, code_bits=cb) for M in MS]
        ax.errorbar([M / N for M in MS], [r["reward_rate"] for r in rs],
                    yerr=[r["reward_rate_sem"] for r in rs], color=col, marker="o", ms=2.5,
                    lw=1.2, elinewidth=0.6, label=f"N = {N}")
    ax.axvline(0.138, color=ACCENT, ls="--", lw=0.8)
    ax.axhline(0.5, color=MUTED, lw=0.5)
    ax.set(xscale="log", xlabel="memory load  M / N", ylabel="reward rate",
           title="Classical agent: load")
    panel_label(ax, "A")

    for ax, key, ylabel, letter in [(axes[1], "reward_rate", "reward rate", "B"),
                                    (axes[2], "trials_to_recover", "trials to recover", "C")]:
        for cb, N, col in zip(CODE_BITS, Ns, cols):
            rs = [one(res, "classical_grid", M=M, code_bits=cb) for M in MS]
            ax.errorbar(MS, [r[key] for r in rs], yerr=[r[key + "_sem"] for r in rs], color=col,
                        marker="o", ms=2.5, lw=1.2, elinewidth=0.6, label=f"classical, N = {N}")
        rs = [one(res, "hop_grid", M=M, beta=2.0, L=L_CLASSICAL) for M in MS]
        ax.errorbar(MS, [r[key] for r in rs], yerr=[r[key + "_sem"] for r in rs],
                    color=MODEL_COLORS["HOP"], marker="s", ms=2.5, lw=1.2, elinewidth=0.6,
                    label="modern, β = 2")
        ax.set_xscale("log", base=2)
        ax.set_xticks(MS)
        ax.set_xticklabels(MS)
        ax.set(xlabel="stored episodes M", ylabel=ylabel)
        panel_label(ax, letter)
    axes[1].set_title("Classical vs. modern")
    axes[2].legend(fontsize=5, loc="upper left")

    ax = axes[3]
    series = [("HOP", {"beta": 2.0}, MODEL_COLORS["HOP"], "s", "modern, β = 2"),
              ("HOP", {"beta": 8.0}, "#0b5c40", "s", "modern, β = 8"),
              ("HOPC", {}, MODEL_COLORS["HOPC"], "o", "classical, N = 128")]
    for model, match, col, mk, lab in series:
        rs = [[r for r in pick(res, "noise", cue_noise=z, **match) if r["model"] == model][0]
              for z in NOISES]
        ax.errorbar(NOISES, [r["reward_rate"] for r in rs], yerr=[r["reward_rate_sem"] for r in rs],
                    color=col, marker=mk, ms=2.5, lw=1.2, elinewidth=0.6, label=lab)
    ax.axhline(0.5, color=MUTED, lw=0.5)
    ax.set(xlabel="cue noise (P[key symbol corrupted])", ylabel="reward rate",
           title="Cue noise (M = 16, L = 2)")
    ax.legend(fontsize=5.5, loc="lower left")
    panel_label(ax, "D")
    save(fig, os.path.join(FIGURES, "fig3_capacity_behaviour"))


def summary(res):
    print("\nRepresentative models:")
    for r in pick(res, "representative"):
        print(f"  {r['model']:4s} recover={r['trials_to_recover']:.2f}  persev={r['perseveration']:.3f}"
              f"  reward={r['reward_rate']:.3f}")
    print("\nHopfield L=2: trials to recover / perseveration / reward rate")
    for beta in BETAS:
        row = [one(res, "hop_grid", M=M, beta=beta, L=2) for M in MS]
        print(f"  beta={beta:<4g}", "  ".join(
            f"M{M}:{r['trials_to_recover']:.1f}/{r['perseveration']:.2f}/{r['reward_rate']:.3f}"
            for M, r in zip(MS, row)))
    print("\nQ-learning theta=6:")
    for r in pick(res, "q_grid", theta=THETA):
        print(f"  alpha={r['params']['alpha']:<5g} recover={r['trials_to_recover']:.2f}"
              f" persev={r['perseveration']:.3f} reward={r['reward_rate']:.3f}")
    print("\nStickiness:")
    for r in pick(res, "kappa_grid"):
        print(f"  kappa={r['params']['kappa']:<5g} recover={r['trials_to_recover']:.2f}"
              f" persev={r['perseveration']:.3f} reward={r['reward_rate']:.3f}")
    print("\nHybrid lambda:")
    for r in pick(res, "lambda_grid"):
        print(f"  lam={r['params']['lam']:<5g} recover={r['trials_to_recover']:.2f}"
              f" persev={r['perseveration']:.3f} reward={r['reward_rate']:.3f}")
    print("\nClassical agent reward rate (rows N, cols M):")
    for cb in CODE_BITS:
        row = [one(res, "classical_grid", M=M, code_bits=cb) for M in MS]
        print(f"  N={(L_CLASSICAL + 2) * cb:<4d}", "  ".join(f"{r['reward_rate']:.3f}" for r in row))
    print("\nCue noise reward rate:")
    for r in pick(res, "noise"):
        print(f"  {r['model']:5s} beta={r['params'].get('beta', '-')!s:4s} "
              f"noise={r['agent_kwargs']['cue_noise']}: {r['reward_rate']:.3f}")


def main():
    set_style()
    if PLOT_ONLY:
        with open(OUT) as f:
            res = json.load(f)
    else:
        res = run()
    summary(res)
    figure_behaviour(res)
    figure_capacity(res)


if __name__ == "__main__":
    main()
