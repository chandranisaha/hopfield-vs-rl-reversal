"""Demo (one to two minutes): the whole project in miniature.

1. Four agents (Q-learning, Q + stickiness, Hopfield episodic memory, hybrid) play the
   same probabilistic reversal session.
2. What the Hopfield memory retrieves: softmax weights over stored episodes for a blurred
   (low beta) and a sharp (high beta) memory.
3. Model fitting and recovery: data are simulated from the Hopfield agent, all four
   models are fitted by maximum likelihood, and BIC picks the generating model.

Usage:  python demo.py [seed]          Output: figures/demo.png + a table in the terminal
"""

import os
import sys

import numpy as np

from hopfield_rl.agents import ModernEpisodicMemory, make_agent
from hopfield_rl.envs import Environment
from hopfield_rl.experiments import FIGURES
from hopfield_rl.fitting import fit_model
from hopfield_rl.likelihood import LABELS, ChoiceData, Likelihood
from hopfield_rl.plotting import INK2, MODEL_COLORS, MUTED, panel_label, plt, save, set_style
from hopfield_rl.simulate import reversal_metrics, run_session

SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 1
PARAMS = {
    "Q": {"alpha": 0.4, "theta": 6.0},
    "QK": {"alpha": 0.4, "theta": 6.0, "kappa": 1.0},
    "HOP": {"beta": 4.0, "M": 8, "L": 2, "theta": 6.0},
    "HYB": {"alpha": 0.4, "theta": 6.0, "beta": 4.0, "M": 8, "L": 2, "lam": 0.5},
}


def main():
    set_style()
    fig = plt.figure(figsize=(7.2, 5.0), constrained_layout=True)
    grid = fig.add_gridspec(4, 2, width_ratios=[2.2, 1.0])

    # ---------------------------------------------------------------- 1. behaviour
    print("1. Four agents on the same reversal task (1000 trials, better arm pays 0.8)\n")
    print(f"   {'model':16s} reward rate   trials to recover   perseveration")
    sims = {}
    show = 300
    for row, (m, p) in enumerate(PARAMS.items()):
        env = Environment(n_trials=1000, seed=SEED)          # identical schedule for all
        sim = run_session(make_agent(m, p), env, seed=SEED + 100)
        sims[m] = sim
        met = reversal_metrics(sim["choices"], sim["rewards"], sim["better_arm"], sim["reversals"])
        print(f"   {LABELS[m]:16s} {met['reward_rate']:.3f}         {met['trials_to_recover']:5.1f}"
              f"               {met['perseveration']:.2f}")
        ax = fig.add_subplot(grid[row, 0])
        t = np.arange(show)
        ax.fill_between(t, 0, 1, where=sim["better_arm"][:show] == 1, color="#e1e0d9", lw=0,
                        step="mid")
        c, r = sim["choices"][:show], sim["rewards"][:show]
        ax.scatter(t[r == 1], c[r == 1] * 0.8 + 0.1, s=7, color=MODEL_COLORS[m], lw=0)
        ax.scatter(t[r == 0], c[r == 0] * 0.8 + 0.1, s=14, marker="|", color=INK2, lw=0.6)
        ax.set_yticks([0.1, 0.9])
        ax.set_yticklabels(["arm A", "arm B"])
        ax.set(xlim=(0, show), ylim=(-0.05, 1.05))
        ax.grid(False)
        ax.set_title(f"{LABELS[m]}   (reward rate {met['reward_rate']:.2f})")
        if row == 3:
            ax.set_xlabel("trial   (shaded: arm B is better; dot: rewarded choice, tick: unrewarded)")
        else:
            ax.set_xticklabels([])
    panel_label(fig.axes[0], "A")

    # ---------------------------------------------------------------- 2. retrieval
    sim = sims["HOP"]
    t_probe = int(sim["reversals"][3]) + 6                   # shortly after a reversal
    for row, beta in enumerate((0.5, 4.0)):
        mem = ModernEpisodicMemory(L=2, M=16, beta=beta)
        for t in range(t_probe):
            mem.store(sim["choices"][t], sim["rewards"][t])
        order = np.argsort((np.arange(mem.M) - mem._ptr) % mem.M)   # oldest -> newest
        w = mem.retrieval_weights()[order]
        match = (mem.keys @ mem._encode(mem.history))[order]
        ax = fig.add_subplot(grid[row, 1])
        ax.bar(np.arange(mem.M), w, color=[MODEL_COLORS["HOP"] if k == 2 else
                                           ("#8fd9bd" if k == 1 else "#c3c2b7") for k in match],
               width=0.8)
        ax.set(title=f"Retrieval weights, β = {beta:g}", ylabel="weight", ylim=(0, 0.5))
        ax.set_xticks([0, mem.M - 1])
        ax.set_xticklabels(["oldest", "newest"])
        ax.grid(axis="x", visible=False)
        V = mem.values()
        ax.text(0.03, 0.95, f"V(A) = {V[0]:.2f}, V(B) = {V[1]:.2f}", transform=ax.transAxes,
                va="top", fontsize=6.5, color=INK2)
        if row == 0:
            panel_label(ax, "B")
        else:
            ax.set_xlabel("stored episodes; key match with the current\n"
                          "history: dark = 2/2, light = 1/2, grey = 0/2", fontsize=6)

    # ---------------------------------------------------------------- 3. fitting
    print("\n2. Fit all four models to the choices of the Hopfield agent "
          f"(true: {PARAMS['HOP']})\n")
    lik = Likelihood(ChoiceData(sim["choices"], sim["rewards"]))
    rng = np.random.default_rng(SEED)
    fits = {m: fit_model(lik, m, n_starts=2 if m in ("HOP", "HYB") else 8, rng=rng) for m in PARAMS}
    print(f"   {'model':16s}    NLL      BIC   fitted parameters")
    for m, f in fits.items():
        pars = ", ".join(f"{k}={v:.2f}" if isinstance(v, float) else f"{k}={v}"
                         for k, v in f["params"].items())
        print(f"   {LABELS[m]:16s} {f['nll']:7.1f}  {f['bic']:7.1f}   {pars}")
    best = min(fits, key=lambda m: fits[m]["bic"])
    print(f"\n   Lowest BIC: {LABELS[best]}  (chance level NLL = {1000 * np.log(2):.1f})")

    ax = fig.add_subplot(grid[2:, 1])
    names = list(fits)
    bic = np.array([fits[m]["bic"] for m in names])
    ax.barh(range(len(names)), bic - bic.min(), color=[MODEL_COLORS[m] for m in names], height=0.6)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels([LABELS[m] for m in names])
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.axvline(0, color=MUTED, lw=0.8)
    ax.set(xlabel="BIC − best BIC  (0 = selected model)",
           title="Model fits to Hopfield-agent data")
    panel_label(ax, "C")
    save(fig, os.path.join(FIGURES, "demo"))
    print(f"\nFigure written to {os.path.join(FIGURES, 'demo.png')}")


if __name__ == "__main__":
    main()
