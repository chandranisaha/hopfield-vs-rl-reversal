"""Step 4 of the proposal: parameter and model recovery (Wilson & Collins 2019).

For each of the four models, simulate data sets (1000 trials of the reversal task) with
known random parameters, fit ALL four models to each data set, and report
  * parameter recovery: true vs. recovered parameters of the generating model;
  * model recovery: the confusion matrix P(best-fitting model | generating model),
    with the winner chosen by BIC (AIC is reported as well), and its inversion
    P(generating model | best-fitting model).

Outputs: results/recovery.json, results/recovery_summary.json,
         figures/fig4_parameter_recovery.*, figures/fig5_model_recovery.*
Usage:   python scripts/03_recovery.py [--quick] [--plot-only]
"""

import json
import os
import sys

import numpy as np
from scipy.stats import pearsonr, spearmanr

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from hopfield_rl.experiments import FIGURES, RESULTS, pmap, recovery_task
from hopfield_rl.plotting import (INK, INK2, MODEL_COLORS, MUTED, SEQ_CMAP, panel_label, plt, save,
                                  set_style)

QUICK = "--quick" in sys.argv
PLOT_ONLY = "--plot-only" in sys.argv
OUT = os.path.join(RESULTS, "recovery.json")
MODELS4 = ["Q", "QK", "HOP", "HYB"]
LABEL = {"Q": "Q-learning", "QK": "Q + stickiness", "HOP": "Hopfield", "HYB": "Hybrid"}
SYMBOL = {"alpha": "α", "theta": "θ", "kappa": "κ", "beta": "β", "lam": "λ", "L": "L", "M": "M"}
PARAMS = {"Q": ["alpha", "theta"], "QK": ["alpha", "theta", "kappa"],
          "HOP": ["beta", "theta", "L", "M"], "HYB": ["alpha", "theta", "beta", "lam", "L", "M"]}


def run():
    n = 12 if QUICK else 100
    parts = os.path.join(RESULTS, "recovery_quick_parts" if QUICK else "recovery_parts")
    tasks = [{"gen_model": m, "seed": [77, i, j], "fit_models": MODELS4,
              "n_starts": 10, "n_starts_grid": 4,
              "cache": os.path.join(parts, f"{m}_{j:03d}.json")}     # resumable
             for i, m in enumerate(MODELS4) for j in range(n)]
    res = pmap(recovery_task, tasks, desc="recovery")
    os.makedirs(RESULTS, exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(res, f)
    return res


def confusion(res, criterion):
    C = np.zeros((4, 4))
    for r in res:
        scores = [r["fits"][m][criterion] for m in MODELS4]
        C[MODELS4.index(r["gen_model"]), int(np.argmin(scores))] += 1
    return C


def summarise(res):
    summary = {"n_per_model": {m: sum(r["gen_model"] == m for r in res) for m in MODELS4},
               "parameter_recovery": {}, "confusion": {}}
    for m in MODELS4:
        rows = [r for r in res if r["gen_model"] == m]
        summary["parameter_recovery"][m] = {}
        for p in PARAMS[m]:
            true = np.array([r["true"][p] for r in rows], dtype=float)
            fit = np.array([r["fits"][m]["params"][p] for r in rows], dtype=float)
            entry = {"spearman": float(spearmanr(true, fit)[0])}
            if p in ("L", "M"):
                entry["exact_match"] = float(np.mean(true == fit))
            else:
                entry["pearson"] = float(pearsonr(true, fit)[0])
                entry["median_abs_error"] = float(np.median(np.abs(true - fit)))
            summary["parameter_recovery"][m][p] = entry
        summary["parameter_recovery"][m]["median_nll_spread_between_starts"] = float(
            np.median([r["fits"][m]["start_spread"] for r in rows]))
    for crit in ("bic", "aic"):
        C = confusion(res, crit)
        summary["confusion"][crit] = {
            "counts": C.tolist(),
            "p_fit_given_sim": (C / C.sum(axis=1, keepdims=True)).tolist(),
            "p_sim_given_fit": (C / np.maximum(C.sum(axis=0, keepdims=True), 1)).tolist(),
            "accuracy": float(np.trace(C) / C.sum()),
        }
    # two-family view: incremental (Q, Q+kappa) vs. models with an episodic memory
    C = confusion(res, "bic")
    fam = np.array([[C[:2, :2].sum(), C[:2, 2:].sum()], [C[2:, :2].sum(), C[2:, 2:].sum()]])
    summary["confusion"]["bic_family"] = {
        "rows_cols": ["incremental (Q, Q+kappa)", "episodic (Hopfield, Hybrid)"],
        "p_fit_given_sim": (fam / fam.sum(axis=1, keepdims=True)).tolist()}
    with open(os.path.join(RESULTS, "recovery_summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    return summary


def figure_parameters(res, summary):
    panels = [(m, p) for m in MODELS4 for p in PARAMS[m]]
    fig, axes = plt.subplots(3, 5, figsize=(7.2, 4.4), constrained_layout=True)
    axes = axes.ravel()
    rng = np.random.default_rng(0)
    for ax, (m, p) in zip(axes, panels):
        rows = [r for r in res if r["gen_model"] == m]
        true = np.array([r["true"][p] for r in rows], dtype=float)
        fit = np.array([r["fits"][m]["params"][p] for r in rows], dtype=float)
        stats = summary["parameter_recovery"][m][p]
        if p in ("L", "M"):
            levels = sorted(set(true) | set(fit))
            pos = {v: i for i, v in enumerate(levels)}
            x = np.array([pos[v] for v in true]) + rng.uniform(-0.18, 0.18, len(true))
            y = np.array([pos[v] for v in fit]) + rng.uniform(-0.18, 0.18, len(fit))
            ax.plot([-0.5, len(levels) - 0.5], [-0.5, len(levels) - 0.5], color=MUTED, lw=0.8)
            ax.scatter(x, y, s=5, color=MODEL_COLORS[m], alpha=0.6, lw=0)
            ax.set_xticks(range(len(levels)))
            ax.set_xticklabels([int(v) for v in levels])
            ax.set_yticks(range(len(levels)))
            ax.set_yticklabels([int(v) for v in levels])
            note = f"exact {100 * stats['exact_match']:.0f}%"
        else:
            # a few fits run to the parameter bound; they are drawn as triangles at the
            # top edge so that the bulk of the points stays readable
            top = 1.5 * true.max()
            out = fit > top
            lo, hi = min(true.min(), fit.min()), top if out.any() else max(true.max(), fit.max())
            ax.plot([lo, min(hi, true.max() * 1.05)], [lo, min(hi, true.max() * 1.05)], color=MUTED, lw=0.8)
            ax.scatter(true[~out], fit[~out], s=5, color=MODEL_COLORS[m], alpha=0.6, lw=0)
            if out.any():
                ax.scatter(true[out], np.full(out.sum(), top), s=9, marker="^",
                           color=MODEL_COLORS[m], lw=0, clip_on=False)
            note = f"ρ = {stats['spearman']:.2f}" + (f"  ({out.sum()} off scale)" if out.any() else "")
        ax.set_title(f"{LABEL[m]}: {SYMBOL[p]}", fontsize=7.5)
        ax.text(0.04, 0.96, note, transform=ax.transAxes, va="top", fontsize=6.5, color=INK2)
    for ax in axes[len(panels):]:
        ax.axis("off")
    fig.supxlabel("true (simulated) value", fontsize=8, color=INK2)
    fig.supylabel("recovered (fitted) value", fontsize=8, color=INK2)
    save(fig, os.path.join(FIGURES, "fig4_parameter_recovery"))


def heat(ax, P, title, xlabel, ylabel):
    ax.imshow(P, cmap=SEQ_CMAP, vmin=0, vmax=1)
    for i in range(P.shape[0]):
        for j in range(P.shape[1]):
            ax.text(j, i, f"{P[i, j]:.2f}", ha="center", va="center", fontsize=7,
                    color="white" if P[i, j] > 0.55 else INK)
    names = [LABEL[m].replace(" + stickiness", "+κ").replace("-learning", "") for m in MODELS4]
    ax.set_xticks(range(4))
    ax.set_xticklabels(names, fontsize=6.5)
    ax.set_yticks(range(4))
    ax.set_yticklabels(names, fontsize=6.5)
    ax.set(title=title, xlabel=xlabel, ylabel=ylabel)
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)


def figure_models(summary):
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.5), constrained_layout=True)
    heat(axes[0], np.array(summary["confusion"]["bic"]["p_fit_given_sim"]),
         "Confusion matrix (BIC)", "best-fitting model", "simulated model")
    heat(axes[1], np.array(summary["confusion"]["bic"]["p_sim_given_fit"]),
         "Inversion matrix (BIC)", "best-fitting model", "simulated model")
    heat(axes[2], np.array(summary["confusion"]["aic"]["p_fit_given_sim"]),
         "Confusion matrix (AIC)", "best-fitting model", "simulated model")
    for ax, letter in zip(axes, "ABC"):
        panel_label(ax, letter)
    save(fig, os.path.join(FIGURES, "fig5_model_recovery"))


def main():
    set_style()
    if PLOT_ONLY:
        with open(OUT) as f:
            res = json.load(f)
    else:
        res = run()
    summary = summarise(res)
    print(json.dumps(summary, indent=1))
    figure_parameters(res, summary)
    figure_models(summary)


if __name__ == "__main__":
    main()
