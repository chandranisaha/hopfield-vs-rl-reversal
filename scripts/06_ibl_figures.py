"""Model comparison on the real mouse data: statistics and figures.

  * per-mouse 5-fold cross-validated log-likelihood of every model, BIC as a check;
  * sign test across mice for the best model (plus effect sizes);
  * observed vs. model choice curves aligned to block switches (held-out one-step-ahead
    predictions and posterior-predictive simulations), on low-contrast trials where
    the stimulus is ambiguous.

Outputs: results/ibl_summary.json, figures/fig6_ibl_comparison.*, figures/fig7_ibl_switch.*
"""

import glob
import json
import os
import sys

import numpy as np
from scipy.stats import binomtest, wilcoxon

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from hopfield_rl import ibl
from hopfield_rl.experiments import DATA, FIGURES, RESULTS
from hopfield_rl.likelihood import LABELS
from hopfield_rl.plotting import (DATA_COLOR, INK2, MODEL_COLORS, MUTED, panel_label, plt, save,
                                  set_style)

FOUR = ["Q", "QK", "HOP", "HYB"]                 # the four models of the proposal
PRE, POST = 10, 40
LOW_CONTRAST = 0.0625                            # "ambiguous" trials: 0 % and 6.25 % contrast
SHORT = {"STIM": "No history", "Q": "Q", "QK": "Q+κ", "HOP": "Hopfield", "HYB": "Hybrid",
         "HOPC": "Classical", "HOPK": "Hopfield+κ", "HYBK": "Hybrid+κ"}


def load():
    table = ibl.load_table(os.path.join(DATA, ibl.TABLE_NAME))
    mice = []
    for path in sorted(glob.glob(os.path.join(RESULTS, "ibl", "*.json"))):
        with open(path) as f:
            s = json.load(f)
        s["arrays"] = dict(np.load(path.replace(".json", ".npz")))
        s["table"] = table[table.subject == s["subject"]].reset_index(drop=True)
        assert len(s["table"]) == s["n_trials"]
        mice.append(s)
    return table, mice


def sign_test(a, b):
    """Two-sided sign test of a > b across mice (ties dropped)."""
    wins, losses = int(np.sum(a > b)), int(np.sum(a < b))
    p = binomtest(wins, wins + losses, 0.5).pvalue if wins + losses else 1.0
    return wins, losses, float(p)


def aligned(mouse, values, low_only=True):
    """Mean of ``values`` (per-trial P[choice towards the new block side]) at every
    offset from the block switches of one mouse."""
    t = mouse["table"]
    idx, new_right = ibl.block_switches(t)
    p_left = t.p_left.to_numpy()
    sess = t.session.to_numpy()
    use = np.abs(t.signed_contrast.to_numpy()) <= LOW_CONTRAST if low_only else np.ones(len(t), bool)
    offsets = np.arange(-PRE, POST)
    acc = np.full((len(idx), len(offsets)), np.nan)
    for i, (t0, right) in enumerate(zip(idx, new_right)):
        for j, d in enumerate(offsets):
            k = t0 + d
            if k < 0 or k >= len(t) or sess[k] != sess[t0] or not use[k]:
                continue
            # stay inside the old block (d < 0) or the new block (d >= 0)
            if p_left[k] != (p_left[t0 - 1] if d < 0 else p_left[t0]):
                continue
            if d < 0 and np.any(p_left[k:t0] != p_left[t0 - 1]):
                continue
            if d >= 0 and np.any(p_left[t0:k + 1] != p_left[t0]):
                continue
            acc[i, j] = values[k] if right else 1.0 - values[k]
    with np.errstate(invalid="ignore"):
        return offsets, np.nanmean(acc, axis=0)


def main():
    if hasattr(sys.stdout, "reconfigure"):          # Greek letters on a Windows console
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    set_style()
    table, mice = load()
    models = [m for m in ["STIM", "Q", "QK", "HOP", "HYB", "HOPC", "HOPK", "HYBK"]
              if m in mice[0]["models"]]
    n = len(mice)
    cv = {m: np.array([s["models"][m]["cv_ll_per_trial"] for s in mice]) for m in models}
    cv_tot = {m: np.array([s["models"][m]["cv_ll"] for s in mice]) for m in models}
    bic = {m: np.array([s["models"][m]["bic"] for s in mice]) for m in models}
    ntr = np.array([s["n_trials"] for s in mice])

    # ---------------------------------------------------------------- statistics
    summary = {"n_mice": n, "n_trials_total": int(ntr.sum()),
               "n_sessions_total": int(sum(s["n_sessions"] for s in mice)),
               "mice": [{"subject": s["subject"], "lab": s["lab"], "n_trials": s["n_trials"],
                         "n_sessions": s["n_sessions"],
                         "accuracy": float(s["table"].reward.mean())} for s in mice],
               "models": {}}
    for m in models:
        summary["models"][m] = {
            "label": LABELS[m],
            "mean_cv_ll_per_trial": float(cv[m].mean()),
            "mean_choice_probability": float(np.exp(cv[m]).mean()),
            "mean_delta_vs_Q_per_1000_trials": float(1000 * (cv[m] - cv["Q"]).mean()),
            "sem_delta_vs_Q_per_1000_trials": float(1000 * (cv[m] - cv["Q"]).std(ddof=1) / np.sqrt(n)),
            "mean_delta_vs_STIM_per_1000_trials": float(1000 * (cv[m] - cv["STIM"]).mean()),
            "sum_cv_ll": float(cv_tot[m].sum()), "sum_bic": float(bic[m].sum()),
            "k": mice[0]["models"][m]["k"],
        }

    def compare(group, name):
        best_cv = [group[int(np.argmax([cv[m][i] for m in group]))] for i in range(n)]
        best_bic = [group[int(np.argmin([bic[m][i] for m in group]))] for i in range(n)]
        winner = max(group, key=lambda m: cv[m].mean())
        tests = {}
        for m in group:
            if m == winner:
                continue
            wins, losses, p = sign_test(cv[winner], cv[m])
            d = cv[winner] - cv[m]
            tests[m] = {"mice_winner_better": wins, "mice_other_better": losses, "sign_test_p": p,
                        "wilcoxon_p": float(wilcoxon(d).pvalue),
                        "mean_delta_ll_per_1000_trials": float(1000 * d.mean()),
                        "mean_delta_ll_total_per_mouse": float((cv_tot[winner] - cv_tot[m]).mean())}
        summary[name] = {"models": group, "best_by_mean_cv": winner,
                         "mice_won_cv": {m: best_cv.count(m) for m in group},
                         "mice_won_bic": {m: best_bic.count(m) for m in group},
                         "best_by_summed_bic": min(group, key=lambda m: bic[m].sum()),
                         "winner_vs_others": tests,
                         "per_mouse_best_cv": best_cv, "per_mouse_best_bic": best_bic}

    compare(FOUR, "four_models")
    compare([m for m in models if m != "STIM"], "all_history_models")
    # direct tests of the proposal's question
    for a, b in [("HOP", "Q"), ("HOP", "QK"), ("HYB", "Q"), ("HYB", "QK"), ("HYB", "HOP"),
                 ("QK", "Q"), ("Q", "STIM"), ("HYBK", "QK"), ("HOPK", "QK"), ("HYBK", "HYB"),
                 ("HOPC", "HOP")]:
        if a in cv and b in cv:
            wins, losses, p = sign_test(cv[a], cv[b])
            summary.setdefault("pairwise", {})[f"{a}_vs_{b}"] = {
                "mice_first_better": wins, "mice_second_better": losses, "sign_test_p": p,
                "mean_delta_ll_per_1000_trials": float(1000 * (cv[a] - cv[b]).mean())}
    # fitted parameters (full-data fits)
    summary["parameters"] = {}
    for m in models:
        names = list(mice[0]["models"][m]["params"])
        summary["parameters"][m] = {
            p: {"median": float(np.median([s["models"][m]["params"][p] for s in mice])),
                "q25": float(np.percentile([s["models"][m]["params"][p] for s in mice], 25)),
                "q75": float(np.percentile([s["models"][m]["params"][p] for s in mice], 75)),
                "values": [s["models"][m]["params"][p] for s in mice]} for p in names}

    # ---------------------------------------------------------------- switch-aligned curves
    curves = {"data": [], **{f"cv_{m}": [] for m in models}, **{f"sim_{m}": [] for m in models}}
    for s in mice:
        choice = s["table"].choice_right.to_numpy().astype(float)
        offsets, c = aligned(s, choice)
        curves["data"].append(c)
        for m in models:
            curves[f"cv_{m}"].append(aligned(s, s["arrays"][f"cv_p_{m}"])[1])
            curves[f"sim_{m}"].append(aligned(s, s["arrays"][f"sim_{m}"])[1])
    curves = {k: np.array(v) for k, v in curves.items()}
    post = offsets >= 0
    summary["switch_curves"] = {
        "offsets": offsets.tolist(), "low_contrast_max": LOW_CONTRAST,
        "data_mean": np.nanmean(curves["data"], axis=0).tolist(),
        "rmse_heldout_prediction": {}, "rmse_simulation": {}}
    for m in models:
        for kind, key in (("cv", "rmse_heldout_prediction"), ("sim", "rmse_simulation")):
            diff = np.nanmean(curves[f"{kind}_{m}"], axis=0) - np.nanmean(curves["data"], axis=0)
            summary["switch_curves"][key][m] = float(np.sqrt(np.nanmean(diff[post] ** 2)))
        summary["switch_curves"][f"sim_mean_{m}"] = np.nanmean(curves[f"sim_{m}"], axis=0).tolist()
        summary["switch_curves"][f"cv_mean_{m}"] = np.nanmean(curves[f"cv_{m}"], axis=0).tolist()
    with open(os.path.join(RESULTS, "ibl_summary.json"), "w") as f:
        json.dump(summary, f, indent=1)

    # ---------------------------------------------------------------- figure 6
    fig, axes = plt.subplots(1, 4, figsize=(7.2, 2.3), constrained_layout=True,
                             gridspec_kw={"width_ratios": [1.0, 1.35, 0.9, 0.9]})
    ax = axes[0]                          # psychometric curves by block
    levels = np.unique(table.signed_contrast)
    xpos = np.sign(levels) * np.array([0, 1, 2, 3, 4])[np.searchsorted(
        np.unique(np.abs(levels)), np.abs(levels))]
    block_cols = {0.8: "#2a78d6", 0.5: MUTED, 0.2: "#eb6834"}
    best4 = summary["four_models"]["best_by_mean_cv"]
    for pl, col in block_cols.items():
        obs, pred = [], []
        for s in mice:
            t = s["table"]
            sel = np.isclose(t.p_left.to_numpy(), pl)
            sc = t.signed_contrast.to_numpy()
            obs.append([t.choice_right.to_numpy()[sel & (sc == c)].mean() for c in levels])
            pred.append([s["arrays"][f"cv_p_{best4}"][sel & (sc == c)].mean() for c in levels])
        obs, pred = np.array(obs), np.array(pred)
        ax.plot(xpos, np.nanmean(pred, axis=0), color=col, lw=1.2)
        ax.errorbar(xpos, np.nanmean(obs, axis=0), yerr=np.nanstd(obs, axis=0, ddof=1) / np.sqrt(n),
                    fmt="o", color=col, ms=2.8, elinewidth=0.7,
                    label=f"P(left) = {pl}")
    ax.set_xticks([-4, -2, 0, 2, 4])
    ax.set_xticklabels(["-100", "-12.5", "0", "12.5", "100"])
    ax.set(xlabel="signed contrast (%)", ylabel="P(choose right)", ylim=(0, 1),
           title="Block prior")
    ax.legend(fontsize=5, loc="lower right", title=f"dots: mice\nlines: {SHORT[best4]}",
              title_fontsize=5, handletextpad=0.2, borderpad=0.2)
    panel_label(ax, "A")

    ax = axes[1]                          # CV log-likelihood gain over the no-history model
    show = [m for m in models if m != "STIM"]
    rng = np.random.default_rng(0)
    for j, m in enumerate(show):
        gain = 1000 * (cv[m] - cv["STIM"])
        ax.scatter(j + rng.uniform(-0.16, 0.16, n), gain, s=7, color=MODEL_COLORS[m], alpha=0.55, lw=0)
        ax.errorbar(j, gain.mean(), yerr=gain.std(ddof=1) / np.sqrt(n), fmt="_", color=DATA_COLOR,
                    ms=11, mew=1.4, elinewidth=1.0, capsize=0)
    if len(show) > 4:
        ax.axvline(3.5, color=MUTED, lw=0.6, ls=":")
    ax.set_xticks(range(len(show)))
    ax.set_xticklabels([SHORT[m] for m in show], fontsize=5.8, rotation=30, ha="right")
    ax.set(ylabel="held-out log-lik. gain over\nno-history model (nats / 1000 trials)",
           title="5-fold CV")
    panel_label(ax, "B")

    ax = axes[2]                          # paired differences to Q + kappa
    ref = "QK"
    others = [m for m in FOUR if m != ref]
    for j, m in enumerate(others):
        d = 1000 * (cv[m] - cv[ref])
        ax.scatter(j + rng.uniform(-0.15, 0.15, n), d, s=7, color=MODEL_COLORS[m], alpha=0.55, lw=0)
        ax.errorbar(j, d.mean(), yerr=d.std(ddof=1) / np.sqrt(n), fmt="_", color=DATA_COLOR, ms=11,
                    mew=1.4, elinewidth=1.0)
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.set_xticks(range(len(others)))
    ax.set_xticklabels([SHORT[m] for m in others], fontsize=5.8, rotation=30, ha="right")
    ax.set(ylabel="Δ held-out log-lik. vs. Q+κ\n(nats / 1000 trials)", title="Paired")
    panel_label(ax, "C")

    ax = axes[3]                          # mice won
    won_cv = summary["four_models"]["mice_won_cv"]
    won_bic = summary["four_models"]["mice_won_bic"]
    x = np.arange(4)
    ax.bar(x - 0.2, [won_cv[m] for m in FOUR], width=0.36, color=[MODEL_COLORS[m] for m in FOUR])
    ax.bar(x + 0.2, [won_bic[m] for m in FOUR], width=0.36, color=[MODEL_COLORS[m] for m in FOUR],
           alpha=0.45)
    for xi, m in zip(x, FOUR):
        ax.text(xi - 0.2, won_cv[m] + 0.2, str(won_cv[m]), ha="center", fontsize=6, color=INK2)
        ax.text(xi + 0.2, won_bic[m] + 0.2, str(won_bic[m]), ha="center", fontsize=6, color=INK2)
    ax.set_xticks(x)
    ax.set_xticklabels([SHORT[m] for m in FOUR], fontsize=5.8, rotation=30, ha="right")
    ax.set(ylabel="mice best fit", title="Winner", ylim=(0, n + 1.5))
    ax.set_yticks(range(0, n + 1, 2))
    ax.text(0.98, 0.97, "solid: CV   pale: BIC", transform=ax.transAxes, ha="right", va="top",
            fontsize=6, color=INK2)
    panel_label(ax, "D")
    save(fig, os.path.join(FIGURES, "fig6_ibl_comparison"))

    # ---------------------------------------------------------------- figure 7
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.3), constrained_layout=True)
    d_mean = np.nanmean(curves["data"], axis=0)
    d_sem = np.nanstd(curves["data"], axis=0, ddof=1) / np.sqrt(n)
    for ax, kind, title in [(axes[0], "cv", "Held-out predictions"),
                            (axes[1], "sim", "Simulated models")]:
        ax.fill_between(offsets, d_mean - d_sem, d_mean + d_sem, color=DATA_COLOR, alpha=0.15, lw=0)
        ax.plot(offsets, d_mean, color=DATA_COLOR, lw=1.4, label="mice")
        for m in ["STIM"] + FOUR:
            ax.plot(offsets, np.nanmean(curves[f"{kind}_{m}"], axis=0), color=MODEL_COLORS[m],
                    lw=1.1, label=SHORT[m])
        ax.axvline(-0.5, color=MUTED, lw=0.8, ls=":")
        ax.axhline(0.5, color=MUTED, lw=0.5)
        ax.set(xlabel="trials from block switch", title=title, xlim=(-PRE, POST - 1))
    axes[0].set_ylabel("P(choice towards new block side)\n0 % and 6.25 % contrast trials")
    axes[1].legend(fontsize=5.5, loc="lower right", ncol=2)
    panel_label(axes[0], "A")
    panel_label(axes[1], "B")
    ax = axes[2]
    for m in ["STIM"] + FOUR:
        ax.plot(offsets, np.nanmean(curves[f"sim_{m}"], axis=0) - d_mean, color=MODEL_COLORS[m],
                lw=1.1, label=SHORT[m])
    ax.fill_between(offsets, -d_sem, d_sem, color=DATA_COLOR, alpha=0.15, lw=0)
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.axvline(-0.5, color=MUTED, lw=0.8, ls=":")
    ax.set(xlabel="trials from block switch", ylabel="simulated model − mice",
           title="Residuals", xlim=(-PRE, POST - 1))
    panel_label(ax, "C")
    save(fig, os.path.join(FIGURES, "fig7_ibl_switch"))

    # ---------------------------------------------------------------- console report
    print(f"{n} mice, {ntr.sum()} trials")
    for m in models:
        v = summary["models"][m]
        print(f"  {LABELS[m]:24s} CV LL/trial {v['mean_cv_ll_per_trial']:.4f}  "
              f"Δ vs Q {v['mean_delta_vs_Q_per_1000_trials']:+7.2f} ± "
              f"{v['sem_delta_vs_Q_per_1000_trials']:.2f} /1000 trials   sum BIC {v['sum_bic']:.0f}")
    for key in ("four_models", "all_history_models"):
        v = summary[key]
        print(f"\n{key}: best = {v['best_by_mean_cv']}; mice won (CV) {v['mice_won_cv']}; "
              f"(BIC) {v['mice_won_bic']}")
        for m, t in v["winner_vs_others"].items():
            print(f"   vs {m:5s}: {t['mice_winner_better']}/{n} mice, sign test p = "
                  f"{t['sign_test_p']:.4g}, Δ = {t['mean_delta_ll_per_1000_trials']:.2f} nats/1000 trials")
    print("\npairwise:", json.dumps(summary["pairwise"], indent=1))
    print("\nswitch-curve RMSE (simulation):", summary["switch_curves"]["rmse_simulation"])
    print("switch-curve RMSE (held-out):", summary["switch_curves"]["rmse_heldout_prediction"])
    for m in models:
        print(m, {p: (round(v["median"], 3), round(v["q25"], 3), round(v["q75"], 3))
                  for p, v in summary["parameters"][m].items()})


if __name__ == "__main__":
    main()
