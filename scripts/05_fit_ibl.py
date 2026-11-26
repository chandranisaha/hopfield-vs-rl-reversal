"""Fit all models to the real mouse data (IBL) and cross-validate.

For every mouse and every model:
  * maximum-likelihood fit on all sessions (parameters, NLL, BIC as a check);
  * 5-fold cross-validation over held-out SESSIONS: all parameters, including the
    discrete L and M, are re-fitted on the training sessions of each fold, and the
    log-likelihood of the held-out sessions is recorded trial by trial;
  * a posterior-predictive simulation: the fitted model plays the mouse's own stimulus
    sequence (reward = choice matches the stimulus side) to obtain model choices around
    block switches that are not conditioned on the mouse's actual choices.

In every model the stimulus enters as a logistic term (one weight per contrast level and
a constant bias); the history mechanism supplies theta * (V(right) - V(left)).

Outputs: results/<tag>/<mouse>.json|npz and results/<tag>_fits.json   (tag = "ibl")
Usage:   python scripts/05_fit_ibl.py [--quick] [--mice N] [--table csv --tag name]
"""

import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from hopfield_rl import ibl
from hopfield_rl.agents import make_agent
from hopfield_rl.experiments import DATA, RESULTS, pmap
from hopfield_rl.fitting import cross_validate, fit_model, session_folds
from hopfield_rl.likelihood import MODELS, Likelihood

QUICK = "--quick" in sys.argv
MAIN_MODELS = ["STIM", "Q", "QK", "HOP", "HYB"]       # no-history reference + the four models
EXTRA_MODELS = ["HOPC", "HOPK", "HYBK"]               # classical variant, stickiness controls
ALL_MODELS = MAIN_MODELS + EXTRA_MODELS
L_GRID = (1, 2, 3)
M_GRID = (2, 4, 8, 16, 32, 64, 128)
HYBRID_TOP_CELLS = 6      # hybrids: only the 6 cells preferred by the nested episodic model
N_FOLDS = 5
N_SIM = 10
# random starts: (full-data fit, each CV fold); per (L, M) cell for the episodic models.
# The hybrids additionally start from the optima of their nested models in every cell.
STARTS = {"STIM": (3, 3), "Q": (12, 8), "QK": (12, 8), "HOP": (3, 2), "HOPK": (3, 2),
          "HOPC": (3, 2), "HYB": (1, 0), "HYBK": (1, 0)}
# optional: --table <csv> --tag <name> run the identical pipeline on another trial table
# (used for the validation of the pipeline on synthetic mice)
TABLE = sys.argv[sys.argv.index("--table") + 1] if "--table" in sys.argv else os.path.join(DATA, ibl.TABLE_NAME)
TAG = sys.argv[sys.argv.index("--tag") + 1] if "--tag" in sys.argv else "ibl"
OUT_DIR = os.path.join(RESULTS, TAG + ("_quick" if QUICK else ""))


def simulate_on_stimuli(model, params, w, data, stim_right, n_rep, seed):
    """Mean simulated P(choose right) per trial when the fitted model performs the task
    itself on the mouse's stimulus sequence."""
    rng = np.random.default_rng(seed)
    ext = data.X @ w
    total = np.zeros(data.n_trials)
    agent = make_agent(model, params)
    for _ in range(n_rep):
        for sl in data.session_slices():
            agent.reset()
            for t in range(sl.start, sl.stop):
                c = agent.choose(rng, ext[t])
                agent.update(c, int(c == stim_right[t]))
                total[t] += c
    return total / n_rep


def mouse_folds(data, index):
    """Session folds of a mouse: identical for every model."""
    return session_folds(data.session, N_FOLDS, np.random.default_rng([2026, index]))


def fit_one(task):
    """Fit one model to one mouse (full data + cross-validation + simulation)."""
    subject, index, model, quick, table_path, out_dir = task
    out = os.path.join(out_dir, "parts", f"{subject}__{model}")
    if os.path.exists(out + ".json"):                 # finished earlier: resume
        return out
    t_start = time.time()
    table = ibl.load_table(table_path)
    mt = table[table.subject == subject].reset_index(drop=True)
    data = ibl.to_choice_data(mt, ibl.contrast_levels(mt))   # the mouse's own contrast levels
    lik = Likelihood(data)
    folds = mouse_folds(data, index)
    rng = np.random.default_rng([2027, index, ALL_MODELS.index(model)])
    grid = dict(L_grid=(1, 2), M_grid=(2, 8, 32)) if quick else dict(L_grid=L_GRID, M_grid=M_GRID)
    if model in ("HYB", "HYBK"):
        grid["hybrid_top_cells"] = HYBRID_TOP_CELLS
    n_full, n_cv = STARTS[model]
    if quick:
        n_full, n_cv = min(n_full, 3), min(n_cv, 2)
    full = fit_model(lik, model, n_starts=n_full, rng=rng, **grid)
    ll, p1, fold_fits = cross_validate(lik, model, folds, n_starts=n_cv, rng=rng, **grid)
    sim = simulate_on_stimuli(model, full["params"], full["stim_weights"], data,
                              mt.stim_right.to_numpy(), 2 if quick else N_SIM,
                              seed=[7, index, ALL_MODELS.index(model)])
    summary = {
        "params": full["params"], "stim_weights": full["stim_weights"].tolist(),
        "nll": full["nll"], "bic": full["bic"], "aic": full["aic"], "k": full["k"],
        "start_spread": float(np.max(full["start_nlls"]) - np.min(full["start_nlls"])),
        "profile": {f"{L},{M}": v for (L, M), v in full["profile"].items()},
        "cv_ll": float(ll.sum()), "cv_ll_per_trial": float(ll.mean()),
        "cv_ll_by_fold": [float(ll[folds == f].sum()) for f in range(N_FOLDS)],
        "fold_params": [f["params"] for f in fold_fits],
        "minutes": (time.time() - t_start) / 60,
    }
    os.makedirs(os.path.dirname(out), exist_ok=True)
    np.savez_compressed(out + ".npz", cv_ll=ll, cv_p=p1, sim=sim, folds=folds)
    with open(out + ".json", "w") as f:
        json.dump(summary, f, indent=1)
    return out


def main():
    table = ibl.load_table(TABLE)
    subjects = sorted(table.subject.unique())
    if "--mice" in sys.argv:
        subjects = subjects[:int(sys.argv[sys.argv.index("--mice") + 1])]
    models = MAIN_MODELS if QUICK else ALL_MODELS
    print(f"fitting {len(models)} models to {len(subjects)} mice ({len(table)} trials)", flush=True)
    # slowest models first so that the workers stay balanced
    order = sorted(models, key=lambda m: -(len(MODELS[m][0]) + 10 * MODELS[m][1]))
    tasks = [(s, i, m, QUICK, TABLE, OUT_DIR) for m in order for i, s in enumerate(subjects)]
    pmap(fit_one, tasks, desc="fits")

    # merge the per-model parts into one record (+ one array file) per mouse
    res = []
    for s in subjects:
        mt = table[table.subject == s]
        rec = {"subject": s, "lab": str(mt.lab.iloc[0]), "n_trials": int(len(mt)),
               "n_sessions": int(mt.session.nunique()),
               "contrast_levels": ibl.contrast_levels(mt), "models": {}}
        arrays = {}
        for m in models:
            part = os.path.join(OUT_DIR, "parts", f"{s}__{m}")
            with open(part + ".json") as f:
                rec["models"][m] = json.load(f)
            npz = np.load(part + ".npz")
            arrays["folds"] = npz["folds"]
            arrays[f"cv_ll_{m}"], arrays[f"cv_p_{m}"], arrays[f"sim_{m}"] = npz["cv_ll"], npz["cv_p"], npz["sim"]
        np.savez_compressed(os.path.join(OUT_DIR, f"{s}.npz"), **arrays)
        with open(os.path.join(OUT_DIR, f"{s}.json"), "w") as f:
            json.dump(rec, f, indent=1)
        res.append(rec)
    with open(os.path.join(RESULTS, f"{TAG}_fits_quick.json" if QUICK else f"{TAG}_fits.json"), "w") as f:
        json.dump(res, f, indent=1)
    for r in res:
        row = "  ".join(f"{m}:{v['cv_ll_per_trial']:.4f}" for m, v in r["models"].items())
        mins = sum(v["minutes"] for v in r["models"].values())
        print(f"{r['subject']:14s} n={r['n_trials']:6d}  {row}  ({mins:.1f} cpu-min)")


if __name__ == "__main__":
    main()
