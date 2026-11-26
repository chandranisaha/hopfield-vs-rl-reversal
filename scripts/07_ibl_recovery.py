"""Model recovery in the regime of the real data ("can we trust the mouse fits?").

The recovery analysis of script 03 uses the simulated reversal task.  The mouse task is
different (a stimulus drives most choices), so the check is repeated here under the
conditions of the real data: for every mouse, each of the four models - with the
parameters fitted to that mouse - performs the task on the mouse's own stimulus
sequence (same sessions, same trials; reward = choice matches the stimulus side).  All
four models are then fitted to each synthetic data set exactly as they are fitted to
the mice, and the winner is chosen by BIC.

Requires: results/ibl/*.json (script 05).
Outputs:  results/ibl_recovery.json, results/ibl_recovery_summary.json,
          figures/fig8_ibl_recovery.*
Usage:    python scripts/07_ibl_recovery.py [--plot-only]
"""

import json
import os
import sys

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")          # before numpy is imported (also in the workers)

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from hopfield_rl import ibl
from hopfield_rl.agents import make_agent
from hopfield_rl.experiments import DATA, FIGURES, RESULTS, pmap
from hopfield_rl.fitting import fit_model
from hopfield_rl.likelihood import ChoiceData, Likelihood
from hopfield_rl.plotting import INK, SEQ_CMAP, panel_label, plt, save, set_style

FOUR = ["Q", "QK", "HOP", "HYB"]
NAMES = ["Q", "Q+κ", "Hopfield", "Hybrid"]
L_GRID = (1, 2, 3)
M_GRID = (2, 4, 8, 16, 32, 64, 128)
STARTS = {"Q": 12, "QK": 12, "HOP": 3, "HYB": 1}
OUT = os.path.join(RESULTS, "ibl_recovery.json")
PARTS = os.path.join(RESULTS, "ibl_recovery_parts")


def recover(task):
    subject, index, gen = task
    part = os.path.join(PARTS, f"{subject}__{gen}.json")
    if os.path.exists(part):                 # resumable: finished tasks are skipped
        with open(part) as f:
            return json.load(f)
    table = ibl.load_table(os.path.join(DATA, ibl.TABLE_NAME))
    mt = table[table.subject == subject].reset_index(drop=True)
    real = ibl.to_choice_data(mt, ibl.contrast_levels(mt))
    with open(os.path.join(RESULTS, "ibl", f"{subject}.json")) as f:
        fitted = json.load(f)["models"][gen]
    # the fitted model performs the task on the mouse's stimulus sequence
    rng = np.random.default_rng([99, index, FOUR.index(gen)])
    ext = real.X @ np.array(fitted["stim_weights"])
    stim_right = mt.stim_right.to_numpy()
    choices = np.empty(real.n_trials, dtype=np.int64)
    agent = make_agent(gen, fitted["params"])
    for sl in real.session_slices():
        agent.reset()
        for t in range(sl.start, sl.stop):
            c = agent.choose(rng, ext[t])
            agent.update(c, int(c == stim_right[t]))
            choices[t] = c
    lik = Likelihood(ChoiceData(choices, (choices == stim_right).astype(int), real.session, real.X))
    fits = {}
    for m in FOUR:
        fit = fit_model(lik, m, n_starts=STARTS[m], rng=rng, L_grid=L_GRID, M_grid=M_GRID,
                        hybrid_top_cells=6 if m == "HYB" else None)
        fits[m] = {"params": fit["params"], "nll": fit["nll"], "bic": fit["bic"]}
    out = {"subject": subject, "gen_model": gen, "true": fitted["params"], "fits": fits,
           "accuracy": float(np.mean(choices == stim_right))}
    os.makedirs(PARTS, exist_ok=True)
    with open(part + ".tmp", "w") as f:
        json.dump(out, f)
    os.replace(part + ".tmp", part)
    return out


def main():
    set_style()
    if "--plot-only" in sys.argv:
        with open(OUT) as f:
            res = json.load(f)
    else:
        table = ibl.load_table(os.path.join(DATA, ibl.TABLE_NAME))
        subjects = sorted(table.subject.unique())
        # hybrids are the slowest fits: spread them evenly over the workers
        tasks = [(s, i, g) for g in FOUR for i, s in enumerate(subjects)]
        res = pmap(recover, tasks, desc="synthetic mice")
        with open(OUT, "w") as f:
            json.dump(res, f, indent=1)

    C = np.zeros((4, 4))
    for r in res:
        C[FOUR.index(r["gen_model"]), int(np.argmin([r["fits"][m]["bic"] for m in FOUR]))] += 1
    P = C / C.sum(axis=1, keepdims=True)
    fam = np.array([[C[:2, :2].sum(), C[:2, 2:].sum()], [C[2:, :2].sum(), C[2:, 2:].sum()]])
    summary = {"models": FOUR, "counts": C.tolist(), "p_fit_given_sim": P.tolist(),
               "accuracy": float(np.trace(C) / C.sum()),
               "family_p_fit_given_sim": (fam / fam.sum(axis=1, keepdims=True)).tolist(),
               "n_per_model": int(C.sum(axis=1)[0])}
    with open(os.path.join(RESULTS, "ibl_recovery_summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    print(json.dumps(summary, indent=1))

    fig, ax = plt.subplots(figsize=(2.6, 2.3), constrained_layout=True)
    ax.imshow(P, cmap=SEQ_CMAP, vmin=0, vmax=1)
    for i in range(4):
        for j in range(4):
            ax.text(j, i, f"{P[i, j]:.2f}", ha="center", va="center", fontsize=7,
                    color="white" if P[i, j] > 0.55 else INK)
    ax.set_xticks(range(4))
    ax.set_xticklabels(NAMES, fontsize=6.5)
    ax.set_yticks(range(4))
    ax.set_yticklabels(NAMES, fontsize=6.5)
    ax.set(xlabel="best-fitting model (BIC)", ylabel="simulated model",
           title="Recovery on the mouse task")
    ax.grid(False)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    save(fig, os.path.join(FIGURES, "fig8_ibl_recovery"))


if __name__ == "__main__":
    main()
