"""Maximum-likelihood fitting (proposal Step 3) and model-comparison utilities.

Continuous parameters are fitted by bounded minimisation of the negative
log-likelihood with ``scipy.optimize.minimize`` (L-BFGS-B, exact analytic gradients)
from many random starts.  The integer memory parameters L (key length) and M (memory
window) cannot be handled by a gradient-based optimiser, so they are fitted by an outer
grid search: the continuous parameters are optimised inside every (L, M) cell and the
best cell is kept (a profile likelihood, i.e. still the joint maximum-likelihood
estimate on the grid).

The likelihood surface has several local minima (most visibly for the hybrid, which
contains the incremental and the episodic model as the special cases lambda = 0 and
lambda = 1).  In addition to the random starts, the hybrid is therefore also started
from the optima of its two nested models fitted to the same data, which guarantees
that the optimiser explores the neighbourhood of both.
"""

from __future__ import annotations

import itertools

import numpy as np
from scipy.optimize import minimize

from .likelihood import BOUNDS, INIT_RANGE, MODELS, STIM_BOUNDS, STIM_INIT, n_free_params

DEFAULT_L_GRID = (1, 2, 3)
DEFAULT_M_GRID = (2, 4, 8, 16, 32, 64)
NESTED = {"HYB": ("Q", "HOP"), "HYBK": ("QK", "HOPK")}   # hybrid -> (incremental, episodic)


def _bounds_and_inits(model, n_stim):
    names = MODELS[model][0]
    bounds = [STIM_BOUNDS] * n_stim + [BOUNDS[n] for n in names]
    inits = [STIM_INIT] * n_stim + [INIT_RANGE[n] for n in names]
    return bounds, inits


def _fit_cell(lik, model, disc, rows, n_starts, rng, extra_starts=()):
    """Multi-start bounded optimisation of the continuous parameters for fixed (L, M)."""
    n_stim = lik.data.n_stim
    bounds, inits = _bounds_and_inits(model, n_stim)
    n = lik.T if rows is None else int(np.sum(rows))
    if len(bounds) == 0:                       # nothing to optimise
        f = lik.nll(model, np.zeros(0), disc, rows)
        return np.zeros(0), f, [f]

    def objective(x):                          # mean NLL per trial and its exact gradient
        f, g = lik.nll_and_grad(model, x, disc, rows)
        return f / n, g / n

    lo = np.array([b[0] for b in bounds])
    hi = np.array([b[1] for b in bounds])
    starts = [np.clip(np.asarray(x0, dtype=float), lo, hi) for x0 in extra_starts]
    starts += [np.array([rng.uniform(a, b) for a, b in inits]) for _ in range(n_starts)]
    best_x, best_f, all_f = None, np.inf, []
    for x0 in starts:
        res = minimize(objective, x0, method="L-BFGS-B", jac=True, bounds=bounds,
                       options={"maxiter": 300, "ftol": 1e-12, "gtol": 1e-7})
        f = float(res.fun) * n
        all_f.append(f)
        if np.isfinite(f) and f < best_f:
            best_x, best_f = res.x, f
    return best_x, best_f, all_f


def _hybrid_starts(lik, model, x_inc, x_epi):
    """Starting points of the hybrid built from the optima of its nested models:
    (i) nearly pure incremental, (ii) nearly pure episodic, (iii) both value signals
    at their individually fitted strengths:  theta = theta_inc + theta_epi,
    lambda = theta_epi / theta."""
    inc_model, epi_model = NESTED[model]
    w_inc, p_inc = lik.unpack(inc_model, x_inc)
    w_epi, p_epi = lik.unpack(epi_model, x_epi)
    beta = max(p_epi["beta"], 0.5)
    names = MODELS[model][0]

    def build(w, theta, lam, kappa):
        p = {"alpha": p_inc["alpha"], "theta": theta, "beta": beta, "lam": lam, "kappa": kappa}
        return np.concatenate([w, [p[name] for name in names]])

    k_inc, k_epi = p_inc.get("kappa", 0.0), p_epi.get("kappa", 0.0)
    total = p_inc["theta"] + p_epi["theta"]
    lam_mix = p_epi["theta"] / total if total > 0 else 0.5
    return [build(w_inc, p_inc["theta"], 0.02, k_inc),
            build(w_epi, p_epi["theta"], 0.98, k_epi),
            build(0.5 * (w_inc + w_epi), total, lam_mix, 0.5 * (k_inc + k_epi))]


def fit_model(lik, model, rows=None, n_starts=10, rng=None, L_grid=DEFAULT_L_GRID,
              M_grid=DEFAULT_M_GRID, disc=None, n_starts_nested=6, hybrid_top_cells=None):
    """Fit one model to the data behind ``lik`` (a ``Likelihood`` object).

    rows     : optional boolean mask of the trials that enter the likelihood (training set)
    n_starts : random starts (per (L, M) cell for the episodic models)
    disc     : fix (L, M) instead of searching the grid
    hybrid_top_cells : if set, a hybrid model is optimised only in the (L, M) cells in
               which its nested episodic model reaches the highest likelihood (the
               episodic model is fitted in every cell first).  Used on the large mouse
               data sets to save computation; None searches every cell.
    Returns a dict with the estimates, the minimum NLL, AIC, BIC and diagnostics.
    """
    rng = np.random.default_rng() if rng is None else rng
    episodic = MODELS[model][1]
    cells = [None] if not episodic else ([tuple(disc)] if disc is not None
                                         else list(itertools.product(L_grid, M_grid)))
    x_inc = None
    if model in NESTED:
        x_inc = _fit_cell(lik, NESTED[model][0], None, rows, n_starts_nested, rng)[0]
    x_epi = {}
    if model in NESTED:
        f_epi = {}
        for cell in cells:
            x_epi[cell], f_epi[cell], _ = _fit_cell(lik, NESTED[model][1], cell, rows, 2, rng)
        if hybrid_top_cells is not None:
            cells = sorted(cells, key=lambda c: f_epi[c])[:hybrid_top_cells]
    best = None
    profile = {}
    for cell in cells:
        extra = ()
        if model in NESTED:
            extra = _hybrid_starts(lik, model, x_inc, x_epi[cell])
        x, f, all_f = _fit_cell(lik, model, cell, rows, n_starts, rng, extra)
        if cell is not None:
            profile[cell] = f
        if best is None or f < best["nll"]:
            best = {"x": x, "nll": f, "disc": cell, "start_nlls": all_f}
    n = lik.T if rows is None else int(np.sum(rows))
    k = n_free_params(model, lik.data.n_stim)
    w, p = lik.unpack(model, best["x"])
    params = {name: float(v) for name, v in p.items()}
    if best["disc"] is not None:
        params["L"], params["M"] = int(best["disc"][0]), int(best["disc"][1])
    best.update({
        "model": model, "params": params, "stim_weights": np.asarray(w, dtype=float),
        "n": n, "k": k,
        "aic": 2.0 * k + 2.0 * best["nll"],
        "bic": k * np.log(n) + 2.0 * best["nll"],
        "profile": profile,
    })
    return best


def cross_validate(lik, model, folds, n_starts=5, rng=None, **fit_kwargs):
    """K-fold cross-validated log-likelihood over held-out sessions.

    folds : (T,) int array assigning every trial to a fold (all trials of a session
            share a fold).  For each fold the model (including L and M) is fitted on
            the other folds and evaluated on the held-out trials.
    Returns (per-trial held-out log-likelihood, per-trial held-out P(arm 1), fits).
    """
    rng = np.random.default_rng() if rng is None else rng
    ll = np.full(lik.T, np.nan)
    p1 = np.full(lik.T, np.nan)
    fits = []
    for f in np.unique(folds):
        test = folds == f
        fit = fit_model(lik, model, rows=~test, n_starts=n_starts, rng=rng, **fit_kwargs)
        ll[test] = lik.trial_loglik(model, fit["x"], fit["disc"])[test]
        p1[test] = lik.p_arm1(model, fit["x"], fit["disc"])[test]
        fits.append(fit)
    return ll, p1, fits


def session_folds(session, n_folds=5, rng=None):
    """Assign whole sessions to folds (shuffled, sizes as equal as possible)."""
    rng = np.random.default_rng() if rng is None else rng
    ids = np.unique(session)
    perm = rng.permutation(len(ids))
    fold_of = np.empty(len(ids), dtype=np.int64)
    fold_of[perm] = np.arange(len(ids)) % n_folds
    return fold_of[np.searchsorted(ids, session)]
