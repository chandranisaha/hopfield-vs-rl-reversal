"""Running agents on the environment and measuring reversal-aligned behaviour."""

from __future__ import annotations

import numpy as np


def run_session(agent, env, seed=None):
    """Let ``agent`` play one session of ``env``.

    Returns a dict with the choices, rewards, the better arm on every trial and the
    reversal trials.  The same function is used for every model (shared interface).
    """
    rng = np.random.default_rng(seed)
    agent.reset()
    env.reset()
    T = env.n_trials
    choices = np.empty(T, dtype=np.int64)
    rewards = np.empty(T, dtype=np.int64)
    for t in range(T):
        c = agent.choose(rng)
        r = env.step(c)
        agent.update(c, r)
        choices[t] = c
        rewards[t] = r
    return {"choices": choices, "rewards": rewards, "better_arm": env.better_arm.copy(),
            "reversals": env.reversals.copy()}


def aligned_curve(choices, better_arm, reversals, pre=10, post=30):
    """P(choose the NEW better arm) at every offset around the reversals.

    Offset 0 is the first trial after the switch.  Offsets that would fall into a
    neighbouring block (or outside the session) are excluded.  Returns
    (offsets, mean, n_reversals_contributing).
    """
    T = len(choices)
    offsets = np.arange(-pre, post)
    hits = np.full((len(reversals), len(offsets)), np.nan)
    bounds = np.concatenate([[0], reversals, [T]])
    for i, t0 in enumerate(reversals):
        new_arm = better_arm[t0]
        lo, hi = bounds[i], bounds[i + 2]  # start of previous block, end of new block
        for j, d in enumerate(offsets):
            t = t0 + d
            if lo <= t < hi:
                hits[i, j] = float(choices[t] == new_arm)
    n = np.sum(~np.isnan(hits), axis=0)
    with np.errstate(invalid="ignore"):
        mean = np.where(n > 0, np.nansum(hits, axis=0) / np.maximum(n, 1), np.nan)
    return offsets, mean, n


def reversal_metrics(choices, rewards, better_arm, reversals, window=5, criterion=4,
                     persev_window=10):
    """Behavioural signatures of the proposal (Section 5, item 1).

    trials_to_recover : mean number of trials after a reversal until the agent "mostly
        picks the new good arm": the first post-reversal trial that starts a run of
        ``window`` trials with at least ``criterion`` choices of the new better arm
        (searched inside the new block; if never reached, the block length is used).
    perseveration : fraction of choices of the OLD arm in the first ``persev_window``
        trials after a reversal.
    persev_run : mean number of consecutive old-arm choices right after a reversal.
    reward_rate : mean reward per trial (ceiling 0.8).
    p_better : fraction of trials on which the currently better arm was chosen.
    """
    T = len(choices)
    bounds = np.concatenate([reversals, [T]])
    ttr, persev, run = [], [], []
    for i, t0 in enumerate(reversals):
        end = bounds[i + 1]
        new_arm = better_arm[t0]
        block = (choices[t0:end] == new_arm).astype(int)
        n = len(block)
        if n < window:
            continue
        csum = np.concatenate([[0], np.cumsum(block)])
        wins = csum[window:] - csum[:-window]          # hits in [k, k + window)
        ok = np.nonzero(wins >= criterion)[0]
        ttr.append(ok[0] if len(ok) else n)
        persev.append(1.0 - block[:persev_window].mean())
        old = np.nonzero(block == 1)[0]
        run.append(old[0] if len(old) else n)
    return {
        "trials_to_recover": float(np.mean(ttr)) if ttr else np.nan,
        "perseveration": float(np.mean(persev)) if persev else np.nan,
        "persev_run": float(np.mean(run)) if run else np.nan,
        "reward_rate": float(np.mean(rewards)),
        "p_better": float(np.mean(choices == better_arm)),
    }
