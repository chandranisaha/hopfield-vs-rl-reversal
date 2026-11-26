"""The task environment: a two-armed bandit with an (optional) reversal schedule.

Proposal, Section 3: "the better arm pays with probability 0.8 (the other 0.2) and
the better arm switches every 20-40 trials; 1000 trials per agent".

A single class covers both the reversal task and the fixed-probability task used for
the Step-1 sanity check (``block_range=None`` means the better arm never switches).
"""

from __future__ import annotations

import numpy as np


class Environment:
    """Two-armed probabilistic (reversal) bandit.

    Parameters
    ----------
    n_trials : int
        Number of trials in the session.
    p_high, p_low : float
        Reward probability of the currently better / worse arm.
    block_range : (int, int) or None
        Block lengths are drawn uniformly from the integers in this closed interval.
        ``None`` gives one block for the whole session (no reversals).
    seed : int, SeedSequence or None
        Seeds both the reversal schedule and the reward draws.
    first_better : {0, 1} or None
        Arm that is better in the first block (random if None).

    Attributes
    ----------
    better_arm : (n_trials,) int array, the better arm on every trial
    p_reward   : (n_trials, 2) float array, reward probability of each arm
    reversals  : int array, trial indices at which a new block starts (first trial of
                 the new block; trial 0 is not a reversal)
    """

    n_arms = 2

    def __init__(self, n_trials=1000, p_high=0.8, p_low=0.2, block_range=(20, 40),
                 seed=None, first_better=None):
        self.n_trials = int(n_trials)
        self.p_high = float(p_high)
        self.p_low = float(p_low)
        self.block_range = block_range
        self._rng = np.random.default_rng(seed)

        better = int(self._rng.integers(2)) if first_better is None else int(first_better)
        better_arm = np.empty(self.n_trials, dtype=np.int64)
        reversals = []
        t = 0
        while t < self.n_trials:
            if block_range is None:
                length = self.n_trials
            else:
                lo, hi = block_range
                length = int(self._rng.integers(lo, hi + 1))
            better_arm[t:t + length] = better
            t += length
            if t < self.n_trials:
                reversals.append(t)
            better = 1 - better
        self.better_arm = better_arm
        self.reversals = np.asarray(reversals, dtype=np.int64)
        self.p_reward = np.where(
            better_arm[:, None] == np.arange(2)[None, :], self.p_high, self.p_low
        )
        self.t = 0

    def reset(self):
        """Restart the session (same schedule, reward draws continue from the RNG)."""
        self.t = 0

    def step(self, choice):
        """Deliver the reward (0/1) for ``choice`` on the current trial and advance."""
        if self.t >= self.n_trials:
            raise RuntimeError("session is over")
        reward = int(self._rng.random() < self.p_reward[self.t, int(choice)])
        self.t += 1
        return reward
