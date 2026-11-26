"""Classical and modern Hopfield networks on toy patterns (proposal Step 1).

Used to validate the retrieval implementation against the known results:
* classical network [Hopfield 1982]: retrieval of stored +/-1 patterns from corrupted
  cues, failing near P = 0.14 N stored random patterns [Amit et al. 1985];
* modern continuous network [Ramsauer et al. 2021]: update xi <- X^T softmax(beta X xi),
  with a far larger capacity and a retrieval sharpness set by beta.
"""

from __future__ import annotations

import numpy as np


def random_patterns(n_patterns, n_units, rng):
    """Unbiased random +/-1 patterns, shape (n_patterns, n_units)."""
    return rng.choice(np.array([-1.0, 1.0]), size=(n_patterns, n_units))


def corrupt(patterns, flip_prob, rng):
    """Flip every unit independently with probability ``flip_prob``."""
    flips = rng.random(np.shape(patterns)) < flip_prob
    return np.where(flips, -np.asarray(patterns), patterns)


def overlap(a, b):
    """Overlap m = (1/N) sum_j a_j b_j (1 = identical, 0 = unrelated)."""
    return np.mean(np.asarray(a) * np.asarray(b), axis=-1)


class ClassicalHopfield:
    """Binary Hopfield network with the Hebbian rule W = (1/N) sum_i xi_i xi_i^T.

    Self-connections are removed (W_jj = 0) as in Hopfield (1982).
    """

    def __init__(self, n_units):
        self.N = int(n_units)
        self.W = np.zeros((self.N, self.N))

    def store(self, patterns):
        X = np.atleast_2d(np.asarray(patterns, dtype=float))
        self.W = X.T @ X / self.N
        np.fill_diagonal(self.W, 0.0)
        return self

    def energy(self, state):
        return -0.5 * state @ self.W @ state

    def retrieve(self, cues, max_iter=100, mode="sync", rng=None):
        """Iterate xi <- sign(W xi) from the cue(s) until a fixed point.

        mode='sync'  : all units updated together (fast, batched over cues)
        mode='async' : one unit at a time in random order (the original dynamics, which
                       is guaranteed to converge because the energy never increases)
        A unit whose field is exactly zero keeps its state.
        """
        states = np.array(cues, dtype=float)
        single = states.ndim == 1
        states = np.atleast_2d(states)
        if mode == "sync":
            for _ in range(max_iter):
                h = states @ self.W
                new = np.where(h > 0, 1.0, np.where(h < 0, -1.0, states))
                if np.array_equal(new, states):
                    break
                states = new
        elif mode == "async":
            rng = np.random.default_rng() if rng is None else rng
            for b in range(states.shape[0]):
                s = states[b]
                for _ in range(max_iter):
                    changed = False
                    for j in rng.permutation(self.N):
                        h = self.W[j] @ s
                        new = s[j] if h == 0 else (1.0 if h > 0 else -1.0)
                        if new != s[j]:
                            s[j] = new
                            changed = True
                    if not changed:
                        break
        else:
            raise ValueError(mode)
        return states[0] if single else states


class ModernHopfield:
    """Modern continuous Hopfield network: xi <- X^T softmax(beta X xi)."""

    def __init__(self, beta=1.0):
        self.beta = float(beta)
        self.X = None

    def store(self, patterns):
        self.X = np.atleast_2d(np.asarray(patterns, dtype=float))
        return self

    def attention(self, cues):
        """Softmax weights over stored patterns, shape (n_cues, n_patterns)."""
        a = self.beta * (np.atleast_2d(cues) @ self.X.T)
        a -= a.max(axis=1, keepdims=True)
        w = np.exp(a)
        return w / w.sum(axis=1, keepdims=True)

    def retrieve(self, cues, n_iter=3):
        states = np.array(cues, dtype=float)
        single = states.ndim == 1
        states = np.atleast_2d(states)
        for _ in range(n_iter):
            new = self.attention(states) @ self.X
            if np.allclose(new, states, atol=1e-10):
                break
            states = new
        return states[0] if single else states
