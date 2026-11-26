"""Likelihood of a choice sequence under every model (proposal Step 3).

"The softmax probability of each observed choice, using the model's value estimates
before that trial."  With two arms the softmax is a logistic function of the logit
difference between arm 1 and arm 0:

    logit_t = x_t . w  +  theta * (V_t(1) - V_t(0))  [+ kappa * (+1/-1 for the last choice)]

``x_t . w`` is the stimulus term used on the mouse data (absent in the simulated task).

Two implementations are provided and tested against each other:
* ``agent_logit_diff``: literal and slow, it replays the data through the same Agent
  objects that are used for simulation;
* ``Likelihood``: fast and exact, it uses closed forms (an exponential filter for the
  Q-values, sufficient statistics of the episodic memory) so that a model can be
  evaluated thousands of times during fitting.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import lfilter
from scipy.special import expit

from .agents import ClassicalEpisodicMemory, make_agent

Q0 = 0.5      # initial Q-value of both arms
EPS = 1e-6    # epsilon of the episodic value formula

# name -> (continuous parameters, uses episodic memory with discrete (L, M))
MODELS = {
    "STIM": ((), False),                                   # no history term (reference)
    "Q":    (("alpha", "theta"), False),                   # Model 1
    "QK":   (("alpha", "theta", "kappa"), False),          # Model 2
    "HOP":  (("beta", "theta"), True),                     # Model 3 (modern)
    "HYB":  (("alpha", "theta", "beta", "lam"), True),     # Model 4
    "HOPC": (("theta",), True),                            # Model 3 (classical variant)
    "HOPK": (("beta", "theta", "kappa"), True),            # supplementary: Model 3 + kappa
    "HYBK": (("alpha", "theta", "beta", "lam", "kappa"), True),  # supplementary: Model 4 + kappa
}
LABELS = {"STIM": "No history", "Q": "Q-learning", "QK": "Q + stickiness", "HOP": "Hopfield",
          "HYB": "Hybrid", "HOPC": "Hopfield (classical)", "HOPK": "Hopfield + stickiness",
          "HYBK": "Hybrid + stickiness"}

BOUNDS = {"alpha": (0.0, 1.0), "theta": (0.0, 50.0), "kappa": (-5.0, 5.0),
          "beta": (0.0, 10.0), "lam": (0.0, 1.0)}
INIT_RANGE = {"alpha": (0.05, 0.95), "theta": (0.5, 10.0), "kappa": (-1.0, 1.5),
              "beta": (0.0, 4.0), "lam": (0.1, 0.9)}
STIM_BOUNDS = (-30.0, 30.0)
STIM_INIT = (0.0, 4.0)


def n_free_params(model, n_stim=0):
    """Number of free parameters, counting the discrete M and L (proposal: 'Free
    parameters: beta, M, L, temperature')."""
    cont, episodic = MODELS[model]
    return n_stim + len(cont) + (2 if episodic else 0)


# --------------------------------------------------------------------------------------
# Data container
# --------------------------------------------------------------------------------------
@dataclass
class ChoiceData:
    """Trial-by-trial data of one subject.

    choices, rewards : (T,) int arrays with values in {0, 1}
    session : (T,) int array, session index of every trial (non-decreasing); agents are
              reset at the start of every session
    X : optional (T, p) stimulus design matrix (contribution to the logit of arm 1)
    """

    choices: np.ndarray
    rewards: np.ndarray
    session: np.ndarray = None
    X: np.ndarray = None

    def __post_init__(self):
        self.choices = np.asarray(self.choices, dtype=np.int64)
        self.rewards = np.asarray(self.rewards, dtype=np.int64)
        T = len(self.choices)
        if self.session is None:
            self.session = np.zeros(T, dtype=np.int64)
        self.session = np.asarray(self.session, dtype=np.int64)
        if np.any(np.diff(self.session) < 0):
            raise ValueError("session index must be non-decreasing")
        if not (np.isin(self.choices, (0, 1)).all() and np.isin(self.rewards, (0, 1)).all()):
            raise ValueError("choices and rewards must be 0/1")
        if self.X is not None:
            self.X = np.asarray(self.X, dtype=float)
            assert self.X.shape[0] == T

    @property
    def n_trials(self):
        return len(self.choices)

    @property
    def n_stim(self):
        return 0 if self.X is None else self.X.shape[1]

    def session_slices(self):
        """List of slices, one per session."""
        T = self.n_trials
        if T == 0:
            return []
        starts = np.concatenate([[0], np.nonzero(np.diff(self.session) != 0)[0] + 1])
        ends = np.concatenate([starts[1:], [T]])
        return [slice(int(a), int(b)) for a, b in zip(starts, ends)]


# --------------------------------------------------------------------------------------
# Literal (slow) implementation through the Agent interface
# --------------------------------------------------------------------------------------
def agent_logit_diff(model, params, data, **agent_kwargs):
    """History logit difference (arm 1 minus arm 0) before every trial, obtained by
    replaying the observed choices and rewards through the corresponding Agent."""
    out = np.empty(data.n_trials)
    agent = make_agent(model, params, **agent_kwargs)
    for sl in data.session_slices():
        agent.reset()
        for t in range(sl.start, sl.stop):
            z = agent.logits()
            out[t] = z[1] - z[0]
            agent.update(int(data.choices[t]), int(data.rewards[t]))
    return out


# --------------------------------------------------------------------------------------
# Building blocks of the fast implementation
# --------------------------------------------------------------------------------------
def episodic_statistics(choices, rewards, L, M):
    """Sufficient statistics of the modern episodic memory for ONE session.

    For trial t, arm a and match level m (number of the L key positions on which a
    stored episode agrees with the current history, i.e. k_i^T q_t):
        n[t, a, m] = number of episodes among the last M with choice a and k_i^T q_t = m
        s[t, a, m] = summed reward of those episodes
    Then for any beta:  V_t(a) = sum_m e^{beta m} s[t,a,m] / (sum_m e^{beta m} n[t,a,m] + eps Z_t).
    """
    c = np.asarray(choices, dtype=np.int64)
    r = np.asarray(rewards, dtype=np.int64)
    T = len(c)
    sym = 2 * c + r
    # E[u, d-1] = 1 if the symbols of trials u and u-d are equal
    E = np.zeros((T, M), dtype=np.int8)
    for d in range(1, min(M, T - 1) + 1):
        E[d:, d - 1] = sym[d:] == sym[:-d]
    # match[t, d-1] = k_{t-d}^T q_t = sum_{l=1..L} 1[sym[t-l] == sym[t-d-l]]
    match = np.zeros((T, M), dtype=np.int64)
    for lag in range(1, min(L, T - 1) + 1):
        match[lag:] += E[:-lag]
    t_idx = np.arange(T)[:, None]
    src = t_idx - np.arange(1, M + 1)[None, :]       # index of the stored episode
    valid = src >= 0
    src_c = c[np.clip(src, 0, None)]
    src_r = r[np.clip(src, 0, None)]
    width = 2 * (L + 1)
    flat = (t_idx * width + src_c * (L + 1) + match)[valid]
    n_stat = np.bincount(flat, minlength=T * width).reshape(T, 2, L + 1).astype(float)
    s_stat = np.bincount(flat, weights=src_r[valid].astype(float),
                         minlength=T * width).reshape(T, 2, L + 1)
    return n_stat, s_stat


def episodic_values(n_stat, s_stat, beta, eps=EPS):
    """V_t(a) of the modern episodic memory for all trials, shape (T, 2)."""
    L = n_stat.shape[2] - 1
    e = np.exp(beta * (np.arange(L + 1) - L))   # common factor e^{-beta L} cancels
    den = n_stat @ e
    num = s_stat @ e
    Z = den.sum(axis=1, keepdims=True)
    return num / np.where(den > 0, den + eps * Z, 1.0)


def classical_values(data, L, M, code_bits=32, value_bits=32, seed=0):
    """V_t(a) of the classical-Hopfield episodic memory (replayed, parameter-free)."""
    V = np.empty((data.n_trials, 2))
    mem = ClassicalEpisodicMemory(L, M, code_bits=code_bits, value_bits=value_bits, seed=seed)
    for sl in data.session_slices():
        mem.reset()
        for t in range(sl.start, sl.stop):
            V[t] = mem.values()
            mem.store(int(data.choices[t]), int(data.rewards[t]))
    return V


# --------------------------------------------------------------------------------------
# Fast likelihood
# --------------------------------------------------------------------------------------
class Likelihood:
    """Exact one-step-ahead choice likelihood of every model for one data set."""

    def __init__(self, data, classical_kwargs=None):
        self.data = data
        self.T = data.n_trials
        self.sgn = 2.0 * data.choices - 1.0               # +1 if arm 1 was chosen
        self.classical_kwargs = classical_kwargs or {}
        self._stats = {}
        self._classical = {}
        self._prepare_q()
        self._prepare_prev()

    # ---- precomputations ------------------------------------------------------------
    def _prepare_prev(self):
        """+1 / -1 if the previous choice (same session) was arm 1 / arm 0, else 0."""
        prev = np.zeros(self.T)
        prev[1:] = self.sgn[:-1]
        for sl in self.data.session_slices():
            prev[sl.start] = 0.0
        self.prev = prev

    def _prepare_q(self):
        """Index arrays that turn the Q-learning recursion into one linear filter per arm."""
        c, sess = self.data.choices, self.data.session
        self._q = []
        for a in range(2):
            idx = np.nonzero(c == a)[0]                    # trials on which arm a was chosen
            r_sub = self.data.rewards[idx].astype(float)
            s_sub = sess[idx]
            n = len(idx)
            if n:
                first = np.concatenate([[0], np.nonzero(np.diff(s_sub) != 0)[0] + 1])
                first_of = np.repeat(first, np.diff(np.concatenate([first, [n]])))
            else:
                first_of = np.zeros(0, dtype=np.int64)
            k = np.arange(n) - first_of + 1                # rank of the choice within its session
            j0 = first_of - 1                              # last choice of arm a before the session
            before = np.concatenate([[0], np.cumsum(c == a)[:-1]])   # arm-a choices before trial t
            last = before - 1
            valid = last >= 0
            valid[valid] = s_sub[last[valid]] == sess[valid]
            self._q.append((r_sub, k, j0, np.clip(last, 0, None), valid))

    def stats(self, L, M):
        key = (int(L), int(M))
        if key not in self._stats:
            parts = [episodic_statistics(self.data.choices[sl], self.data.rewards[sl], *key)
                     for sl in self.data.session_slices()]
            self._stats[key] = (np.concatenate([p[0] for p in parts]),
                                np.concatenate([p[1] for p in parts]))
        return self._stats[key]

    def classical(self, L, M):
        key = (int(L), int(M))
        if key not in self._classical:
            self._classical[key] = classical_values(self.data, *key, **self.classical_kwargs)
        return self._classical[key]

    # ---- value signals ---------------------------------------------------------------
    def q_values(self, alpha, q0=Q0):
        """Q_t(a) before every trial, shape (T, 2).

        For one arm, Q after its j-th choice is an exponential filter of the rewards
        obtained on that arm; the filter is run once over all sessions and the state
        carried over from earlier sessions is removed analytically (reset to q0).
        """
        Q = np.full((self.T, 2), q0)
        for a, (r_sub, k, j0, last, valid) in enumerate(self._q):
            if len(r_sub) == 0:
                continue
            y = lfilter([alpha], [1.0, -(1.0 - alpha)], r_sub)
            decay = (1.0 - alpha) ** k
            y_prev = np.where(j0 >= 0, y[np.clip(j0, 0, None)], 0.0)
            q_after = y - decay * y_prev + decay * q0
            Q[:, a] = np.where(valid, q_after[last], q0)
        return Q

    def epi_values(self, beta, L, M):
        return episodic_values(*self.stats(L, M), beta)

    # ---- logits ----------------------------------------------------------------------
    def history_logit(self, model, p, disc=None):
        """theta * (V(1) - V(0)) [+ kappa * prev] before every trial."""
        if model == "STIM":
            return np.zeros(self.T)
        if model in ("Q", "QK"):
            Q = self.q_values(p["alpha"])
            z = p["theta"] * (Q[:, 1] - Q[:, 0])
        elif model in ("HOP", "HOPK"):
            V = self.epi_values(p["beta"], *disc)
            z = p["theta"] * (V[:, 1] - V[:, 0])
        elif model in ("HYB", "HYBK"):
            Q = self.q_values(p["alpha"])
            V = self.epi_values(p["beta"], *disc)
            z = p["theta"] * ((1.0 - p["lam"]) * (Q[:, 1] - Q[:, 0])
                              + p["lam"] * (V[:, 1] - V[:, 0]))
        elif model == "HOPC":
            V = self.classical(*disc)
            z = p["theta"] * (V[:, 1] - V[:, 0])
        else:
            raise ValueError(model)
        if "kappa" in p:
            z = z + p["kappa"] * self.prev
        return z

    def unpack(self, model, x):
        """Split a parameter vector into (stimulus weights, dict of model parameters)."""
        ns = self.data.n_stim
        names = MODELS[model][0]
        return np.asarray(x[:ns]), dict(zip(names, x[ns:]))

    def logit(self, model, x, disc=None):
        w, p = self.unpack(model, x)
        z = self.history_logit(model, p, disc)
        if self.data.X is not None:
            z = z + self.data.X @ w
        return z

    def trial_loglik(self, model, x, disc=None):
        """log P(observed choice) for every trial."""
        return -np.logaddexp(0.0, -self.sgn * self.logit(model, x, disc))

    def p_arm1(self, model, x, disc=None):
        """Model probability of choosing arm 1 on every trial (one-step-ahead)."""
        return 1.0 / (1.0 + np.exp(-self.logit(model, x, disc)))

    def nll(self, model, x, disc=None, rows=None):
        """Negative log-likelihood summed over all trials (or the trials in ``rows``)."""
        ll = self.trial_loglik(model, x, disc)
        return -float(ll.sum() if rows is None else ll[rows].sum())

    # ---- exact gradients (used by the optimiser) -------------------------------------
    def q_values_and_grad(self, alpha, q0=Q0):
        """Q_t(a) and dQ_t(a)/dalpha.  Differentiating Q <- Q + alpha (r - Q) gives the
        recursion dQ <- (1 - alpha) dQ + (r - Q), again a linear filter per arm."""
        Q = np.full((self.T, 2), q0)
        dQ = np.zeros((self.T, 2))
        for a, (r_sub, k, j0, last, valid) in enumerate(self._q):
            if len(r_sub) == 0:
                continue
            den = [1.0, -(1.0 - alpha)]
            y = lfilter([alpha], den, r_sub)
            decay = (1.0 - alpha) ** k
            j0c = np.clip(j0, 0, None)
            q_after = y - decay * np.where(j0 >= 0, y[j0c], 0.0) + decay * q0
            q_before = np.empty_like(q_after)
            q_before[1:] = q_after[:-1]
            q_before[k == 1] = q0                       # first choice of the arm in a session
            f = lfilter([1.0], den, r_sub - q_before)
            dq_after = f - decay * np.where(j0 >= 0, f[j0c], 0.0)
            Q[:, a] = np.where(valid, q_after[last], q0)
            dQ[:, a] = np.where(valid, dq_after[last], 0.0)
        return Q, dQ

    def epi_values_and_grad(self, beta, L, M, eps=EPS):
        """V_t(a) of the modern episodic memory and dV_t(a)/dbeta."""
        n_stat, s_stat = self.stats(L, M)
        m = np.arange(L + 1) - L
        e = np.exp(beta * m)
        de = m * e
        den, num = n_stat @ e, s_stat @ e
        dden, dnum = n_stat @ de, s_stat @ de
        has = den > 0
        D = np.where(has, den + eps * den.sum(axis=1, keepdims=True), 1.0)
        dD = dden + eps * dden.sum(axis=1, keepdims=True)
        V = num / D
        dV = np.where(has, (dnum * D - num * dD) / D ** 2, 0.0)
        return V, dV

    def logit_and_jacobian(self, model, x, disc=None):
        """Logit of arm 1 on every trial and its derivative w.r.t. every parameter,
        shapes (T,) and (T, len(x)); parameter order as in ``x``."""
        ns = self.data.n_stim
        names = MODELS[model][0]
        w, p = self.unpack(model, x)
        J = np.zeros((self.T, ns + len(names)))
        z = np.zeros(self.T)
        if ns:
            J[:, :ns] = self.data.X
            z += self.data.X @ w
        col = {name: ns + i for i, name in enumerate(names)}
        if model in ("Q", "QK"):
            Q, dQ = self.q_values_and_grad(p["alpha"])
            diff = Q[:, 1] - Q[:, 0]
            J[:, col["alpha"]] = p["theta"] * (dQ[:, 1] - dQ[:, 0])
        elif model in ("HOP", "HOPK"):
            V, dV = self.epi_values_and_grad(p["beta"], *disc)
            diff = V[:, 1] - V[:, 0]
            J[:, col["beta"]] = p["theta"] * (dV[:, 1] - dV[:, 0])
        elif model in ("HYB", "HYBK"):
            Q, dQ = self.q_values_and_grad(p["alpha"])
            V, dV = self.epi_values_and_grad(p["beta"], *disc)
            dq, dv = Q[:, 1] - Q[:, 0], V[:, 1] - V[:, 0]
            lam = p["lam"]
            diff = (1.0 - lam) * dq + lam * dv
            J[:, col["alpha"]] = p["theta"] * (1.0 - lam) * (dQ[:, 1] - dQ[:, 0])
            J[:, col["beta"]] = p["theta"] * lam * (dV[:, 1] - dV[:, 0])
            J[:, col["lam"]] = p["theta"] * (dv - dq)
        elif model == "HOPC":
            V = self.classical(*disc)
            diff = V[:, 1] - V[:, 0]
        elif model == "STIM":
            return z, J
        else:
            raise ValueError(model)
        z += p["theta"] * diff
        J[:, col["theta"]] = diff
        if "kappa" in col:
            z += p["kappa"] * self.prev
            J[:, col["kappa"]] = self.prev
        return z, J

    def nll_and_grad(self, model, x, disc=None, rows=None):
        """Negative log-likelihood and its exact gradient."""
        z, J = self.logit_and_jacobian(model, x, disc)
        u = self.sgn * z
        loss = np.logaddexp(0.0, -u)
        g = -self.sgn * expit(-u)                      # d loss / d logit
        if rows is not None:
            loss, g, J = loss[rows], g[rows], J[rows]
        return float(loss.sum()), J.T @ g
