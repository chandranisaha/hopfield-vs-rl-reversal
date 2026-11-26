"""The agents of the proposal behind one interface (``choose`` / ``update``).

Notation (proposal, Section 4): on trial t the agent makes a choice c_t and receives a
reward r_t in {0, 1}.  In code the two arms are indexed 0 and 1.

Model 1  QLearningAgent       Q_{t+1}(c_t) = Q_t(c_t) + alpha (r_t - Q_t(c_t)),
                              P(c_t = k) ∝ exp(theta Q_t(k))
Model 2  QPerseverationAgent  Model 1 + bonus kappa for repeating the last choice
Model 3  HopfieldAgent        episodic memory; modern (softmax) or classical (Hebbian)
Model 4  HybridAgent          V = (1 - lambda) Q + lambda V_epi

Every agent exposes ``logits()`` (the history-dependent log-odds of each arm, computed
from what the agent knew *before* the current trial).  ``choose`` turns them into a
choice with a softmax, optionally adding an external logit for arm 1 (used on the mouse
data, where the visual stimulus enters every model as a logistic term).
"""

from __future__ import annotations

import numpy as np

N_ARMS = 2
N_SYMBOLS = 4  # a (choice, reward) pair takes one of 4 values: symbol = 2 * choice + reward


def softmax(z):
    z = np.asarray(z, dtype=float)
    z = z - z.max()
    p = np.exp(z)
    return p / p.sum()


# --------------------------------------------------------------------------------------
# Agent interface
# --------------------------------------------------------------------------------------
class Agent:
    """Common interface: every experiment runs unchanged on every model."""

    name = "agent"

    def reset(self):
        raise NotImplementedError

    def logits(self):
        """History-dependent logit of each arm, shape (2,), before the current trial."""
        raise NotImplementedError

    def update(self, choice, reward):
        """Learn from the outcome of the current trial."""
        raise NotImplementedError

    def choice_probs(self, ext_logit=0.0):
        z = np.array(self.logits(), dtype=float)
        z[1] += ext_logit
        return softmax(z)

    def choose(self, rng, ext_logit=0.0):
        """Sample a choice (0 or 1) from the softmax policy."""
        return int(rng.random() < self.choice_probs(ext_logit)[1])


# --------------------------------------------------------------------------------------
# Models 1 and 2: incremental RL
# --------------------------------------------------------------------------------------
class QLearningAgent(Agent):
    """Model 1.  Free parameters: learning rate alpha, inverse temperature theta."""

    name = "Q"

    def __init__(self, alpha, theta, q0=0.5):
        self.alpha = float(alpha)
        self.theta = float(theta)
        self.q0 = float(q0)
        self.reset()

    def reset(self):
        self.Q = np.full(N_ARMS, self.q0)

    def values(self):
        return self.Q.copy()

    def logits(self):
        return self.theta * self.Q

    def update(self, choice, reward):
        self.Q[choice] += self.alpha * (reward - self.Q[choice])


class QPerseverationAgent(QLearningAgent):
    """Model 2.  P(c_t = k) ∝ exp(theta Q_t(k) + kappa 1[k = c_{t-1}])."""

    name = "Q+kappa"

    def __init__(self, alpha, theta, kappa, q0=0.5):
        self.kappa = float(kappa)
        super().__init__(alpha, theta, q0)

    def reset(self):
        super().reset()
        self.prev_choice = None

    def logits(self):
        z = self.theta * self.Q
        if self.prev_choice is not None:
            z[self.prev_choice] += self.kappa
        return z

    def update(self, choice, reward):
        super().update(choice, reward)
        self.prev_choice = int(choice)


# --------------------------------------------------------------------------------------
# Episodic memories (the value part of Models 3 and 4)
# --------------------------------------------------------------------------------------
class _HistoryMixin:
    """Keeps the last L (choice, reward) symbols, most recent first."""

    def _reset_history(self):
        self.history = []

    def _push_history(self, choice, reward):
        self.history.insert(0, 2 * int(choice) + int(reward))
        del self.history[self.L:]

    def _cue_symbols(self):
        """Symbols used to cue the memory; optionally corrupted (cue noise)."""
        syms = list(self.history)
        if self.cue_noise > 0.0:
            for lag in range(len(syms)):
                if self._noise_rng.random() < self.cue_noise:
                    # replace by one of the three other symbols
                    syms[lag] = (syms[lag] + int(self._noise_rng.integers(1, N_SYMBOLS))) % N_SYMBOLS
        return syms


class ModernEpisodicMemory(_HistoryMixin):
    """Modern-Hopfield episodic memory (proposal Model 3, modern variant).

    After each trial one episode is stored: key k_t = the previous L choice-reward
    pairs, one-hot encoded (4 units per lag, all-zero for lags before the session
    start), and value (c_t, r_t).  At decision time the current history q_t is the query:

        w_i  = softmax_i(beta k_i^T q_t)            over the last M episodes
        V(a) = sum_i w_i r_i 1[c_i = a] / (sum_i w_i 1[c_i = a] + eps)

    This is the retrieval rule of the modern continuous Hopfield network [Ramsauer et
    al. 2021]: dot-product similarity, softmax separation with inverse temperature beta,
    projection onto the stored values.
    """

    def __init__(self, L, M, beta, eps=1e-6, cue_noise=0.0, seed=0):
        self.L = int(L)
        self.M = int(M)
        self.beta = float(beta)
        self.eps = float(eps)
        self.cue_noise = float(cue_noise)
        self._seed = seed
        self.reset()

    def reset(self):
        self._reset_history()
        self._noise_rng = np.random.default_rng(self._seed)
        self.keys = np.zeros((self.M, N_SYMBOLS * self.L))
        self.choices = np.zeros(self.M, dtype=np.int64)
        self.rewards = np.zeros(self.M)
        self.count = 0   # number of stored episodes (<= M)
        self._ptr = 0    # ring-buffer write position: the oldest episode is overwritten

    def _encode(self, symbols):
        key = np.zeros(N_SYMBOLS * self.L)
        for lag, sym in enumerate(symbols):
            key[N_SYMBOLS * lag + sym] = 1.0
        return key

    def retrieval_weights(self):
        """Softmax weights over the stored episodes for the current query."""
        q = self._encode(self._cue_symbols())
        return softmax(self.beta * (self.keys[:self.count] @ q))

    def values(self):
        V = np.zeros(N_ARMS)
        if self.count == 0:
            return V
        w = self.retrieval_weights()
        c = self.choices[:self.count]
        r = self.rewards[:self.count]
        for a in range(N_ARMS):
            sel = c == a
            V[a] = np.sum(w[sel] * r[sel]) / (np.sum(w[sel]) + self.eps)
        return V

    def store(self, choice, reward):
        self.keys[self._ptr] = self._encode(self.history)  # uncorrupted key
        self.choices[self._ptr] = int(choice)
        self.rewards[self._ptr] = float(reward)
        self._ptr = (self._ptr + 1) % self.M
        self.count = min(self.count + 1, self.M)
        self._push_history(choice, reward)


class ClassicalEpisodicMemory(_HistoryMixin):
    """Classical-Hopfield episodic memory (proposal Model 3, classical variant).

    A binary (+1/-1) Hebbian network W = (1/N) sum_i xi_i xi_i^T over concatenated
    (key, value) patterns of the last M episodes [Hopfield 1982].

    Pattern layout (N = L * code_bits + 2 * value_bits units):
        key    : for every lag, the one-hot (choice, reward) symbol is expanded into a
                 fixed random +/-1 codeword of ``code_bits`` units (so that different
                 symbols are nearly orthogonal, as one-hot codes are)
        choice : +u for arm 1, -u for arm 0   (u a fixed random +/-1 vector)
        reward : +v for reward, -v for no reward
    Retrieval: to evaluate arm a the network is cued with the key part and the candidate
    action a, the reward units start at 0 (unknown); all units are then updated with
    xi <- sign(W xi) until nothing changes, and the completed reward part is read out:
    V(a) = (1 + overlap with v) / 2.

    Integer arithmetic is used throughout (the 1/N factor does not change the sign), so
    removing the oldest episode from W is exact.
    """

    def __init__(self, L, M, code_bits=32, value_bits=32, cue_noise=0.0, seed=0, max_iter=20):
        self.L = int(L)
        self.M = int(M)
        self.code_bits = int(code_bits)
        self.value_bits = int(value_bits)
        self.cue_noise = float(cue_noise)
        self.max_iter = int(max_iter)
        self._seed = seed
        self.n_key = self.L * self.code_bits
        self.N = self.n_key + 2 * self.value_bits
        code_rng = np.random.default_rng(np.random.SeedSequence([12345, int(seed)]))
        self.key_codes = code_rng.choice(np.array([-1, 1]), size=(self.L, N_SYMBOLS, self.code_bits))
        self.u = code_rng.choice(np.array([-1, 1]), size=self.value_bits)
        self.v = code_rng.choice(np.array([-1, 1]), size=self.value_bits)
        self._choice_slice = slice(self.n_key, self.n_key + self.value_bits)
        self._reward_slice = slice(self.n_key + self.value_bits, self.N)
        self.reset()

    def reset(self):
        self._reset_history()
        self._noise_rng = np.random.default_rng(self._seed)
        self.patterns = np.zeros((self.M, self.N), dtype=np.int64)
        self.W = np.zeros((self.N, self.N), dtype=np.int64)  # N * W, zero diagonal
        self.count = 0
        self._ptr = 0

    def _encode_key(self, symbols):
        key = np.zeros(self.n_key, dtype=np.int64)
        for lag, sym in enumerate(symbols):
            key[lag * self.code_bits:(lag + 1) * self.code_bits] = self.key_codes[lag, sym]
        return key

    def _pattern(self, symbols, choice, reward):
        xi = np.zeros(self.N, dtype=np.int64)
        xi[:self.n_key] = self._encode_key(symbols)
        xi[self._choice_slice] = self.u if choice == 1 else -self.u
        xi[self._reward_slice] = self.v if reward == 1 else -self.v
        return xi

    def complete(self, state):
        """Iterate xi <- sign(W xi) (synchronous; a unit with zero field keeps its state)."""
        state = state.copy()
        for _ in range(self.max_iter):
            h = self.W @ state
            new = np.where(h > 0, 1, np.where(h < 0, -1, state))
            if np.array_equal(new, state):
                break
            state = new
        return state

    def values(self):
        V = np.zeros(N_ARMS)
        key = self._encode_key(self._cue_symbols())
        for a in range(N_ARMS):
            cue = np.zeros(self.N, dtype=np.int64)
            cue[:self.n_key] = key
            cue[self._choice_slice] = self.u if a == 1 else -self.u
            out = self.complete(cue)
            V[a] = 0.5 * (1.0 + (out[self._reward_slice] @ self.v) / self.value_bits)
        return V

    def store(self, choice, reward):
        xi = self._pattern(self.history, int(choice), int(reward))
        if self.count == self.M:  # forget the oldest episode
            old = self.patterns[self._ptr]
            self.W -= np.outer(old, old)
        self.W += np.outer(xi, xi)
        np.fill_diagonal(self.W, 0)
        self.patterns[self._ptr] = xi
        self._ptr = (self._ptr + 1) % self.M
        self.count = min(self.count + 1, self.M)
        self._push_history(choice, reward)


def make_memory(variant, L, M, beta=1.0, eps=1e-6, cue_noise=0.0, seed=0,
                code_bits=32, value_bits=32):
    if variant == "modern":
        return ModernEpisodicMemory(L, M, beta, eps=eps, cue_noise=cue_noise, seed=seed)
    if variant == "classical":
        return ClassicalEpisodicMemory(L, M, code_bits=code_bits, value_bits=value_bits,
                                       cue_noise=cue_noise, seed=seed)
    raise ValueError(f"unknown variant {variant!r}")


# --------------------------------------------------------------------------------------
# Models 3 and 4
# --------------------------------------------------------------------------------------
class HopfieldAgent(Agent):
    """Model 3.  Free parameters: beta (modern variant only), M, L, theta."""

    name = "Hopfield"

    def __init__(self, beta, M, L, theta, variant="modern", **memory_kwargs):
        self.theta = float(theta)
        self.variant = variant
        self.memory = make_memory(variant, L, M, beta=beta, **memory_kwargs)

    def reset(self):
        self.memory.reset()

    def values(self):
        return self.memory.values()

    def logits(self):
        return self.theta * self.memory.values()

    def update(self, choice, reward):
        self.memory.store(choice, reward)


class HybridAgent(Agent):
    """Model 4.  V = (1 - lambda) Q + lambda V_epi, softmax with theta."""

    name = "Hybrid"

    def __init__(self, alpha, theta, beta, M, L, lam, q0=0.5, variant="modern", **memory_kwargs):
        self.alpha = float(alpha)
        self.theta = float(theta)
        self.lam = float(lam)
        self.q0 = float(q0)
        self.memory = make_memory(variant, L, M, beta=beta, **memory_kwargs)
        self.reset()

    def reset(self):
        self.Q = np.full(N_ARMS, self.q0)
        self.memory.reset()

    def values(self):
        return (1.0 - self.lam) * self.Q + self.lam * self.memory.values()

    def logits(self):
        return self.theta * self.values()

    def update(self, choice, reward):
        self.Q[choice] += self.alpha * (reward - self.Q[choice])
        self.memory.store(choice, reward)


class KappaWrapper(Agent):
    """Adds a perseveration bonus kappa to any agent (supplementary control models)."""

    def __init__(self, agent, kappa):
        self.agent = agent
        self.kappa = float(kappa)
        self.name = agent.name + "+kappa"
        self.prev_choice = None

    def reset(self):
        self.agent.reset()
        self.prev_choice = None

    def logits(self):
        z = np.array(self.agent.logits(), dtype=float)
        if self.prev_choice is not None:
            z[self.prev_choice] += self.kappa
        return z

    def update(self, choice, reward):
        self.agent.update(choice, reward)
        self.prev_choice = int(choice)


# --------------------------------------------------------------------------------------
# Factory used by simulation, recovery and posterior-predictive code
# --------------------------------------------------------------------------------------
def make_agent(model, params, **kwargs):
    """Build an agent from a model name and a parameter dict.

    model: 'Q', 'QK', 'HOP', 'HYB', 'HOPC' (classical Hopfield), 'HOPK', 'HYBK', 'STIM'.
    """
    p = params
    if model == "STIM":            # no history term at all (reference model)
        return QLearningAgent(alpha=0.0, theta=0.0)
    if model == "Q":
        return QLearningAgent(p["alpha"], p["theta"])
    if model == "QK":
        return QPerseverationAgent(p["alpha"], p["theta"], p["kappa"])
    if model == "HOP":
        return HopfieldAgent(p["beta"], p["M"], p["L"], p["theta"], **kwargs)
    if model == "HOPC":
        return HopfieldAgent(0.0, p["M"], p["L"], p["theta"], variant="classical", **kwargs)
    if model == "HYB":
        return HybridAgent(p["alpha"], p["theta"], p["beta"], p["M"], p["L"], p["lam"], **kwargs)
    if model == "HOPK":
        return KappaWrapper(HopfieldAgent(p["beta"], p["M"], p["L"], p["theta"], **kwargs), p["kappa"])
    if model == "HYBK":
        return KappaWrapper(
            HybridAgent(p["alpha"], p["theta"], p["beta"], p["M"], p["L"], p["lam"], **kwargs),
            p["kappa"])
    raise ValueError(f"unknown model {model!r}")
