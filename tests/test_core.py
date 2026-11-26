"""Unit tests: run with  python -m pytest tests -q"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from hopfield_rl.agents import (ClassicalEpisodicMemory, HopfieldAgent, ModernEpisodicMemory,
                                QLearningAgent, QPerseverationAgent, make_agent)
from hopfield_rl.envs import Environment
from hopfield_rl.fitting import fit_model
from hopfield_rl.hopfield import ClassicalHopfield, ModernHopfield, corrupt, overlap, random_patterns
from hopfield_rl.likelihood import ChoiceData, Likelihood, agent_logit_diff
from hopfield_rl.simulate import aligned_curve, reversal_metrics, run_session


# ---------------------------------------------------------------- environment
def test_environment_schedule():
    env = Environment(n_trials=1000, seed=3)
    bounds = np.concatenate([[0], env.reversals, [1000]])
    lengths = np.diff(bounds)
    assert np.all((lengths[:-1] >= 20) & (lengths[:-1] <= 40))     # last block may be cut
    assert np.all(env.better_arm[env.reversals] != env.better_arm[env.reversals - 1])
    assert np.allclose(env.p_reward[np.arange(1000), env.better_arm], 0.8)
    assert np.allclose(env.p_reward[np.arange(1000), 1 - env.better_arm], 0.2)


def test_environment_reward_rate():
    env = Environment(n_trials=20000, block_range=None, seed=0, first_better=1)
    r = [env.step(1) for _ in range(20000)]
    assert abs(np.mean(r) - 0.8) < 0.01


# ---------------------------------------------------------------- worked examples
def test_q_update_worked_example():
    agent = QLearningAgent(alpha=0.2, theta=5.0)
    agent.update(0, 1)
    assert np.isclose(agent.Q[0], 0.6) and np.isclose(agent.Q[1], 0.5)
    agent = QLearningAgent(alpha=0.2, theta=5.0)
    agent.update(0, 0)
    assert np.isclose(agent.Q[0], 0.4)


def test_softmax_worked_example():
    agent = QLearningAgent(alpha=0.2, theta=5.0)
    agent.Q = np.array([0.6, 0.4])
    assert np.isclose(agent.choice_probs()[0], 1 / (1 + np.exp(-1.0)))   # 0.731
    agent.theta = 1.0
    assert np.isclose(agent.choice_probs()[0], 1 / (1 + np.exp(-0.2)))   # 0.55


def test_perseveration_bonus():
    agent = QPerseverationAgent(alpha=0.0, theta=3.0, kappa=1.2)
    assert np.allclose(agent.choice_probs(), 0.5)
    agent.update(1, 0)   # alpha = 0: values unchanged, only stickiness
    assert np.isclose(agent.choice_probs()[1], 1 / (1 + np.exp(-1.2)))


def test_modern_memory_toy_example():
    """Explainer toy example: L=1, M=3, beta=2 -> weights .47/.47/.06, V(A)=.5, V(B)~1."""
    mem = ModernEpisodicMemory(L=1, M=3, beta=2.0)
    A, B = 0, 1
    key_A_rew = mem._encode([2 * A + 1])
    key_B_unrew = mem._encode([2 * B + 0])
    mem.keys[:] = [key_A_rew, key_A_rew, key_B_unrew]
    mem.choices[:] = [A, A, B]
    mem.rewards[:] = [1, 0, 1]
    mem.count = 3
    mem.history = [2 * A + 1]            # current situation: chose A, rewarded
    w = mem.retrieval_weights()
    e2 = np.exp(2.0)
    assert np.allclose(w, np.array([e2, e2, 1.0]) / (2 * e2 + 1))
    V = mem.values()
    assert np.isclose(V[A], 0.5, atol=1e-5)
    assert np.isclose(V[B], 1.0, atol=1e-4)


def test_memory_window_forgets():
    mem = ModernEpisodicMemory(L=1, M=4, beta=0.0)
    for _ in range(10):
        mem.store(0, 1)
    for _ in range(4):
        mem.store(1, 0)
    V = mem.values()           # only the last 4 episodes (arm 1, unrewarded) remain
    assert mem.count == 4 and V[0] == 0.0 and V[1] == 0.0
    mem.store(0, 1)
    assert np.isclose(mem.values()[0], 1.0, atol=1e-5)


# ---------------------------------------------------------------- Hopfield networks
def test_classical_hopfield_retrieval_and_capacity():
    rng = np.random.default_rng(0)
    N = 400
    low = random_patterns(int(0.05 * N), N, rng)
    net = ClassicalHopfield(N).store(low)
    out = net.retrieve(corrupt(low, 0.1, rng))
    assert np.mean(overlap(out, low)) > 0.99
    high = random_patterns(int(0.3 * N), N, rng)
    net = ClassicalHopfield(N).store(high)
    out = net.retrieve(corrupt(high, 0.1, rng))
    assert np.mean(overlap(out, high)) < 0.7


def test_classical_async_matches_sync_below_capacity():
    rng = np.random.default_rng(1)
    X = random_patterns(5, 100, rng)
    net = ClassicalHopfield(100).store(X)
    cue = corrupt(X[0], 0.1, rng)
    assert np.array_equal(net.retrieve(cue, mode="async", rng=rng), X[0])
    assert np.array_equal(net.retrieve(cue, mode="sync"), X[0])


def test_modern_hopfield_exceeds_classical_capacity():
    rng = np.random.default_rng(2)
    N = 64
    X = random_patterns(10 * N, N, rng)      # load P/N = 10 >> 0.14
    net = ModernHopfield(beta=1.0).store(X)
    out = np.sign(net.retrieve(corrupt(X[:200], 0.1, rng)))
    assert np.mean(overlap(out, X[:200])) > 0.99


def test_classical_memory_pattern_completion():
    mem = ClassicalEpisodicMemory(L=2, M=8, seed=1)
    assert np.allclose(mem.values(), 0.5)            # empty memory: no information
    seq = [(0, 1), (0, 1), (0, 1), (0, 1)]
    for c, r in seq:
        mem.store(c, r)
    V = mem.values()                                  # history (A rew, A rew): A was rewarded
    assert V[0] == 1.0


# ---------------------------------------------------------------- likelihood
def _random_data(rng, T=400, n_sessions=3, with_x=True):
    c = rng.integers(0, 2, T)
    r = rng.integers(0, 2, T)
    sess = np.sort(rng.integers(0, n_sessions, T))
    X = rng.normal(size=(T, 2)) if with_x else None
    return ChoiceData(c, r, sess, X)


def test_episodic_statistics_bruteforce():
    rng = np.random.default_rng(5)
    data = _random_data(rng, T=120, n_sessions=1, with_x=False)
    lik = Likelihood(data)
    for L, M, beta in [(1, 3, 2.0), (2, 8, 0.7), (3, 50, 4.0)]:
        V_fast = lik.epi_values(beta, L, M)
        V_agent = np.empty((data.n_trials, 2))
        mem = ModernEpisodicMemory(L, M, beta)
        for t in range(data.n_trials):
            V_agent[t] = mem.values()
            mem.store(data.choices[t], data.rewards[t])
        assert np.allclose(V_fast, V_agent, atol=1e-10)


def test_fast_likelihood_matches_agents():
    rng = np.random.default_rng(7)
    data = _random_data(rng)
    lik = Likelihood(data)
    cases = [
        ("Q", {"alpha": 0.37, "theta": 4.2}, None),
        ("Q", {"alpha": 1.0, "theta": 2.0}, None),
        ("Q", {"alpha": 0.0, "theta": 2.0}, None),
        ("QK", {"alpha": 0.21, "theta": 3.0, "kappa": 0.8}, None),
        ("HOP", {"beta": 1.7, "theta": 5.0}, (2, 8)),
        ("HOP", {"beta": 0.0, "theta": 5.0}, (1, 2)),
        ("HOP", {"beta": 9.0, "theta": 2.5}, (3, 64)),
        ("HYB", {"alpha": 0.6, "theta": 6.0, "beta": 2.2, "lam": 0.35}, (3, 16)),
        ("HOPK", {"beta": 1.1, "theta": 3.0, "kappa": -0.4}, (1, 4)),
        ("HYBK", {"alpha": 0.15, "theta": 2.0, "beta": 3.0, "lam": 0.7, "kappa": 0.5}, (2, 32)),
        ("HOPC", {"theta": 2.0}, (2, 8)),
    ]
    for model, p, disc in cases:
        full = dict(p)
        if disc is not None:
            full["L"], full["M"] = disc
        slow = agent_logit_diff(model, full, data)
        fast = lik.history_logit(model, p, disc)
        assert np.allclose(slow, fast, atol=1e-8), model


def test_nll_is_sum_of_choice_log_probs():
    rng = np.random.default_rng(11)
    data = _random_data(rng, T=60, n_sessions=2)
    lik = Likelihood(data)
    w = np.array([0.8, -0.3])
    p = {"alpha": 0.3, "theta": 3.0, "kappa": 0.5}
    x = np.concatenate([w, [p["alpha"], p["theta"], p["kappa"]]])
    agent = make_agent("QK", p)
    total = 0.0
    for sl in data.session_slices():
        agent.reset()
        for t in range(sl.start, sl.stop):
            probs = agent.choice_probs(ext_logit=float(data.X[t] @ w))
            total += np.log(probs[data.choices[t]])
            agent.update(data.choices[t], data.rewards[t])
    assert np.isclose(lik.nll("QK", x), -total)
    assert np.isclose(lik.p_arm1("QK", x)[0], 1 / (1 + np.exp(-data.X[0] @ w)))


# ---------------------------------------------------------------- simulation + fitting
def test_metrics_on_ideal_agent():
    better = np.repeat([0, 1, 0], 30)
    reversals = np.array([30, 60])
    choices = better.copy()
    choices[30:33] = 0                      # perseverates for 3 trials after 1st reversal
    rewards = np.ones(90, dtype=int)
    m = reversal_metrics(choices, rewards, better, reversals)
    assert np.isclose(m["persev_run"], 1.5)             # (3 + 0) / 2
    assert np.isclose(m["perseveration"], 0.15)         # (0.3 + 0) / 2
    assert np.isclose(m["trials_to_recover"], 1.0)      # (2 + 0) / 2
    off, mean, n = aligned_curve(choices, better, reversals, pre=5, post=10)
    assert mean[off == -1][0] == 0.0 and mean[off == 0][0] == 0.5 and mean[off == 5][0] == 1.0


def test_fit_recovers_q_learning_parameters():
    env = Environment(n_trials=1000, seed=21)
    sim = run_session(QLearningAgent(alpha=0.4, theta=6.0), env, seed=22)
    lik = Likelihood(ChoiceData(sim["choices"], sim["rewards"]))
    fit = fit_model(lik, "Q", n_starts=5, rng=np.random.default_rng(0))
    assert abs(fit["params"]["alpha"] - 0.4) < 0.15
    assert abs(fit["params"]["theta"] - 6.0) < 2.0
    assert fit["nll"] <= lik.nll("Q", np.array([0.4, 6.0])) + 1e-6
    assert max(fit["start_nlls"]) - min(fit["start_nlls"]) < 1e-3   # all starts agree


def test_fit_recovers_hopfield_discrete_parameters():
    env = Environment(n_trials=1000, seed=31)
    sim = run_session(HopfieldAgent(beta=3.0, M=8, L=2, theta=6.0), env, seed=32)
    lik = Likelihood(ChoiceData(sim["choices"], sim["rewards"]))
    fit = fit_model(lik, "HOP", n_starts=3, rng=np.random.default_rng(0))
    assert fit["params"]["M"] == 8
    assert fit["nll"] <= lik.nll("HOP", np.array([3.0, 6.0]), (2, 8)) + 1e-6


def test_analytic_gradient_matches_finite_differences():
    from scipy.optimize import approx_fprime
    rng = np.random.default_rng(13)
    data = _random_data(rng, T=300, n_sessions=3)
    lik = Likelihood(data)
    rows = rng.random(300) < 0.7
    w = [0.8, -0.3]
    cases = [
        ("STIM", w, None),
        ("Q", w + [0.37, 4.2], None),
        ("QK", w + [0.21, 3.0, 0.8], None),
        ("HOP", w + [1.7, 5.0], (2, 8)),
        ("HYB", w + [0.6, 6.0, 2.2, 0.35], (3, 16)),
        ("HOPK", w + [1.1, 3.0, -0.4], (1, 4)),
        ("HYBK", w + [0.15, 2.0, 3.0, 0.7, 0.5], (2, 32)),
        ("HOPC", w + [2.0], (2, 8)),
    ]
    for model, x, disc in cases:
        x = np.array(x)
        for r in (None, rows):
            f, g = lik.nll_and_grad(model, x, disc, r)
            assert np.isclose(f, lik.nll(model, x, disc, r))
            num = approx_fprime(x, lambda v: lik.nll(model, v, disc, r), 1e-6)
            assert np.allclose(g, num, rtol=1e-4, atol=1e-4), (model, g, num)


# ---------------------------------------------------------------- mouse data preparation
def _fake_ibl_session(code_right=-1):
    import pandas as pd
    # 6 trials: stimulus side R R L L R L ; choice R L L R (no-go) L
    stim_right = np.array([1, 1, 0, 0, 1, 0], dtype=bool)
    chose_right = np.array([1, 0, 0, 1, 0, 0], dtype=bool)
    nogo = np.array([0, 0, 0, 0, 1, 0], dtype=bool)
    contrast = np.array([1.0, 0.25, 0.0, 0.125, 0.0625, 1.0])
    choice = np.where(chose_right, code_right, -code_right)
    choice[nogo] = 0
    return pd.DataFrame({
        "contrastLeft": np.where(stim_right, np.nan, contrast),
        "contrastRight": np.where(stim_right, contrast, np.nan),
        "choice": choice,
        "feedbackType": np.where((chose_right == stim_right) & ~nogo, 1, -1),
        "probabilityLeft": [0.5, 0.5, 0.8, 0.8, 0.2, 0.2],
        "response_times": np.arange(6) + 0.5, "stimOn_times": np.arange(6.0),
    })


def test_ibl_session_table_infers_choice_coding():
    from hopfield_rl import ibl
    meta = {"subject": "m", "lab": "l", "id": "e", "start_time": "2020-01-01T10:00:00"}
    for code in (-1, 1):
        tab = ibl.session_table(_fake_ibl_session(code), meta)
        assert len(tab) == 5                                   # the no-go trial is dropped
        assert tab.choice_right.tolist() == [1, 0, 0, 1, 0]
        assert tab.reward.tolist() == [1, 0, 1, 0, 1]
        assert tab.signed_contrast.tolist() == [1.0, 0.25, -0.0, -0.125, -1.0]
        assert tab.stim_right.tolist() == [1, 1, 0, 0, 0]
        assert (tab.choice_code_right == code).all()


def test_ibl_design_matrix_and_block_switches():
    import pandas as pd
    from hopfield_rl import ibl
    t = pd.DataFrame({
        "signed_contrast": [1.0, -0.25, 0.0, 0.25, -1.0, 0.0],
        "choice_right": [1, 0, 1, 1, 0, 0], "reward": [1, 1, 0, 1, 1, 1],
        "session": [0, 0, 0, 0, 1, 1], "p_left": [0.5, 0.8, 0.2, 0.2, 0.2, 0.8]})
    levels = ibl.contrast_levels(t)
    assert levels == [0.25, 1.0]
    data = ibl.to_choice_data(t, levels)
    assert np.array_equal(data.X, [[0, 1, 1], [-1, 0, 1], [0, 0, 1], [1, 0, 1], [0, -1, 1], [0, 0, 1]])
    idx, new_right = ibl.block_switches(t)
    # 0.5 -> 0.8 is not a switch between biased blocks; the session boundary is not one either
    assert idx.tolist() == [2, 5] and new_right.tolist() == [1, 0]
