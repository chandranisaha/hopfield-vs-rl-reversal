"""Hopfield episodic memory vs. incremental RL in reversal learning (INCM 2026).

Modules
-------
envs        Environment (reward probabilities + reversal schedule)
agents      Agent interface (choose / update) and the models of the proposal
hopfield    Classical (1982) and modern (2021) Hopfield networks on toy patterns
simulate    Running agents on environments, reversal-aligned behavioural measures
likelihood  Fast, exact likelihoods of a choice sequence under every model
fitting     Bounded multi-start maximum likelihood, model comparison utilities
ibl         Download and preparation of the IBL mouse behaviour data
"""

__version__ = "1.0.0"
