# Do Hopfield Episodic Memories Explain Adaptation in Reversal Learning Better than Incremental RL?

INCM 2026 course project — Chandrani Saha (2024113002) and Inesh Roy (2024112015).

The project compares two ways in which an agent can adapt when the better option
switches without warning:

* **incremental RL** — Q-learning keeps one running value per option (Models 1–2);
* **episodic memory** — a Hopfield associative memory stores individual
  (history → choice, reward) episodes and retrieves the ones that resemble the current
  situation (Model 3), alone or mixed with Q-learning (Model 4).

It has three parts: (i) simulation of the agents on a probabilistic reversal task,
(ii) parameter and model recovery, (iii) fits to real mouse choices from the
International Brain Laboratory (IBL). The full write-up is in
[report/report.pdf](report/report.pdf).

## Layout

```
hopfield_rl/            the library
  envs.py               Environment: two-armed bandit with a reversal schedule
  agents.py             Agent interface (choose / update) and all models
  hopfield.py           classical (1982) and modern (2021) Hopfield networks on toy patterns
  simulate.py           running agents; reversal-aligned behavioural measures
  likelihood.py         exact choice likelihood of every model (+ analytic gradients)
  fitting.py            bounded multi-start maximum likelihood, grid over (L, M), cross-validation
  ibl.py                download and cleaning of the IBL mouse data
  experiments.py        simulation / recovery building blocks, parallel map
  plotting.py           shared figure style
scripts/                one script per analysis step (run in order)
tests/test_core.py      unit tests
demo.py                 short end-to-end demo
data/                   IBL trial table (created by scripts/04_download_ibl.py)
results/                numbers behind every figure (JSON / NPZ)
figures/                all figures (PDF + PNG)
report/                 the report (LaTeX source + PDF)
docs/                   project proposal and presentation slides
```

## Setup

Python 3.10 or newer.

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows   (Linux/macOS: source .venv/bin/activate)
pip install -r requirements.txt
python -m pytest tests -q         # 20 tests, about 10 s
```

## Demo

```bash
python demo.py
```

Runs in one to two minutes: the four agents play one reversal session, the retrieval
weights of the Hopfield memory are shown for a blurred and a sharp memory, and all four
models are fitted to the Hopfield agent's choices (BIC selects the generating model).
Output: a table in the terminal and `figures/demo.png`.

## Reproducing every result

Run the scripts in order from the project root. Each one writes its numbers to
`results/` and its figures to `figures/`. All random seeds are fixed.

| Step | Command | What it does | Time* |
|---|---|---|---|
| 1 | `python scripts/01_sanity_checks.py` | Q-learning converges to the true arm values; classical Hopfield capacity (≈ 0.14 N) and cue noise; modern Hopfield capacity and β | 3 min |
| 2 | `python scripts/02_behaviour.py` | all models on the reversal task over parameter grids; reversal-aligned curves; trials to recover, perseveration, reward rate; classical-memory agent vs. load and cue noise | 15 min |
| 3 | `python scripts/03_recovery.py` | parameter recovery and model recovery (confusion matrix), 100 simulated data sets per model | 60 min |
| 4 | `python scripts/04_download_ibl.py` | downloads the mouse data (see below) | network |
| 5 | `python scripts/05_fit_ibl.py` | fits all models to every mouse; 5-fold cross-validation over sessions; posterior-predictive simulations | about 4 h |
| 6 | `python scripts/06_ibl_figures.py` | model comparison statistics (CV log-likelihood, BIC, sign tests) and figures | 1 min |
| 7 | `python scripts/07_ibl_recovery.py` | model recovery repeated on the mouse task (synthetic mice with known mechanism) | about 45 min |

\* on a 12-thread laptop with 6 worker processes. Set the environment variable
`HRL_WORKERS` to change the number of worker processes (each needs about 200 MB of
memory). Scripts 2, 3 and 7 accept `--plot-only` to redraw figures from stored
results; scripts 1–3 and 5 accept `--quick` for a fast, low-precision test run. Scripts 5
and 7 are resumable: finished fits are stored in `results/ibl/parts/` and
`results/ibl_recovery_parts/` and skipped when the script is started again. Keep the
computer awake while script 5 runs.

## Data

**Simulated task (main data set).** Two arms; the better arm pays with probability 0.8
and the other with 0.2; the better arm switches every 20–40 trials; 1000 trials per
agent. Generated on the fly by `hopfield_rl/envs.py`.

**Real data: IBL mouse behaviour.** `scripts/04_download_ibl.py` downloads behavioural
trial tables of the biased-block task (`biasedChoiceWorld`) from the public IBL
behaviour data release (tag `2021_Q1_IBL_et_al_Behaviour`; The International Brain
Laboratory 2021, eLife 10:e63711) through the
[ONE API](https://int-brain-lab.github.io/ONE/) from
`https://openalyx.internationalbrainlab.org`. No registration is needed for the script
as written: it uses the public read-only credentials documented by the IBL
(user `intbrainlab`, password `international`). The IBL has announced that this shared
account will be retired; if the login fails, create a free account at
<https://openalyx.internationalbrainlab.org/signup> and put its user name and password
in `hopfield_rl/ibl.py` (`connect`).

* Selection (fixed in code before any model was fitted): mice with at least 15
  biased-block sessions in the release; laboratories are visited round-robin in
  alphabetical order, taking their mice alphabetically, until 15 mice are selected; for
  each mouse the 15 most recent biased-block sessions are used.
* Inclusion: a session needs at least 200 completed trials and both biased block
  types; a mouse needs at least 10 such sessions. Trials without a response are dropped.
* Result: 14 mice from 9 laboratories, 208 sessions, 142,219 trials (one of the 15
  selected mice has no trial tables on the server).
* Only the per-session trial tables are downloaded (about 60 kB each, about 15 MB in
  total). They are cached in `data/ibl_cache/`; the selected sessions are listed in
  `data/ibl_sessions.csv`; the cleaned table used by all analyses is
  `data/ibl_trials.csv.gz` (one row per trial: mouse, session, signed contrast, stimulus
  side, choice, reward, block prior, response time). The download is resumable.

The IBL data are released under a CC-BY 4.0 licence.

## Models in one place

On trial *t* the agent chooses *c_t* and receives reward *r_t* ∈ {0, 1}.

| Model | Value signal | Free parameters |
|---|---|---|
| 1 Q-learning | Q(c) ← Q(c) + α (r − Q(c)); P(c = k) ∝ exp(θ Q(k)) | α, θ |
| 2 Q + stickiness | adds κ to the logit of the previously chosen arm | α, θ, κ |
| 3 Hopfield (modern) | key = previous L (choice, reward) pairs, one-hot; w_i = softmax_i(β k_iᵀq) over the last M episodes; V(a) = Σ w_i r_i 1[c_i = a] / (Σ w_i 1[c_i = a] + ε) | β, M, L, θ |
| 3c Hopfield (classical) | binary Hebbian network over (key, choice, reward) patterns, cue with key and candidate action, iterate sign(Wξ), read the reward units | M, L, θ |
| 4 Hybrid | V = (1 − λ) Q + λ V_epi | α, θ, β, M, L, λ |

Fixed constants: Q₀ = 0.5, ε = 10⁻⁶. On the mouse data every model additionally has
one stimulus weight per contrast level and a constant side bias (logistic stimulus
term); "arms" are the left and right choices and the reward is the water reward. The
supplementary models "Hopfield + κ" and "Hybrid + κ" add the stickiness term of Model 2
to Models 3 and 4.

## Main results

* **Sanity checks.** Q-values converge to the true arm values (0.790 / 0.191 for 0.8 / 0.2).
  The classical Hopfield network breaks down around 0.14 N stored patterns; the modern
  network (β = 1) retrieves perfectly up to 50 N.
* **Reversal task.** A Hopfield memory with a short window (M = 2) earns as much as the
  best Q-learner (reward rate 0.687 vs. 0.686). Long windows and blurred retrieval
  (low β) cause interference; a sharp memory spanning several blocks adapts fast again.
  The classical-memory agent is limited by interference between similar episodes, not
  by the 0.14 N capacity.
* **Recovery.** BIC identifies the generating model in 92.5 % of 400 simulated data
  sets; parameters of Models 1–3 are recovered with Spearman ρ ≥ 0.92.
* **Mice (14 mice, 142,219 trials, 5-fold CV over sessions).** Plain Q-learning is no
  better than a model without history. The Hopfield model beats it in 14/14 mice, but
  Q + stickiness is the best of the four proposed models (9/14 mice; its lead over
  Hopfield is not significant, sign test p = 0.42). The fitted memory uses M = 2 and
  β = 0 in most mice, i.e. a two-trial outcome average without similarity-based
  retrieval. Adding stickiness to the Hopfield / hybrid model gives the best account
  (better than Q + stickiness in 12/14 and 13/14 mice). The Hopfield model reproduces
  the choice curve after block switches best.
* **Recovery on the mouse task.** Q, Q + stickiness and Hopfield are identified in
  14/14 synthetic mice each; the hybrid is confused with Hopfield (6/14 correct).
