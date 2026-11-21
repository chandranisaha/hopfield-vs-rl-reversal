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

