"""International Brain Laboratory (IBL) mouse behaviour: download and trial table.

Data: sessions of the biased-block task (``biasedChoiceWorld``) from the public IBL
behaviour data release (tag ``2021_Q1_IBL_et_al_Behaviour``; IBL 2021, eLife 10:e63711),
accessed through the ONE API on the public server.  Only the small per-session trial
tables (``_ibl_trials.table.pqt``) are downloaded; no neural data.

Task: a grating appears on the left or right at contrast {100, 25, 12.5, 6.25, 0} %.
After 90 unbiased trials the probability that the stimulus is on the left alternates
between 0.8 and 0.2 in blocks of 20-100 trials.  A correct wheel turn is rewarded.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

from .likelihood import ChoiceData

BASE_URL = "https://openalyx.internationalbrainlab.org"
TAG = "2021_Q1_IBL_et_al_Behaviour"
TABLE_NAME = "ibl_trials.csv.gz"


def connect(cache_dir):
    """ONE client for the public IBL database (public read-only credentials)."""
    from one.api import ONE
    return ONE(base_url=BASE_URL, username="intbrainlab", password="international",
               silent=True, cache_dir=cache_dir)


def list_sessions(one):
    """All biased-block sessions of the behaviour data release, as a DataFrame."""
    ses = one.alyx.rest("sessions", "list", tag=TAG, task_protocol="biased")
    df = pd.DataFrame([{k: s[k] for k in ("id", "subject", "lab", "start_time", "number",
                                           "task_protocol")} for s in ses])
    df = df[df.task_protocol.str.contains("biasedChoiceWorld")]
    return df.sort_values(["subject", "start_time"]).reset_index(drop=True)


def select_sessions(sessions, n_mice=15, n_sessions=15):
    """Deterministic, lab-balanced selection (made before looking at any behaviour).

    Mice with at least ``n_sessions`` biased-block sessions are eligible.  Laboratories
    are visited round-robin in alphabetical order, taking their mice in alphabetical
    order, until ``n_mice`` are selected.  For every mouse the ``n_sessions`` most
    recent biased-block sessions are used (the best-trained behaviour).
    """
    counts = sessions.groupby("subject").size()
    eligible = sessions[sessions.subject.isin(counts[counts >= n_sessions].index)]
    by_lab = {lab: sorted(g.subject.unique()) for lab, g in eligible.groupby("lab")}
    chosen = []
    while len(chosen) < n_mice and any(by_lab.values()):
        for lab in sorted(by_lab):
            if by_lab[lab] and len(chosen) < n_mice:
                chosen.append(by_lab[lab].pop(0))
    out = [eligible[eligible.subject == s].sort_values("start_time").tail(n_sessions)
           for s in sorted(chosen)]
    return pd.concat(out).reset_index(drop=True)


def session_table(trials, meta):
    """Clean one session's trial table.

    Output columns: signed_contrast (>0: stimulus on the right), stim_right, choice_right,
    reward, p_left (block prior), rt, plus session metadata.  No-go trials (no wheel
    response) are dropped.  The sign convention of ``choice`` is not assumed: it is
    inferred from the data (a rewarded trial means the chosen side equals the stimulus
    side) and checked for consistency on every trial.
    """
    t = pd.DataFrame(trials).copy()
    stim_right = t.contrastLeft.isna().to_numpy()
    both = t.contrastLeft.isna() == t.contrastRight.isna()
    if both.any():
        raise ValueError("trials with stimulus on both / neither side")
    contrast = np.where(stim_right, t.contrastRight, t.contrastLeft).astype(float)
    go = t.choice.to_numpy() != 0
    rewarded = t.feedbackType.to_numpy() == 1
    chose_right_from_feedback = rewarded == stim_right
    raw = t.choice.to_numpy()
    # which raw code (+1 or -1) corresponds to a rightward choice?
    agree_pos = np.mean((raw[go] == 1) == chose_right_from_feedback[go])
    code_right = 1 if agree_pos > 0.5 else -1
    if not np.array_equal(raw[go] == code_right, chose_right_from_feedback[go]):
        raise ValueError("choice, feedback and stimulus side are inconsistent")
    out = pd.DataFrame({
        "subject": meta["subject"], "lab": meta["lab"], "eid": meta["id"],
        "date": str(meta["start_time"])[:10],
        "trial": np.arange(len(t)),
        "signed_contrast": np.where(stim_right, contrast, -contrast),
        "stim_right": stim_right.astype(int),
        "choice_right": (raw == code_right).astype(int),
        "reward": rewarded.astype(int),
        "p_left": t.probabilityLeft.to_numpy().round(2),
        "rt": (t.response_times - t.stimOn_times).to_numpy(),
    })
    out["choice_code_right"] = code_right
    return out[go].reset_index(drop=True)


def _cached_file(cache_dir, meta):
    """Where ONE stores the trial table of a session inside its cache directory."""
    return os.path.join(cache_dir, str(meta["lab"]), "Subjects", str(meta["subject"]),
                        str(meta["start_time"])[:10], f"{int(meta['number']):03d}", "alf",
                        "_ibl_trials.table.pqt")


def download(data_dir, n_mice=15, n_sessions=15, min_trials=200, min_sessions=10, log=print):
    """Download the selected sessions and write the combined trial table.

    Resumable: the session selection is stored in ``ibl_sessions.csv`` and trial tables
    already present in the ONE cache are read from disk without contacting the server.

    Inclusion rules (fixed before any model was fitted):
      * session: at least ``min_trials`` completed trials and both biased block types;
      * mouse: at least ``min_sessions`` such sessions.
    """
    cache_dir = os.path.join(data_dir, "ibl_cache")
    sessions_file = os.path.join(data_dir, "ibl_sessions.csv")
    one = None
    if os.path.exists(sessions_file):
        selected = pd.read_csv(sessions_file)
        log(f"using the stored selection of {len(selected)} sessions ({sessions_file})")
    else:
        one = connect(cache_dir)
        sessions = list_sessions(one)
        log(f"{len(sessions)} biasedChoiceWorld sessions from {sessions.subject.nunique()} "
            f"mice in the release")
        selected = select_sessions(sessions, n_mice, n_sessions)
        os.makedirs(data_dir, exist_ok=True)
        selected.to_csv(sessions_file, index=False)
    tables, skipped = [], []
    for i, meta in selected.iterrows():
        local = _cached_file(cache_dir, meta)
        tab, err = None, None
        for attempt in range(3):
            try:
                if os.path.exists(local):
                    trials = pd.read_parquet(local)
                else:
                    one = connect(cache_dir) if one is None else one
                    trials = one.load_dataset(meta["id"], "_ibl_trials.table.pqt", collection="alf")
                tab = session_table(trials, meta)
                break
            except Exception as e:  # network or memory hiccup: try again
                err = e
        if tab is None:
            skipped.append((meta["subject"], meta["id"], repr(err)[:120]))
            log(f"  [{i + 1}/{len(selected)}] {meta['subject']} {meta['id']}: FAILED {err!r}")
            continue
        ok = len(tab) >= min_trials and set(np.unique(tab.p_left)) >= {0.2, 0.8}
        log(f"  [{i + 1}/{len(selected)}] {meta['subject']} {str(meta['start_time'])[:10]}: "
            f"{len(tab)} trials{'' if ok else '  (excluded: too short or no biased blocks)'}")
        if ok:
            tables.append(tab)
        else:
            skipped.append((meta["subject"], meta["id"], f"{len(tab)} trials / no blocks"))
    table = pd.concat(tables, ignore_index=True)
    n_ok = table.groupby("subject").eid.nunique()
    for subject in n_ok[n_ok < min_sessions].index:
        log(f"  mouse {subject} excluded: only {n_ok[subject]} valid sessions")
        skipped.append((subject, "all", f"only {n_ok[subject]} valid sessions"))
    table = table[table.subject.isin(n_ok[n_ok >= min_sessions].index)].reset_index(drop=True)
    table["session"] = table.groupby("subject")["eid"].transform(lambda s: pd.factorize(s)[0])
    table.to_csv(os.path.join(data_dir, TABLE_NAME), index=False)
    return table, skipped


def load_table(path):
    return pd.read_csv(path)


def contrast_levels(table):
    """Sorted non-zero contrast magnitudes present in the table."""
    c = np.unique(np.abs(table.signed_contrast.to_numpy()))
    return [float(x) for x in c if x > 0]


def to_choice_data(mouse_table, levels):
    """ChoiceData for one mouse.  Arm 1 = right, arm 0 = left.

    Stimulus design (logit of a rightward choice): one weight per contrast level
    (regressor +1 / -1 if a stimulus of that contrast is on the right / left) plus a
    constant side bias.  A 0 % contrast trial has only the bias term, so on those
    trials the history term carries all the remaining information.
    """
    sc = mouse_table.signed_contrast.to_numpy()
    X = np.zeros((len(sc), len(levels) + 1))
    for j, lev in enumerate(levels):
        X[:, j] = np.sign(sc) * np.isclose(np.abs(sc), lev)
    X[:, -1] = 1.0
    return ChoiceData(mouse_table.choice_right.to_numpy(), mouse_table.reward.to_numpy(),
                      mouse_table.session.to_numpy(), X)


def block_switches(mouse_table):
    """Indices (rows of the mouse table) of the first trial after a switch between the
    biased blocks (0.8 <-> 0.2), with the newly more likely side (1 = right)."""
    p = mouse_table.p_left.to_numpy()
    s = mouse_table.session.to_numpy()
    idx = np.nonzero((p[1:] != p[:-1]) & (s[1:] == s[:-1]) & (p[:-1] != 0.5))[0] + 1
    new_side_right = (p[idx] < 0.5).astype(int)
    return idx, new_side_right
