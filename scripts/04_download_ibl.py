"""Download the IBL subset (real mouse data) and build the trial table.

15 mice x their 15 most recent biased-block sessions from the public IBL behaviour
data release, behaviour tables only (about 60 kB per session).  Needs internet access;
uses the public IBL credentials through the ONE API.

Output: data/ibl_trials.csv.gz, data/ibl_sessions.csv (the session selection) and
        data/ibl_download_log.txt; raw files in data/ibl_cache/.  The script is resumable:
        run it again after an interruption and only the missing sessions are fetched.
Usage:  python scripts/04_download_ibl.py [--mice 15] [--sessions 15]
"""

import os
import sys
import warnings

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
warnings.filterwarnings("ignore")

from hopfield_rl import ibl
from hopfield_rl.experiments import DATA


def arg(name, default):
    return int(sys.argv[sys.argv.index(name) + 1]) if name in sys.argv else default


def main():
    os.makedirs(DATA, exist_ok=True)
    lines = []

    def log(msg):
        print(msg, flush=True)
        lines.append(msg)

    table, skipped = ibl.download(DATA, n_mice=arg("--mice", 15), n_sessions=arg("--sessions", 15),
                                  log=log)
    log(f"\nSaved {len(table)} trials, {table.subject.nunique()} mice, "
        f"{table.groupby('subject').session.nunique().sum()} sessions")
    log(table.groupby(["lab", "subject"]).agg(sessions=("session", "nunique"),
                                              trials=("trial", "size"),
                                              accuracy=("reward", "mean")).round(3).to_string())
    log(f"choice code for 'right' in the raw data: {sorted(table.choice_code_right.unique())}")
    log(f"skipped sessions: {skipped}")
    with open(os.path.join(DATA, "ibl_download_log.txt"), "w") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    main()
