"""Hider variants (v7 settings) replayed on the freshest matches: every candidate seat plays my agent against the rivals' recorded quotes.

    python3 analysis/v9_tune.py [rows=2000] [stride=10] [data=data/v8_since.jsonl.gz]
A hider shows nothing at step 0, so this replay is faithful for it (validated on v5 / v7 in section 12).  Old prior kept: it assumes rivals do not
know my dice, which is true for a hider.
"""
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import replay_policy as rp  # noqa: E402

OLD = {0: (1.0, []), 1: (1.0, [0.72]), 2: (0.75, [-0.16, 0.26]), 3: (0.26, [0.05, -0.03, 0.26])}
N = None
V7 = dict(hide0=True, lam=30.0, clip0=5.0, _prior=OLD)
A = {**V7, "hide_w": 3.0, "lam": 100.0}
VARIANTS = {
    "v7 (40/44, lam 30)": V7,
    "hide quote 40.5/43.5": {**V7, "hide_w": 3.0},
    "A = 40.5/43.5 + lam 100": A,
    "A + lam 300": {**A, "lam": 300.0},
    "B = A + take .5/.5/.5": {**A, "take": (N, 0.5, 0.5, 0.5)},
    "C = B + half step 3 1.5": {**A, "take": (N, 0.5, 0.5, 0.5), "half": (2.0, 2.0, 2.0, 1.5)},
    "A + half step 3 1.5": {**A, "half": (2.0, 2.0, 2.0, 1.5)},
    "v8 (informed, for reference)": {},
}
if __name__ == "__main__":
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
    every = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    data = sys.argv[3] if len(sys.argv) > 3 else "data/v8_since.jsonl.gz"
    rows = rp.select_rows(data, "all", limit, every)
    print(f"{len(rows)} seat-matches ({len(rows) * 200} rounds), stride {every}, {data}")
    with ProcessPoolExecutor() as pool:
        rp.report(rp.evaluate("agent.py", VARIANTS, rows, pool), "v7 (40/44, lam 30)")
