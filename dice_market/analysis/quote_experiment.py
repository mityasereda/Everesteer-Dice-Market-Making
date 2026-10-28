"""Quote widths, skew and learner settings on top of the best take margins, replayed on my real v5 matches.

    python3 analysis/quote_experiment.py [matches=200]
"""
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import replay_policy as rp  # noqa: E402

N = None
T = {"take": (N, 0.5, 0.25, 0.25)}
PRIOR_ALL = {0: (1.0, []), 1: (1.0, [0.57]), 2: (0.55, [0.05, 0.19]), 3: (0.30, [0.10, -0.06, 0.26])}
VARIANTS = {
    "off (= v5)": {},
    "take (best margins)": dict(T),
    "take + half 2,2,1.5,1.0": {**T, "half": (2.0, 2.0, 1.5, 1.0)},
    "take + half 2,2,2.0,1.0": {**T, "half": (2.0, 2.0, 2.0, 1.0)},
    "take + half 2,2,1.0,0.5": {**T, "half": (2.0, 2.0, 1.0, 0.5)},
    "take + half 2,2,1.0,1.5": {**T, "half": (2.0, 2.0, 1.0, 1.5)},
    "take + half 2,1.5,1.0,1.0": {**T, "half": (2.0, 1.5, 1.0, 1.0)},
    "take + half 2,1.0,1.0,1.0": {**T, "half": (2.0, 1.0, 1.0, 1.0)},
    "take + skew 0.10": {**T, "skew": 0.10},
    "take + skew 0.50": {**T, "skew": 0.50},
    "take + lam 100": {**T, "lam": 100.0},
    "take + prior refit (all seats)": {**T, "_prior": PRIOR_ALL},
    "take + prior refit + lam 100": {**T, "_prior": PRIOR_ALL, "lam": 100.0},
    "take + mode both, prior refit, lam 100": {**T, "_prior": PRIOR_ALL, "lam": 100.0, "mode": "both", "lam_seat": 300.0},
}


def main():
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    rows = rp.select_rows("data/mine_all.jsonl.gz", "mine", limit)
    print(f"{len(rows)} real v5 matches ({len(rows) * 200} rounds)")
    with ProcessPoolExecutor() as pool:
        res = rp.evaluate("agent.py", VARIANTS, rows, pool)
    rp.report(res, "off (= v5)")


if __name__ == "__main__":
    main()
