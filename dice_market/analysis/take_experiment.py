"""Take-logic variants replayed on my real v5 matches (paired: same rounds, same rivals' recorded quotes).

    python3 analysis/take_experiment.py [matches=200]
"""
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import replay_policy as rp  # noqa: E402

N = None
VARIANTS = {
    "off (= v5)": {},
    "margin 1.5 / .75 / .75": {"take": (N, 1.5, 0.75, 0.75)},
    "margin 1.0 / .5 / .5": {"take": (N, 1.0, 0.5, 0.5)},
    "margin 0.5 / .25 / .25": {"take": (N, 0.5, 0.25, 0.25)},
    "margin 0 at every step": {"take": (N, 0.0, 0.0, 0.0)},
    "step 1 only, margin 0.5": {"take": (N, 0.5, N, N)},
    "step 2 only, margin 0.5": {"take": (N, N, 0.5, N)},
    "step 3 only, margin 0.5": {"take": (N, N, N, 0.5)},
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
