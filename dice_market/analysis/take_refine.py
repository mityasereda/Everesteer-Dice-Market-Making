"""Refine take margins on the wide-step-2 setting (tuning set = my first 200 v5 matches), then validate on the rest.

    python3 analysis/take_refine.py
"""
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import replay_policy as rp  # noqa: E402

N = None
W = {"half": (2.0, 2.0, 2.0, 1.0)}
REFINE = {
    "off (= v5)": {},
    "wide step 2, take .5/.25/.25": {**W, "take": (N, 0.5, 0.25, 0.25)},
    "  margins .25/.25/.25": {**W, "take": (N, 0.25, 0.25, 0.25)},
    "  margins .75/.25/.25": {**W, "take": (N, 0.75, 0.25, 0.25)},
    "  margins .5/.5/.25": {**W, "take": (N, 0.5, 0.5, 0.25)},
    "  margins .5/1.0/.25": {**W, "take": (N, 0.5, 1.0, 0.25)},
    "  margins .5/.25/.5": {**W, "take": (N, 0.5, 0.25, 0.5)},
    "  step 1 only .5": {**W, "take": (N, 0.5, N, N)},
    "  steps 1-2 .5/.5": {**W, "take": (N, 0.5, 0.5, N)},
    "  no take, wide step 2 only": {**W},
    "  step 3 half 1.25": {**W, "take": (N, 0.5, 0.25, 0.25), "half": (2.0, 2.0, 2.0, 1.25)},
}
VALIDATE = {
    "off (= v5)": {},
    "take .5/.25/.25": {"take": (N, 0.5, 0.25, 0.25)},
    "wide step 2 only": {**W},
    "wide step 2 + take .5/.25/.25": {**W, "take": (N, 0.5, 0.25, 0.25)},
}


def main():
    with ProcessPoolExecutor() as pool:
        tune = rp.select_rows("data/mine_all.jsonl.gz", "mine", 200)
        print(f"TUNING SET: {len(tune)} matches")
        rp.report(rp.evaluate("agent.py", REFINE, tune, pool), "off (= v5)")
        val = rp.select_rows("data/mine_all.jsonl.gz", "mine", 10 ** 6, skip=200)
        print(f"\nVALIDATION SET (never used for tuning): {len(val)} matches")
        rp.report(rp.evaluate("agent.py", VALIDATE, val, pool), "off (= v5)")


if __name__ == "__main__":
    main()
