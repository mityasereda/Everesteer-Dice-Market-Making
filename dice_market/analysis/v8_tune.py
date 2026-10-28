"""Later-step settings of v8 (widths, skew, take margins), tuned on one sample of 28 Oct matches and checked on another.

    python3 analysis/v8_tune.py [rows=1200] [stride=10]      # tuning sample;  stride 13 gives a mostly different validation sample
"""
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import replay_policy as rp  # noqa: E402

N = None
VARIANTS = {
    "v8 (defaults)": {},
    "half 2,2,2,1.5": dict(half=(2.0, 2.0, 2.0, 1.5)),
    "half 2,2,2,0.5": dict(half=(2.0, 2.0, 2.0, 0.5)),
    "half 2,2,1.5,1": dict(half=(2.0, 2.0, 1.5, 1.0)),
    "half 2,1.5,2,1": dict(half=(2.0, 1.5, 2.0, 1.0)),
    "skew 0.10": dict(skew=0.10),
    "skew 0.5": dict(skew=0.5),
    "skew 1.0": dict(skew=1.0),
    "take .25/.25/.25": dict(take=(N, 0.25, 0.25, 0.25)),
    "take 1/.5/.5": dict(take=(N, 1.0, 0.5, 0.5)),
    "take .5/.5/.5": dict(take=(N, 0.5, 0.5, 0.5)),
    "take 0/0/0": dict(take=(N, 0.0, 0.0, 0.0)),
    "lam 300": dict(lam=300.0),
    "lam 30": dict(lam=30.0),
}

if __name__ == "__main__":
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else 1200
    every = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    rows = rp.select_rows("data/day28.jsonl.gz", "all", limit, every)
    print(f"{len(rows)} seat-matches ({len(rows) * 200} rounds), stride {every}")
    with ProcessPoolExecutor() as pool:
        rp.report(rp.evaluate("agent.py", VARIANTS, rows, pool), "v8 (defaults)")
