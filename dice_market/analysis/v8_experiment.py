"""v8 variants replayed on 28 Oct matches (rivals' recorded quotes held fixed; every candidate seat plays my agent in turn).

    python3 analysis/v8_experiment.py [seat_matches=1000] [stride=24] [skip=0]
Per-step edge per round and total, paired against the v7-equivalent settings. Exact at step 0; approximate later, and it cannot
measure what revealing my dice costs at steps 1 to 3 (rivals do not react), so only compare variants that reveal the same.
"""
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import replay_policy as rp  # noqa: E402

OLD = {0: (1.0, []), 1: (1.0, [0.72]), 2: (0.75, [-0.16, 0.26]), 3: (0.26, [0.05, -0.03, 0.26])}
CLIP5 = {0: (1.0, []), 1: (1.0, [0.47]), 2: (0.30, [0.05, 0.21]), 3: (0.29, [0.12, -0.07, 0.26])}
V7 = dict(hide0=True, lam=30.0, clip0=5.0, _prior=OLD)
VARIANTS = {
    "v7 (hide, old prior, lam 30)": V7,
    "v8 (levels P=6, new prior, lam 100)": {},
    "v8, lam 30": dict(lam=30.0),
    "v8, old prior + clip 5 + lam 30": dict(lam=30.0, clip0=5.0, _prior=OLD),
    "v8, clip 5 (prior fit at 5)": dict(clip0=5.0, _prior=CLIP5),
    "smooth lean k=2 (v6-like), new prior": dict(step0_levels=None, step0_k=2.0),
    "levels P=5": dict(step0_levels=5.0),
    "levels P=6.5": dict(step0_levels=6.5),
    "levels P=7": dict(step0_levels=7.0),
    "levels P=8": dict(step0_levels=8.0),
    "v8 without take logic": dict(take=(None, None, None, None)),
}


def main():
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
    every = int(sys.argv[2]) if len(sys.argv) > 2 else 24
    skip = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    rows = rp.select_rows("data/day28.jsonl.gz", "all", limit + skip, every)[skip:]
    print(f"{len(rows)} seat-matches ({len(rows) * 200} rounds)")
    with ProcessPoolExecutor() as pool:
        res = rp.evaluate("agent.py", VARIANTS, rows, pool)
    rp.report(res, "v7 (hide, old prior, lam 30)")


if __name__ == "__main__":
    main()
