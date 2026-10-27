"""Coordinate-descent tuning of the half widths (steps 1 to 3) and the position skew of agent.py.

    python3 analysis/tune_spreads.py agent.py [n=80] [passes=2]

Variants are written to .tune_variants/ (git-ignored) and scored on the same seeds against the TRAIN mixes (adaptive
copies of real agents). Score = mean pairwise win rate over the train mixes, with mean P&L per round shown beside it.
half[0] is not tuned: while hide0 is on the step-0 quote is the default 40 / 44.
Results against these opponents are provisional: they are copies of real agents' average behaviour (see tourney.py).
"""
import re
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import tourney  # noqa: E402

TRAIN = ["pro_adapt", "fieldA_adapt", "gate_adapt"]
SEED = 7000
GRID = {1: [1.0, 1.5, 2.0], 2: [1.0, 1.5, 1.75, 2.0], 3: [0.5, 0.75, 1.0, 1.5], "skew": [0.1, 0.25, 0.4, 0.6]}
VAR = ROOT / ".tune_variants"


def variant(src, half, skew, tag):
    p = f"half=({', '.join(f'{h:.2f}' for h in half)}),"
    out, k = re.subn(r"half=\([^)]*\),", p, src, count=1)
    out, k2 = re.subn(r"skew=[0-9.]+,", f"skew={skew:.2f},", out, count=1)
    assert k == 1 and k2 == 1
    VAR.mkdir(exist_ok=True)
    path = VAR / f"v_{tag}.py"
    path.write_text(out)
    return path


def score(path, n, pool, mixes=TRAIN, seed=SEED):
    rs = [tourney.run(path, m, n, 200, seed, pool) for m in mixes]
    return sum(r["win"] for r in rs) / len(rs), sum(r["pnl"] for r in rs) / len(rs), rs


if __name__ == "__main__":
    src = Path(sys.argv[1]).read_text()
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 80
    passes = int(sys.argv[3]) if len(sys.argv) > 3 else 2
    half = [2.0, 2.0, 1.5, 1.5]
    skew = 0.4
    cache = {}
    with ProcessPoolExecutor() as pool:
        def ev(h, s):
            key = (tuple(h), s)
            if key not in cache:
                cache[key] = score(variant(src, h, s, f"{'_'.join(map(str, h))}_{s}"), n, pool)[:2]
            return cache[key]

        base = ev(half, skew)
        print(f"start half={half} skew={skew}  win {base[0]:.3f}  P&L {base[1]:+.3f}", flush=True)
        for ps in range(passes):
            for dim in (3, 2, 1, "skew"):
                best = (ev(half, skew), half[dim] if dim != "skew" else skew)
                for v in GRID[dim]:
                    h2, s2 = list(half), skew
                    if dim == "skew":
                        s2 = v
                    else:
                        h2[dim] = v
                    r = ev(h2, s2)
                    print(f"  pass {ps + 1} {dim!s:>4} = {v:<5}  win {r[0]:.3f}  P&L {r[1]:+.3f}", flush=True)
                    if (r[0], r[1]) > best[0]:
                        best = (r, v)
                if dim == "skew":
                    skew = best[1]
                else:
                    half[dim] = best[1]
                print(f"pass {ps + 1}: best {dim} -> {best[1]}   now half={half} skew={skew}  win {best[0][0]:.3f}  P&L {best[0][1]:+.3f}", flush=True)
    print("FINAL half =", tuple(half), " skew =", skew)
