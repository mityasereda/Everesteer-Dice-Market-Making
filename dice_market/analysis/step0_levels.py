"""Step-0 quote families against today's field: the coarse three-level quote the top 40 use, versus the default and a smooth lean.

    python3 analysis/step0_levels.py data/day28.jsonl.gz [matches=240] [workers=6]

For every seat-round of a sample of real matches, replace that seat's step-0 quote by each candidate (rivals' recorded quotes held fixed;
nobody has seen anything at step 0, so this part is exact), run the real engine, and score
  cross  = edge of the step-0 fills (value - price on buys, price - value on sells), and
  picked = edge of the fills where a rival's recorded step-1 quote picks off the candidate's resting step-0 quote
           (approximate: rivals' step-1 quotes were reactions to the quote the seat really sent).
The three-level family is: own sum < 7 -> mid 42 - P, own sum > 7 -> mid 42 + P, own sum == 7 -> 40/44, width 4.
What revealing costs at steps 1-3 cannot be measured here.  Train on even matches, report on odd ones.
"""
import gzip
import json
import random
import sys
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import dicemarket as dm  # noqa: E402


def snap(x):
    return round(x * 2) / 2


def linear(k, w=4.0):
    def f(own):
        c = 42.0 + k * (own - 7.0)
        b = min(max(snap(c - w / 2), 12.0), 72.0 - w)
        return (b, b + w)
    return f


def levels(p_low, p_high=None, w=4.0, extreme=None):
    p_high = p_low if p_high is None else p_high

    def f(own):
        if own == 7:
            return (40.0, 44.0)
        if extreme and abs(own - 7) < extreme:
            return (40.0, 44.0)
        c = 42.0 - p_low if own < 7 else 42.0 + p_high
        b = min(max(snap(c - w / 2), 12.0), 72.0 - w)
        return (b, b + w)
    return f


CANDS = {"default 40/44": lambda own: (40.0, 44.0), "smooth lean k=2.0": linear(2.0), "smooth lean k=2.5": linear(2.5)}
for p in (3, 4, 5, 6, 6.5, 7, 8, 9, 10, 12):
    CANDS[f"3 levels P={p:g}"] = levels(p)
CANDS["3 levels P=7, only own<=5 / >=9"] = levels(7, extreme=2)
import random as _r
_rr = _r.Random(11)


def decoy(p, w=4.0, flip=None):
    """Far quote on a side chosen independently of my dice (flip=None: coin toss; flip=True: always the opposite side of my dice)."""
    def f(own):
        up = _rr.random() < 0.5 if flip is None else (own < 7) if flip else (own > 7)
        if flip is not None and own == 7:
            up = _rr.random() < 0.5
        b = min(max(snap(42.0 + (p if up else -p) - w / 2), 12.0), 72.0 - w)
        return (b, b + w)
    return f


for p in (4, 5, 6, 7):
    CANDS[f"decoy: coin-toss side, P={p}"] = decoy(p)
CANDS["decoy: opposite side to my dice, P=6"] = decoy(6, flip=True)
CANDS["3 levels low 8 / high 6"] = levels(8, 6)
for w in (2.0, 3.0, 4.0):
    for sh in (-2.0, -1.0, 0.0, 1.0, 2.0):
        CANDS[f"hider quote width {w:g} shifted {sh:+g}"] = (lambda w, sh: (lambda own: (40.0 + sh + (4.0 - w) / 2, 44.0 + sh - (4.0 - w) / 2)))(w, sh)
for bid, ask in ((40.0, 43.5), (40.5, 44.0), (40.5, 43.0), (41.0, 44.0), (40.0, 43.0), (41.0, 43.0), (39.5, 43.0), (41.0, 44.5), (40.5, 43.5), (40.0, 44.0)):
    CANDS[f"hider quote {bid:g}/{ask:g}"] = (lambda b, a: (lambda own: (b, a)))(bid, ask)
for c in (2, 3, 4):
    for p in (6, 7, 8):
        CANDS[f"push only when |dice-7|>={c}, P={p}"] = levels(p, extreme=c)


def work(args):
    idx, line = args
    m = json.loads(line)
    rng = random.Random(idx)
    out = {name: [0.0, 0.0, 0, 0.0] for name in CANDS}   # cross edge, picked edge, seat-rounds, sum of squared match means
    per = {name: [0.0, 0] for name in CANDS}
    for r in m["rounds"][::3]:
        d, v = r["d"], r["v"]
        rec0 = {i: (b / 2.0, a / 2.0) for i, (b, a) in enumerate(r["st"][0]["s"])}
        rec1 = {i: (b / 2.0, a / 2.0) for i, (b, a) in enumerate(r["st"][1]["s"])}
        for me in range(5):
            if m["kinds"][me] == "house":
                continue
            own = d[me] + d[me + 6]
            for name, pol in CANDS.items():
                q = pol(own)
                orders = dict(rec0)
                orders[me] = q
                book = {i: {"bid": None, "ask": None} for i in range(5)}
                pos = {i: 0 for i in range(5)}
                cross = picked = 0.0
                for t in dm.match_step(book, orders, rng, pos, 0):
                    if t["buyer"] == me:
                        cross += v - t["price"]
                    if t["seller"] == me:
                        cross += t["price"] - v
                for t in dm.match_step(book, rec1, rng, pos, 1):
                    if t["kind"] != "take":
                        continue
                    if t["seller"] == me and t["price"] == q[1]:
                        picked += t["price"] - v
                    if t["buyer"] == me and t["price"] == q[0]:
                        picked += v - t["price"]
                o = out[name]
                o[0] += cross; o[1] += picked; o[2] += 1
                per[name][0] += cross + picked; per[name][1] += 1
    for name, (e, k) in per.items():
        out[name][3] = (e / k) ** 2 if k else 0.0
    return idx, out


if __name__ == "__main__":
    path = sys.argv[1]
    n_match = int(sys.argv[2]) if len(sys.argv) > 2 else 240
    workers = int(sys.argv[3]) if len(sys.argv) > 3 else 6
    STRIDE = int(sys.argv[4]) if len(sys.argv) > 4 else 5
    lines = []
    with gzip.open(path, "rt") as f:
        for i, line in enumerate(f):
            if i % STRIDE == 0:                          # spread the sample over the whole day
                lines.append((len(lines), line))
            if len(lines) >= n_match:
                break
    with Pool(workers) as pool:
        res = pool.map(work, lines, chunksize=4)
    acc = {part: {n: [0.0, 0.0, 0, 0.0] for n in CANDS} for part in ("train", "test")}   # cross, picked, n, per-match total^2
    for idx, out in res:
        part = "train" if idx % 2 == 0 else "test"
        for n, (c, p, k, q) in out.items():
            a = acc[part][n]
            a[0] += c; a[1] += p; a[2] += k; a[3] += q
    print(f"{len(lines)} matches sampled (every {STRIDE}th of the day, every 3rd round); per seat-round edge, rivals fixed")
    print(f"{'step-0 quote':36s}{'train total':>12s} | {'TEST cross':>11s}{'picked@1':>10s}{'total':>9s}{'+-se':>7s}")
    rows = []
    for n in CANDS:
        tr, te = acc["train"][n], acc["test"][n]
        nm = len(lines) / 2
        mean_sq = te[3] / nm
        mean = (te[0] + te[1]) / te[2]
        se = max(mean_sq - mean ** 2, 0) ** 0.5 / nm ** 0.5
        rows.append(((tr[0] + tr[1]) / tr[2], n, te[0] / te[2], te[1] / te[2], mean, se))
    for tr, n, c, p, t, se in sorted(rows, reverse=True):
        print(f"{n:36s}{tr:+12.3f} | {c:+11.3f}{p:+10.3f}{t:+9.3f}{se:7.3f}")
