"""Wider search of step-0 quotes with a train/test split, plus the later-step cost of revealing.

    python3 analysis/step0_search.py data/new.jsonl.gz
Train on even-numbered matches, report on odd-numbered ones (the grid is searched on the first half only).
Then compare steps 1-3 edge of seat-rounds that quoted informed at step 0 vs the default 40/44.
"""
import gzip
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import dicemarket as dm  # noqa: E402

rng = random.Random(0)


def policy(k, w, shift=0.0):
    def f(own):
        c = 42.0 + k * (own - 7.0) + shift
        bid = round((c - w / 2) * 2) / 2
        bid = min(max(bid, 12.0), 72.0 - w)
        return (bid, bid + w)
    return f


CANDS = {"default": lambda own: (40.0, 44.0)}
for k in (1.0, 1.5, 2.0, 2.5, 3.0, 4.0):
    for w in (2.0, 3.0, 4.0):
        CANDS[f"k={k:g} w={w:g}"] = policy(k, w)
acc = {part: defaultdict(lambda: [0.0, 0.0, 0]) for part in ("train", "test")}   # name -> sum edge, sum edge^2, n (per match)
later = {"informed": [0.0, 0], "default": [0.0, 0]}
with gzip.open(sys.argv[1], "rt") as f:
    for idx, line in enumerate(f):
        m = json.loads(line)
        part = "train" if idx % 2 == 0 else "test"
        edges = defaultdict(float)
        n = 0
        for r in m["rounds"][::2]:
            d, v = r["d"], r["v"]
            rec = {i: (b / 2.0, a / 2.0) for i, (b, a) in enumerate(r["st"][0]["s"])}
            # later-step edge by step-0 style (real trades, all candidate seats)
            le = defaultdict(float)
            for st in r["st"][1:]:
                for b, se, p, k in st["t"]:
                    le[b] += v - p / 4.0
                    le[se] -= v - p / 4.0
            for me in range(5):
                if m["kinds"][me] == "house":
                    continue
                key = "default" if rec[me] == (40.0, 44.0) else "informed"
                later[key][0] += le[me]; later[key][1] += 1
                own = d[me] + d[me + 6]
                n += 1
                for name, pol in CANDS.items():
                    orders = dict(rec)
                    orders[me] = pol(own)
                    book = {i: {"bid": None, "ask": None} for i in range(5)}
                    for t in dm.match_step(book, orders, rng, {i: 0 for i in range(5)}, 0):
                        if t["buyer"] == me:
                            edges[name] += v - t["price"]
                        if t["seller"] == me:
                            edges[name] += t["price"] - v
        if n:
            for name in CANDS:
                a = acc[part][name]
                a[0] += edges[name] / n; a[1] += (edges[name] / n) ** 2; a[2] += 1
rows = []
for name in CANDS:
    tr, te = acc["train"][name], acc["test"][name]
    se = ((te[1] / te[2] - (te[0] / te[2]) ** 2) / te[2]) ** 0.5
    rows.append((tr[0] / tr[2], name, te[0] / te[2], se))
print(f"{'step-0 quote':20s}{'train edge':>12s}{'TEST edge':>12s}{'+-se':>8s}")
for tr, name, te, se in sorted(rows, reverse=True):
    print(f"{name:20s}{tr:+12.3f}{te:+12.3f}{se:8.3f}")
print("\nsteps 1-3 edge per round, by how the seat quoted at step 0 (all candidate seats, real trades):")
for k, (e, n) in later.items():
    print(f"  {k:9s} {e / n:+.3f}  ({n} seat-rounds)")
