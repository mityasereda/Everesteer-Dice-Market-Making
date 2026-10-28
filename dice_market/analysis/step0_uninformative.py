"""Step-0 quotes that reveal nothing about my dice (their position does not depend on them): which loses least?

    python3 analysis/step0_uninformative.py data/late.jsonl.gz [max_matches=1500]
Step 0 replay against the rivals' recorded step-0 quotes is exact (nobody can react yet).
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
MAXM = int(sys.argv[2]) if len(sys.argv) > 2 else 1500
CANDS = {f"({b:g}, {b + w:g})": (b, b + w) for b, w in
         [(40, 4), (39, 4), (41, 4), (38, 4), (42, 4), (40.5, 3), (41, 2), (41.5, 1), (41.75, 0.5), (40.5, 4), (39.5, 4)]}
acc = defaultdict(float); n = 0
with gzip.open(sys.argv[1], "rt") as f:
    for line in f:
        m = json.loads(line)
        if m["at"] < "2026-10-27T22:30:00":
            continue
        for r in m["rounds"][::4]:
            v = r["v"]
            rec = {i: (b / 2.0, a / 2.0) for i, (b, a) in enumerate(r["st"][0]["s"])}
            for me in range(5):
                if m["kinds"][me] == "house":
                    continue
                n += 1
                for name, q in CANDS.items():
                    orders = dict(rec); orders[me] = q
                    book = {i: {"bid": None, "ask": None} for i in range(5)}
                    for t in dm.match_step(book, orders, rng, {i: 0 for i in range(5)}, 0):
                        if t["buyer"] == me:
                            acc[name] += v - t["price"]
                        if t["seller"] == me:
                            acc[name] += t["price"] - v
        MAXM -= 1
        if MAXM <= 0:
            break
print(f"{n} seat-rounds replayed (matches after 22:30 UTC on 27 Oct)")
for name, e in sorted(acc.items(), key=lambda kv: -kv[1]):
    print(f"  quote {name:14s} step-0 edge per round {e / n:+.3f}")
