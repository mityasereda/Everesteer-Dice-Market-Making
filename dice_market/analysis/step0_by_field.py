"""Step-0 quote edge, split by how many of the four rivals quoted the default 40/44 (the hiders).

    python3 analysis/step0_by_field.py data/new.jsonl.gz [more.jsonl.gz ...]
The share of hiders keeps falling, so the quote has to stay good when almost everyone is informed.
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


def policy(k, w):
    def f(own):
        bid = min(max(round((42.0 + k * (own - 7.0) - w / 2) * 2) / 2, 12.0), 72.0 - w)
        return (bid, bid + w)
    return f


CANDS = {"default 40/44": lambda own: (40.0, 44.0)}
for k, w in ((1.0, 4.0), (1.25, 4.0), (1.5, 4.0), (1.75, 4.0), (2.0, 4.0), (1.5, 3.0), (1.5, 3.5)):
    CANDS[f"k={k:g} w={w:g}"] = policy(k, w)
acc = defaultdict(lambda: defaultdict(float)); cnt = defaultdict(int)
for path in sys.argv[1:]:
    with gzip.open(path, "rt") as f:
        for line in f:
            m = json.loads(line)
            for r in m["rounds"][::2]:
                d, v = r["d"], r["v"]
                rec = {i: (b / 2.0, a / 2.0) for i, (b, a) in enumerate(r["st"][0]["s"])}
                for me in range(5):
                    if m["kinds"][me] == "house":
                        continue
                    hiders = sum(1 for j in range(5) if j != me and rec[j] == (40.0, 44.0))
                    g = "0-1 hiders" if hiders <= 1 else "2 hiders" if hiders == 2 else "3-4 hiders"
                    cnt[g] += 1
                    own = d[me] + d[me + 6]
                    for name, pol in CANDS.items():
                        orders = dict(rec); orders[me] = pol(own)
                        book = {i: {"bid": None, "ask": None} for i in range(5)}
                        for t in dm.match_step(book, orders, rng, {i: 0 for i in range(5)}, 0):
                            if t["buyer"] == me:
                                acc[g][name] += v - t["price"]
                            if t["seller"] == me:
                                acc[g][name] += t["price"] - v
groups = ["0-1 hiders", "2 hiders", "3-4 hiders"]
print(f"{'step-0 quote':16s}" + "".join(f"{g + ' (n=' + str(cnt[g]) + ')':>26s}" for g in groups))
for name in CANDS:
    print(f"{name:16s}" + "".join(f"{acc[g][name] / cnt[g]:+26.3f}" for g in groups))
