"""Profile every agent in the replay dataset (compact format, streamed).

    python3 analysis/profile.py data/replays.jsonl.gz out.json [sample_every=8]

Match level (all matches): pairwise win rate and mean P&L per round of each agent label.
Quote level (every `sample_every`-th match), per label and step 0..3:
  n, mean spread, slope of the mid on the agent's own dice sum (1 = shows its dice, 0 = hides them),
  RMSE of the mid as an estimate of the final value, and the trade edge (value - price for buys, price - value
  for sells) split by trade kind.
"""
import gzip
import json
import sys
from collections import defaultdict

path, out = sys.argv[1], sys.argv[2]
EVERY = int(sys.argv[3]) if len(sys.argv) > 3 else 8

match = defaultdict(lambda: {"n": 0, "pnl": 0.0, "wins": 0.0, "games": 0, "kind": ""})
Q = defaultdict(lambda: [[0.0] * 12 for _ in range(4)])  # per label, step: n, spread, sx, sy, sxx, sxy, syy, sv, svv, ...
E = defaultdict(lambda: [[0.0] * 6 for _ in range(4)])   # per label, step: take n, take edge, cross n, cross edge, ...
DEFAULT0 = defaultdict(float)
n_rows = 0
with gzip.open(path, "rt") as f:
    for line in f:
        m = json.loads(line)
        n_rows += 1
        labels, kinds, seats = m["labels"], m["kinds"], m["seats"]
        pnl = [s["pnl_per_round"] for s in seats]
        for i, l in enumerate(labels):
            d = match[l]
            d["n"] += 1
            d["pnl"] += pnl[i]
            d["kind"] = kinds[i]
            for j in range(len(labels)):
                if j != i:
                    d["games"] += 1
                    d["wins"] += 1.0 if pnl[i] > pnl[j] else 0.5 if pnl[i] == pnl[j] else 0.0
        if n_rows % EVERY:
            continue
        for r in m["rounds"]:
            v, dice = r["v"], r["d"]
            for s, st in enumerate(r["st"]):
                for i, (b, a) in enumerate(st["s"]):
                    mid = (b + a) / 4.0
                    own = dice[i] + dice[i + 6]
                    x, y, vv = own - 7.0, mid - 42.0, v - 42.0
                    q = Q[labels[i]][s]
                    q[0] += 1
                    q[1] += (a - b) / 2.0
                    q[2] += x
                    q[3] += y
                    q[4] += x * x
                    q[5] += x * y
                    q[6] += y * y
                    q[7] += (mid - v) ** 2
                    q[8] += vv * y
                    q[9] += vv * vv
                    if s == 0 and b == 80 and a == 88:
                        DEFAULT0[labels[i]] += 1
                for bu, se, p, k in st["t"]:
                    price = p / 4.0
                    c = 0 if k == "t" else 2
                    e = E[labels[bu]][s]
                    e[c] += 1
                    e[c + 1] += v - price
                    e = E[labels[se]][s]
                    e[c] += 1
                    e[c + 1] += price - v
json.dump({"rows": n_rows, "every": EVERY, "match": match, "Q": Q, "E": E, "default0": DEFAULT0}, open(out, "w"))
print("done", n_rows, "matches")
