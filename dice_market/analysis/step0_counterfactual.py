"""What would my step-0 quote have earned against the rivals' RECORDED step-0 quotes?

    python3 analysis/step0_counterfactual.py data/new.jsonl.gz [max_matches=400]

At step 0 nobody has seen anything, so rivals' quotes cannot depend on mine (they can only depend on past rounds).
For every seat in every sampled match, I take that seat's dice as "mine", keep the other four recorded step-0 quotes,
swap in an alternative quote and run the engine's own matching (dicemarket.match_step). Edge = sum over my step-0
fills of (value - price) for buys and (price - value) for sells. This is step 0 only: it ignores what a revealed
quote tells rivals at later steps, and rivals who adapted to one particular seat's habit.
"""
import gzip
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import dicemarket as dm  # noqa: E402

path = sys.argv[1]
MAXM = int(sys.argv[2]) if len(sys.argv) > 2 else 400
rng = random.Random(0)


def policy(k, w, shift=0.0):
    """centre = 42 + k * (own dice - 7) + shift ; total width w."""
    def f(own):
        c = 42.0 + k * (own - 7.0) + shift
        bid = round((c - w / 2) * 2) / 2
        bid = min(max(bid, 12.0), 72.0 - w)
        return (bid, bid + w)
    return f


CANDS = {"default 40/44 (current)": lambda own: (40.0, 44.0)}
for k in (0.5, 1.0, 1.5):
    for w in (2.0, 3.0, 4.0):
        CANDS[f"centre 42+{k}*(dice-7), width {w:g}"] = policy(k, w)
tot = defaultdict(float); n = 0
with gzip.open(path, "rt") as f:
    for line in f:
        m = json.loads(line)
        if "candidate" not in m["kinds"] and "you" not in m["kinds"]:
            continue
        for r in m["rounds"][::2]:
            d, v = r["d"], r["v"]
            rec = {i: (b / 2.0, a / 2.0) for i, (b, a) in enumerate(r["st"][0]["s"])}
            for me in range(5):
                if m["kinds"][me] == "house":
                    continue
                own = d[me] + d[me + 6]
                n += 1
                for name, pol in CANDS.items():
                    orders = dict(rec)
                    orders[me] = pol(own)
                    book = {i: {"bid": None, "ask": None} for i in range(5)}
                    trades = dm.match_step(book, orders, rng, {i: 0 for i in range(5)}, 0)
                    for t in trades:
                        if t["buyer"] == me:
                            tot[name] += v - t["price"]
                        if t["seller"] == me:
                            tot[name] += t["price"] - v
        MAXM -= 1
        if MAXM <= 0:
            break
print(f"{n} seat-rounds replayed")
for name, e in sorted(tot.items(), key=lambda kv: -kv[1]):
    print(f"  {name:38s} step-0 edge per round {e / n:+.3f}")
