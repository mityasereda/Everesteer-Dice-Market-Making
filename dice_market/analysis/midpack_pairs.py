"""Who earns from whom: edge per round of each agent group, split by the type of the counterparty.

    python3 analysis/midpack_pairs.py data/day28.jsonl.gz data/midpack.json data/midpack_pairs.json

Types (from midpack.json): HT = the two top-14 hiders, H = other hiders, IA = informed top 14, IB = informed 15-40,
IC = informed 41+, ME = my label.  A trade's edge is value - price for the buyer and price - value for the seller.
Also splits the groups' steps-1..3 edge by how many top hiders (HT) sit at the table.
"""
import gzip
import json
import sys
from collections import defaultdict

prof = json.load(open(sys.argv[2]))


def typ(label):
    v = prof.get(label)
    if label == "fanaberysty":
        return "ME"
    if not v or not v["rank"]:
        return "other"
    hid = v["default0"] > 0.8
    if hid:
        return "HT" if v["rank"] <= 14 else "H"
    return "IA" if v["rank"] <= 14 else "IB" if v["rank"] <= 40 else "IC"


T = {l: typ(l) for l in prof}
EDGE = defaultdict(float)     # (actor type, counterparty type, step) -> sum edge
EXP = defaultdict(float)      # actor type -> seat-rounds
TAB = defaultdict(lambda: [0.0, 0.0])   # (actor type, n top hiders at table) -> [edge steps 1-3, seat-rounds]
n = 0
with gzip.open(sys.argv[1], "rt") as f:
    for line in f:
        m = json.loads(line)
        n += 1
        ty = [T.get(l, "other") for l in m["labels"]]
        R = len(m["rounds"])
        nht = sum(1 for t in ty if t == "HT")
        for t in ty:
            EXP[t] += R
        for i, t in enumerate(ty):
            TAB[(t, nht - (1 if t == "HT" else 0))][1] += R
        for r in m["rounds"]:
            v = r["v"]
            for s, st in enumerate(r["st"]):
                for bu, se, p, k in st["t"]:
                    e = v - p / 4.0
                    EDGE[(ty[bu], ty[se], s)] += e
                    EDGE[(ty[se], ty[bu], s)] -= e
                    if s >= 1:
                        TAB[(ty[bu], nht - (1 if ty[bu] == "HT" else 0))][0] += e
                        TAB[(ty[se], nht - (1 if ty[se] == "HT" else 0))][0] -= e
out = {"edge": {f"{a}|{b}|{s}": x / EXP[a] for (a, b, s), x in EDGE.items()},
       "tab": {f"{a}|{k}": (x[0] / x[1] if x[1] else None, x[1]) for (a, k), x in TAB.items()}, "exp": EXP}
json.dump(out, open(sys.argv[3], "w"), indent=1)
print("done", n, "matches")
actors = ["ME", "IB", "IA", "HT", "H", "IC"]
cps = ["ME", "HT", "H", "IA", "IB", "IC", "other"]
print("edge per round (all four steps) of the ROW group, earned from the COLUMN group")
print(f"{'':6s}" + "".join(f"{c:>8s}" for c in cps) + "     total")
for a in actors:
    row = [sum(EDGE.get((a, c, s), 0.0) for s in range(4)) / EXP[a] for c in cps]
    print(f"{a:6s}" + "".join(f"{x:+8.2f}" for x in row) + f"  {sum(row):+8.2f}")
print("\nsteps 1-3 edge per round by number of top hiders (HT) at the table (others than the agent itself)")
for a in actors:
    print(f"{a:6s}" + "".join(f"   {k} HT: " + (f"{TAB[(a, k)][0] / TAB[(a, k)][1]:+.2f} (n={TAB[(a, k)][1] / 200:.0f} matches)" if TAB[(a, k)][1] > 2000 else "   n/a") for k in range(3)))
