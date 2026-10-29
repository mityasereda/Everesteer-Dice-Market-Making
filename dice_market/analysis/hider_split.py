"""Hiding against pushing at the same table: my edge per round by the number of hiders among my four rivals.

    python3 analysis/hider_split.py data/mine_all2.jsonl.gz
A rival is a hider in a match if it quoted the default 40/44 in more than 60% of that match's step-0 quotes (an informed agent hits 40/44
about 1 round in 6 by chance). Versions by upload time: v5 (before 22:33 on 27 Oct, v6 window excluded), v7 (08:13 to 11:27 on 28 Oct), v8 (after).
Different versions met different fields, so this only partly controls for the field.
"""
import gzip
import json
import sys
from collections import defaultdict

import numpy as np

ME = "fanaberysty"
V6_FROM, V6_TO = "2026-10-27T22:33:26", "2026-10-27T22:50:30"
V7_FROM, V8_FROM = "2026-10-28T08:13:48", "2026-10-28T11:27:44"
G = defaultdict(lambda: defaultdict(lambda: np.zeros(8)))     # version -> n hiders -> matches, wins, games, step edges 0..3, rank sum
with gzip.open(sys.argv[1], "rt") as f:
    for line in f:
        m = json.loads(line)
        if ME not in m["labels"]:
            continue
        at = m["at"][:19]
        ver = "v5" if at < V6_FROM else "v6" if at < V6_TO else "v7" if V7_FROM <= at < V8_FROM else "v8" if at >= V8_FROM else None
        if ver is None or ver == "v6":
            continue
        me = m["labels"].index(ME)
        pn = [s["pnl_per_round"] for s in m["seats"]]
        R = m["rounds"]
        dflt = np.zeros(5)
        for r in R:
            for j in range(5):
                dflt[j] += tuple(r["st"][0]["s"][j]) == (80, 88)
        nh = sum(1 for j in range(5) if j != me and dflt[j] / len(R) > 0.6)
        e = np.zeros(4)
        for r in R:
            v = r["v"]
            for s, st in enumerate(r["st"]):
                for bu, se, p, k in st["t"]:
                    if bu == me:
                        e[s] += v - p / 4.0
                    if se == me:
                        e[s] += p / 4.0 - v
        a = G[ver][nh]
        a[0] += 1
        for j in range(5):
            if j != me:
                a[1] += (pn[me] > pn[j]) + 0.5 * (pn[me] == pn[j]); a[2] += 1
        a[3:7] += e / len(R)
        a[7] += 1 + sum(1 for j in range(5) if pn[j] > pn[me])
print(f"{'version':8s}{'hiders':>7s}{'matches':>9s}{'win':>7s}{'rank':>6s}{'step0':>7s}{'s1-3':>7s}{'total':>7s}")
for ver in ("v5", "v7", "v8"):
    for nh in range(5):
        a = G[ver].get(nh)
        if a is None or a[0] < 3:
            continue
        print(f"{ver:8s}{nh:7d}{a[0]:9.0f}{a[1] / a[2]:7.3f}{a[7] / a[0]:6.2f}{a[3] / a[0]:+7.2f}{a[4:7].sum() / a[0]:+7.2f}{a[3:7].sum() / a[0]:+7.2f}")
    tot = sum(G[ver].values())
    print(f"{ver:8s}{'all':>7s}{tot[0]:9.0f}{tot[1] / tot[2]:7.3f}{tot[7] / tot[0]:6.2f}{tot[3] / tot[0]:+7.2f}{tot[4:7].sum() / tot[0]:+7.2f}{tot[3:7].sum() / tot[0]:+7.2f}\n")
