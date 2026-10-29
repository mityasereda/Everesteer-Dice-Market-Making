"""Where v8 wins and loses on the real board.

    python3 analysis/v8_matches.py data/mine_all2.jsonl.gz [from=2026-10-28T11:27:44] [to=...]
1. 25-match blocks over time: rank, win rate, step-0 and steps 1-3 edge, share of rivals that quote informed at step 0.
2. Step-0 edge by how many rivals at the table quote the default 40/44 (the hiders pay, the informed do not).
3. Per step: my quote's RMSE against the true value vs the other candidates' quotes; width (printed doubled: price units x2); OLS of my mid on own dice,
   my position and the others' previous mid (columns in that order) (compare with ranks 15-40: step 1 1.2 / 1.3, step 2 0.7 / 0.85 and skew -1, step 3 0.2 / 0.95).
4. Per step and role: my edge and fills per round against the average candidate.
"""
import gzip
import json
import sys
from collections import defaultdict

import numpy as np

ME = "fanaberysty"
FROM = sys.argv[2] if len(sys.argv) > 2 else "2026-10-28T11:27:44"
TO = sys.argv[3] if len(sys.argv) > 3 else "9999"
ms = []
with gzip.open(sys.argv[1], "rt") as f:
    for line in f:
        m = json.loads(line)
        if FROM <= m["at"][:19] < TO and ME in m["labels"]:
            ms.append(m)
ms.sort(key=lambda m: m["at"])
print(len(ms), "matches", ms[0]["at"][:19], "to", ms[-1]["at"][:19])

BL = defaultdict(lambda: np.zeros(12))     # block -> n matches, rank, wins, games, pnl, step edges 0..3, informed rival share, rounds
S0 = defaultdict(lambda: np.zeros(3))      # number of default-quoting rivals -> my step-0 edge, my step-0 fills, rounds
X = [[] for _ in range(4)]                 # per step: my regression rows
Y = [[] for _ in range(4)]
SE = np.zeros((4, 2)); CNT = np.zeros(4); WID = np.zeros((4, 2))
ROLE = defaultdict(lambda: np.zeros(4))    # (step, role) -> my edge, my fills, field edge, field fills
NFLD = 0; NME = 0
for k, m in enumerate(ms):
    me = m["labels"].index(ME)
    pn = [s["pnl_per_round"] for s in m["seats"]]
    b = BL[k // 25]
    b[0] += 1; b[1] += 1 + sum(1 for j in range(5) if pn[j] > pn[me])
    for j in range(5):
        if j != me:
            b[2] += pn[me] > pn[j] if pn[me] != pn[j] else 0.5; b[3] += 1
    b[4] += pn[me]
    cands = [j for j in range(5) if j != me and m["kinds"][j] == "candidate"]
    pos = np.zeros(5)
    for r in m["rounds"]:
        v, d = r["v"], r["d"]
        own = d[me] + d[me + 6]
        nd = sum(1 for j in range(5) if j != me and tuple(r["st"][0]["s"][j]) == (80, 88))
        b[10] += sum(1 for j in range(5) if j != me and tuple(r["st"][0]["s"][j]) != (80, 88)) / 4.0
        b[11] += 1
        pos = np.zeros(5)
        for s, st in enumerate(r["st"]):
            q = st["s"]
            mids = [(a + c) / 4.0 for a, c in q]
            SE[s, 0] += (mids[me] - v) ** 2
            SE[s, 1] += np.mean([(mids[j] - v) ** 2 for j in cands]) if cands else 0
            CNT[s] += 1
            WID[s, 0] += (q[me][1] - q[me][0]) / 2.0
            WID[s, 1] += np.mean([(q[j][1] - q[j][0]) / 2.0 for j in cands]) if cands else 0
            om = (sum(((r["st"][s - 1]["s"][j][0] + r["st"][s - 1]["s"][j][1]) / 4.0) for j in range(5) if j != me) / 4.0 - 42) if s else 0.0
            X[s].append([1.0, own - 7, pos[me], om]); Y[s].append(mids[me] - 42)
            prev = r["st"][s - 1]["s"] if s else None
            for bu, se, p, kind in st["t"]:
                e = v - p / 4.0
                if kind == "c":
                    rb = rs = 2
                else:
                    sr = prev[se][1] * 2 == p
                    br = prev[bu][0] * 2 == p
                    if sr and not br:
                        rb, rs = 0, 1
                    elif br and not sr:
                        rb, rs = 1, 0
                    else:
                        rb = rs = None
                for who, role, sign in ((bu, rb, 1), (se, rs, -1)):
                    if role is None:
                        continue
                    if who == me:
                        ROLE[(s, role)][0] += sign * e; ROLE[(s, role)][1] += 1
                        b[5 + s] += sign * e
                        if s == 0:
                            S0[nd][0] += sign * e; S0[nd][1] += 1
                    elif who in cands:
                        ROLE[(s, role)][2] += sign * e; ROLE[(s, role)][3] += 1
                pos[bu] += 1; pos[se] -= 1
            if s == 0:
                S0[nd][2] += 1
        NME += 1; NFLD += len(cands)
print("\n1. blocks of 25 matches")
print(f"{'block':>8s}{'rank':>7s}{'win':>7s}{'P&L':>7s}{'step0':>7s}{'s1':>6s}{'s2':>6s}{'s3':>6s}{'s1-3':>7s}{'informed rivals':>17s}")
for i, b in sorted(BL.items()):
    rounds = b[0] * 200
    print(f"{i * 25 + 1:>4d}-{i * 25 + int(b[0]):<3d}{b[1] / b[0]:7.2f}{b[2] / b[3]:7.3f}{b[4] / b[0]:+7.2f}{b[5] / rounds:+7.2f}{b[6] / rounds:+6.2f}{b[7] / rounds:+6.2f}{b[8] / rounds:+6.2f}{(b[6] + b[7] + b[8]) / rounds:+7.2f}{b[10] / b[11]:17.2f}")
print("\n2. my step-0 edge by number of rivals at the default 40/44")
for nd, a in sorted(S0.items()):
    print(f"   {nd} default rivals: {a[2]:7.0f} rounds ({a[2] / sum(v[2] for v in S0.values()):.0%})   edge {a[0] / a[2]:+.2f} per round   fills {a[1] / a[2]:.2f}")
print("\n3. my quote against the other candidates, by step")
print(f"{'':8s}{'my RMSE':>9s}{'rivals':>8s}{'my width':>10s}{'rivals':>8s}   OLS of my mid: own dice, position, others' previous mid")
for s in range(4):
    x, y = np.array(X[s]), np.array(Y[s])
    if s == 0:
        x = x[:, :2]
    beta = np.linalg.lstsq(x, y, rcond=None)[0]
    print(f"step {s:<3d}{(SE[s, 0] / CNT[s]) ** .5:9.2f}{(SE[s, 1] / CNT[s]) ** .5:8.2f}{WID[s, 0] / CNT[s] * 2:10.2f}{WID[s, 1] / CNT[s] * 2:8.2f}   "
          + "  ".join(f"{c:+.2f}" for c in beta[1:]))
print("\n4. edge per round (me | average candidate) and my fills per round, by step and role")
for s in range(4):
    print(f"step {s}: " + "   ".join(f"{role} {ROLE[(s, ri)][0] / NME:+.2f} | {ROLE[(s, ri)][2] / NFLD:+.2f} ({ROLE[(s, ri)][1] / NME:.2f} fills)" for ri, role in enumerate(("taker", "resting", "cross"))))
