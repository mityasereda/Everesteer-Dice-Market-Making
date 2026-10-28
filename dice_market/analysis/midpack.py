"""Profile every agent in a recent dataset, to compare leaderboard tiers (the top 14 against ranks 15-40, the ones to beat).

    python3 analysis/midpack.py data/day28.jsonl.gz data/lb_now.json data/midpack.json

Per label (all matches in the file), per step 0..3:
  n rows, mean width, share of the default 40/44 quote at step 0,
  an OLS fit of the quote mid (minus 42) on [1, own dice - 7, position before the step, mean of the other seats' mids at the previous step - 42],
  RMSE of the mid as an estimate of the final value,
  trade edge per round by role (taker / resting / cross) and fills per round.
Match level: pairwise win rate and P&L per round.  Step-0 default share and width are also split by time (first / second half of the file).
"""
import gzip
import json
import sys
from collections import defaultdict

import numpy as np

path, lb_path, out = sys.argv[1], sys.argv[2], sys.argv[3]
lb = {r["label"]: r for r in json.load(open(lb_path))}

NF = 4                                           # features: const, own, position, others' previous mid
XTX = defaultdict(lambda: np.zeros((4, NF, NF)))
XTY = defaultdict(lambda: np.zeros((4, NF)))
YTY = defaultdict(lambda: np.zeros(4))
CNT = defaultdict(lambda: np.zeros(4))
WID = defaultdict(lambda: np.zeros(4))
SE = defaultdict(lambda: np.zeros(4))            # squared error of the mid vs the final value
DEF0 = defaultdict(float)
HALF = defaultdict(lambda: np.zeros((2, 3)))     # per half of the file: n, default at step 0, width at step 0
EDGE = defaultdict(lambda: np.zeros((4, 3)))     # step, role -> sum of edge
FILL = defaultdict(lambda: np.zeros((4, 3)))
ROUNDS = defaultdict(int)
MATCH = defaultdict(lambda: {"n": 0, "pnl": 0.0, "wins": 0.0, "games": 0})

t0 = None
rows = []
with gzip.open(path, "rt") as f:
    for line in f:
        rows.append(line)
n_total = len(rows)

for k, line in enumerate(rows):
    m = json.loads(line)
    labels = m["labels"]
    half = 0 if k < n_total / 2 else 1
    pnl = [s["pnl_per_round"] for s in m["seats"]]
    for i, l in enumerate(labels):
        d = MATCH[l]
        d["n"] += 1
        d["pnl"] += pnl[i]
        for j in range(5):
            if j != i:
                d["games"] += 1
                d["wins"] += 1.0 if pnl[i] > pnl[j] else 0.5 if pnl[i] == pnl[j] else 0.0
    rd = m["rounds"]
    R = len(rd)
    B = np.array([[st["s"] for st in r["st"]] for r in rd], dtype=float)     # R,4,5,2
    V = np.array([r["v"] for r in rd], dtype=float)
    D = np.array([r["d"] for r in rd], dtype=float)
    MID = (B[..., 0] + B[..., 1]) / 4.0                                       # R,4,5
    WIDTH = (B[..., 1] - B[..., 0]) / 2.0
    NET = np.zeros((R, 4, 5))
    for ri, r in enumerate(rd):
        for s, st in enumerate(r["st"]):
            prev = r["st"][s - 1]["s"] if s else None
            for bu, se, p, kind in st["t"]:
                NET[ri, s, bu] += 1
                NET[ri, s, se] -= 1
                if kind == "c":
                    roles = (2, 2)
                else:
                    sell_rest = prev[se][1] * 2 == p
                    buy_rest = prev[bu][0] * 2 == p
                    if sell_rest and not buy_rest:
                        roles = (0, 1)
                    elif buy_rest and not sell_rest:
                        roles = (1, 0)
                    else:
                        continue
                e = V[ri] - p / 4.0
                lb_, ls_ = labels[bu], labels[se]
                EDGE[lb_][s, roles[0]] += e
                FILL[lb_][s, roles[0]] += 1
                EDGE[ls_][s, roles[1]] -= e
                FILL[ls_][s, roles[1]] += 1
    POS = np.cumsum(NET, axis=1) - NET                                        # position before the step
    for i, l in enumerate(labels):
        ROUNDS[l] += R
        own = D[:, i] + D[:, i + 6]
        mid, wid = MID[:, :, i], WIDTH[:, :, i]
        default0 = (B[:, 0, i, 0] == 80) & (B[:, 0, i, 1] == 88)
        DEF0[l] += default0.sum()
        h = HALF[l]
        h[half] += (R, default0.sum(), wid[:, 0].sum())
        for s in range(4):
            om = (MID[:, s - 1, :].sum(axis=1) - MID[:, s - 1, i]) / 4.0 - 42 if s else np.zeros(R)
            X = np.stack([np.ones(R), own - 7, POS[:, s, i], om], axis=1)
            y = mid[:, s] - 42
            XTX[l][s] += X.T @ X
            XTY[l][s] += X.T @ y
            YTY[l][s] += y @ y
            CNT[l][s] += R
            WID[l][s] += wid[:, s].sum()
            SE[l][s] += ((mid[:, s] - V) ** 2).sum()

res = {}
for l in MATCH:
    n = ROUNDS[l]
    d = MATCH[l]
    r = {"rank": lb.get(l, {}).get("rank"), "elo": lb.get(l, {}).get("elo"), "agent": lb.get(l, {}).get("agent"),
         "matches": d["n"], "rounds": n, "pnl": d["pnl"] / d["n"], "win": d["wins"] / d["games"],
         "default0": DEF0[l] / n, "steps": []}
    hh = HALF[l]
    r["half"] = [{"default0": (hh[h, 1] / hh[h, 0]) if hh[h, 0] else None, "width0": (hh[h, 2] / hh[h, 0]) if hh[h, 0] else None,
                  "rounds": hh[h, 0]} for h in range(2)]
    for s in range(4):
        cnt = CNT[l][s]
        # drop unused features (position at step 0, others' mid at step 0) by a tiny ridge
        A = XTX[l][s] + 1e-6 * np.eye(NF) * max(cnt, 1.0)
        if s == 0:
            A[2, 2] += 1e9
            A[3, 3] += 1e9
        beta = np.linalg.solve(A, XTY[l][s])
        resid = YTY[l][s] - 2 * beta @ XTY[l][s] + beta @ XTX[l][s] @ beta
        r["steps"].append({
            "width": WID[l][s] / cnt, "rmse": (SE[l][s] / cnt) ** 0.5,
            "const": beta[0], "b_own": beta[1], "b_pos": beta[2], "b_oth": beta[3], "fit_sd": (max(resid, 0) / cnt) ** 0.5,
            "edge": (EDGE[l][s] / n).tolist(), "fills": (FILL[l][s] / n).tolist()})
    res[l] = r
json.dump(res, open(out, "w"), indent=1, default=float)
print("done", n_total, "matches,", len(res), "labels")
