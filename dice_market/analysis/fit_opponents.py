"""Fit imitation models of real agents from the replay dataset.

    python3 analysis/fit_opponents.py data/replays.jsonl.gz opponents/fits.json [every=3] [label1,label2,...]

For each chosen agent label and step s, regress the agent's quote mid (minus 42) on what it could see:
  1, its dice sum - 7, the others' summed mids at each earlier step, its own previous mid, its position.
Also keeps the residual sd, the empirical distribution of quote widths, and at step 0 the share of default 40 / 44 quotes.
The imitation agents (opponents/imit_*.py, made by analysis/make_opponents.py) replay these fits. They copy each
agent's AVERAGE behaviour; the real agents may adapt during a match in ways these static copies do not.
"""
import gzip
import json
import sys
from collections import Counter, defaultdict

import numpy as np

path, out = sys.argv[1], sys.argv[2]
EVERY = int(sys.argv[3]) if len(sys.argv) > 3 else 3
LABELS = sys.argv[4].split(",") if len(sys.argv) > 4 else ["Docsdadi", "zudvarnoki", "MPP", "billhwang", "momarci13", "yurko_mud", "Joshua_Karat", "leobgg", "house pro"]
RIDGE = 1e-3


def dim(s):
    return 2 + s + (1 if s >= 1 else 0) + (1 if s >= 1 else 0)   # 1, own, S_0..S_{s-1}, prev mid, pos


acc = {(l, s): [np.zeros((dim(s), dim(s))), np.zeros(dim(s)), 0.0, 0] for l in LABELS for s in range(4)}
widths = {(l, s): Counter() for l in LABELS for s in range(4)}
p0 = {l: [0, 0] for l in LABELS}   # default 40/44 count, total, at step 0
n_rows = 0
with gzip.open(path, "rt") as f:
    for line in f:
        n_rows += 1
        if n_rows % EVERY:
            continue
        m = json.loads(line)
        mine = [(i, l) for i, l in enumerate(m["labels"]) if l in LABELS]
        if not mine:
            continue
        R = m["rounds"]
        S = np.array([[st["s"] for st in r["st"]] for r in R], float)       # (R,4,5,2) ticks
        M = (S[..., 0] + S[..., 1]) / 4.0 - 42.0                             # (R,4,5)
        W = (S[..., 1] - S[..., 0]) / 2.0                                    # (R,4,5) full width in price
        D = np.array([r["d"] for r in R], float)
        pos = np.zeros((len(R), 4, 5))
        for ri, r in enumerate(R):
            p = [0] * 5
            for s, st in enumerate(r["st"]):
                pos[ri, s] = p
                for b, se, _, _ in st["t"]:
                    p[b] += 1
                    p[se] -= 1
        for i, l in mine:
            own = D[:, i] + D[:, i + 6] - 7.0
            for s in range(4):
                default = (S[:, s, i, 0] == 80) & (S[:, s, i, 1] == 88)
                if s == 0:
                    p0[l][0] += int(default.sum())
                    p0[l][1] += len(R)
                keep = ~default if s == 0 else np.ones(len(R), bool)
                cols = [np.ones(len(R)), own]
                for t in range(s):
                    cols.append(M[:, t, :].sum(1) - M[:, t, i])
                if s >= 1:
                    cols.append(M[:, s - 1, i])
                    cols.append(pos[:, s, i])
                X, y = np.stack(cols, 1)[keep], M[keep, s, i]
                a = acc[(l, s)]
                a[0] += X.T @ X
                a[1] += X.T @ y
                a[2] += float(y @ y)
                a[3] += len(y)
                widths[(l, s)].update(W[keep, s, i].tolist())

fits = {}
for l in LABELS:
    steps = []
    for s in range(4):
        A, b, yy, n = acc[(l, s)]
        w = np.linalg.solve(A + RIDGE * np.eye(len(b)), b)
        tot = sum(widths[(l, s)].values())
        if n == 0:   # an agent that always sends the default quote at this step: nothing to fit
            steps.append({"w": [0.0] * len(b), "sd": 0.0, "n": 0, "widths": {"4.0": 1.0}})
            continue
        sse = yy - 2 * w @ b + w @ A @ w
        steps.append({"w": [round(float(v), 5) for v in w], "sd": round(float(max(sse, 0) / n) ** 0.5, 4), "n": n,
                      "widths": {str(k): round(v / tot, 5) for k, v in sorted(widths[(l, s)].items())}})
    fits[l] = {"p0": round(p0[l][0] / p0[l][1], 4), "steps": steps}
json.dump(fits, open(out, "w"), indent=1)
for l in LABELS:
    print(f"{l:14s} default-step0 {fits[l]['p0']:.2f}  resid sd by step {[fits[l]['steps'][s]['sd'] for s in range(4)]}  n {[fits[l]['steps'][s]['n'] for s in range(4)][0]}")
