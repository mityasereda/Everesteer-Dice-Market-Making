"""Fit the agent's starting weights (PRIOR) on recent real matches.

    python3 analysis/fit_prior.py data/recent_tight.jsonl.gz [every=2] [informed]
For each step s, OLS of V - 42 on: my dice sum - 7 and, for each earlier step t, the SUM over the other four seats of
(mid - 42), clipped like the agent (+-5 at step 0, +-15 later). Train on even-numbered matches, test on odd ones.
Candidate seats only. Prints the PRIOR line for agent.py.
"""
import gzip
import json
import sys

import numpy as np

path, EVERY = sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 2
INFORMED = len(sys.argv) > 3 and sys.argv[3] == "informed"   # keep only seat-rounds whose own step-0 quote was NOT 40 / 44
CLIP0, CLIP = 5.0, 15.0
acc = {s: [0, 0, 0, 0] for s in range(1, 4)}
test = {s: [] for s in range(1, 4)}
n = 0
with gzip.open(path, "rt") as f:
    for line in f:
        n += 1
        if n % EVERY:
            continue
        m = json.loads(line)
        R = m["rounds"]
        S = np.array([[st["s"] for st in r["st"]] for r in R], float)
        M = (S[..., 0] + S[..., 1]) / 4.0 - 42.0
        D = np.array([r["d"] for r in R], float)
        V = np.array([r["v"] for r in R], float) - 42.0
        train = (n // EVERY) % 2 == 0
        for i in range(5):
            if m["kinds"][i] == "house":
                continue
            own = D[:, i] + D[:, i + 6] - 7.0
            keep = ~((S[:, 0, i, 0] == 80) & (S[:, 0, i, 1] == 88)) if INFORMED else np.ones(len(R), bool)
            for s in range(1, 4):
                cols = [np.ones(len(R)), own]
                for t in range(s):
                    c = CLIP0 if t == 0 else CLIP
                    cols.append(np.clip(np.delete(M[:, t, :], i, axis=1), -c, c).sum(1))
                X = np.stack(cols, 1)[keep]
                Vk = V[keep]
                if train:
                    a = acc[s]
                    a[0] = a[0] + X.T @ X if not np.isscalar(a[0]) else X.T @ X
                    a[1] = a[1] + X.T @ Vk if not np.isscalar(a[1]) else X.T @ Vk
                else:
                    test[s].append((X, Vk))
W = {}
for s in range(1, 4):
    w = np.linalg.solve(acc[s][0], acc[s][1])
    W[s] = w
    err = sum(float(((V - X @ w) ** 2).sum()) for X, V in test[s]); cnt = sum(len(V) for X, V in test[s])
    base = sum(float(((V - X[:, 1]) ** 2).sum()) for X, V in test[s])
    print(f"step {s}: weights [const, own, S_0..] = {np.round(w, 3).tolist()}   held-out RMSE {(err / cnt) ** .5:.3f}   (own dice only {(base / cnt) ** .5:.3f})")
if INFORMED:
    json.dump({str(s): [float(W[s][1]), [float(x) for x in W[s][2:]]] for s in range(1, 4)}, open("data/prior_informed.json", "w"))
print("\nPRIOR = {0: (1.0, []), " + ", ".join(f"{s}: ({W[s][1]:.2f}, {[round(float(x), 2) for x in W[s][2:]]})" for s in range(1, 4)) + "}")
