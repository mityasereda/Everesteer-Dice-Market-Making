"""How well can the value be estimated from my dice plus the others' quotes? Offline OLS on the real field.

    python3 analysis/decode.py data/replays.jsonl.gz [every=4]
For each step s and each seat's point of view, regress V - 42 on: my dice sum - 7, and for each earlier step t the
sum over the other seats of (mid - 42). Prints RMSE by step for several feature sets.
"""
import gzip, json, sys
import numpy as np

path, EVERY = sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 4
FS = {  # name -> function(step) -> list of earlier steps whose summed other-mids are used
    "own dice only": lambda s: [],
    "+ summed mids of others": lambda s: list(range(s)),
}
acc = {(name, s): [np.zeros((2 + s, 2 + s)), np.zeros(2 + s), 0.0, 0] for name in FS for s in range(4)}
acc_floor = [0.0, 0]
n_rows = 0
with gzip.open(path, "rt") as f:
    for line in f:
        n_rows += 1
        if n_rows % EVERY:
            continue
        m = json.loads(line)
        R = m["rounds"]
        S = np.array([[st["s"] for st in r["st"]] for r in R], float)      # (rounds, 4, 5, 2) ticks
        mid = (S[..., 0] + S[..., 1]) / 4.0 - 42.0                          # (rounds, 4, 5)
        D = np.array([r["d"] for r in R], float)
        V = np.array([r["v"] for r in R], float) - 42.0
        for i in range(5):
            own = D[:, i] + D[:, i + 6] - 7.0
            for s in range(4):
                oth = [mid[:, t, :].sum(1) - mid[:, t, i] for t in range(s)]
                for name, fn in FS.items():
                    cols = [np.ones(len(R)), own] + [oth[t] for t in fn(s)]
                    X = np.stack(cols, 1)
                    a = acc[(name, s)]
                    k = X.shape[1]
                    a[0] = a[0] if a[0].shape[0] == k else np.zeros((k, k))
                    a[1] = a[1] if a[1].shape[0] == k else np.zeros(k)
                    a[0] += X.T @ X
                    a[1] += X.T @ V
                    a[2] += float(V @ V)
                    a[3] += len(V)
print("matches used", n_rows // EVERY)
for s in range(4):
    for name in FS:
        A, b, vv, n = acc[(name, s)]
        w = np.linalg.solve(A, b)
        rmse = ((vv - 2 * w @ b + w @ A @ w) / n) ** 0.5
        print(f"step {s}  {name:26s} RMSE {rmse:5.2f}  weights {np.round(w, 2).tolist()}")
print("reference: knowing all 10 visible dice -> RMSE 2.42; own two dice only -> 5.40")
