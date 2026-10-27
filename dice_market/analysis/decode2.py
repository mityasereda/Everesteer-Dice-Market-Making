"""Compare offline decode feature sets (OLS on the real field), train on even matches, test on odd ones."""
import gzip, json, sys
import numpy as np

path, EVERY = sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 8
SETS = ["sum", "sorted", "sum+sorted", "sum+abs"]


def feats(kind, s, own, M, i):
    """own (R,), M (R,4,5) mids minus 42, i = my seat. Returns (R, k)."""
    cols = [np.ones_like(own), own]
    for t in range(1, s):
        o = np.delete(M[:, t, :], i, axis=1)                 # (R, 4) the others' mids at step t
        if kind in ("sum", "sum+sorted", "sum+abs"):
            cols.append(o.sum(1))
        if kind in ("sorted", "sum+sorted"):
            cols.extend(np.sort(o, axis=1).T)
        if kind == "sum+abs":
            cols.append(np.abs(o).sum(1))
    if s >= 1 and kind in ("sum", "sum+sorted", "sum+abs"):
        o0 = np.delete(M[:, 0, :], i, axis=1)
        cols.append(o0.sum(1))
    return np.stack(cols, 1)


stats = {}
n_rows = 0
with gzip.open(path, "rt") as f:
    for line in f:
        n_rows += 1
        if n_rows % EVERY:
            continue
        m = json.loads(line)
        part = "test" if (n_rows // EVERY) % 2 else "train"
        R = m["rounds"]
        S = np.array([[st["s"] for st in r["st"]] for r in R], float)
        M = (S[..., 0] + S[..., 1]) / 4.0 - 42.0
        D = np.array([r["d"] for r in R], float)
        V = np.array([r["v"] for r in R], float) - 42.0
        for i in range(5):
            own = D[:, i] + D[:, i + 6] - 7.0
            for kind in SETS:
                for s in (2, 3):
                    X = feats(kind, s, own, M, i)
                    a = stats.setdefault((kind, s, part), [0, 0, 0.0, 0])
                    a[0] = a[0] + X.T @ X if not np.isscalar(a[0]) else X.T @ X
                    a[1] = a[1] + X.T @ V if not np.isscalar(a[1]) else X.T @ V
                    a[2] += float(V @ V); a[3] += len(V)
                    if part == "test":
                        stats.setdefault((kind, s, "testX"), []).append((X, V))
for kind in SETS:
    for s in (2, 3):
        A, b, _, _ = stats[(kind, s, "train")]
        w = np.linalg.solve(A, b)
        err = sum(float(((V - X @ w) ** 2).sum()) for X, V in stats[(kind, s, "testX")])
        n = sum(len(V) for X, V in stats[(kind, s, "testX")])
        print(f"step {s} {kind:11s} held-out RMSE {(err / n) ** .5:5.3f}  weights {np.round(w, 2).tolist()}")
