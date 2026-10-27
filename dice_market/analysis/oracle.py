"""Upper bound: if I knew each seat's identity, how much better is the decode? Weights each other seat's mid by the
inverse of that agent's own step-2 MSE (from profile.json). Train on even matches, test on odd."""
import gzip, json, sys
import numpy as np

prof = json.load(open("data/profile.json"))["Q"]
mse = {l: (q[2][7] / q[2][0] if q[2][0] > 200 else 12.0) for l, q in prof.items()}   # step-2 mid vs value, MSE
mse1 = {l: (q[1][7] / q[1][0] if q[1][0] > 200 else 36.0) for l, q in prof.items()}
EVERY, n_rows, acc, test = 8, 0, [0, 0, 0.0, 0], []


def feat(M, D, i, labels):
    own = D[:, i] + D[:, i + 6] - 7.0
    oth = [j for j in range(5) if j != i]
    w2 = np.array([1 / mse[labels[j]] for j in oth]); w2 /= w2.sum()
    w1 = np.array([1 / mse1[labels[j]] for j in oth]); w1 /= w1.sum()
    return np.stack([np.ones_like(own), own, M[:, 2, oth] @ w2, M[:, 1, oth] @ w1, M[:, 2, oth].sum(1)], 1)


with gzip.open("data/replays.jsonl.gz", "rt") as f:
    for line in f:
        n_rows += 1
        if n_rows % EVERY:
            continue
        m = json.loads(line); R = m["rounds"]
        S = np.array([[st["s"] for st in r["st"]] for r in R], float)
        M = (S[..., 0] + S[..., 1]) / 4.0 - 42.0
        D = np.array([r["d"] for r in R], float)
        V = np.array([r["v"] for r in R], float) - 42.0
        for i in range(5):
            X = feat(M, D, i, m["labels"])
            if (n_rows // EVERY) % 2 == 0:
                acc[0] = acc[0] + X.T @ X if acc[3] else X.T @ X
                acc[1] = acc[1] + X.T @ V if acc[3] else X.T @ V
                acc[3] += len(V)
            else:
                test.append((X, V))
w = np.linalg.solve(acc[0], acc[1])
err = sum(float(((V - X @ w) ** 2).sum()) for X, V in test); n = sum(len(V) for X, V in test)
print("oracle (identity-weighted) step-3 held-out RMSE", round((err / n) ** .5, 3), " weights", np.round(w, 2).tolist())
print("pooled sum decode was 3.275; best single agents' own step-3 mid about 2.95; floor 2.42")
