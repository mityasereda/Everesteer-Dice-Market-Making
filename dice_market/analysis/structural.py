"""Structural decode: per agent label, learn dice-sum S_j ~ a + b * (step-1 mid - 42); then
V-hat = 42 + (my dice - 7) + sum_j (a_j + b_j x_j).  Train per-label on even matches, test on odd ones.
Compared with the pooled decode (step-2 RMSE 4.33)."""
import gzip, json
import numpy as np
from collections import defaultdict

EVERY = 8
st = defaultdict(lambda: np.zeros(5))   # per label: n, sx, sy, sxx, sxy
tests, n_rows = [], 0
with gzip.open("data/replays.jsonl.gz", "rt") as f:
    for line in f:
        n_rows += 1
        if n_rows % EVERY:
            continue
        m = json.loads(line); R = m["rounds"]
        S = np.array([[st_["s"][:] for st_ in r["st"][:2]] for r in R], float)    # (R,2,5,2)
        M = (S[..., 0] + S[..., 1]) / 4.0 - 42.0
        D = np.array([r["d"] for r in R], float)
        V = np.array([r["v"] for r in R], float) - 42.0
        sums = np.stack([D[:, j] + D[:, j + 6] - 7.0 for j in range(5)], 1)          # (R,5)
        if (n_rows // EVERY) % 2 == 0:
            for j, l in enumerate(m["labels"]):
                x, y = M[:, 1, j], sums[:, j]
                a = st[l]; a += [len(x), x.sum(), y.sum(), (x * x).sum(), (x * y).sum()]
        else:
            tests.append((m["labels"], M[:, 1, :], sums, V))


def fit(l):
    n, sx, sy, sxx, sxy = st[l] if st[l][0] > 300 else st["__pool__"]
    den = sxx - sx * sx / n
    b = (sxy - sx * sy / n) / den if den > 1e-6 else 0.0
    return (sy - b * sx) / n, b


for l in list(st):   # pooled fallback for rare agents
    st["__pool__"] += st[l]
err_s, err_p, n = 0.0, 0.0, 0
for labels, M1, sums, V in tests:
    ab = [fit(l) for l in labels]
    for i in range(5):
        own = sums[:, i]
        est = own + sum(ab[j][0] + ab[j][1] * M1[:, j] for j in range(5) if j != i)
        err_s += float(((V - est) ** 2).sum())
        n += len(V)
print("structural per-label decode, step-2 information: held-out RMSE", round((err_s / n) ** .5, 3), "  (pooled linear decode 4.33; floor 2.42)")
for l in ["billhwang", "bruno", "zudvarnoki", "Docsdadi", "MPP", "yurko_mud", "zeus", "house pro", "house reader"]:
    if l in st and st[l][0] > 300:
        a, b = fit(l)
        print(f"  {l:14s} S_j-7 = {a:+.2f} + {b:.2f} * (mid1-42)")
