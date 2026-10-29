"""Do public trades carry information about the value beyond own dice and rivals' quote mids?

    python3 analysis/trade_info.py data/day28.jsonl.gz [every=4]
OLS of V - 42 on the v8 features (own dice - 7, sums of the other seats' mids at each earlier step, step-0 mids clipped at +-3) with and
without trade features per earlier step: A = signed take count (+1 when the taker bought, -1 when the taker sold, aggressor =
the side whose own quote did NOT rest), B = sum of (price - 42) over the step's take trades, C = sum of (price - 42) over its crosses.
Informed seat-rounds only (own step-0 quote not the default), candidate seats, train on even matches, test on odd ones.
"""
import gzip
import json
import sys

import numpy as np

path = sys.argv[1]
EVERY = int(sys.argv[2]) if len(sys.argv) > 2 else 4
CLIP0 = 3.0
tr = {s: [None] * 4 for s in range(1, 4)}
te = {s: [] for s in range(1, 4)}
n = 0
with gzip.open(path, "rt") as f:
    for line in f:
        n += 1
        if n % EVERY:
            continue
        m = json.loads(line)
        R = m["rounds"]
        nr = len(R)
        S = np.array([[st["s"] for st in r["st"]] for r in R], float)
        M = (S[..., 0] + S[..., 1]) / 4.0 - 42.0
        D = np.array([r["d"] for r in R], float)
        V = np.array([r["v"] for r in R], float) - 42.0
        TA = np.zeros((nr, 4)); TB = np.zeros((nr, 4)); TC = np.zeros((nr, 4))
        for ri, r in enumerate(R):
            for s, st in enumerate(r["st"]):
                prev = r["st"][s - 1]["s"] if s else None
                for bu, se, p, k in st["t"]:
                    if k == "c":
                        TC[ri, s] += p / 4.0 - 42
                    else:
                        buy_rest = prev[bu][0] * 2 == p
                        TA[ri, s] += -1 if buy_rest else 1
                        TB[ri, s] += p / 4.0 - 42
        train = (n // EVERY) % 2 == 0
        for i in range(5):
            if m["kinds"][i] == "house":
                continue
            keep = ~((S[:, 0, i, 0] == 80) & (S[:, 0, i, 1] == 88))
            own = D[:, i] + D[:, i + 6] - 7.0
            for s in range(1, 4):
                base = [np.ones(nr), own]
                for t in range(s):
                    c = CLIP0 if t == 0 else 15.0
                    base.append(np.clip(np.delete(M[:, t, :], i, axis=1), -c, c).sum(1))
                extra = []
                for t in range(s):
                    extra += [TA[:, t], TB[:, t], TC[:, t]]
                Xb = np.stack(base, 1)[keep]
                Xe = np.stack(base + extra, 1)[keep]
                Vk = V[keep]
                if train:
                    a = tr[s]
                    if a[0] is None:
                        a[0], a[1], a[2], a[3] = Xb.T @ Xb, Xe.T @ Xe, Xb.T @ Vk, Xe.T @ Vk
                    else:
                        a[0] += Xb.T @ Xb; a[1] += Xe.T @ Xe; a[2] += Xb.T @ Vk; a[3] += Xe.T @ Vk
                else:
                    te[s].append((Xb, Xe, Vk))
for s in range(1, 4):
    wb = np.linalg.lstsq(tr[s][0], tr[s][2], rcond=1e-10)[0]; we = np.linalg.lstsq(tr[s][1], tr[s][3], rcond=1e-10)[0]
    cnt = sum(len(v) for _, _, v in te[s])
    eb = sum(float(((v - xb @ wb) ** 2).sum()) for xb, _, v in te[s]); ee = sum(float(((v - xe @ we) ** 2).sum()) for _, xe, v in te[s])
    print(f"step {s}: held-out RMSE {np.sqrt(eb / cnt):.3f} without trade features, {np.sqrt(ee / cnt):.3f} with ({cnt} seat-rounds); trade weights "
          + str(np.round(we[len(wb):], 3).tolist()))
