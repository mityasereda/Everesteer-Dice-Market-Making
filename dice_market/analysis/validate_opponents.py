"""How faithful are the imitation agents? Play them against each other and compare with the real agents.

    python3 analysis/validate_opponents.py data/profile.json [matches=150]

Tables are random 5-seat draws from the nine imitations. For each imitation the script reports, next to the real
agent's number from profile.json: mid RMSE vs the value by step, mean quote width at step 3, trades per round,
edge per round at steps 2 + 3, and pairwise win rate / P&L per round (real ones come from the whole leaderboard,
so they include opponents that are not these nine: compare the ORDER, not the levels).
"""
import json
import random
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import dicemarket as dm  # noqa: E402
SHORT = {"Docsdadi": "docsdadi", "zudvarnoki": "zudvarnoki", "MPP": "mpp", "billhwang": "billhwang",
         "momarci13": "momarci13", "yurko_mud": "yurko", "Joshua_Karat": "joshua", "leobgg": "leobgg", "house pro": "housepro"}
LABELS = list(SHORT)


def one(seed):
    rng = random.Random(f"val:{seed}")
    pick = rng.sample(LABELS, 5)
    trace = []
    res = dm.play_match([str(ROOT / "opponents" / f"imit_{SHORT[l]}.py") for l in pick], rounds=200, seed=seed, trace=trace)
    out = {}
    for i, l in enumerate(pick):
        st = [[0.0, 0, 0.0, 0.0] for _ in range(4)]    # sq err, n, width sum, edge sum
        for r in trace:
            v = r["value"]
            for s, h in enumerate(r["history"]):
                b, a = h["sent"][i]
                st[s][0] += ((b + a) / 2 - v) ** 2
                st[s][1] += 1
                st[s][2] += a - b
                for t in h["trades"]:
                    if t["buyer"] == i:
                        st[s][3] += v - t["price"]
                    if t["seller"] == i:
                        st[s][3] += t["price"] - v
        wins = sum(1.0 if res[i]["pnl"] > res[j]["pnl"] else 0.5 if res[i]["pnl"] == res[j]["pnl"] else 0.0
                   for j in range(5) if j != i)
        out[l] = (st, res[i]["pnl_per_round"], res[i]["trades"] / 200.0, wins)
    return out


if __name__ == "__main__":
    prof = json.load(open(sys.argv[1]))
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 150
    tot = defaultdict(lambda: [[0.0, 0, 0.0, 0.0] for _ in range(4)])
    pnl, trades, wins, games = defaultdict(float), defaultdict(float), defaultdict(float), defaultdict(int)
    with ProcessPoolExecutor() as pool:
        for out in pool.map(one, range(n)):
            for l, (st, p, tr, w) in out.items():
                for s in range(4):
                    for k in range(4):
                        tot[l][s][k] += st[s][k]
                pnl[l] += p; trades[l] += tr; wins[l] += w; games[l] += 4
    M, Q, E = prof["match"], prof["Q"], prof["E"]
    print(f"{'agent':13s}|{'mid RMSE s1 s2 s3  (real)':>30s} |{'width s3':>10s}{'real':>6s} |{'trades/rd':>10s}{'real':>6s} |{'edge s2+s3':>11s}{'real':>6s} |{'win%':>6s}{'real':>6s} |{'P&L/rd':>8s}{'real':>7s}")
    for l in LABELS:
        t = tot[l]
        ims = [(t[s][0] / t[s][1]) ** 0.5 for s in (1, 2, 3)]
        real = [(Q[l][s][7] / Q[l][s][0]) ** 0.5 for s in (1, 2, 3)]
        rn = Q[l][0][0]
        n_games = games[l] / 4   # matches this agent played here
        e_im = (t[2][3] + t[3][3]) / t[2][1]
        e_re = sum(E[l][s][1] + E[l][s][3] for s in (2, 3)) / rn
        tr_re = sum(E[l][s][0] + E[l][s][2] for s in range(4)) / rn
        print(f"{l:13s}|{ims[0]:6.2f}{ims[1]:5.2f}{ims[2]:5.2f}  ({real[0]:.2f} {real[1]:.2f} {real[2]:.2f}) |{t[3][2] / t[3][1]:10.2f}{Q[l][3][1] / Q[l][3][0]:6.2f} |"
              f"{trades[l] / n_games:10.2f}{tr_re:6.2f} |{e_im:+11.2f}{e_re:+6.2f} |"
              f"{100 * wins[l] / games[l]:6.0f}{100 * M[l]['wins'] / M[l]['games']:6.0f} |{pnl[l] / n_games:+8.2f}{M[l]['pnl'] / M[l]['n']:+7.2f}")
