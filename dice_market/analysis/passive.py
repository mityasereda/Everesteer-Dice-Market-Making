"""Score an agent's value estimate on recorded real matches, as a passive observer.

    python3 analysis/passive.py agent.py [sample_every=150]
For each sampled match and each seat, the agent sees the real dice and the real quotes of the other seats and
reports its estimate at each step. Reports RMSE vs the true value by step, early rounds (<30) vs later.
Opponents did not react to this agent, so this measures estimation accuracy, not profit.
"""
import gzip, importlib.util, json, sys
from concurrent.futures import ProcessPoolExecutor

AGENT = sys.argv[1]
EVERY = int(sys.argv[2]) if len(sys.argv) > 2 else 150
MODES = {f"lam {l}": dict(prior="keep", lam=float(l)) for l in (60, 100, 200, 400)}


def load():
    spec = importlib.util.spec_from_file_location("cand", AGENT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def one(args):
    line, mode = args
    cfg = MODES[mode]
    m = json.loads(line)
    mod = load()
    if cfg["prior"] is None:
        mod.PRIOR = None
    mod.PARAMS["lam"] = cfg["lam"]
    out = {}
    for i in range(5):
        ag = mod.Agent(i, 5)
        for ri, r in enumerate(m["rounds"]):
            d, v = r["d"], r["v"]
            ag.new_round(ri, {i: d[i], i + 6: d[i + 6]})
            hist = []
            for s, st in enumerate(r["st"]):
                e = ag.estimate(s, hist)
                k = (s, 0 if ri < 30 else 1)
                a = out.setdefault(k, [0.0, 0])
                a[0] += (v - e) ** 2
                a[1] += 1
                hist.append({"sent": {j: (b / 2.0, a_ / 2.0) for j, (b, a_) in enumerate(st["s"])}, "quotes": {},
                             "trades": [{"buyer": b, "seller": se, "price": p / 4.0, "kind": "take" if k_ == "t" else "cross"}
                                        for b, se, p, k_ in st["t"]]})
            ag.end_round({"value": v, "dice": d, "pnl": [], "history": hist})
    return out


if __name__ == "__main__":
    lines, n = [], 0
    with gzip.open("data/replays.jsonl.gz", "rt") as f:
        for line in f:
            n += 1
            if n % EVERY == 0:
                lines.append(line)
    print("sampled matches:", len(lines))
    with ProcessPoolExecutor() as pool:
        for mode in MODES:
            tot = {}
            for out in pool.map(one, [(l, mode) for l in lines], chunksize=4):
                for k, (se, c) in out.items():
                    a = tot.setdefault(k, [0.0, 0]); a[0] += se; a[1] += c
            row = "  ".join(f"s{s}: {(tot[(s,0)][0]/tot[(s,0)][1])**.5:4.2f}/{(tot[(s,1)][0]/tot[(s,1)][1])**.5:4.2f}" for s in range(4))
            print(f"{mode:22s} RMSE early/later  {row}")
