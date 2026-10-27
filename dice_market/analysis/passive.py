"""Score an agent's value estimate on recorded real matches, as a passive observer.

    python3 analysis/passive.py agent.py [data=data/recent_tight.jsonl.gz] [every=12] [offset=6] [informed]
Takes every `every`-th match (those with index % every == offset), feeds the agent each seat's real dice and the other
seats' real quotes, and reports RMSE of its estimate against the true value, by step, for rounds < 30 and >= 30.
The prior in agent.py was fitted on matches with index % 4 == 0 of the same file (see fit_prior.py), so offset 6 with
every 12 (index % 4 == 2) is held out. Opponents did not react to this agent: this measures estimation, not profit.
"""
import gzip
import importlib.util
import json
import sys
from concurrent.futures import ProcessPoolExecutor

AGENT = sys.argv[1]
DATA = sys.argv[2] if len(sys.argv) > 2 else "data/recent_tight.jsonl.gz"
EVERY = int(sys.argv[3]) if len(sys.argv) > 3 else 12
OFFSET = int(sys.argv[4]) if len(sys.argv) > 4 else 6
INFORMED = len(sys.argv) > 5 and sys.argv[5] == "informed"   # score only seats that quote informed (default share < 10%) in the match
OLD_PRIOR = {0: (1.0, []), 1: (1.0, [0.72]), 2: (0.75, [-0.16, 0.26]), 3: (0.26, [0.05, -0.03, 0.26])}
MODES = {"as is": {}}   # use the agent file exactly as it is (no parameter overrides)
if INFORMED:
    try:
        _pi = json.load(open("data/prior_informed.json"))
        MODES["prior refitted on informed seats"] = dict(prior_dict={0: (1.0, []), **{int(k): (v[0], v[1]) for k, v in _pi.items()}})
    except FileNotFoundError:
        pass


def one(args):
    line, mode = args
    cfg = dict(MODES[mode])
    m = json.loads(line)
    spec = importlib.util.spec_from_file_location("cand", AGENT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if cfg.pop("prior", "new") == "old":
        mod.PRIOR = OLD_PRIOR
    if "prior_dict" in cfg:
        mod.PRIOR = cfg.pop("prior_dict")
    mod.PARAMS.update(cfg)
    out = {}
    for i in range(5):
        if m["kinds"][i] == "house":
            continue
        if INFORMED and sum(1 for r in m["rounds"][::10] if tuple(r["st"][0]["s"][i]) == (80, 88)) > 2:
            continue
        ag = mod.Agent(i, 5)
        for ri, r in enumerate(m["rounds"]):
            d, v = r["d"], r["v"]
            ag.new_round(ri, {i: d[i], i + 6: d[i + 6]})
            hist = []
            for s, st in enumerate(r["st"]):
                e = ag.estimate(s, hist)
                a = out.setdefault((s, 0 if ri < 30 else 1 if ri < 100 else 2), [0.0, 0])
                a[0] += (v - e) ** 2
                a[1] += 1
                hist.append({"sent": {j: (b / 2.0, a_ / 2.0) for j, (b, a_) in enumerate(st["s"])}, "quotes": {},
                             "trades": [{"buyer": b, "seller": se, "price": p / 4.0, "kind": "take" if k_ == "t" else "cross"}
                                        for b, se, p, k_ in st["t"]]})
            ag.end_round({"value": v, "dice": d, "pnl": [], "history": hist})
    return out


if __name__ == "__main__":
    lines = []
    with gzip.open(DATA, "rt") as f:
        for n, line in enumerate(f):
            if n % EVERY == OFFSET:
                lines.append(line)
    print("held-out matches:", len(lines), "from", DATA)
    print(f"{'design':38s}   RMSE by step, rounds 0-29 / 30-99 / 100-199")
    with ProcessPoolExecutor() as pool:
        for mode in MODES:
            tot = {}
            for out in pool.map(one, [(l, mode) for l in lines], chunksize=4):
                for k, (se, c) in out.items():
                    a = tot.setdefault(k, [0.0, 0]); a[0] += se; a[1] += c
            row = "  ".join(f"s{s}: " + "/".join(f"{(tot[(s, b)][0] / tot[(s, b)][1]) ** .5:4.2f}" for b in range(3)) for s in range(2, 4))
            print(f"{mode:38s} {row}", flush=True)
