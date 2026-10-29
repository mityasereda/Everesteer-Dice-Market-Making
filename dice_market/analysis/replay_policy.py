"""Replay real rounds with the rivals' quotes held fixed and my agent deciding its own quotes.

    python3 analysis/replay_policy.py agent.py data/mine_all.jsonl.gz [seats=mine|all] [max_matches] ...
For every round of every chosen match the rivals' RECORDED quotes at each step are fed to the real matching engine
(dicemarket.match_step) together with the quote my agent chooses from the history it would have seen. This is exact for
one step (nobody can react within a step) and an approximation across steps (rivals cannot react to my different
quotes). It is good for changes that reveal no new information (taking, widths, skew) and wrong for ones that do
(revealing my dice at step 0). A baseline check replays my real agent on its real matches: it must reproduce the real edges.
"""
import gzip
import importlib.util
import json
import random
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import dicemarket as dm  # noqa: E402
import replays  # noqa: E402

ME = "fanaberysty"
V6_FROM, V6_TO = "2026-10-27T22:33:26", "2026-10-27T22:50:30"


def load_agent(path):
    spec = importlib.util.spec_from_file_location("cand_" + str(abs(hash(path))), path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def edges_of(match_row, seat, mod, overrides, rng_seed):
    """Per-step edge of `seat` over one match when my agent plays that seat against the recorded rivals."""
    m = replays.expand(match_row)
    overrides = dict(overrides)
    if "_prior" in overrides:                      # replace the module-level PRIOR for this variant
        mod.PRIOR = overrides.pop("_prior")
    mod.PARAMS.update(overrides)
    ag = mod.Agent(seat, 5)
    rng = random.Random(rng_seed)
    e = [0.0] * 4
    n = 0
    for ri, r in enumerate(m["rounds"]):
        d, v = r["dice"], r["value"]
        ag.new_round(ri, {seat: d[seat], seat + 6: d[seat + 6]})
        book = {i: {"bid": None, "ask": None} for i in range(5)}
        pos = {i: 0 for i in range(5)}
        hist = []
        for s, st in enumerate(r["steps"]):
            orders = {j: dm.clean_quote(tuple(q)) for j, q in st["sent"].items() if j != seat}
            orders[seat] = dm.clean_quote(ag.quote(s, [dict(h) for h in hist]))
            trades = dm.match_step(book, orders, rng, pos, s)
            for t in trades:
                if t["buyer"] == seat:
                    e[s] += v - t["price"]
                if t["seller"] == seat:
                    e[s] += t["price"] - v
            hist.append({"sent": orders, "quotes": dm.public_book(book), "trades": trades})
        ag.end_round({"value": v, "dice": d, "pnl": [], "history": hist})
        n += 1
    return [x / n for x in e]


def job(args):
    path, overrides, row, seat, seed = args
    mod = load_agent(path)
    return edges_of(row, seat, mod, overrides, seed)


def select_rows(data, which, limit, every=1, skip=0, since="", until="9999"):
    """The first `limit` eligible seat-matches after skipping the first `skip` (so tuning and validation sets are disjoint)."""
    rows = []
    seen = 0
    with gzip.open(data, "rt") as f:
        for n, line in enumerate(f):
            if n % every:
                continue
            m = json.loads(line)
            if which == "mine":
                if ME not in m["labels"] or V6_FROM <= m["at"] < V6_TO or not (since <= m["at"][:19] < until):
                    continue
                seen += 1
                if seen <= skip:
                    continue
                rows.append((line, m["labels"].index(ME)))
            else:
                for i, k in enumerate(m["kinds"]):
                    if k == "candidate":
                        rows.append((line, i))
            if len(rows) >= limit:
                break
    return rows


def evaluate(path, variants, rows, pool):
    """variants: {name: overrides}. Returns {name: list of per-match edge vectors} (same matches, same seeds)."""
    out = {}
    for name, ov in variants.items():
        jobs = [(path, ov, json.loads(line), seat, 1000 + k) for k, (line, seat) in enumerate(rows)]
        out[name] = list(pool.map(job, jobs, chunksize=4))
    return out


def report(res, base="off"):
    import statistics
    names = list(res)
    print(f"{'variant':34s}" + "".join(f"{'step ' + str(s):>9s}" for s in range(4)) + f"{'total':>9s}{'vs ' + base:>10s}{'+-se':>7s}")
    b = res[base]
    for name in names:
        v = res[name]
        steps = [statistics.mean(x[s] for x in v) for s in range(4)]
        tot = [sum(x) for x in v]
        diff = [sum(x) - sum(y) for x, y in zip(v, b)]
        se = statistics.stdev(diff) / len(diff) ** 0.5 if len(diff) > 1 else 0.0
        print(f"{name:34s}" + "".join(f"{x:+9.3f}" for x in steps) + f"{statistics.mean(tot):+9.3f}{statistics.mean(diff):+10.3f}{se:7.3f}")


if __name__ == "__main__":
    path, data = sys.argv[1], sys.argv[2]
    which = sys.argv[3] if len(sys.argv) > 3 else "mine"
    limit = int(sys.argv[4]) if len(sys.argv) > 4 else 10 ** 9
    rows = select_rows(data, which, limit)
    print(f"{len(rows)} seat-matches ({which})")
    variants = {"off": {}, "take 0.5 all steps": {"take": (None, 0.5, 0.5, 0.5)}}
    with ProcessPoolExecutor() as pool:
        report(evaluate(path, variants, rows, pool))
