"""Local tournament harness: play many 5-seat matches of one agent against a mix of opponents.

    python3 tourney.py agent.py                         # the default suite
    python3 tourney.py agent.py --mix gate_imit -n 100
    python3 tourney.py agent.py --pool imit -n 300      # random tables from the nine imitations plus me

Opponent families
  house   the pack's house and example agents (far weaker than the real field)
  imit    opponents/imit_*.py   static linear copies of real agents, fitted from the replays
  adapt   opponents/adapt_*.py  my learning estimator with a real agent's spread widths, hiding and skew
Both copy families are approximations of real agents' AVERAGE behaviour: results against them are provisional.

Seats are shuffled per match and dice depend on the seed alone, so variants of an agent run with the same --seed are
compared on identical dice and opponent behaviour. A match counts a "silent forced quote" when the agent returned an
invalid quote that the engine replaced by 40 / 44 without logging an error.
The gate mixes mirror the upload test (me, the boss, three example agents); the boss is the house-pro copy and
"P(beat boss)" is my win rate against it, "P(pass)" the implied chance of beating it in at least 2 of 3 matches.
"""
import argparse
import random
import statistics
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import dicemarket as dm

ROOT = Path(__file__).resolve().parent
SHORT = ["docsdadi", "zudvarnoki", "mpp", "billhwang", "momarci13", "yurko", "joshua", "leobgg", "housepro"]
FILES = {
    "fair_value": "examples/fair_value.py",
    "wide": "examples/wide_quoter.py",
    "random": "examples/random_quoter.py",
    "house_fair_value": "house/house_fair_value.py",
    "house_wide": "house/house_wide_market_maker.py",
    "house_reader": "house/house_reader.py",
    "house_hider": "house/house_hider.py",
}
for _f in sorted((ROOT / "opponents").glob("*.py")):   # every generated opponent file is available by its stem
    FILES[_f.stem] = f"opponents/{_f.name}"
VERSIONS = ["v0", "v1", "v2", "v3", "v4", "v5"]   # interim versions of my own agent, opponents/ver_*.py (analysis/make_versions.py)
for _v in VERSIONS:
    FILES[f"ver_{_v}"] = f"opponents/ver_{_v}.py"
A, B = ["docsdadi", "zudvarnoki", "mpp", "billhwang"], ["momarci13", "yurko", "joshua", "leobgg"]
MIXES = {
    "weak": ["fair_value", "fair_value", "wide", "random"],
    "readers": ["house_reader", "house_reader", "house_hider", "house_fair_value"],
    "mixed": ["house_reader", "house_hider", "fair_value", "house_wide"],
    "hard": ["house_reader", "house_reader", "house_hider", "house_hider"],
    "self": ["me", "me", "me", "me"],
    "gate_imit": ["imit_housepro", "fair_value", "wide", "random"],
    "gate_adapt": ["adapt_housepro", "fair_value", "wide", "random"],
    "fieldA_imit": [f"imit_{s}" for s in A],
    "fieldB_imit": [f"imit_{s}" for s in B],
    "fieldA_adapt": [f"adapt_{s}" for s in A],
    "fieldB_adapt": [f"adapt_{s}" for s in B],
    "pro_adapt": ["adapt_housepro", "adapt_docsdadi", "adapt_zudvarnoki", "adapt_billhwang"],
    "fieldA_adapt30": [f"adapt30_{s}" for s in A],
    "fieldB_adapt30": [f"adapt30_{s}" for s in B],
    "pro_adapt30": ["adapt30_housepro", "adapt30_docsdadi", "adapt30_zudvarnoki", "adapt30_billhwang"],
    "gate_adapt30": ["adapt30_housepro", "fair_value", "wide", "random"],
    "pro_imit": ["imit_housepro", "imit_docsdadi", "imit_zudvarnoki", "imit_billhwang"],
    "ver_all": [f"ver_{v}" for v in ("v0", "v1", "v2", "v3")],
    **{f"vs_{v}": [f"ver_{v}"] * 4 for v in VERSIONS},
    "ver_and_field": ["ver_v1", "ver_v3", "adapt_housepro", "adapt_zudvarnoki"],
    "rA_imit": ["rimit_zeus", "rimit_zudvarnoki", "rimit_bruno", "rimit_gyurmatag"],
    "rB_imit": ["rimit_yurko_mud", "rimit_hegemonia", "rimit_jk007", "rimit_kart7"],
    "rC_imit": ["rimit_sk1", "rimit_marco_milanesi_", "rimit_housepro", "rimit_zeus"],
    "gate_rimit": ["rimit_housepro", "fair_value", "wide", "random"],
    "radA": ["adaptr_zeus", "adaptr_zudvarnoki", "adaptr_bruno", "adaptr_gyurmatag"],
    "radB": ["adaptr_yurko_mud", "adaptr_hegemonia", "adaptr_jk007", "adaptr_kart7"],
    "radC": ["adaptr_sk1", "adaptr_marco_milanesi_", "adaptr_housepro", "adaptr_zeus"],
    "gate_rad": ["adaptr_housepro", "fair_value", "wide", "random"],
    "ver_old": [f"ver_{v}" for v in ("v1", "v2", "v3", "v4")],
}
SUITE = ["weak", "readers", "hard", "gate_imit", "gate_adapt", "fieldA_imit", "fieldB_imit", "fieldA_adapt", "fieldB_adapt",
         "pro_adapt", "self"]


POOLS = {"imit": [f"imit_{s}" for s in SHORT], "adapt": [f"adapt_{s}" for s in SHORT], "ver": [f"ver_{v}" for v in VERSIONS],
         "rimit": ["rimit_zeus", "rimit_zudvarnoki", "rimit_bruno", "rimit_gyurmatag", "rimit_yurko_mud", "rimit_hegemonia",
                  "rimit_jk007", "rimit_kart7", "rimit_sk1", "rimit_marco_milanesi_", "rimit_housepro"],
         "radapt": [n.replace("rimit_", "adaptr_") for n in ["rimit_zeus", "rimit_zudvarnoki", "rimit_bruno", "rimit_gyurmatag", "rimit_yurko_mud",
                   "rimit_hegemonia", "rimit_jk007", "rimit_kart7", "rimit_sk1", "rimit_marco_milanesi_", "rimit_housepro"]],
         "all": [f"adapt_{s}" for s in SHORT] + [f"ver_{v}" for v in VERSIONS]}


def _path(name, me):
    return str(me) if name == "me" else str(ROOT / FILES[name])


def play_counted(paths, rounds, seed):
    """play_match, also counting per seat the invalid quotes the engine silently replaced by 40 / 44."""
    real, n, calls, silent = dm.clean_quote, len(paths), [0], [0] * len(paths)

    def counting(q):
        c = real(q)
        seat = calls[0] % n
        calls[0] += 1
        if q is not None and c == dm.DEFAULT_QUOTE:
            try:
                same = (float(q[0]), float(q[1])) == dm.DEFAULT_QUOTE
            except Exception:
                same = False
            if not same:
                silent[seat] += 1
        return c

    dm.clean_quote = counting
    try:
        res = dm.play_match(paths, rounds=rounds, seed=seed)
    finally:
        dm.clean_quote = real
    return res, silent


def one_match(args):
    me, names, seed, rounds = args
    rng = random.Random(f"order:{seed}")
    order = ["me"] + list(names)
    rng.shuffle(order)
    res, silent = play_counted([_path(n, me) for n in order], rounds, seed)
    i = order.index("me")
    pnl = res[i]["pnl_per_round"]
    wins = {}
    for j, n in enumerate(order):
        if j != i:
            d = pnl - res[j]["pnl_per_round"]
            wins.setdefault(n, []).append(1.0 if d > 0 else 0.5 if d == 0 else 0.0)
    ahead = sum(1 for j in range(len(order)) if j != i and res[j]["pnl_per_round"] > pnl)
    tied = sum(1 for j in range(len(order)) if j != i and res[j]["pnl_per_round"] == pnl)
    first = 1.0 / (1 + tied) if ahead == 0 else 0.0          # a tie for first shares the win
    return pnl, wins, (res[i]["errors"], res[i]["timeouts"], res[i]["first_error"]), silent[i], first, 1 + ahead + tied / 2.0


def run(me, mix, n, rounds=200, seed0=0, pool=None):
    names = MIXES[mix]
    assert len(names) == 4, f"mix {mix} has {len(names)} opponents: a table must have 5 seats (4 opponents)"
    jobs = [(me, names, seed0 + k, rounds) for k in range(n)]
    out = list(pool.map(one_match, jobs, chunksize=2)) if pool else [one_match(j) for j in jobs]
    pnls = [o[0] for o in out]
    wins = {}
    for o in out:
        for k, v in o[1].items():
            wins.setdefault(k, []).extend(v)
    allw = [x for v in wins.values() for x in v]
    se = statistics.stdev(pnls) / len(pnls) ** 0.5 if len(pnls) > 1 else 0.0
    return {"pnl": statistics.mean(pnls), "se": se, "win": statistics.mean(allw),
            "first": statistics.mean(o[4] for o in out), "rank": statistics.mean(o[5] for o in out),
            "per": {k: statistics.mean(v) for k, v in wins.items()},
            "errors": sum(o[2][0] for o in out), "timeouts": sum(o[2][1] for o in out),
            "first_error": next((o[2][2] for o in out if o[2][2]), None), "silent": sum(o[3] for o in out)}


def pool_run(me, family, n, rounds=200, seed0=0, pool=None):
    """Random 5-seat tables drawn from the nine copies plus me. Returns {name: (win rate, P&L per round, tables)}."""
    names = POOLS[family] + ["me"]
    jobs = [(me, names, seed0 + k, rounds) for k in range(n)]
    out = list(pool.map(pool_match, jobs, chunksize=2))
    acc = {}
    for table in out:
        for name, w, p in table:
            a = acc.setdefault(name, [0.0, 0.0, 0])
            a[0] += w
            a[1] += p
            a[2] += 1
    return {k: (v[0] / (4 * v[2]), v[1] / v[2], v[2]) for k, v in acc.items()}


def pool_match(args):
    me, names, seed, rounds = args
    rng = random.Random(f"pool:{seed}")
    order = rng.sample(names, 5)
    res = dm.play_match([_path(n, me) for n in order], rounds=rounds, seed=seed)
    pn = [r["pnl_per_round"] for r in res]
    return [(order[i], sum(1.0 if pn[i] > pn[j] else 0.5 if pn[i] == pn[j] else 0.0 for j in range(5) if j != i), pn[i])
            for i in range(5)]


def fmt(mix, r):
    detail = "  ".join(f"{k.replace('imit_', 'i_').replace('adapt_', 'a_')} {v:.2f}" for k, v in sorted(r["per"].items()))
    boss = next((v for k, v in r["per"].items() if k.endswith("housepro")), None)
    gate = ""
    if boss is not None and mix.startswith("gate"):
        gate = f"   P(beat boss) {boss:.2f}  P(pass 2 of 3) {3 * boss ** 2 - 2 * boss ** 3:.2f}"
    return (f"{mix:13s} pairwise win {r['win']:.3f}   1st {r['first']:.3f}   avg rank {r['rank']:.2f}   P&L/round {r['pnl']:+.3f} +- {r['se']:.3f}   errors {r['errors']}   timeouts {r['timeouts']}   "
            f"silent-forced {r['silent']}{gate}\n{'':13s} [{detail}]" + (f"\n{r['first_error']}" if r["first_error"] else ""))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("agent")
    ap.add_argument("--mix", default="suite")
    ap.add_argument("--pool", choices=["imit", "adapt", "ver", "rimit", "radapt", "all"])
    ap.add_argument("-n", type=int, default=60)
    ap.add_argument("--rounds", type=int, default=200)
    ap.add_argument("--seed", type=int, default=1000)
    a = ap.parse_args()
    me = Path(a.agent).resolve()
    with ProcessPoolExecutor() as pool:
        if a.pool:
            res = pool_run(me, a.pool, a.n, a.rounds, a.seed, pool)
            print(f"pool '{a.pool}' plus me ({len(POOLS[a.pool]) + 1} agents), {a.n} random 5-seat tables")
            for name, (w, p, t) in sorted(res.items(), key=lambda kv: -kv[1][0]):
                print(f"  {name:18s} win rate {100 * w:5.1f}%   P&L/round {p:+.3f}   tables {t}")
            return
        mixes = SUITE if a.mix == "suite" else list(MIXES) if a.mix == "all" else [a.mix]
        for m in mixes:
            print(fmt(m, run(me, m, a.n, a.rounds, a.seed, pool)))


if __name__ == "__main__":
    main()
