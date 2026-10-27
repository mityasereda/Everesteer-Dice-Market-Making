"""Local tournament harness: play many 5-seat matches of one agent against a mix of opponents.

    python3 tourney.py agent.py                      # every mix, 60 matches each
    python3 tourney.py agent.py --mix readers -n 100
    python3 tourney.py agent.py --mix self           # 5 copies of the agent

Seats are shuffled per match, so seat position does not matter. Dice depend on the seed alone, so every agent
sees the same dice for a given seed. Reports P&L per round (mean +- standard error), and the share of pairwise
games won, which is what the Elo rating uses.
"""
import argparse
import random
import statistics
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import dicemarket as dm

ROOT = Path(__file__).resolve().parent
FILES = {
    "fair_value": "examples/fair_value.py",
    "wide": "examples/wide_quoter.py",
    "random": "examples/random_quoter.py",
    "house_fair_value": "house/house_fair_value.py",
    "house_wide": "house/house_wide_market_maker.py",
    "house_reader": "house/house_reader.py",
    "house_hider": "house/house_hider.py",
}
MIXES = {
    "weak": ["fair_value", "fair_value", "wide", "random"],
    "readers": ["house_reader", "house_reader", "house_hider", "house_fair_value"],
    "mixed": ["house_reader", "house_hider", "fair_value", "house_wide"],
    "hard": ["house_reader", "house_reader", "house_hider", "house_hider"],
    "self": ["me", "me", "me", "me"],
}


def _path(name, me):
    return str(me) if name == "me" else str(ROOT / FILES[name])


def one_match(args):
    me, names, seed, rounds = args
    rng = random.Random(f"order:{seed}")
    order = ["me"] + list(names)
    rng.shuffle(order)
    res = dm.play_match([_path(n, me) for n in order], rounds=rounds, seed=seed)
    mine = [i for i, n in enumerate(order) if n == "me"]
    i = mine[0]
    pnl = res[i]["pnl_per_round"]
    wins = {}
    for j, n in enumerate(order):
        if j != i:
            d = pnl - res[j]["pnl_per_round"]
            wins.setdefault(n, []).append(1.0 if d > 0 else 0.5 if d == 0 else 0.0)
    return pnl, wins, res[i]["errors"], res[i]["timeouts"]


def run(me, mix, n, rounds=200, seed0=0, pool=None):
    names = MIXES[mix]
    jobs = [(me, names, seed0 + k, rounds) for k in range(n)]
    out = list(pool.map(one_match, jobs)) if pool else [one_match(j) for j in jobs]
    pnls = [o[0] for o in out]
    wins, errs = {}, sum(o[2] + o[3] for o in out)
    for o in out:
        for k, v in o[1].items():
            wins.setdefault(k, []).extend(v)
    allw = [x for v in wins.values() for x in v]
    se = statistics.stdev(pnls) / len(pnls) ** 0.5 if len(pnls) > 1 else 0.0
    return statistics.mean(pnls), se, statistics.mean(allw), {k: statistics.mean(v) for k, v in wins.items()}, errs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("agent")
    ap.add_argument("--mix", default="all")
    ap.add_argument("-n", type=int, default=60)
    ap.add_argument("--rounds", type=int, default=200)
    ap.add_argument("--seed", type=int, default=1000)
    a = ap.parse_args()
    mixes = list(MIXES) if a.mix == "all" else [a.mix]
    with ProcessPoolExecutor() as pool:
        for m in mixes:
            mean, se, win, per, errs = run(Path(a.agent).resolve(), m, a.n, a.rounds, a.seed, pool)
            detail = "  ".join(f"{k} {v:.2f}" for k, v in sorted(per.items()))
            print(f"{m:8s} P&L/round {mean:+.3f} +- {se:.3f}   win rate {win:.3f}   errors {errs}   [{detail}]")


if __name__ == "__main__":
    main()
