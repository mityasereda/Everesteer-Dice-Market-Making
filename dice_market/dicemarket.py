"""Dice Market Making, game v3: engine and local match runner. See AGENTS.md for the rules.

Play a local match (2 to 6 agents):
    python dicemarket.py agent.py examples/fair_value.py examples/wide_quoter.py examples/random_quoter.py
Options: --rounds N (default 200), --seed S (default 0)
Check your agent before you submit:
    python dicemarket.py --check agent.py

The game, in short:
- 12 dice are rolled each round. The value is their sum (mean 42). Seat i sees dice i and i + 6.
  With 5 seats, dice 5 and 11 are seen by nobody.
- 4 quoting steps per round. Each step, every seat must send a bid and an ask for 1 unit, at most 4 apart.
  Prices are rounded to 0.5. A missing or invalid quote is replaced by 40 / 44 (the long-run mean, 4 wide).
- The quotes rest in one book that clears at the end of the round. Each step, in this order:
  1. take: new orders that cross a resting quote from an earlier step trade at the RESTING price. Your old quote
     is still live here, so a stale quote can be picked off before your new one replaces it;
  2. replace: your new quote replaces your old one;
  3. cross: new orders that cross each other trade at the midpoint.
- There is no noise trader. Every point one seat wins, another seat loses.
- Positions settle at the round's value and reset to zero. Limit: 5 units long or short within a round.
- After each round, every agent learns the value and all 12 dice.
- An agent may ship weights.npz next to agent.py: the engine loads it into the module global WEIGHTS.

The tournament uses exactly this engine, with 5 seats per table.
"""
import argparse
import copy
import importlib.util
import random
import sys
import time
import traceback
from pathlib import Path

N_DICE = 12
MEAN = 3.5 * N_DICE     # 42
PRICE_RANGE = (float(N_DICE), 6.0 * N_DICE)  # 12 to 72
ROUNDS = 200            # rounds per match
STEPS = 4               # quoting steps per round
MAX_WIDTH = 4.0         # ask - bid at most this
TICK = 0.5              # prices are rounded to this, so a quote cannot carry hidden digits
POSITION_LIMIT = 5      # max |position| within a round
DEFAULT_QUOTE = (MEAN - MAX_WIDTH / 2, MEAN + MAX_WIDTH / 2)  # posted for a seat that fails to quote
CALL_TIME_LIMIT = 0.05  # seconds; a slower call is ignored and counted as a timeout


def seen_dice(seat):
    """The dice a seat sees: {die index: value} is built from these indices."""
    return (seat, seat + 6)


def load_bot(path, seat, n_seats):
    """Import the agent (class Agent) from a file under a unique module name.

    If weights.npz sits next to the file, it is loaded into the module global WEIGHTS (a dict of arrays).
    """
    spec = importlib.util.spec_from_file_location(f"dicemarket_bot_{seat}_{abs(hash(str(path)))}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    weights = Path(path).with_name("weights.npz")
    if weights.exists():  # after the module ran, so its own "WEIGHTS = None" line does not overwrite them
        import numpy as np
        with np.load(weights, allow_pickle=False) as z:
            mod.WEIGHTS = {k: z[k] for k in z.files}
    return mod.Agent(seat, n_seats)


class Seat:
    """Wraps one agent: times every call and turns exceptions and slow calls into 'no answer'."""

    def __init__(self, bot, name):
        self.bot, self.name = bot, name
        self.errors = self.timeouts = 0
        self.first_error = None

    def call(self, method, *args):
        t0 = time.perf_counter()
        try:
            out = getattr(self.bot, method)(*args)
        except Exception:
            self.errors += 1
            if self.first_error is None:
                self.first_error = traceback.format_exc(limit=3)
            return None
        if time.perf_counter() - t0 > CALL_TIME_LIMIT:
            self.timeouts += 1
            return None
        return out


def clean_quote(q):
    """An agent's answer to quote(): (bid, ask) inside the price range, bid < ask, at most MAX_WIDTH apart.

    Prices are rounded to TICK first. Anything else (None, an error, a slow call, a stopped agent, a bad number,
    too wide) becomes DEFAULT_QUOTE.
    """
    try:
        bid, ask = round(float(q[0]) / TICK) * TICK, round(float(q[1]) / TICK) * TICK
    except (TypeError, ValueError, IndexError, KeyError, OverflowError):
        return DEFAULT_QUOTE
    lo, hi = PRICE_RANGE
    if not (lo <= bid < ask <= hi) or ask - bid > MAX_WIDTH + 1e-9:  # a NaN fails the first test
        return DEFAULT_QUOTE
    return bid, ask


def match_step(book, orders, rng, pos, step):
    """One step on the resting book. Every order is for 1 unit. Returns a list of trades.

    book: {seat: {"bid": (price, step posted) | None, "ask": ...}}, changed in place.
    orders: {seat: (bid, ask)}. pos: {seat: position}, read for the limits.
    1) take: in random seat order, a new bid takes the best resting ask of another seat at or below it, at the
       resting price (a new ask likewise takes the best resting bid). Resting quotes are the ones from earlier steps.
    2) replace: the new quote replaces the seat's old one; unfilled new sides start resting.
    3) cross: resting orders that now cross trade at the older order's price, or at the midpoint if both are new.
    A fill that would take a seat past the position limit does not happen; that resting side is removed.
    """
    trades = []

    def room(seat, side):
        return pos[seat] < POSITION_LIMIT if side == "buy" else pos[seat] > -POSITION_LIMIT

    def fill(buyer, seller, price, kind):
        trades.append({"buyer": buyer, "seller": seller, "price": round(price, 6), "kind": kind})
        pos[buyer] += 1
        pos[seller] -= 1

    takers = list(orders)
    rng.shuffle(takers)
    left = {s: {"bid": o[0], "ask": o[1]} for s, o in orders.items()}
    for s in takers:  # 1) take
        b = left[s]["bid"]
        if room(s, "buy"):
            asks = sorted((q["ask"][0], rng.random(), t) for t, q in book.items() if t != s and q["ask"] and q["ask"][0] <= b)
            for price, _, t in asks:
                if room(t, "sell"):
                    fill(s, t, price, "take"); book[t]["ask"] = None; left[s]["bid"] = None
                    break
                book[t]["ask"] = None
        a = left[s]["ask"]
        if room(s, "sell"):
            bids = sorted((-q["bid"][0], rng.random(), t) for t, q in book.items() if t != s and q["bid"] and q["bid"][0] >= a)
            for negp, _, t in bids:
                if room(t, "buy"):
                    fill(t, s, -negp, "take"); book[t]["bid"] = None; left[s]["ask"] = None
                    break
                book[t]["bid"] = None
    for s in orders:  # 2) replace
        book[s] = {"bid": None if left[s]["bid"] is None else (left[s]["bid"], step),
                   "ask": None if left[s]["ask"] is None else (left[s]["ask"], step)}
    while True:  # 3) cross
        bids = sorted((-q["bid"][0], rng.random(), t) for t, q in book.items() if q["bid"])
        asks = sorted((q["ask"][0], rng.random(), t) for t, q in book.items() if q["ask"])
        hit = next(((bt, at) for _, _, bt in bids for _, _, at in asks
                    if bt != at and book[bt]["bid"][0] >= book[at]["ask"][0]), None)
        if hit is None:
            return trades
        bt, at = hit
        (bp, bstep), (ap, astep) = book[bt]["bid"], book[at]["ask"]
        if not room(bt, "buy"):
            book[bt]["bid"] = None
            continue
        if not room(at, "sell"):
            book[at]["ask"] = None
            continue
        price = (bp + ap) / 2 if bstep == astep else (bp if bstep < astep else ap)
        fill(bt, at, price, "cross"); book[bt]["bid"] = None; book[at]["ask"] = None


def public_book(book):
    """The resting quotes everyone sees: {seat: (bid, ask)} with None for an empty side."""
    return {s: (q["bid"][0] if q["bid"] else None, q["ask"][0] if q["ask"] else None)
            for s, q in book.items() if q["bid"] or q["ask"]}


def play_match(bot_paths, rounds=ROUNDS, seed=0, log=False, trace=None):
    """Play one match. Returns per-seat results: name, total P&L, P&L per round, trades, errors, timeouts.

    If trace is a list, the full record of each round is appended to it, for replays and tests.
    """
    rng = random.Random(seed)                 # matching ties
    dice_rng = random.Random(f"dice:{seed}")  # the dice depend on the seed alone, never on the orders
    n = len(bot_paths)
    seats = [Seat(load_bot(p, i, n), Path(p).stem) for i, p in enumerate(bot_paths)]
    total = [0.0] * n
    n_trades = [0] * n
    for r in range(rounds):
        dice = [dice_rng.randint(1, 6) for _ in range(N_DICE)]
        value = sum(dice)
        pos = {i: 0 for i in range(n)}
        cash = [0.0] * n
        for i, s in enumerate(seats):
            s.call("new_round", r, {d: dice[d] for d in seen_dice(i)})
        history = []
        book = {i: {"bid": None, "ask": None} for i in range(n)}  # clears every round
        for step in range(STEPS):
            orders = {i: clean_quote(s.call("quote", step, copy.deepcopy(history))) for i, s in enumerate(seats)}
            trades = match_step(book, orders, rng, pos, step)
            for t in trades:
                cash[t["buyer"]] -= t["price"]
                cash[t["seller"]] += t["price"]
                n_trades[t["buyer"]] += 1
                n_trades[t["seller"]] += 1
            history.append({"sent": orders, "quotes": public_book(book), "trades": trades})
            if log and r == 0:
                print(f"round 0 step {step}: trades {trades}")
        pnl = [cash[i] + pos[i] * value for i in range(n)]
        for i in range(n):
            total[i] += pnl[i]
        result = {"value": value, "dice": list(dice), "pnl": pnl, "history": history}
        for s in seats:
            s.call("end_round", copy.deepcopy(result))
        if trace is not None:
            trace.append(dict(copy.deepcopy(result), positions=[pos[i] for i in range(n)]))
    return [{"seat": i, "name": s.name, "pnl": total[i], "pnl_per_round": total[i] / rounds, "trades": n_trades[i],
             "errors": s.errors, "timeouts": s.timeouts, "first_error": s.first_error}
            for i, s in enumerate(seats)]


def check(path):
    """Pre-submission check: load, play 50 rounds against the example agents, report problems."""
    here = Path(__file__).resolve().parent / "examples"
    others = [here / "fair_value.py", here / "wide_quoter.py", here / "random_quoter.py", here / "fair_value.py"]
    t0 = time.perf_counter()
    res = play_match([path] + others, rounds=50, seed=12345)
    me = res[0]
    print(f"{me['name']}: P&L per round {me['pnl_per_round']:+.3f}, trades {me['trades']}, "
          f"errors {me['errors']}, timeouts {me['timeouts']}, {time.perf_counter() - t0:.1f}s for 50 rounds")
    if me["first_error"]:
        print("first error:\n" + me["first_error"])
    ok = me["errors"] == 0 and me["timeouts"] == 0
    print("CHECK PASSED" if ok else "CHECK FAILED: fix errors/timeouts before submitting")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bots", nargs="+")
    ap.add_argument("--rounds", type=int, default=ROUNDS)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log", action="store_true")
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    if a.check:
        sys.exit(0 if check(a.bots[0]) else 1)
    if not 2 <= len(a.bots) <= 6:
        sys.exit("play with 2 to 6 agents")
    res = play_match(a.bots, a.rounds, a.seed, a.log)
    for r in res:
        print(f"seat {r['seat']} {r['name']:20s} P&L/round {r['pnl_per_round']:+8.3f}  trades {r['trades']:5d}  "
              f"errors {r['errors']}  timeouts {r['timeouts']}")


if __name__ == "__main__":
    main()
