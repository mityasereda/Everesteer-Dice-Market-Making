"""Read the replay dataset (replays.jsonl.gz) into the full match shape described in AGENTS.md.

    import replays
    for match in replays.load("replays.jsonl.gz"):
        match["rounds"][0]["steps"][0]["quotes"]  # the resting book, as in the live `history`

The file stores each match in a compact form (format 2): quote prices as integer half-point ticks, trade prices as
integer quarter points (a cross trades at a midpoint), quotes as lists by seat, the book as the sides that are
gone, trades as [buyer, seller, price, kind], and no profit or position, because both follow from the trades.
expand() rebuilds them exactly. compact() is the inverse, used by the server. Both accept the
old full form (no "f" key), so a file with both forms loads.
"""
import gzip
import json

TICK = 0.5


def _p(x):
    """A price in ticks -> the price; None stays None."""
    return None if x is None else x * TICK


def _t(x):
    return None if x is None else int(round(x / TICK))


def _q(x):
    """A trade price -> quarter points (a cross trades at the midpoint of two tick prices)."""
    return int(round(x * 4))


def compact(match):
    """The compact row of one match in the full shape. Idempotent on a compact row."""
    if match.get("f") == 2:
        return match
    out = {k: v for k, v in match.items() if k != "rounds"}
    out["f"] = 2
    rounds = []
    for r in match["rounds"]:
        n = len(r["pnl"])
        steps = []
        for st in r["steps"]:
            sent = {int(s): q for s, q in st["sent"].items()}
            book = {int(s): q for s, q in st["quotes"].items()}
            gone = []
            for s in range(n):
                q = book.get(s, (None, None))
                for side in (0, 1):
                    if q[side] is None:
                        gone.append([s, side])
            steps.append({"s": [[_t(sent[s][0]), _t(sent[s][1])] for s in range(n)], "x": gone,
                          "t": [[t["buyer"], t["seller"], _q(t["price"]), t["kind"][0]] for t in st["trades"]]})
        rounds.append({"v": r["value"], "d": r["dice"], "st": steps})
    out["rounds"] = rounds
    return out


def expand(row):
    """One match in the full shape (value, dice, pnl, positions, steps with sent, quotes, trades with kind)."""
    if row.get("f") != 2:
        return row
    out = {k: v for k, v in row.items() if k not in ("rounds", "f")}
    rounds = []
    for r in row["rounds"]:
        n = len(r["st"][0]["s"]) if r["st"] else 0
        pos, cash = [0] * n, [0.0] * n
        steps = []
        for st in r["st"]:
            sent = {s: (_p(q[0]), _p(q[1])) for s, q in enumerate(st["s"])}
            gone = {(s, side) for s, side in st["x"]}
            quotes = {}
            for s in range(n):
                q = (None if (s, 0) in gone else sent[s][0], None if (s, 1) in gone else sent[s][1])
                if q[0] is not None or q[1] is not None:
                    quotes[s] = q
            trades = []
            for b, sl, p, kind in st["t"]:
                price = p / 4
                trades.append({"buyer": b, "seller": sl, "price": price, "kind": "take" if kind == "t" else "cross"})
                pos[b] += 1
                pos[sl] -= 1
                cash[b] -= price
                cash[sl] += price
            steps.append({"sent": sent, "quotes": quotes, "trades": trades})
        rounds.append({"value": r["v"], "dice": r["d"], "pnl": [round(cash[i] + pos[i] * r["v"], 6) for i in range(n)],
                       "positions": pos, "steps": steps})
    out["rounds"] = rounds
    return out


def load(path):
    """Every match in a dataset file, in the full shape, in file order."""
    with gzip.open(path, "rt") as f:
        for line in f:
            if line.strip():
                yield expand(json.loads(line))
