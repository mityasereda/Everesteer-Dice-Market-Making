"""Analyse my real leaderboard matches (replays downloaded by `dmapi.py replay`, in data/mine/*.json).

    python3 analysis/my_matches.py [agent.py] [replay folder=data/mine]

1. Reproduce: feed my agent the recorded history and check it re-sends exactly the quotes it sent live.
2. Where the money goes: edge per round by step and by role (taker / resting / midpoint cross), me vs the field.
3. How accurate was my estimate at each step, vs the other seats' own quote mids.
4. Early vs late rounds of a match (is the learning phase the problem?).
"""
import glob
import importlib.util
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
AGENT = sys.argv[1] if len(sys.argv) > 1 else str(ROOT / "agent.py")


def load_agent():
    spec = importlib.util.spec_from_file_location("mine", AGENT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def norm(step):
    """JSON gives string seat keys: make them ints."""
    return {"sent": {int(k): tuple(v) for k, v in step["sent"].items()},
            "quotes": {int(k): tuple(v) for k, v in step["quotes"].items()},
            "trades": step["trades"]}


def roles(step_idx, steps):
    """For each trade of a step: 'taker' / 'resting' for each side of a take, or 'cross'."""
    prev = steps[step_idx - 1]["quotes"] if step_idx else {}
    out = []
    for t in steps[step_idx]["trades"]:
        if t["kind"] == "cross":
            out.append((t, "cross", "cross"))
            continue
        sell_rest = prev.get(t["seller"], (None, None))[1] == t["price"]
        buy_rest = prev.get(t["buyer"], (None, None))[0] == t["price"]
        if sell_rest and not buy_rest:
            out.append((t, "taker", "resting"))      # buyer took the seller's resting ask
        elif buy_rest and not sell_rest:
            out.append((t, "resting", "taker"))      # seller hit the buyer's resting bid
        else:
            out.append((t, "?", "?"))
    return out


FOLDER = sys.argv[2] if len(sys.argv) > 2 else str(ROOT / "data" / "mine")
files = sorted(glob.glob(str(Path(FOLDER) / "*.json")))
matches = [json.load(open(f)) for f in files]
mod = load_agent()
exact = total = 0
edge_me = defaultdict(float); edge_fld = defaultdict(float); n_me = n_fld = 0
trades_me = trades_fld = 0
se_est = [[0.0, 0] for _ in range(4)]; se_oth = [[0.0, 0] for _ in range(4)]; se_best = [[0.0, 0] for _ in range(4)]
bands = {"rounds 0-29": [0.0, 0], "rounds 30-99": [0.0, 0], "rounds 100-199": [0.0, 0]}
width_me = [[0.0, 0] for _ in range(4)]; width_fld = [[0.0, 0] for _ in range(4)]
for m in matches:
    me = [t["seat"] for t in m["table"] if t["kind"] == "you"][0]
    kinds = {t["seat"]: t["kind"] for t in m["table"]}
    ag = mod.Agent(me, 5)
    for ri, r in enumerate(m["rounds"]):
        d, v = r["dice"], r["value"]
        steps = [norm(s) for s in r["steps"]]
        ag.new_round(ri, {me: d[me], me + 6: d[me + 6]})
        hist = []
        for s, st in enumerate(steps):
            q = ag.quote(s, hist)
            est = ag.estimate(s, hist) if s else None
            total += 1
            exact += (round(q[0] * 2) / 2, round(q[1] * 2) / 2) == tuple(st["sent"][me])
            if s >= 1:
                se_est[s][0] += (v - est) ** 2; se_est[s][1] += 1
            oth = [((st["sent"][j][0] + st["sent"][j][1]) / 2 - v) ** 2 for j in range(5) if j != me and kinds[j] == "candidate"]
            for e in oth:
                se_oth[s][0] += e; se_oth[s][1] += 1
            for j in range(5):
                w = st["sent"][j][1] - st["sent"][j][0]
                tgt = width_me if j == me else width_fld
                if j == me or kinds[j] == "candidate":
                    tgt[s][0] += w; tgt[s][1] += 1
            hist.append(st)
        for s in range(4):
            for t, rb, rs in roles(s, steps):
                for who, role, sign in ((t["buyer"], rb, 1), (t["seller"], rs, -1)):
                    e = sign * (v - t["price"])
                    if who == me:
                        edge_me[(s, role)] += e; trades_me += 1
                    elif kinds[who] == "candidate":
                        edge_fld[(s, role)] += e; trades_fld += 1
        n_me += 1
        n_fld += sum(1 for j in range(5) if j != me and kinds[j] == "candidate")
        band = "rounds 0-29" if ri < 30 else "rounds 30-99" if ri < 100 else "rounds 100-199"
        bands[band][0] += r["pnl"][me]; bands[band][1] += 1
        ag.end_round({"value": v, "dice": d, "pnl": r["pnl"], "history": hist})

print(f"1. reproduction: {exact}/{total} of my live quotes re-created exactly by the current agent.py ({100 * exact / total:.1f}%)")
print(f"\n2. edge per round by step and role   (edge = value - price on buys, price - value on sells; {len(matches)} matches, {n_me} rounds)")
print(f"   {'':16s}" + "".join(f"{'step ' + str(s):>22s}" for s in range(4)))
for role in ("taker", "resting", "cross"):
    row_me = "".join(f"  me {edge_me[(s, role)] / n_me:+6.2f} fld {edge_fld[(s, role)] / n_fld:+6.2f}" for s in range(4))
    print(f"   {role:16s}{row_me}")
tot_me = sum(edge_me.values()) / n_me; tot_f = sum(edge_fld.values()) / n_fld
print(f"   total per round: me {tot_me:+.3f}, average candidate seat {tot_f:+.3f}   | trades per round: me {trades_me / n_me:.2f}, candidate {trades_fld / n_fld:.2f}")
print("\n3. RMSE vs the true value by step: my estimate | the other candidates' quote mids")
print("   " + "   ".join(f"step {s}: {((se_est[s][0] / se_est[s][1]) ** .5 if se_est[s][1] else float('nan')):.2f} | {(se_oth[s][0] / se_oth[s][1]) ** .5:.2f}" for s in range(4)))
print("\n4. my P&L per round by phase of the match:  " + "   ".join(f"{k}: {a / n:+.2f}" for k, (a, n) in bands.items()))
print("\n5. mean quote width (ask - bid) by step: me " + " ".join(f"{width_me[s][0] / width_me[s][1]:.2f}" for s in range(4)) + "  | other candidates " + " ".join(f"{width_fld[s][0] / width_fld[s][1]:.2f}" for s in range(4)))
