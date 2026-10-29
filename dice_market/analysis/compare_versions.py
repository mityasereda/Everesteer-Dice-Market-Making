"""Detailed comparison of my versions on real leaderboard matches (all matches under my handle, from the bulk dataset).

    python3 analysis/compare_versions.py data/mine_all.jsonl.gz data/recent_tight.jsonl.gz
Groups: v5 first 25 matches, v5 later matches, v6 (25 matches, 22:33-22:50 UTC). For each group: result, strength of the
opponents (their pairwise win rate in the recent window), step 0 / steps 1-3 edge by role, fills, and the edge per fill.
"""
import gzip
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import replays  # noqa: E402

ME = "fanaberysty"
V6_FROM, V6_TO = "2026-10-27T22:33:26", "2026-10-27T22:50:30"
V7_FROM = "2026-10-28T08:13:48"
V8_FROM = "2026-10-28T11:27:44"

# opponent strength proxy: each label's pairwise win rate in the recent window
strength = defaultdict(lambda: [0.0, 0])
with gzip.open(sys.argv[2], "rt") as f:
    for line in f:
        m = json.loads(line)
        pn = [s["pnl_per_round"] for s in m["seats"]]
        for i, l in enumerate(m["labels"]):
            for j in range(5):
                if j != i:
                    strength[l][1] += 1; strength[l][0] += 1.0 if pn[i] > pn[j] else 0.5 if pn[i] == pn[j] else 0.0
rate = {l: a / b for l, (a, b) in strength.items() if b >= 200}

rows = []
with gzip.open(sys.argv[1], "rt") as f:
    for line in f:
        rows.append(replays.expand(json.loads(line)))
rows.sort(key=lambda m: m["at"])
v5 = [m for m in rows if m["at"] < V6_FROM]
v7 = [m for m in rows if V7_FROM <= m["at"] < V8_FROM]
v8 = [m for m in rows if m["at"] >= V8_FROM]
groups = {"v5, first 25 matches": v5[:25], "v5, matches 26-192": v5[25:], "v6, all 25 matches": [m for m in rows if V6_FROM <= m["at"] < V6_TO],
          "v7, first 25 matches": v7[:25], "v7, all matches": v7,
          "v8, first 25 matches": v8[:25], "v8, matches 26-100": v8[25:100], "v8, matches 101+": v8[100:], "v8, all matches": v8}


def roles(steps, s):
    prev = steps[s - 1]["quotes"] if s else {}
    for t in steps[s]["trades"]:
        if t["kind"] == "cross":
            yield t, "cross", "cross"
            continue
        sr = (prev.get(t["seller"]) or (None, None))[1] == t["price"]
        br = (prev.get(t["buyer"]) or (None, None))[0] == t["price"]
        pair = ("taker", "resting") if sr and not br else ("resting", "taker") if br and not sr else ("?", "?")
        yield t, pair[0], pair[1]


def analyse(ms):
    me_e = defaultdict(float); fl_e = defaultdict(float); me_n = defaultdict(int)
    nr = nfl = 0; ranks = []; wins = games = 0; pnl = 0.0; opp = []; informed = []; ph = defaultdict(lambda: [0.0, 0])
    absend = 0.0
    for m in ms:
        i = m["labels"].index(ME)
        pn = [s["pnl_per_round"] for s in m["seats"]]
        ranks.append(1 + sum(1 for j in range(5) if pn[j] > pn[i])); pnl += pn[i]
        for j in range(5):
            if j != i:
                games += 1; wins += pn[i] > pn[j]
                if m["labels"][j] in rate: opp.append(rate[m["labels"][j]])
        for ri, r in enumerate(m["rounds"]):
            v = r["value"]; nr += 1; nfl += sum(1 for j in range(5) if j != i and m["kinds"][j] == "candidate")
            informed.append(sum(1 for j in range(5) if j != i and tuple(r["steps"][0]["sent"][j]) != (40.0, 44.0)) / 4)
            absend += abs(r["positions"][i])
            tot = 0.0
            for s in range(4):
                for t, rb, rs in roles(r["steps"], s):
                    for who, role, sign in ((t["buyer"], rb, 1), (t["seller"], rs, -1)):
                        e = sign * (v - t["price"])
                        if who == i:
                            me_e[(s, role)] += e; me_n[(s, role)] += 1; tot += e
                        elif m["kinds"][who] == "candidate":
                            fl_e[(s, role)] += e
            b = ph["0-29" if ri < 30 else "30-99" if ri < 100 else "100-199"]; b[0] += tot; b[1] += 1
    return dict(n=len(ms), nr=nr, nfl=nfl, rank=sum(ranks) / len(ms), win=wins / games, pnl=pnl / len(ms), opp=sum(opp) / len(opp),
                inf=sum(informed) / len(informed), me=me_e, fl=fl_e, cnt=me_n, ph=ph, absend=absend / nr)


res = {k: analyse(v) for k, v in groups.items() if v}
print(f"{'':24s}" + "".join(f"{k:>22s}" for k in res))
for lab, key, fmt in (("matches", "n", "{:d}"), ("average rank", "rank", "{:.2f}"), ("pairwise win rate", "win", "{:.3f}"), ("P&L per round", "pnl", "{:+.2f}"),
                      ("opponent strength*", "opp", "{:.3f}"), ("rivals informed at step 0", "inf", "{:.2f}"), ("mean |end position|", "absend", "{:.2f}")):
    print(f"{lab:24s}" + "".join(f"{fmt.format(r[key]):>22s}" for r in res.values()))
print("  * mean pairwise win rate of the rivals in the recent window (0.50 = average)\n")
for step_label, steps in (("step 0", (0,)), ("steps 1-3", (1, 2, 3))):
    print(f"edge per round, {step_label}:   me / average candidate")
    for role in ("taker", "resting", "cross"):
        line = f"  {role:22s}"
        for r in res.values():
            me = sum(r["me"][(s, role)] for s in steps) / r["nr"]; fl = sum(r["fl"][(s, role)] for s in steps) / r["nfl"]
            line += f"{me:+9.2f} /{fl:+6.2f}  "
        print(line)
    line = f"  {'total':22s}"
    for r in res.values():
        me = sum(r["me"][(s, ro)] for s in steps for ro in ("taker", "resting", "cross", "?")) / r["nr"]
        fl = sum(r["fl"][(s, ro)] for s in steps for ro in ("taker", "resting", "cross", "?")) / r["nfl"]
        line += f"{me:+9.2f} /{fl:+6.2f}  "
    print(line)
print("\nedge per round by step (total over roles), me / average candidate:")
for s in range(4):
    line = f"  step {s:17d}"
    for r in res.values():
        me = sum(r["me"][(s, ro)] for ro in ("taker", "resting", "cross", "?")) / r["nr"]; fl = sum(r["fl"][(s, ro)] for ro in ("taker", "resting", "cross", "?")) / r["nfl"]
        line += f"{me:+9.2f} /{fl:+6.2f}  "
    print(line)
print("\nmy fills per round and edge per fill, steps 1-3:")
for role in ("taker", "resting"):
    line = f"  {role:22s}"
    for r in res.values():
        c = sum(r["cnt"][(s, role)] for s in (1, 2, 3)); e = sum(r["me"][(s, role)] for s in (1, 2, 3))
        line += f"{c / r['nr']:8.2f} fills {e / max(c, 1):+6.2f}  "
    print(line)
print("\nmy P&L-style edge per round by phase of the match:")
for p in ("0-29", "30-99", "100-199"):
    print(f"  rounds {p:11s}" + "".join(f"{r['ph'][p][0] / r['ph'][p][1]:+22.2f}" for r in res.values()))
