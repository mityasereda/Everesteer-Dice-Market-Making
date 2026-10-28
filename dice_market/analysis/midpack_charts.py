"""Figure 6: the agents ranked 15-40 (the ones to beat for the top 20), from the midpack*.json files and step0_levels.txt.

    python3 analysis/midpack_charts.py data ../docs/figures

Colour roles (same palette as charts.py): aqua = top 14, blue = ranks 15-40, gray = ranks 41+, orange = my agent.
"""
import json
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

D, OUT = Path(sys.argv[1]), Path(sys.argv[2])
prof = json.load(open(D / "midpack.json"))
curves = json.load(open(D / "midpack_step0.json"))

SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BLUE, ORANGE, AQUA, GRAY, LIGHTBLUE = "#2a78d6", "#eb6834", "#1baf7a", "#b3b1a8", "#8fbbee"
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
                     "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
                     "text.color": INK, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": INK2,
                     "axes.edgecolor": AXIS, "axes.linewidth": 1.0, "font.size": 10})


def group(l, v):
    if l == "fanaberysty":
        return "me (hider, v7)"
    k = v["rank"]
    if not k:
        return None
    hid = v["default0"] > 0.8
    base = "top 14" if k <= 14 else "15-25" if k <= 25 else "26-40" if k <= 40 else "41+"
    return f"{base} {'hiders' if hid else 'informed'}"


G = {}
for l, v in prof.items():
    g = group(l, v)
    if g:
        G.setdefault(g, []).append((l, v))


def wavg(items, f):
    w = np.array([v["rounds"] for _, v in items], float)
    return float((w * np.array([f(v) for _, v in items], float)).sum() / w.sum())


fig, ax = plt.subplots(2, 2, figsize=(13, 9.6))
fig.subplots_adjust(hspace=0.40, wspace=0.30, left=0.16, right=0.97, top=0.89, bottom=0.09)

# 1. where each group's edge comes from
a = ax[0, 0]
order = ["me (hider, v7)", "41+ informed", "41+ hiders", "26-40 informed", "15-25 informed", "top 14 hiders", "top 14 informed"]
rows = []
for g in order:
    it = G[g]
    s0 = wavg(it, lambda v: v["steps"][0]["edge"][2] + v["steps"][1]["edge"][1])      # cross at step 0 + my step-0 quote picked off at step 1
    tot = wavg(it, lambda v: sum(sum(v["steps"][s]["edge"]) for s in range(4)))
    rows.append((g, s0, tot - s0, tot, len(it)))
y = np.arange(len(rows))
a.barh(y, [r[1] for r in rows], color=ORANGE, height=0.55, label="step-0 quote: crossed at step 0 + picked off at step 1")
a.barh(y, [r[2] for r in rows], left=[r[1] if r[1] * r[2] > 0 else 0.0 for r in rows], color=BLUE, height=0.55, label="everything else: takes and later quotes")
for i, (g, s0, rest, tot, n) in enumerate(rows):
    a.plot([tot], [i], marker="D", color=INK, ms=5, zorder=5)
    a.text(max(s0 + rest if s0 * rest > 0 else rest, s0, tot, 0) + 0.04, i, f"{tot:+.2f}", fontsize=9, color=INK, va="center", fontweight="bold")
a.set_yticks(y)
a.set_yticklabels([f"{r[0]}\n(n={r[4]})" if r[0] != order[0] else r[0] for r in rows], fontsize=8.5)
a.axvline(0, color=AXIS, lw=1)
a.set_xlabel("edge per round; diamond and number = total", labelpad=2)
a.set_title("1. Hiding costs ~1.2 at the start; informed give back most of it later", loc="left", fontsize=10.5, color=INK)
a.legend(loc="upper center", bbox_to_anchor=(0.45, -0.13), fontsize=8, frameon=False, ncol=1)
a.set_xlim(-1.3, 1.65)
a.grid(axis="x", color=GRID, lw=0.6)
a.set_axisbelow(True)

# 2. step-0 quote curves
a = ax[0, 1]
own = np.arange(2, 13)


def pooled(labels):
    n = np.zeros(13); m = np.zeros(13)
    for l in labels:
        c = curves[l]
        for o in range(13):
            n[o] += c["n"][o]; m[o] += (c["bid"][o] + c["ask"][o]) / 2 * c["n"][o]
    return m[2:13] / np.maximum(n[2:13], 1)


OFFS = {"top 14 informed": 0.55, "15-25 informed": -0.45, "26-40 informed": 0.35, "41+ informed": -0.45}
sel = {"top 14 informed": (AQUA, "-"), "15-25 informed": (BLUE, "-"), "26-40 informed": (LIGHTBLUE, "-"), "41+ informed": (GRAY, "-")}
for g, (c, ls) in sel.items():
    labels = [l for l, v in G[g] if v["matches"] >= 100]
    yv = pooled(labels)
    a.plot(own, yv, color=c, lw=2.2, ls=ls)
    a.text(12.15, yv[-1] + OFFS[g], g.replace(" informed", ""), color=INK2, fontsize=8.5, va="center")
a.plot(own, 42 + 0 * own, color=ORANGE, lw=2.2)
a.text(12.15, 42, "me / hiders (40/44)", color=ORANGE, fontsize=8.5, va="center")
a.plot(own, 35 + own, color=MUTED, lw=1.2, ls="--")
a.text(8.2, 37.6, "dashed: fair value given\nmy two dice (35 + sum)", color=MUTED, fontsize=8, va="center")
a.set_xlim(1.8, 14.3)
a.set_xlabel("my own two dice (sum)")
a.set_ylabel("step-0 quote midpoint")
a.set_title("2. Ranks 15-40 quote three levels: low / 40-44 / high", loc="left", fontsize=10.5, color=INK)
a.grid(color=GRID, lw=0.6)
a.set_axisbelow(True)

# 3. push size against Elo
a = ax[1, 0]
pts = []
for l, v in prof.items():
    if not v["rank"] or v["rank"] > 60 or v["matches"] < 150 or l not in curves:
        continue
    c = curves[l]
    lo = [(c["bid"][o] + c["ask"][o]) / 2 for o in range(2, 7)]; wl = [c["n"][o] for o in range(2, 7)]
    hi = [(c["bid"][o] + c["ask"][o]) / 2 for o in range(8, 13)]; wh = [c["n"][o] for o in range(8, 13)]
    push = (np.average(hi, weights=wh) - np.average(lo, weights=wl)) / 2
    pts.append((l, v["rank"], v["elo"], push, v["default0"] > 0.8))
for l, rk, elo, push, hid in pts:
    col = ORANGE if l == "fanaberysty" else AQUA if rk <= 14 else BLUE if rk <= 40 else GRAY
    a.scatter(push, elo, s=44, color=col, marker="s" if hid else "o", edgecolor=SURFACE, lw=0.8, zorder=3)
inf = [(p[3], p[2]) for p in pts if not p[4]]
xs, ys = np.array([q[0] for q in inf]), np.array([q[1] for q in inf])
k, b = np.polyfit(xs, ys, 1)
xx = np.linspace(xs.min(), xs.max(), 20)
a.plot(xx, k * xx + b, color=MUTED, lw=1, ls="--", zorder=1)
a.text(8.9, 1085, f"informed agents: correlation {np.corrcoef(xs, ys)[0, 1]:+.2f}", color=INK2, fontsize=9, ha="right")
a.axhspan(1192, 1368, color=BLUE, alpha=0.07, zorder=0)
a.text(8.9, 1280, "ranks 15-40", color=BLUE, fontsize=8.5, ha="right", alpha=0.9)
for l, rk, elo, push, hid in pts:
    if l in ("fanaberysty", "leobgg", "Rochronaldo", "hermes"):
        a.annotate(l if l != "fanaberysty" else "me", (push, elo), xytext=(5, 4), textcoords="offset points", fontsize=8, color=INK2)
a.set_xlabel("push size at step 0: how far the quote moves from 42 when my dice are not 7")
a.set_ylabel("Elo")
a.set_title("3. The bigger the push, the higher the rank (squares = hiders)", loc="left", fontsize=10.5, color=INK)
a.grid(color=GRID, lw=0.6)
a.set_axisbelow(True)

# 4. exact step-0 search
a = ax[1, 1]
res = {}
for line in open(D / "step0_levels.txt"):
    m = re.match(r"(.+?)\s+([+-]\d\.\d+) \| \s*([+-]\d\.\d+)\s+([+-]\d\.\d+)\s+([+-]\d\.\d+)\s+(\d\.\d+)$", line.strip())
    if m:
        res[m.group(1).strip()] = (float(m.group(5)), float(m.group(6)))
ps = [p for p in (3, 4, 5, 6, 6.5, 7, 8, 9, 10, 12) if f"3 levels P={p:g}" in res]
tot = [res[f"3 levels P={p:g}"][0] for p in ps]
err = [res[f"3 levels P={p:g}"][1] for p in ps]
a.errorbar(ps, tot, yerr=err, color=BLUE, lw=2, marker="o", ms=6, capsize=3)
a.axhline(res["default 40/44"][0], color=ORANGE, lw=2)
a.text(3, res["default 40/44"][0] + 0.07, "default 40/44 (v5, v7)", color=ORANGE, fontsize=9, ha="left")
a.axhline(res["smooth lean k=2.0"][0], color=GRAY, lw=1.6, ls="--")
a.text(12, res["smooth lean k=2.0"][0] + 0.05, "smooth lean k=2 (v6-style)", color=INK2, fontsize=9, ha="right")
a.text(3.0, 0.2, "three levels,\npush P", color=BLUE, fontsize=9, ha="left", va="center")
a.set_xlabel("push size P")
a.set_ylabel("edge per round at steps 0-1")
a.set_title("4. Replay on today's field: P=6 beats the default by ~1.5", loc="left", fontsize=10.5, color=INK)
a.grid(color=GRID, lw=0.6)
a.set_axisbelow(True)

fig.suptitle("The agents ranked 15-40: what they do and what it earns", x=0.07, ha="left", fontsize=14, fontweight="bold", color=INK)
fig.text(0.07, 0.01, "Source: 4,876 leaderboard matches played 28 Oct 2026 00:00-10:36 UTC; ranks from the leaderboard at 10:30 UTC. Panel 4: 976 matches, rivals' recorded quotes held fixed; steps 0-1 only.",
         fontsize=8, color=MUTED)
OUT.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT / "6_ranks_15_to_40.png", dpi=150)
print("saved", OUT / "6_ranks_15_to_40.png")
