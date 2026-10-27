"""Charts of how the top agents play, from data/profile.json (made by analysis/profile.py).

    python3 analysis/charts.py data/profile.json ../docs/figures

Colour roles (dataviz palette, light mode): blue = top candidates, orange = my agent, grays = house agents and context.
"""
import json
import statistics
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

prof = json.load(open(sys.argv[1]))
OUT = Path(sys.argv[2])
OUT.mkdir(parents=True, exist_ok=True)
M, Q, E, DEF = prof["match"], prof["Q"], prof["E"], prof["default0"]

SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BLUE, ORANGE, AQUA, GRAY, LIGHTGRAY = "#2a78d6", "#eb6834", "#1baf7a", "#b3b1a8", "#d9d8d1"
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
                     "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
                     "text.color": INK, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": INK2,
                     "axes.edgecolor": AXIS, "axes.linewidth": 1.0, "font.size": 10})
SOURCE = "Source: 56,160 leaderboard matches, 23-27 Oct 2026. Quote-level figures use every 8th match."

# the 12 strongest candidates with a real sample, by pairwise win rate
cands = [(l, d) for l, d in M.items() if d["kind"] == "candidate" and d["n"] >= 1500]
cands.sort(key=lambda kv: -kv[1]["wins"] / kv[1]["games"])
TOP = [l for l, _ in cands[:12]]
HOUSE = ["house pro", "house reader", "house hider", "house fair value"]


def win(l):
    return 100 * M[l]["wins"] / M[l]["games"]


def pnl(l):
    return M[l]["pnl"] / M[l]["n"]


def frame(title, sub, figsize=(8.4, 5.2)):
    fig, ax = plt.subplots(figsize=figsize)
    fig.subplots_adjust(left=0.2, right=0.95, top=0.80, bottom=0.12)
    fig.text(0.02, 0.945, title, fontsize=13.5, fontweight="bold", color=INK, ha="left", va="center")
    fig.text(0.02, 0.885, sub, fontsize=9.5, color=INK2, ha="left", va="center")
    fig.text(0.02, 0.02, SOURCE, fontsize=7.5, color=MUTED, ha="left", va="center")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    return fig, ax


def hbars(ax, labels, values, colors, fmt, xlim, note=None):
    y = range(len(labels))
    ax.barh(list(y), values, height=0.58, color=colors, zorder=3)
    ax.set_yticks(list(y))
    ax.set_yticklabels(labels, fontsize=9.5)
    ax.invert_yaxis()
    ax.set_xlim(*xlim)
    ax.xaxis.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", length=0, labelsize=8.5)
    for i, v in enumerate(values):
        ax.text(v + (xlim[1] - xlim[0]) * 0.012, i, fmt(i, v), va="center", fontsize=8.8, color=INK2,
                bbox=dict(facecolor=SURFACE, edgecolor="none", pad=1.5), zorder=5)


# 1. who wins -----------------------------------------------------------------------------------------------------
names = TOP + ["house pro"]
vals = [win(l) for l in names]
fig, ax = frame("The best candidates win about two of every three pairwise games",
                "Share of pairwise games won (higher profit than the other seat), agents with 1,500+ matches")
hbars(ax, names, vals, [BLUE] * len(TOP) + [GRAY], lambda i, v: f"{v:.0f}%   {pnl(names[i]):+.2f}/round", (40, 80))
ax.axvline(50, color=MUTED, linewidth=1, linestyle=(0, (4, 3)), zorder=2)
ax.set_ylim(len(names) - 0.5, -1.0)
ax.text(50.4, -0.62, "coin flip (50%)", fontsize=8, color=MUTED, va="center")
fig.text(0.95, 0.885, "blue = candidate, gray = house", fontsize=8.5, color=INK2, ha="right", va="center")
fig.savefig(OUT / "1_who_wins.png", dpi=170)
plt.close(fig)

# 2. step-0 hiding ------------------------------------------------------------------------------------------------
names = TOP + ["house pro", "house hider", "house reader"]
vals = [100 * DEF.get(l, 0) / Q[l][0][0] for l in names]
fig, ax = frame("Almost every strong agent hides its dice at step 0",
                "Share of step-0 quotes that are exactly the default 40 / 44, which shows nothing about the dice")
hbars(ax, names, vals, [BLUE] * len(TOP) + [GRAY] * 3, lambda i, v: f"{v:.0f}%", (0, 112))
ax.set_xticks([0, 25, 50, 75, 100])
fig.text(0.95, 0.885, "blue = candidate, gray = house", fontsize=8.5, color=INK2, ha="right", va="center")
fig.savefig(OUT / "2_step0_hiding.png", dpi=170)
plt.close(fig)

# 3. estimation error by step -------------------------------------------------------------------------------------
def rmse(l, s):
    q = Q[l][s]
    return (q[7] / q[0]) ** 0.5


steps = [0, 1, 2, 3]
fig, ax = frame("By step 3 the best agents estimate the value to within about 3",
                "RMSE of each agent's quote midpoint against the true value, by step (lower is better)",
                figsize=(8.4, 5.4))
fig.subplots_adjust(left=0.09, right=0.72, top=0.80, bottom=0.14)
for l in TOP:
    ax.plot(steps, [rmse(l, s) for s in steps], color=LIGHTGRAY, linewidth=1.1, zorder=2, clip_on=True)
med = [statistics.median(rmse(l, s) for l in TOP) for s in steps]
mine = [5.46, 4.98, 3.54, 3.01]  # my agent's internal estimate, later rounds (analysis/passive.py)
ax.plot(steps, med, color=BLUE, linewidth=2.4, marker="o", markersize=6, markeredgecolor=SURFACE, markeredgewidth=1.5, zorder=4)
ax.plot(steps, mine, color=ORANGE, linewidth=2.4, marker="o", markersize=6, markeredgecolor=SURFACE, markeredgewidth=1.5, zorder=5)
ax.axhline(5.40, color=MUTED, linewidth=1, linestyle=(0, (4, 3)), zorder=1)
ax.axhline(2.42, color=MUTED, linewidth=1, linestyle=(0, (4, 3)), zorder=1)
ax.text(3.04, 5.40, "own two dice only\n(5.40)", fontsize=8.3, color=INK2, va="center")
ax.text(3.04, 2.42, "floor: all 10 visible\ndice known (2.42)", fontsize=8.3, color=INK2, va="center")
ax.text(3.04, med[3] + 0.18, f"top-12 median ({med[3]:.2f})", fontsize=8.8, color=INK, va="center", fontweight="bold")
ax.text(3.04, mine[3] - 0.22, f"my agent v1 ({mine[3]:.2f})", fontsize=8.8, color=INK, va="center", fontweight="bold")
ax.set_ylim(2, 8)
ax.set_xlim(-0.1, 3.1)
ax.set_xticks(steps)
ax.set_xticklabels([f"step {s}" for s in steps])
ax.yaxis.grid(True, color=GRID, linewidth=0.8, zorder=0)
ax.set_axisbelow(True)
ax.tick_params(length=0, labelsize=8.8)
ax.tick_params(axis="y", colors=MUTED)
fig.text(0.02, 0.062, "Gray lines: the 12 individual agents; two peak above 7.9 at step 1 (one reaches 13.9). My line is my internal estimate in rounds 30+; at step 0 I still quote 40 / 44.", fontsize=7.2, color=MUTED, ha="left")
fig.savefig(OUT / "3_estimation_error.png", dpi=170)
plt.close(fig)

# 4. where the edge is --------------------------------------------------------------------------------------------
def edge(l, s):
    n = Q[l][0][0]
    return (E[l][s][1] + E[l][s][3]) / n


groups = [("Top 12 candidates (average)", BLUE, [statistics.mean(edge(l, s) for l in TOP) for s in steps]),
          ("house pro", GRAY, [edge("house pro", s) for s in steps]),
          ("House reader, hider, fair value (average)", "#5f5e5a",
           [statistics.mean(edge(l, s) for l in ("house reader", "house hider", "house fair value")) for s in steps])]
fig, ax = frame("The money is made at steps 2 and 3, and lost there too",
                "Trading edge per round by step: buys earn value - price, sells earn price - value (summed over trades)",
                figsize=(8.8, 5.4))
fig.subplots_adjust(left=0.08, right=0.97, top=0.80, bottom=0.20)
w = 0.26
for k, (name, color, vals) in enumerate(groups):
    xs = [s + (k - 1) * (w + 0.02) for s in steps]
    ax.bar(xs, vals, width=w, color=color, zorder=3, label=name)
    for x, v in zip(xs, vals):
        if abs(v) > 0.04:
            ax.text(x, v + (0.05 if v >= 0 else -0.05), f"{v:+.2f}", ha="center", va="bottom" if v >= 0 else "top",
                    fontsize=7.8, color=INK2)
ax.axhline(0, color=AXIS, linewidth=1.2, zorder=4)
ax.set_xticks(steps)
ax.set_xticklabels([f"step {s}" for s in steps])
ax.set_ylim(-2.35, 1.0)
ax.yaxis.grid(True, color=GRID, linewidth=0.8, zorder=0)
ax.set_axisbelow(True)
ax.spines["left"].set_visible(False)
ax.tick_params(length=0, labelsize=8.8)
ax.tick_params(axis="y", colors=MUTED)
leg = ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=3, frameon=False, fontsize=8.6, handlelength=1.2)
for t in leg.get_texts():
    t.set_color(INK2)
fig.savefig(OUT / "4_where_the_edge_is.png", dpi=170)
plt.close(fig)

# 5. final-step spread --------------------------------------------------------------------------------------------
def spread3(l):
    q = Q[l][3]
    return q[1] / q[0]


items = [(l, spread3(l), BLUE) for l in TOP] + [("house pro", spread3("house pro"), GRAY), ("my agent v1", 3.0, ORANGE)]
items.sort(key=lambda t: t[1])
fig, ax = frame("Several top agents quote about 1 wide at the last step; my v1 quotes 3",
                "Mean full width (ask - bid) of the step-3 quote, maximum allowed is 4", figsize=(8.4, 5.4))
hbars(ax, [t[0] for t in items], [t[1] for t in items], [t[2] for t in items], lambda i, v: f"{v:.2f}", (0, 4.4))
ax.set_xticks([0, 1, 2, 3, 4])
fig.text(0.95, 0.885, "blue = candidate, gray = house, orange = mine", fontsize=8.5, color=INK2, ha="right", va="center")
fig.savefig(OUT / "5_final_step_spread.png", dpi=170)
plt.close(fig)
print("wrote", sorted(p.name for p in OUT.glob("*.png")))
