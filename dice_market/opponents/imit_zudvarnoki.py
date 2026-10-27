"""Imitation of "zudvarnoki" fitted from the replay dataset (analysis/fit_opponents.py). Do not edit by hand.

A static copy of the agent's average behaviour: it does not learn during a match the way the real agent may.
"""
import random

LABEL = 'zudvarnoki'
FIT = {"p0": 0.9589, "steps": [{"w": [0.01597, 1.34081], "sd": 2.3847, "n": 36511, "widths": {"4.0": 1.0}}, {"w": [0.01673, 1.1634, 0.96147, 0.24789, -0.36804], "sd": 2.425, "n": 888400, "widths": {"2.5": 0.09823, "3.0": 0.08657, "3.5": 0.07606, "4.0": 0.73914}}, {"w": [0.01419, 1.25246, -0.64015, 0.42093, -0.04502, -1.06362], "sd": 2.6094, "n": 888400, "widths": {"2.5": 0.04982, "3.0": 0.08021, "3.5": 0.07467, "4.0": 0.7953}}, {"w": [-0.02248, 0.14593, 0.07979, -0.06902, 0.12698, 0.61275, -0.21813], "sd": 1.4774, "n": 888400, "widths": {"0.5": 0.00162, "1.0": 0.942, "1.5": 0.00755, "2.0": 0.00379, "2.5": 0.0035, "3.0": 0.00357, "3.5": 0.0025, "4.0": 0.03546}}]}


class Agent:
    def __init__(self, seat, n_seats):
        self.seat = seat
        self.rng = random.Random(LABEL + ":" + str(seat))
        self.own = 7.0

    def new_round(self, round_no, my_dice):
        self.own = float(sum(my_dice.values()))

    def quote(self, step, history):
        u_default, u_width = self.rng.random(), self.rng.random()    # always two draws, so streams stay aligned
        if step == 0 and u_default < FIT["p0"]:
            return (40.0, 44.0)
        f = FIT["steps"][step]
        x = [1.0, self.own - 7.0]
        for t in range(step):
            sent = history[t]["sent"]
            x.append(sum((q[0] + q[1]) / 2.0 - 42.0 for j, q in sent.items() if j != self.seat))
        if step >= 1:
            q = history[step - 1]["sent"][self.seat]
            pos = sum((tr["buyer"] == self.seat) - (tr["seller"] == self.seat) for h in history for tr in h["trades"])
            x.extend([(q[0] + q[1]) / 2.0 - 42.0, float(pos)])
        mid = 42.0 + sum(w * v for w, v in zip(f["w"], x))
        acc, width = 0.0, 4.0
        for w_, p in sorted((float(k), v) for k, v in f["widths"].items()):
            acc += p
            width = w_
            if u_width <= acc:
                break
        bid = round((mid - width / 2.0) * 2.0) / 2.0
        bid = min(max(bid, 12.0), 72.0 - width)
        return (bid, bid + width)

    def end_round(self, result):
        pass
