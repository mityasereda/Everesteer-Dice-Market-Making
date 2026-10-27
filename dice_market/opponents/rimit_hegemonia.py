"""Imitation of "Hegemonia" fitted from the replay dataset (analysis/fit_opponents.py). Do not edit by hand.

A static copy of the agent's average behaviour: it does not learn during a match the way the real agent may.
"""
import random

LABEL = 'Hegemonia'
FIT = {"p0": 0.431, "steps": [{"w": [-0.01842, 1.40782], "sd": 2.7503, "n": 54278, "widths": {"4.0": 1.0}}, {"w": [0.02786, 1.09579, 0.57392, 0.06531, -0.48334], "sd": 1.6185, "n": 95400, "widths": {"4.0": 1.0}}, {"w": [0.00753, 0.68009, -0.02197, 0.14648, 0.27232, -1.07255], "sd": 2.1013, "n": 95400, "widths": {"4.0": 1.0}}, {"w": [-0.08675, 0.21858, 0.03426, -0.04943, 0.13677, 0.55318, -0.25713], "sd": 1.2081, "n": 95400, "widths": {"1.0": 0.07752, "2.0": 0.86461, "4.0": 0.05787}}]}


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
