"""Imitation of "momarci13" fitted from the replay dataset (analysis/fit_opponents.py). Do not edit by hand.

A static copy of the agent's average behaviour: it does not learn during a match the way the real agent may.
"""
import random

LABEL = 'momarci13'
FIT = {"p0": 0.9478, "steps": [{"w": [0.00144, 1.23009], "sd": 1.6828, "n": 44617, "widths": {"4.0": 1.0}}, {"w": [-0.09573, 0.9503, 0.89265, 0.20451, -0.1942], "sd": 2.0547, "n": 854800, "widths": {"0.5": 0.06268, "1.0": 0.01463, "1.5": 0.00866, "2.0": 0.00813, "2.5": 0.03539, "3.0": 0.05116, "3.5": 0.0792, "4.0": 0.74016}}, {"w": [0.07399, 1.07623, -0.7083, 0.35464, 0.21673, -1.40652], "sd": 2.7243, "n": 854800, "widths": {"0.5": 0.0, "1.5": 0.00221, "2.0": 0.41655, "2.5": 0.03978, "3.0": 0.12576, "3.5": 0.12619, "4.0": 0.28949}}, {"w": [-0.01301, 0.31075, 0.04477, -0.05524, 0.14605, 0.53194, -0.37907], "sd": 1.4009, "n": 854800, "widths": {"0.5": 5e-05, "1.0": 0.02118, "1.5": 0.68135, "2.0": 0.22529, "2.5": 0.04855, "3.0": 0.0003, "4.0": 0.02329}}]}


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
