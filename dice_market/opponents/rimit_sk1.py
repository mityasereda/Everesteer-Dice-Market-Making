"""Imitation of "SK1" fitted from the replay dataset (analysis/fit_opponents.py). Do not edit by hand.

A static copy of the agent's average behaviour: it does not learn during a match the way the real agent may.
"""
import random

LABEL = 'SK1'
FIT = {"p0": 0.1569, "steps": [{"w": [0.0155, 1.68636], "sd": 2.5704, "n": 41314, "widths": {"4.0": 1.0}}, {"w": [-0.03561, 1.43192, 0.57268, -0.35725, 1.03895], "sd": 3.4398, "n": 49000, "widths": {"0.5": 0.00012, "1.0": 8e-05, "1.5": 0.00018, "2.0": 8e-05, "2.5": 0.00039, "3.0": 0.00039, "3.5": 0.00031, "4.0": 0.99845}}, {"w": [-0.19971, 0.54713, -0.01545, 0.12715, 0.36186, -1.54018], "sd": 2.1602, "n": 49000, "widths": {"2.0": 0.00161, "2.5": 0.08847, "3.0": 0.155, "3.5": 0.0892, "4.0": 0.66571}}, {"w": [0.07758, 0.81765, 0.05647, -0.06943, 0.19751, 0.31276, -0.8694], "sd": 1.842, "n": 49000, "widths": {"0.5": 0.09353, "1.0": 0.13133, "1.5": 0.11802, "2.0": 0.08102, "2.5": 0.06782, "3.0": 0.03596, "3.5": 0.07851, "4.0": 0.39382}}]}


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
