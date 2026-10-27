"""Imitation of "leobgg" fitted from the replay dataset (analysis/fit_opponents.py). Do not edit by hand.

A static copy of the agent's average behaviour: it does not learn during a match the way the real agent may.
"""
import random

LABEL = 'leobgg'
FIT = {"p0": 0.9956, "steps": [{"w": [0.0143, 1.33724], "sd": 1.8782, "n": 4179, "widths": {"4.0": 1.0}}, {"w": [0.00609, 1.00015, 0.70846, 0.00796, -0.02703], "sd": 0.6879, "n": 945600, "widths": {"0.5": 0.00029, "1.0": 0.00021, "1.5": 0.00018, "2.0": 0.00018, "2.5": 0.00019, "3.0": 0.03409, "3.5": 0.00044, "4.0": 0.96442}}, {"w": [0.03661, 1.70302, -0.62708, 0.50047, -0.52987, -0.85024], "sd": 2.5533, "n": 945600, "widths": {"0.5": 0.05047, "1.0": 0.02877, "1.5": 0.01843, "2.0": 0.01376, "2.5": 0.01572, "3.0": 0.06472, "3.5": 0.07538, "4.0": 0.73276}}, {"w": [-0.00895, 0.07545, -0.02843, -0.0724, 0.23619, 0.29294, -0.19014], "sd": 2.1275, "n": 945600, "widths": {"0.5": 0.1226, "1.0": 0.04634, "1.5": 0.01827, "2.0": 0.00808, "2.5": 0.00813, "3.0": 0.06357, "3.5": 0.13509, "4.0": 0.59792}}]}


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
