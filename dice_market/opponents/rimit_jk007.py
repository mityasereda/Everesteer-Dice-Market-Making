"""Imitation of "JK007" fitted from the replay dataset (analysis/fit_opponents.py). Do not edit by hand.

A static copy of the agent's average behaviour: it does not learn during a match the way the real agent may.
"""
import random

LABEL = 'JK007'
FIT = {"p0": 0.1667, "steps": [{"w": [0.00972, 1.54924], "sd": 2.2549, "n": 65329, "widths": {"4.0": 1.0}}, {"w": [0.03616, 1.36707, 0.59117, -0.12135, 0.19345], "sd": 1.738, "n": 78400, "widths": {"3.0": 0.00098, "3.5": 0.00148, "4.0": 0.99754}}, {"w": [0.00341, 0.21144, -0.24781, 0.1405, 0.66244, -1.61241], "sd": 2.5008, "n": 78400, "widths": {"3.0": 0.00093, "3.5": 0.00142, "4.0": 0.99765}}, {"w": [0.00765, 0.29976, 0.04892, -0.03199, 0.1354, 0.48207, -0.4493], "sd": 1.502, "n": 78400, "widths": {"3.0": 0.00066, "3.5": 0.00103, "4.0": 0.9983}}]}


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
