"""Imitation of "kart7" fitted from the replay dataset (analysis/fit_opponents.py). Do not edit by hand.

A static copy of the agent's average behaviour: it does not learn during a match the way the real agent may.
"""
import random

LABEL = 'kart7'
FIT = {"p0": 0.4833, "steps": [{"w": [0.01497, 1.63993], "sd": 2.5407, "n": 29454, "widths": {"4.0": 1.0}}, {"w": [-0.00499, 2.11727, 0.54524, -0.20555, -1.41158], "sd": 4.1309, "n": 57000, "widths": {"0.5": 0.00544, "1.0": 0.00667, "1.5": 0.00981, "2.0": 0.00584, "2.5": 0.01563, "3.0": 0.00705, "3.5": 0.00753, "4.0": 0.94204}}, {"w": [0.00812, 0.88841, -0.1684, 0.20073, 0.18637, -1.27907], "sd": 3.2876, "n": 57000, "widths": {"0.5": 0.00505, "1.0": 0.007, "1.5": 0.00654, "2.0": 0.00593, "2.5": 0.00909, "3.0": 0.00507, "3.5": 0.00482, "4.0": 0.95649}}, {"w": [-0.00333, 0.3736, -0.02038, -0.04701, 0.18566, 0.37366, -0.34126], "sd": 1.7345, "n": 57000, "widths": {"0.5": 0.08344, "1.0": 0.04923, "1.5": 0.02498, "2.0": 0.01109, "2.5": 0.00493, "3.0": 0.00172, "3.5": 0.00126, "4.0": 0.82335}}]}


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
