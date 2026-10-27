"""Imitation of "Joshua_Karat" fitted from the replay dataset (analysis/fit_opponents.py). Do not edit by hand.

A static copy of the agent's average behaviour: it does not learn during a match the way the real agent may.
"""
import random

LABEL = 'Joshua_Karat'
FIT = {"p0": 0.9785, "steps": [{"w": [-0.0182, 1.4961], "sd": 2.1238, "n": 6294, "widths": {"4.0": 1.0}}, {"w": [0.03831, 1.07825, 1.21883, -0.07685, 0.0207], "sd": 5.7722, "n": 292200, "widths": {"2.0": 0.17722, "4.0": 0.82278}}, {"w": [0.02097, 0.94206, -0.78134, 0.42959, 0.04949, -0.49618], "sd": 3.7105, "n": 292200, "widths": {"2.0": 0.31416, "4.0": 0.68584}}, {"w": [-0.045, 0.79239, -0.04618, 0.02276, 0.17637, 0.13906, -0.96452], "sd": 2.0013, "n": 292200, "widths": {"0.5": 0.01088, "1.0": 0.31183, "2.0": 0.02389, "4.0": 0.6534}}]}


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
