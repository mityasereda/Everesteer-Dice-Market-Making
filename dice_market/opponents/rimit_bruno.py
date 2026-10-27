"""Imitation of "bruno" fitted from the replay dataset (analysis/fit_opponents.py). Do not edit by hand.

A static copy of the agent's average behaviour: it does not learn during a match the way the real agent may.
"""
import random

LABEL = 'bruno'
FIT = {"p0": 0.0, "steps": [{"w": [0.00147, 1.37979], "sd": 2.4568, "n": 62000, "widths": {"4.0": 1.0}}, {"w": [-0.0102, 1.22628, 0.53988, -0.29736, 1.00644], "sd": 2.0633, "n": 62000, "widths": {"4.0": 1.0}}, {"w": [-0.05747, 0.51932, -0.02423, 0.12826, 0.3353, -1.42374], "sd": 1.9271, "n": 62000, "widths": {"2.5": 0.01174, "3.0": 0.13126, "3.5": 0.09329, "4.0": 0.76371}}, {"w": [0.02872, 0.28413, 0.05048, -0.0534, 0.14345, 0.54776, -0.24238], "sd": 1.0432, "n": 62000, "widths": {"1.0": 0.01111, "2.0": 0.88703, "3.0": 0.02797, "4.0": 0.07389}}]}


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
