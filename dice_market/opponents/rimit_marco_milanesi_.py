"""Imitation of "marco_milanesi_" fitted from the replay dataset (analysis/fit_opponents.py). Do not edit by hand.

A static copy of the agent's average behaviour: it does not learn during a match the way the real agent may.
"""
import random

LABEL = 'marco_milanesi_'
FIT = {"p0": 0.0565, "steps": [{"w": [0.06868, 1.49663], "sd": 2.7749, "n": 70951, "widths": {"4.0": 1.0}}, {"w": [-0.06321, 1.27385, 0.5553, -0.14591, 0.31153], "sd": 1.9741, "n": 75200, "widths": {"1.0": 8e-05, "2.0": 0.00044, "4.0": 0.99948}}, {"w": [0.01819, 0.43432, -0.03433, 0.13933, 0.30859, -1.08946], "sd": 1.791, "n": 75200, "widths": {"1.0": 7e-05, "2.0": 0.00049, "3.0": 0.0002, "3.5": 0.25055, "4.0": 0.7487}}, {"w": [0.06007, 0.26743, 0.05644, -0.07649, 0.16169, 0.59248, -0.21618], "sd": 1.0829, "n": 75200, "widths": {"1.0": 0.0775, "2.0": 0.05125, "2.5": 0.52773, "3.0": 0.09242, "3.5": 7e-05, "4.0": 0.25104}}]}


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
