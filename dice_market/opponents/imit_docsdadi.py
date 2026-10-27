"""Imitation of "Docsdadi" fitted from the replay dataset (analysis/fit_opponents.py). Do not edit by hand.

A static copy of the agent's average behaviour: it does not learn during a match the way the real agent may.
"""
import random

LABEL = 'Docsdadi'
FIT = {"p0": 1.0, "steps": [{"w": [0.0, 0.0], "sd": 0.0, "n": 0, "widths": {"4.0": 1.0}}, {"w": [0.00605, 1.05759, 0.84682, 0.0, 0.09006], "sd": 1.1533, "n": 969400, "widths": {"2.0": 0.00042, "2.5": 0.11942, "3.0": 0.10939, "3.5": 0.07802, "4.0": 0.69275}}, {"w": [0.02395, 0.81192, -0.99017, 0.40876, 0.34651, -0.97744], "sd": 2.4806, "n": 969400, "widths": {"1.5": 0.00051, "2.0": 0.10467, "2.5": 0.17802, "3.0": 0.12816, "3.5": 0.58731, "4.0": 0.00133}}, {"w": [-0.01443, 0.12238, 0.05263, -0.05992, 0.10572, 0.67654, -0.27879], "sd": 1.3998, "n": 969400, "widths": {"0.5": 0.04409, "1.0": 0.95591}}]}


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
