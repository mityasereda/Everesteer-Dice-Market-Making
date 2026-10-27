"""Imitation of "yurko_mud" fitted from the replay dataset (analysis/fit_opponents.py). Do not edit by hand.

A static copy of the agent's average behaviour: it does not learn during a match the way the real agent may.
"""
import random

LABEL = 'yurko_mud'
FIT = {"p0": 0.2423, "steps": [{"w": [-0.05113, 1.52928], "sd": 2.8384, "n": 39856, "widths": {"4.0": 1.0}}, {"w": [0.02141, 1.5784, 0.54244, -0.06374, -0.64758], "sd": 3.8737, "n": 52600, "widths": {"4.0": 1.0}}, {"w": [-0.12784, 0.93163, -0.01825, 0.15792, 0.19938, -1.3376], "sd": 2.7611, "n": 52600, "widths": {"2.5": 0.05051, "3.0": 0.04137, "3.5": 0.02804, "4.0": 0.88008}}, {"w": [0.02949, 0.40243, 0.06561, -0.02978, 0.14411, 0.41248, -0.52152], "sd": 1.3295, "n": 52600, "widths": {"1.5": 0.75606, "2.0": 0.05721, "2.5": 0.12614, "3.0": 0.0393, "3.5": 0.01148, "4.0": 0.00981}}]}


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
