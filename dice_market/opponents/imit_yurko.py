"""Imitation of "yurko_mud" fitted from the replay dataset (analysis/fit_opponents.py). Do not edit by hand.

A static copy of the agent's average behaviour: it does not learn during a match the way the real agent may.
"""
import random

LABEL = 'yurko_mud'
FIT = {"p0": 0.8959, "steps": [{"w": [-0.02575, 1.06525], "sd": 1.9099, "n": 32237, "widths": {"4.0": 1.0}}, {"w": [-0.05326, 1.27708, 0.85721, -0.05088, -0.51006], "sd": 2.8407, "n": 309600, "widths": {"0.5": 0.01582, "1.0": 0.01115, "1.5": 0.00449, "2.0": 0.00368, "2.5": 0.00214, "3.0": 0.08154, "3.5": 0.06866, "4.0": 0.81251}}, {"w": [-0.09563, 0.98133, -0.43223, 0.28423, 0.12792, -1.02905], "sd": 2.7635, "n": 309600, "widths": {"1.5": 6e-05, "2.0": 0.02034, "2.5": 0.1688, "3.0": 0.13645, "3.5": 0.10349, "4.0": 0.57085}}, {"w": [0.00552, 0.23161, -0.0274, -0.00234, 0.12201, 0.51538, -0.65631], "sd": 1.5924, "n": 309600, "widths": {"1.0": 0.04222, "1.5": 0.48397, "2.0": 0.14802, "2.5": 0.22464, "3.0": 0.0642, "3.5": 0.01819, "4.0": 0.01876}}]}


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
