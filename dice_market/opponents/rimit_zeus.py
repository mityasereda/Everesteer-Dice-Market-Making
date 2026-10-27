"""Imitation of "zeus" fitted from the replay dataset (analysis/fit_opponents.py). Do not edit by hand.

A static copy of the agent's average behaviour: it does not learn during a match the way the real agent may.
"""
import random

LABEL = 'zeus'
FIT = {"p0": 0.0572, "steps": [{"w": [-0.01461, 1.69062], "sd": 3.0405, "n": 89189, "widths": {"4.0": 1.0}}, {"w": [0.04958, 1.53166, 0.61434, -0.08651, -0.23738], "sd": 2.6487, "n": 94600, "widths": {"4.0": 1.0}}, {"w": [0.01206, 0.57919, 0.04009, 0.13654, 0.28063, -1.12627], "sd": 1.9561, "n": 94600, "widths": {"2.0": 0.01577, "2.5": 0.05738, "3.0": 0.08448, "3.5": 0.0733, "4.0": 0.76907}}, {"w": [-0.00704, 0.64499, 0.04498, -0.04488, 0.13708, 0.51552, -0.33452], "sd": 1.461, "n": 94600, "widths": {"0.5": 0.00169, "1.0": 0.01433, "1.5": 0.0446, "2.0": 0.1382, "2.5": 0.15149, "3.0": 0.08147, "3.5": 0.04131, "4.0": 0.5269}}]}


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
