"""Imitation of "billhwang" fitted from the replay dataset (analysis/fit_opponents.py). Do not edit by hand.

A static copy of the agent's average behaviour: it does not learn during a match the way the real agent may.
"""
import random

LABEL = 'billhwang'
FIT = {"p0": 0.9275, "steps": [{"w": [0.00483, 1.05185], "sd": 0.842, "n": 80930, "widths": {"4.0": 1.0}}, {"w": [0.12942, 0.45377, 0.8752, 0.21097, 0.31118], "sd": 1.5396, "n": 1115800, "widths": {"0.5": 0.07036, "1.0": 0.0329, "1.5": 0.01089, "2.0": 0.03335, "2.5": 0.00793, "3.0": 0.0305, "3.5": 0.00547, "4.0": 0.8086}}, {"w": [0.13536, 1.04466, -0.43923, 0.36926, 0.11608, -1.35705], "sd": 2.7301, "n": 1115800, "widths": {"0.5": 1e-05, "1.0": 6e-05, "1.5": 0.00031, "2.0": 0.01152, "2.5": 0.05656, "3.0": 0.3037, "3.5": 0.25214, "4.0": 0.37571}}, {"w": [-0.0689, 0.42217, 0.10091, -0.07851, 0.14225, 0.60714, -0.41722], "sd": 1.8405, "n": 1115800, "widths": {"0.5": 0.00053, "1.0": 0.75829, "1.5": 0.20565, "2.0": 0.01595, "2.5": 0.00181, "3.0": 5e-05, "3.5": 0.0, "4.0": 0.01771}}]}


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
