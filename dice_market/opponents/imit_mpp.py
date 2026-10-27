"""Imitation of "MPP" fitted from the replay dataset (analysis/fit_opponents.py). Do not edit by hand.

A static copy of the agent's average behaviour: it does not learn during a match the way the real agent may.
"""
import random

LABEL = 'MPP'
FIT = {"p0": 0.8184, "steps": [{"w": [-0.00069, 0.1782], "sd": 1.1458, "n": 215048, "widths": {"3.5": 0.50336, "4.0": 0.49664}}, {"w": [0.00396, 1.15717, 0.85127, -0.17404, -0.33129], "sd": 1.5726, "n": 1184400, "widths": {"0.5": 0.00301, "1.0": 0.00556, "1.5": 0.00948, "2.0": 0.01945, "2.5": 0.0287, "3.0": 0.04013, "3.5": 0.04136, "4.0": 0.85231}}, {"w": [-0.00063, 1.13518, -0.73768, 0.45393, 0.00854, -1.25252], "sd": 2.1752, "n": 1184400, "widths": {"0.5": 0.00028, "1.0": 0.00106, "1.5": 0.00599, "2.0": 0.07491, "2.5": 0.05225, "3.0": 0.17738, "3.5": 0.48092, "4.0": 0.20721}}, {"w": [-0.02221, -0.02208, 0.0956, -0.09948, 0.12962, 0.71583, -0.44996], "sd": 1.6532, "n": 1184400, "widths": {"0.5": 0.02422, "1.0": 0.37504, "1.5": 0.40913, "2.0": 0.06357, "2.5": 7e-05, "3.0": 0.02884, "4.0": 0.09913}}]}


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
