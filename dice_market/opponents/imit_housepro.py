"""Imitation of "house pro" fitted from the replay dataset (analysis/fit_opponents.py). Do not edit by hand.

A static copy of the agent's average behaviour: it does not learn during a match the way the real agent may.
"""
import random

LABEL = 'house pro'
FIT = {"p0": 1.0, "steps": [{"w": [0.0, 0.0], "sd": 0.0, "n": 0, "widths": {"4.0": 1.0}}, {"w": [0.00436, 1.03042, 0.81573, 0.0, 0.16771], "sd": 0.6591, "n": 992000, "widths": {"2.5": 0.00064, "3.0": 0.14951, "3.5": 0.03528, "4.0": 0.81457}}, {"w": [-0.00048, 1.40731, -0.74063, 0.55323, -0.30195, -0.41538], "sd": 1.9965, "n": 992000, "widths": {"2.0": 0.01856, "2.5": 0.4806, "3.0": 0.49739, "3.5": 0.00345}}, {"w": [-0.00827, 0.04756, 0.10349, -0.07565, 0.09579, 0.75417, -0.00862], "sd": 1.2848, "n": 992000, "widths": {"1.0": 0.55391, "1.5": 0.44567, "2.0": 0.00042}}]}


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
