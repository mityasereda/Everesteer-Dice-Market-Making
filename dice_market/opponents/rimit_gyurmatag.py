"""Imitation of "Gyurmatag" fitted from the replay dataset (analysis/fit_opponents.py). Do not edit by hand.

A static copy of the agent's average behaviour: it does not learn during a match the way the real agent may.
"""
import random

LABEL = 'Gyurmatag'
FIT = {"p0": 0.0227, "steps": [{"w": [0.08642, 1.6865], "sd": 2.8836, "n": 64500, "widths": {"4.0": 1.0}}, {"w": [-0.15253, 1.01635, 0.50244, -0.08234, 0.5678], "sd": 1.8968, "n": 66000, "widths": {"2.5": 0.01035, "3.0": 0.01991, "3.5": 0.0728, "4.0": 0.89694}}, {"w": [-0.12454, 0.56736, 0.02101, 0.14591, 0.32325, -1.02192], "sd": 2.0663, "n": 66000, "widths": {"2.0": 2e-05, "2.5": 0.00689, "3.0": 0.00617, "3.5": 0.0707, "4.0": 0.91623}}, {"w": [-0.05336, 0.30365, 0.04473, -0.05277, 0.17313, 0.41623, -0.18663], "sd": 1.2414, "n": 66000, "widths": {"1.5": 0.0003, "2.0": 0.02009, "2.5": 0.27052, "3.0": 0.05268, "3.5": 0.05988, "4.0": 0.59653}}]}


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
