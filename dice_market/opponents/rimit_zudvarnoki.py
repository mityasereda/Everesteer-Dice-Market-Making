"""Imitation of "zudvarnoki" fitted from the replay dataset (analysis/fit_opponents.py). Do not edit by hand.

A static copy of the agent's average behaviour: it does not learn during a match the way the real agent may.
"""
import random

LABEL = 'zudvarnoki'
FIT = {"p0": 0.0414, "steps": [{"w": [0.05693, 1.62785], "sd": 2.9051, "n": 59625, "widths": {"4.0": 1.0}}, {"w": [-0.03067, 1.68787, 0.64955, -0.31115, 0.98941], "sd": 3.9304, "n": 62200, "widths": {"4.0": 1.0}}, {"w": [-0.18681, 0.7088, 0.03371, 0.13509, 0.18159, -1.10948], "sd": 1.9259, "n": 62200, "widths": {"2.5": 0.05746, "3.0": 0.1, "3.5": 0.06479, "4.0": 0.77775}}, {"w": [0.17951, 0.46614, 0.01087, -0.03184, 0.11686, 0.55432, -0.81017], "sd": 1.047, "n": 62200, "widths": {"0.5": 0.03461, "1.0": 0.21326, "1.5": 0.16643, "2.0": 0.08331, "2.5": 0.06887, "3.0": 0.06013, "3.5": 0.04051, "4.0": 0.33286}}]}


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
