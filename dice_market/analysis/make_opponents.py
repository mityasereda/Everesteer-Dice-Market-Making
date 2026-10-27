"""Write one imitation agent file per fitted real agent: opponents/imit_<name>.py

    python3 analysis/make_opponents.py opponents/fits.json opponents

Each file replays the fit of analysis/fit_opponents.py: the quote mid is the fitted linear function of the agent's
dice, the others' mids at earlier steps, its own previous mid and its position. Mids are deterministic (no injected
noise: the fit residual holds real information the fit missed, so noise would only create exploitable mispricers).
Spread widths and the step-0 default 40 / 44 are drawn from the real agent's distributions with a seeded generator,
exactly two draws per quote call, so variants of my agent are compared on identical opponent behaviour.
"""
import json
import sys
from pathlib import Path

NAMES = {"Docsdadi": "docsdadi", "zudvarnoki": "zudvarnoki", "MPP": "mpp", "billhwang": "billhwang",
         "momarci13": "momarci13", "yurko_mud": "yurko", "Joshua_Karat": "joshua", "leobgg": "leobgg",
         "house pro": "housepro"}

TEMPLATE = '''"""Imitation of "{label}" fitted from the replay dataset (analysis/fit_opponents.py). Do not edit by hand.

A static copy of the agent's average behaviour: it does not learn during a match the way the real agent may.
"""
import random

LABEL = {label!r}
FIT = {fit}


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
'''

fits = json.load(open(sys.argv[1]))
out = Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)
for label, short in NAMES.items():
    (out / f"imit_{short}.py").write_text(TEMPLATE.format(label=label, fit=json.dumps(fits[label])))
print("wrote", ", ".join(f"imit_{s}.py" for s in NAMES.values()))
