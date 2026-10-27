"""House reader (game v3): reads every seat's first quote as a view of its two dice, then quotes tight.

- At step 0 nobody has seen anything yet, so a seat that quotes its own expected value shows its two dice:
  mid - 35 is their sum. The reader adds those sums (clipped to 2..12) to its own two dice, plus 3.5 for
  each die nobody showed. That is its estimate of the value.
- Step 0: it quotes +-2 around its own expected value (it has nothing to read yet, and shows its dice).
- Steps 1 to 3: it quotes +-1 around the estimate. A resting rival quote more than 1 through it is taken
  at its price by the new quote.
- It takes every first quote at face value: a seat that quotes 40 / 44 at step 0 reads as an average pair (7).
"""
N_DICE, HALF0, HALF = 12, 2.0, 1.0


def _mid(q):
    return (float(q[0]) + float(q[1])) / 2


class Agent:
    def __init__(self, seat, n_seats):
        self.seat, self.n = seat, n_seats
        self.mine, self.k = 7.0, 2

    def new_round(self, round_no, my_dice):
        try:
            self.mine, self.k = sum(float(v) for v in my_dice.values()), len(my_dice)
        except Exception:
            self.mine, self.k = 7.0, 2

    def estimate(self, history):
        """My dice, plus each other seat's step-0 mid - 35, plus 3.5 per unseen die."""
        rest, shown = 0.0, self.k
        for s, q in (history[0]["sent"].items() if history else ()):
            if int(s) != self.seat and q:
                rest += min(12.0, max(2.0, _mid(q) - 35.0))
                shown += 2
        return self.mine + rest + 3.5 * (N_DICE - shown)

    def quote(self, step, history):
        try:
            c = self.estimate(history)
            h = HALF0 if not history else HALF
            return (c - h, c + h)
        except Exception:
            return None

    def end_round(self, result):
        pass
