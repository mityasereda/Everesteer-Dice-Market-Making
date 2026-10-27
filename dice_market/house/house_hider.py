"""House hider (game v3): shows nothing at step 0, then reads the table like the house reader.

- Step 0: it quotes 40 / 44 around the long-run mean, so its first quote shows nothing of its dice
  (a reader that takes first quotes at face value reads it as an average pair).
- Steps 1 to 3: it reads every other seat's step-0 mid as its two-dice sum (mid - 35, clipped to 2..12), adds
  its own two dice and 3.5 per unseen die, and quotes +-1 around that. A resting rival quote more than 1 through
  it is taken at its price.
- The price of hiding: at step 0 its 40 / 44 can meet an informed new quote at the midpoint.
"""
N_DICE, MEAN, HALF0, HALF = 12, 42.0, 2.0, 1.0


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
        for s, q in history[0]["sent"].items():
            if int(s) != self.seat and q:
                rest += min(12.0, max(2.0, _mid(q) - 35.0))
                shown += 2
        return self.mine + rest + 3.5 * (N_DICE - shown)

    def quote(self, step, history):
        try:
            if not history:
                return (MEAN - HALF0, MEAN + HALF0)
            c = self.estimate(history)
            return (c - HALF, c + HALF)
        except Exception:
            return None

    def end_round(self, result):
        pass
