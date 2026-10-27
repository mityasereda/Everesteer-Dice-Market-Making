"""House wide market maker (game v3): ignores its dice and the table, quotes 40 / 44 every step.

- 40 / 44 is the widest market allowed around the long-run mean (42).
- It never learns anything, so every seat that knows more than the mean trades against it at a profit.
  It is the reference loser: an agent that cannot beat it has not started.
"""
LO, HI = 40.0, 44.0


class Agent:
    def __init__(self, seat, n_seats):
        pass

    def new_round(self, round_no, my_dice):
        pass

    def quote(self, step, history):
        return (LO, HI)

    def end_round(self, result):
        pass
