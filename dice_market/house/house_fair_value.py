"""House fair value (game v3): quotes +-2 around the expected value from its own two dice. Ignores the table.

- Expected value: my two dice plus 3.5 for each of the other 10.
- A new bid at fair - 2 takes any resting ask at or below it (at the ask's price), so rival quotes more than
  2 through my estimate are taken. That is the only way it reacts to the others.
- Its quote shows its two dice to anyone who reads it: mid - 35 is their sum.
"""
N_DICE, HALF = 12, 2.0


class Agent:
    def __init__(self, seat, n_seats):
        self.fair = 42.0

    def new_round(self, round_no, my_dice):
        try:
            self.fair = sum(float(v) for v in my_dice.values()) + 3.5 * (N_DICE - len(my_dice))
        except Exception:
            self.fair = 42.0

    def quote(self, step, history):
        return (self.fair - HALF, self.fair + HALF)

    def end_round(self, result):
        pass
