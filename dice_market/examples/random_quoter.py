"""Example: a random quote near its own expected value, with a random width up to the maximum."""
import random


class Agent:
    def __init__(self, seat, n_seats):
        self.rng = random.Random(seat)

    def new_round(self, round_no, my_dice):
        self.fair = sum(my_dice.values()) + 3.5 * (12 - len(my_dice))

    def quote(self, step, history):
        mid, half = self.fair + self.rng.uniform(-3, 3), self.rng.uniform(0.5, 2)
        return (mid - half, mid + half)

    def end_round(self, result):
        pass
