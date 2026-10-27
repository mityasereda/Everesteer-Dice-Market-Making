"""Example: quotes +-2 around the expected value given its own two dice. Ignores everyone else."""


class Agent:
    def __init__(self, seat, n_seats):
        self.seat = seat

    def new_round(self, round_no, my_dice):
        self.fair = sum(my_dice.values()) + 3.5 * (12 - len(my_dice))

    def quote(self, step, history):
        return (self.fair - 2, self.fair + 2)

    def end_round(self, result):
        pass
