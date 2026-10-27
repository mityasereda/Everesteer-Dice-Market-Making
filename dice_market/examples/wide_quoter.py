"""Example: ignores its dice and quotes the widest allowed market around the long-run mean (40 / 44)."""


class Agent:
    def __init__(self, seat, n_seats):
        pass

    def new_round(self, round_no, my_dice):
        pass

    def quote(self, step, history):
        return (40.0, 44.0)

    def end_round(self, result):
        pass
