"""Your Dice Market Making trading agent. Rename to agent.py, keep the class name Agent and the four methods.

Rules that matter here (full rules in AGENTS.md):
- Standard library and numpy only. No files, network, threads or subprocesses.
- No reading the game's internal state while it runs. Reading dicemarket.py itself is fine: it is the game.
- Every call must return within 0.05 s.
- Optional: ship weights.npz next to agent.py. The engine loads it into the module global WEIGHTS (dict of arrays).
"""
N_DICE, MEAN, MAX_WIDTH = 12, 42.0, 4.0   # the value is the sum of 12 dice; your quote is at most 4 wide
WEIGHTS = None                             # set by the engine when weights.npz is present


class Agent:
    def __init__(self, seat, n_seats):
        # seat: your seat number (0 .. n_seats-1) for the whole match. You see dice seat and seat + 6.
        self.seat = seat

    def new_round(self, round_no, my_dice):
        # my_dice: {die index: value}, your two dice this round. All 12 dice are rolled fresh every round.
        self.fair = sum(my_dice.values()) + 3.5 * (N_DICE - len(my_dice))

    def quote(self, step, history):
        # step: 0 .. 3. history: earlier steps this round, each
        #   {"sent": {seat: (bid, ask)}, "quotes": {seat: (bid, ask)}, "trades": [{"buyer", "seller", "price", "kind"}]}
        #   "sent" is what each seat sent that step; "quotes" is the resting book after it (a filled side is None).
        #   kind: "take" (a resting quote was hit at its price) or "cross" (two new orders met at the midpoint).
        # Return (bid, ask), 1 unit each side, ask - bid at most 4, prices from 12 to 72. You must quote every step:
        # a missing or invalid quote becomes 40 / 44. Your quote rests until this new one replaces it, and the old
        # one is still live while others send theirs, so a stale quote can be picked off at its price.
        return (self.fair - 2, self.fair + 2)

    def end_round(self, result):
        # result: {"value": int, "dice": all 12 dice, "pnl": [round P&L per seat], "history": all 4 steps}
        pass
