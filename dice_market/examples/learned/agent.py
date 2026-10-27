"""Example: a learned quoting policy for the real engine. It uses its own dice and position only.

The policy is linear, with one set of weights per step. Features: 1, my two dice - 7, my position.
Outputs: the centre of my quote minus 42, and the log of the half width (0.5 to 2).
It does not read the other seats' quotes. Adding features that read the table is your job.
The engine loads weights.npz (key "theta", shape (STEPS, 2, N_FEATURES)) into WEIGHTS. If WEIGHTS is None, the
agent quotes fair value +-2. train.py trains the weights in dicesim and imports features() and policy()
from this file, so the simulator and the engine use the same code.
"""
import numpy as np

WEIGHTS = None
STEPS, N_FEATURES = 4, 3
BASE_THETA = np.zeros((STEPS, 2, N_FEATURES))
BASE_THETA[:, 0, 1] = 1.0          # centre 42 + my dice - 7: the fair value
BASE_THETA[:, 1, 0] = np.log(2.0)  # half width 2


def features(my_sum, pos):
    """my_sum, pos (n,): my two dice, my position. Returns (n, N_FEATURES)."""
    return np.stack([np.ones_like(my_sum, dtype=float), my_sum - 7.0, pos], -1)


def policy(theta, step, feats):
    """Quotes from weights. theta (n, STEPS, 2, F) or (STEPS, 2, F). Returns bid, ask, each (n,)."""
    out = np.einsum("...f,...cf->...c", feats, theta[..., step, :, :])
    centre, half = 42.0 + out[..., 0], np.exp(np.clip(out[..., 1], np.log(0.5), np.log(2.0)))
    return centre - half, centre + half


class Agent:
    def __init__(self, seat, n_seats):
        self.theta = None if WEIGHTS is None else np.asarray(WEIGHTS["theta"], float)
        self.seat = seat

    def new_round(self, round_no, my_dice):
        self.my_sum = float(sum(my_dice.values()))

    def quote(self, step, history):
        if self.theta is None:
            return (self.my_sum + 33.0, self.my_sum + 37.0)
        pos = sum((t["buyer"] == self.seat) - (t["seller"] == self.seat) for h in history for t in h["trades"])
        bid, ask = policy(self.theta, min(step, STEPS - 1), features(np.array([self.my_sum]), np.array([float(pos)])))
        return (float(bid[0]), float(ask[0]))

    def end_round(self, result):
        pass
