"""Train the linear quoting policy of agent.py with evolution strategies (numpy only) in dicesim.

Run:  python train.py [generations, default 120]      Output: weights.npz next to this file.
The default takes about 1 minute. A fixed number of generations gives the same weights on every machine.
Each generation tests POP perturbed copies of the weights. Each copy plays its own group of tables against the
simple example agents. The weights move toward the copies with a high P&L. The policy code is shared with
agent.py. This shows the method only: the result beats fair value, not the boss.
To go further, read the dicesim docstring: self-play, your own frozen weights as opponents (add_opponent), the
house baselines, and any agent file as an opponent.
"""
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))  # dicesim.py sits two folders up
from dicesim import DiceMarketEnv  # noqa: E402
from agent import BASE_THETA, features, policy  # noqa: E402

POP, GROUP, SIGMA, LR, ROUNDS = 64, 64, 0.1, 0.05, 25  # copies, tables per copy, noise size, step size, rounds
OPPONENTS = ["fair_value", "fair_value", "wide", "random"]


def act(theta, obs):
    """Quotes (n, 2) of the policy for a dicesim observation."""
    f = features(obs["dice"].sum(1).astype(float), obs["position"])
    return np.stack(policy(theta, obs["step"], f), -1)


def play(thetas, n_per, seed, rounds=ROUNDS):
    """Play len(thetas) * n_per tables, seats shuffled per table. Returns the mean P&L per round of each theta."""
    rng = np.random.default_rng(seed)
    opp = rng.permuted(np.tile(OPPONENTS, (n_per, 1)), axis=1)  # the same tables for every theta
    env = DiceMarketEnv(len(thetas) * n_per, rounds=rounds, seed=seed, opponents=np.tile(opp, (len(thetas), 1)))
    th = np.repeat(thetas, n_per, axis=0)
    obs, done, pnl = env.reset(), False, 0.0
    while not done:
        obs, r, done, _ = env.step(act(th, obs))
        pnl = pnl + r
    return pnl.reshape(len(thetas), n_per).mean(1) / rounds


def main(generations=120):
    rng = np.random.default_rng(1)
    theta = BASE_THETA.copy()
    t0 = time.time()
    for gen in range(1, generations + 1):
        eps = rng.standard_normal((POP // 2, *theta.shape))
        eps = np.concatenate([eps, -eps])  # antithetic pairs reduce the noise
        pnl = play(theta + SIGMA * eps, GROUP, seed=1000 + gen)  # every copy sees the same dice and opponents
        score = (pnl.argsort().argsort() / (POP - 1)) - 0.5  # rank scores resist outliers
        theta = theta + LR / (SIGMA * POP) * np.tensordot(score, eps, 1)
        if gen % 20 == 0:
            print(f"gen {gen:4d}  mean P&L/round of the copies {pnl.mean():+.3f}  ({time.time() - t0:.0f}s)")
    np.savez(HERE / "weights.npz", theta=theta)
    base, new = play(np.stack([BASE_THETA, theta]), 2048, seed=999_999, rounds=100)  # held-out seed
    print(f"held-out vs {', '.join(OPPONENTS)}: fair value +-2 {base:+.3f}  learned {new:+.3f}  P&L/round")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 120)
