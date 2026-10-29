"""Keep my dice out of my quotes for longer, but still use them to take: replayed on my 299 real matches as a hider (v5 and v7 era).

    python3 analysis/hide_longer.py
The replay charges the accuracy a masked quote gives up but cannot credit the rivals for staying ignorant of my dice (they do not react
to my different quotes), so a gain here is a floor; a loss here does not settle it.
"""
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import replay_policy as rp  # noqa: E402

OLD = {0: (1.0, []), 1: (1.0, [0.72]), 2: (0.75, [-0.16, 0.26]), 3: (0.26, [0.05, -0.03, 0.26])}
V7 = dict(hide0=True, lam=30.0, clip0=5.0, _prior=OLD)
T, F = True, False
VARIANTS = {
    "v7 (hides step 0 only)": V7,
    "mask step 1": {**V7, "mask": (F, T, F, F)},
    "mask step 2": {**V7, "mask": (F, F, T, F)},
    "mask step 3": {**V7, "mask": (F, F, F, T)},
    "mask steps 1-2": {**V7, "mask": (F, T, T, F)},
    "mask steps 2-3": {**V7, "mask": (F, F, T, T)},
    "mask steps 1-3": {**V7, "mask": (F, T, T, T)},
}
if __name__ == "__main__":
    rows = rp.select_rows("data/mine_all2.jsonl.gz", "mine", 10 ** 9, until="2026-10-28T11:27:44")
    print(f"{len(rows)} real hider-era matches ({len(rows) * 200} rounds)")
    with ProcessPoolExecutor() as pool:
        rp.report(rp.evaluate("agent.py", VARIANTS, rows, pool), "v7 (hides step 0 only)")
