"""Frozen v9 defaults against v7 settings, replayed on my own real matches as a hider (v5 and v7 era, rivals' real reactions included).

    python3 analysis/v9_final.py
"""
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import replay_policy as rp  # noqa: E402

OLD = {0: (1.0, []), 1: (1.0, [0.72]), 2: (0.75, [-0.16, 0.26]), 3: (0.26, [0.05, -0.03, 0.26])}
VARIANTS = {
    "v7 settings": dict(hide_w=4.0, lam=30.0, half=(2.0, 2.0, 2.0, 1.0), take=(None, 0.5, 0.25, 0.25)),
    "v9 (agent.py defaults)": {},
    "v9 with 40/44 only": dict(hide_w=4.0),
}
if __name__ == "__main__":
    rows = rp.select_rows("data/mine_all2.jsonl.gz", "mine", 10 ** 9, until="2026-10-28T11:27:44")
    print(f"{len(rows)} real hider-era matches ({len(rows) * 200} rounds)")
    with ProcessPoolExecutor() as pool:
        rp.report(rp.evaluate("agent.py", VARIANTS, rows, pool), "v7 settings")
