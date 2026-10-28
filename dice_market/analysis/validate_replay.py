"""Baseline check of analysis/replay_policy.py: replaying my real v5 on its real matches must reproduce the real edges.

    python3 analysis/validate_replay.py [matches=60]
"""
import json
import statistics
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import replay_policy as rp  # noqa: E402
import replays  # noqa: E402


def main():
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else 60
    rows = rp.select_rows("data/mine_all.jsonl.gz", "mine", limit)
    real = [0.0] * 4
    n = 0
    for line, seat in rows:
        m = replays.expand(json.loads(line))
        for r in m["rounds"]:
            v = r["value"]
            n += 1
            for s, st in enumerate(r["steps"]):
                for t in st["trades"]:
                    if t["buyer"] == seat:
                        real[s] += v - t["price"]
                    if t["seller"] == seat:
                        real[s] += t["price"] - v
    with ProcessPoolExecutor() as pool:
        res = rp.evaluate("opponents/ver_v5.py", {"off": {}}, rows, pool)["off"]
    sim = [statistics.mean(x[s] for x in res) for s in range(4)]
    print(f"{len(rows)} real v5 matches, {n} rounds")
    print("  real edge per round by step:     " + "  ".join(f"{x / n:+.3f}" for x in real) + f"   total {sum(real) / n:+.3f}")
    print("  replayed edge per round by step: " + "  ".join(f"{x:+.3f}" for x in sim) + f"   total {sum(sim):+.3f}")


if __name__ == "__main__":
    main()
