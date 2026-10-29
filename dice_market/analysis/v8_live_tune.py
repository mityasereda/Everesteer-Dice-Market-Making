"""Later-step settings of v8, replayed on my own LIVE v8 matches (rivals' quotes held fixed, but they already reacted to my real
step-0 quote, so changes that reveal nothing new after step 0 are fairly judged here).

    python3 analysis/v8_live_tune.py check                 # baseline: replayed v8 must reproduce its real per-step edges
    python3 analysis/v8_live_tune.py tune [tuning=200]     # first N v8 matches; the remaining ones are the validation set
    python3 analysis/v8_live_tune.py valid
Real v8 per-step edges on the same matches are printed by analysis/v8_matches.py.
"""
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import replay_policy as rp  # noqa: E402

V8_FROM = "2026-10-28T11:27:44"
DATA = "data/mine_all2.jsonl.gz"
N = None
EXTRA = {}
VARIANTS = {"v8 (defaults)": {}}
if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "check"
    tuning = int(sys.argv[2]) if len(sys.argv) > 2 else 200
    if mode == "check":
        rows = rp.select_rows(DATA, "mine", 10 ** 9, since=V8_FROM)
    elif mode == "tune":
        rows = rp.select_rows(DATA, "mine", tuning, since=V8_FROM)
    else:
        rows = rp.select_rows(DATA, "mine", 10 ** 9, since=V8_FROM, skip=tuning)
    exec(open("analysis/v8_variants.py").read(), globals())
    print(f"{len(rows)} real v8 matches ({len(rows) * 200} rounds), mode {mode}")
    with ProcessPoolExecutor() as pool:
        rp.report(rp.evaluate("agent.py", VARIANTS, rows, pool), "v8 (defaults)")
