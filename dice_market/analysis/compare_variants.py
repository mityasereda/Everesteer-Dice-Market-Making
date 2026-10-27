"""Compare spread settings on fresh seeds and on mixes that were not used for tuning.

    python3 analysis/compare_variants.py agent.py [n=150]
"""
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "analysis"))
import tourney  # noqa: E402
from tune_spreads import variant  # noqa: E402

CANDIDATES = {
    "v1 baseline  (2,2,1.5,1.5 skew .40)": ([2.0, 2.0, 1.5, 1.5], 0.40),
    "step2 tight  (2,2,1.0,1.5 skew .25)": ([2.0, 2.0, 1.0, 1.5], 0.25),
    "step2+3 tight(2,2,1.0,1.0 skew .25)": ([2.0, 2.0, 1.0, 1.0], 0.25),
    "top-agent like(2,2,1.0,.75 skew .25)": ([2.0, 2.0, 1.0, 0.75], 0.25),
}
HELD_OUT = ["gate_adapt", "pro_adapt", "fieldB_adapt", "gate_imit", "fieldB_imit", "readers"]
def main():
    src = Path(sys.argv[1]).read_text()
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 150
    with ProcessPoolExecutor() as pool:
        print(f"{'variant':38s}" + "".join(f"{m:>14s}" for m in HELD_OUT) + "   (win rate / P&L per round)")
        for i, (name, (h, s)) in enumerate(CANDIDATES.items()):
            path = variant(src, h, s, f"cmp{i}")
            rs = {m: tourney.run(path, m, n, 200, 11000, pool) for m in HELD_OUT}
            print(f"{name:38s}" + "".join(f"{rs[m]['win']:7.3f}/{rs[m]['pnl']:+.2f}" for m in HELD_OUT), flush=True)
            if "gate_adapt" in rs:
                b = rs["gate_adapt"]["per"]["adapt_housepro"]
                print(f"{'':38s}gate vs adaptive boss: P(beat boss) {b:.2f}  P(pass 2 of 3) {3 * b ** 2 - 2 * b ** 3:.2f}   "
                      f"silent-forced {sum(r['silent'] for r in rs.values())}  errors {sum(r['errors'] for r in rs.values())}", flush=True)


if __name__ == "__main__":
    main()
