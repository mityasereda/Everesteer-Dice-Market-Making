"""Which change causes the loss to v1? Switch the prior, lam and spreads of the current agent one at a time.

    python3 analysis/ablate.py agent.py [n=150]
"""
import re
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import tourney  # noqa: E402

MIXES = ["vs_v1", "ver_old", "pro_adapt", "fieldB_adapt", "gate_adapt"]
# name: (use pooled prior, lam, half widths)
VARIANTS = {
    "B  pooled prior, lam 30": (True, 30.0, (2.0, 2.0, 1.0, 1.0)),
    "F  pooled prior, lam 50": (True, 50.0, (2.0, 2.0, 1.0, 1.0)),
    "G  pooled prior, lam 15": (True, 15.0, (2.0, 2.0, 1.0, 1.0)),
    "current (pooled prior, lam 100)": (True, 100.0, (2.0, 2.0, 1.0, 1.0)),
}


def build(src, prior, lam, half, tag):
    t = re.sub(r"half=\([^)]*\),", f"half=({', '.join(f'{h:.1f}' for h in half)}),", src, count=1)
    t = re.sub(r"lam=[0-9.]+,", f"lam={lam:.1f},", t, count=1)
    if not prior:
        t = re.sub(r"(?m)^PRIOR = \{.*\}$", "PRIOR = None", t, count=1)
    d = ROOT / ".tune_variants"
    d.mkdir(exist_ok=True)
    (d / f"abl_{tag}.py").write_text(t)
    return d / f"abl_{tag}.py"


def main():
    src = Path(sys.argv[1]).read_text()
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 150
    with ProcessPoolExecutor() as pool:
        print(f"{'variant':50s}" + "".join(f"{m:>20s}" for m in MIXES) + "   (pairwise win / P(1st) / avg rank)")
        for i, (name, (prior, lam, half)) in enumerate(VARIANTS.items()):
            path = build(src, prior, lam, half, i)
            rs = [tourney.run(path, m, n, 200, 41000, pool) for m in MIXES]
            print(f"{name:50s}" + "".join(f"  {r['win']:.3f}/{r['first']:.2f}/{r['rank']:.2f}" for r in rs), flush=True)


if __name__ == "__main__":
    main()
