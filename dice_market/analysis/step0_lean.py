"""How far should the step-0 quote lean? Full-match results of v6 variants against several opponent populations.

    python3 analysis/step0_lean.py agent.py [n=100]
Variants differ only in hide0 / step0_k (width stays 4). Same seeds for every variant.
Populations: vs_v5 (four identical hiders that read step-0 mids: the worst case for revealing first), radA/radB/radC
(copies of current leaders, who mostly quote informed), gate_rad (the upload test with a hider boss).
"""
import re
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import tourney  # noqa: E402

MIXES = ["vs_v5", "radA", "radB", "radC", "gate_rad"]
VARIANTS = [("hides at step 0", dict(hide0=True)), ("informed k=1.0", dict(hide0=False, step0_k=1.0)),
            ("informed k=1.25", dict(hide0=False, step0_k=1.25)), ("informed k=1.5 (v6)", dict(hide0=False, step0_k=1.5)),
            ("informed k=1.75", dict(hide0=False, step0_k=1.75))]


def build(src, ov, tag):
    t = src
    for k, v in ov.items():
        t, c = re.subn(rf"{k}=[A-Za-z0-9.]+,", f"{k}={v},", t, count=1)
        assert c == 1, k
    d = ROOT / ".tune_variants"; d.mkdir(exist_ok=True)
    (d / f"lean_{tag}.py").write_text(t)
    return d / f"lean_{tag}.py"


def main():
    src = Path(sys.argv[1]).read_text()
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 100
    with ProcessPoolExecutor() as pool:
        print(f"{'variant':24s}" + "".join(f"{m:>22s}" for m in MIXES) + "    (pairwise win / P(1st) / P&L per round)")
        for i, (name, ov) in enumerate(VARIANTS):
            path = build(src, ov, i)
            rs = [tourney.run(path, m, n, 200, 131000, pool) for m in MIXES]
            print(f"{name:24s}" + "".join(f"   {r['win']:.2f}/{r['first']:.2f}/{r['pnl']:+.2f}" for r in rs), flush=True)


if __name__ == "__main__":
    main()
