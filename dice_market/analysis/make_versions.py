"""Rebuild interim versions of my agent as opponents: opponents/ver_v0.py .. ver_v3.py

    python3 analysis/make_versions.py agent.py opponents

The original files of the early versions no longer exist, so each is the CURRENT agent.py with that version's
parameters (the learning maths is the same; the first version also used numpy for the solve, which was too slow).
  v0  showed its dice at step 0, house-reader prior, lam 30, half widths (2, 2, 1.5, 1.5), skew 0.40
  v1  hides at step 0 (otherwise v0)
  v2  v1 with the pooled prior fitted on real matches and lam 100
  v3  v2 with the step-2 half width 1.0 and skew 0.25
  v4  v3 with the step-3 half width 1.0
The current agent.py (v5) differs from v4 only by lam 30 instead of 100.
"""
import re
import sys
from pathlib import Path

VERSIONS = {
    "v0": dict(half=(2.0, 2.0, 1.5, 1.5), skew=0.40, lam=30.0, hide0=False, prior=False),
    "v1": dict(half=(2.0, 2.0, 1.5, 1.5), skew=0.40, lam=30.0, hide0=True, prior=False),
    "v2": dict(half=(2.0, 2.0, 1.5, 1.5), skew=0.40, lam=100.0, hide0=True, prior=True),
    "v3": dict(half=(2.0, 2.0, 1.0, 1.5), skew=0.25, lam=100.0, hide0=True, prior=True),
    "v4": dict(half=(2.0, 2.0, 1.0, 1.0), skew=0.25, lam=100.0, hide0=True, prior=True),
}
src = Path(sys.argv[1]).read_text()
out = Path(sys.argv[2])


def sub(text, pattern, repl):
    new, k = re.subn(pattern, repl, text, count=1)
    assert k == 1, pattern
    return new


for name, v in VERSIONS.items():
    t = sub(src, r"half=\([^)]*\),", f"half=({', '.join(f'{h:.1f}' for h in v['half'])}),")
    t = sub(t, r"skew=[0-9.]+,", f"skew={v['skew']:.2f},")
    t = sub(t, r"lam=[0-9.]+,", f"lam={v['lam']:.1f},")
    t = sub(t, r"hide0=(True|False),", f"hide0={v['hide0']},")
    if not v["prior"]:
        t = sub(t, r"(?m)^PRIOR = \{.*\}$", "PRIOR = None")
    (out / f"ver_{name}.py").write_text(f"# Interim version {name} of my agent, rebuilt from agent.py by analysis/make_versions.py.\n" + t)
    print("wrote", f"ver_{name}.py", v)
