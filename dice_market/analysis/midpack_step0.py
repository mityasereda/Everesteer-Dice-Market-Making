"""Step-0 quote curve of every profiled agent: mean bid and mean ask by own dice sum (2..12), plus what share of rounds is asymmetric.

    python3 analysis/midpack_step0.py data/day28.jsonl.gz data/midpack.json data/midpack_step0.json
"""
import gzip
import json
import sys
from collections import defaultdict

import numpy as np

prof = json.load(open(sys.argv[2]))
S = defaultdict(lambda: np.zeros((13, 4)))     # own sum -> n, sum bid, sum ask, sum fair-value gap
with gzip.open(sys.argv[1], "rt") as f:
    for line in f:
        m = json.loads(line)
        for r in m["rounds"]:
            d = r["d"]
            for i, l in enumerate(m["labels"]):
                if l not in prof:
                    continue
                own = d[i] + d[i + 6]
                b, a = r["st"][0]["s"][i]
                x = S[l][own]
                x += (1, b / 2.0, a / 2.0, r["v"] - (b + a) / 4.0)
out = {}
for l, a in S.items():
    out[l] = {"n": a[:, 0].tolist(), "bid": (a[:, 1] / np.maximum(a[:, 0], 1)).tolist(),
              "ask": (a[:, 2] / np.maximum(a[:, 0], 1)).tolist(), "value_minus_mid": (a[:, 3] / np.maximum(a[:, 0], 1)).tolist()}
json.dump(out, open(sys.argv[3], "w"))
print("done")
