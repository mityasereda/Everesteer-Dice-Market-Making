"""Collect the most recent matches into one file.

    python3 analysis/make_recent.py CUTOFF out.jsonl.gz in1.jsonl.gz [in2.jsonl.gz ...]
Keeps every match played at or after CUTOFF (ISO time, UTC), de-duplicated by match id.
The field changes fast (the share of default step-0 quotes fell from 95% to 37% in three days), so fits and
opponent copies should use a recent window only.
"""
import gzip
import json
import sys

cutoff, out, ins = sys.argv[1], sys.argv[2], sys.argv[3:]
seen, kept, first, last = set(), 0, None, None
with gzip.open(out, "wt") as w:
    for path in ins:
        with gzip.open(path, "rt") as f:
            for line in f:
                m = json.loads(line)
                if m["at"] < cutoff or m["id"] in seen:
                    continue
                seen.add(m["id"])
                w.write(line)
                kept += 1
                first = min(first or m["at"], m["at"]); last = max(last or m["at"], m["at"])
print(f"kept {kept} matches from {first} to {last}")
