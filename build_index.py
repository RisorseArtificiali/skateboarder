#!/usr/bin/env python3
"""Rebuild results/index.json from the metadata sidecars in results/.

Run after adding new results (manually or via CI): the web page consumes
index.json. Validates each SVG exists and carries the metadata fields.
"""
import glob
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")

REQUIRED = ("benchmark", "prompt", "model", "timestamp", "endpoint")
entries = []
problems = []

for mpath in sorted(glob.glob(os.path.join(RESULTS, "*.json"))):
    if os.path.basename(mpath) == "index.json":
        continue  # our own output
    with open(mpath) as f:
        meta = json.load(f)
    name = os.path.basename(mpath).removesuffix(".json")
    for k in REQUIRED:
        if k not in meta:
            problems.append(f"{name}: missing '{k}'")
    svg_path = os.path.join(RESULTS, f"{name}.svg")
    svg_file = f"results/{name}.svg"
    if meta.get("file") and not os.path.exists(os.path.join(HERE, meta["file"])):
        problems.append(f"{name}: referenced SVG missing: {meta['file']}")
    entry = {"id": name, **{k: meta.get(k) for k in (
        "benchmark", "prompt", "model", "endpoint", "timestamp", "params",
        "usage", "reasoning_tokens", "finish_reason", "svg_len", "has_animation",
        "uses_js", "error", "seconds", "contributor", "cost_usd", "note")}, "svg": svg_file if os.path.exists(svg_path) else None}
    entries.append(entry)

usage = os.path.join(HERE, "results", "index.json")
with open(usage, "w") as f:
    json.dump({"version": 1, "count": len(entries), "entries": entries}, f, indent=2)
print(f"index.json: {len(entries)} entries, {len(problems)} problems")
for p in problems:
    print("  !", p)
sys.exit(1 if problems else 0)
