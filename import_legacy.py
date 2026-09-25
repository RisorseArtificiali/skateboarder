#!/usr/bin/env python3
"""Import existing SVG+metadata from a legacy wave JSON (~/svgs/skateboard/wave*.json)
into the skateboarder repo results/ layout. One-off migration helper."""
import json
import os
import shutil
import sys

SRC = sys.argv[1] if len(sys.argv) > 1 else "/home/maeste/svgs/skateboard"
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results")
os.makedirs(OUT, exist_ok=True)

contributor = input("contributor name [maeste]: ").strip() or "maeste"
total = 0
for wpath in sorted(p for p in os.listdir(SRC) if p.startswith("wave") and p.endswith(".json")):
    with open(os.path.join(SRC, wpath)) as f:
        wave = json.load(f)
    for name, e in wave.items():
        entry = {
            "benchmark": "skateboard",
            "prompt": e.get("variant"),
            "model": e.get("model") or name,
            "endpoint": "https://openrouter.ai/api/v1",
            "timestamp": None,
            "contributor": contributor,
            "params": {"max_tokens": 128000},
        }
        if "error" in e:
            entry["error"] = e["error"]
            entry["finish_reason"] = e.get("finish_reason")
            entry["usage"] = {"total_tokens": e.get("total_tokens")}
            entry["seconds"] = e.get("seconds")
            sid = f"{name} ({wpath})"
        else:
            src_svg = os.path.join(SRC, e["file"])
            if not os.path.exists(src_svg):
                continue
            slug = name.replace(".", "-")
            dst = f"{slug}-{e['variant']}.svg"
            shutil.copy(src_svg, os.path.join(OUT, dst))
            entry["svg"] = f"results/{dst}"
            entry["svg_len"] = e.get("svg_len")
            entry["has_animation"] = e.get("has_animation")
            entry["uses_js"] = e.get("uses_js")
            entry["seconds"] = e.get("seconds")
            entry["usage"] = {"total_tokens": e.get("total_tokens"),
                              "completion_tokens": e.get("completion_tokens"),
                              "prompt_tokens": e.get("prompt_tokens")}
            entry["cost_usd"] = e.get("cost_usd")
            entry["finish_reason"] = e.get("finish_reason")
            if e.get("reasoning_tokens"):
                entry["reasoning_tokens"] = e["reasoning_tokens"]
                entry["note"] = e.get("note")
            sid = dst
        mpath = os.path.join(OUT, sid.replace(".svg", ".json")) if not sid.endswith(")") else os.path.join(OUT, sid.replace(" ", "_").replace("(", "").replace(")", "") + ".json")
        with open(mpath, "w") as f:
            json.dump(entry, f, indent=2)
        total += 1
print(f"imported {total} entries into results/")
