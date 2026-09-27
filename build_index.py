#!/usr/bin/env python3
"""Rebuild results/index.json from the metadata sidecars in results/.

Run after adding new results (manually or via CI): the web page consumes
index.json. Validates each SVG exists and carries the metadata fields.
"""
import glob
import hashlib
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")

REQUIRED = ("benchmark", "prompt", "model", "timestamp", "endpoint")
REASONING_REQUESTS = ("default", "disabled", "effort", "unknown")
REASONING_EFFORTS = ("minimal", "low", "medium", "high", "xhigh", "max")


def slug(value):
    return re.sub(r"[^a-z0-9]+", "-", str(value or "unknown").lower()).strip("-") or "unknown"


def normalize_reasoning(meta):
    reasoning = meta.get("reasoning")
    if isinstance(reasoning, dict) and reasoning.get("requested") in REASONING_REQUESTS:
        normalized = {
            "requested": reasoning["requested"],
            "effort": reasoning.get("effort") if reasoning["requested"] == "effort" else None,
        }
        if isinstance(reasoning.get("provider_default"), dict):
            normalized["provider_default"] = reasoning["provider_default"]
        return normalized

    params = meta.get("params") if isinstance(meta.get("params"), dict) else {}
    if params.get("no_reasoning"):
        return {"requested": "disabled", "effort": None}
    if params.get("reasoning_effort"):
        return {"requested": "effort", "effort": params["reasoning_effort"]}
    return {"requested": "unknown", "effort": None}


def validate_reasoning(meta, name):
    if "reasoning" not in meta:
        return []
    reasoning = meta["reasoning"]
    if not isinstance(reasoning, dict):
        return [f"{name}: reasoning must be an object"]
    requested = reasoning.get("requested")
    if requested not in REASONING_REQUESTS:
        return [f"{name}: invalid requested reasoning: {requested!r}"]
    if "effort" not in reasoning:
        return [f"{name}: reasoning is missing 'effort'"]
    effort = reasoning.get("effort")
    if requested == "effort" and effort not in REASONING_EFFORTS:
        return [f"{name}: invalid reasoning effort: {effort!r}"]
    if requested != "effort" and effort is not None:
        return [f"{name}: reasoning effort is only valid when requested='effort'"]
    if "provider_default" in reasoning and not isinstance(reasoning["provider_default"], dict):
        return [f"{name}: provider_default must be an object"]
    return []


def group_identity(meta):
    params = dict(meta.get("params")) if isinstance(meta.get("params"), dict) else {}
    for key in ("max_tokens", "reasoning", "reasoning_effort", "no_reasoning"):
        params.pop(key, None)
    return {
        "benchmark": meta.get("benchmark"),
        "prompt": meta.get("prompt"),
        "model": meta.get("model"),
        "endpoint": meta.get("endpoint").rstrip("/") if isinstance(meta.get("endpoint"), str) else meta.get("endpoint"),
        "contributor": meta.get("contributor"),
        "hardware": meta.get("hardware"),
        "engine": meta.get("engine"),
        "quantization": meta.get("quantization"),
        "params": params,
    }


def group_id(meta):
    encoded = json.dumps(group_identity(meta), sort_keys=True, separators=(",", ":")).encode()
    digest = hashlib.sha256(encoded).hexdigest()[:12]
    return f"{slug(meta.get('model'))}-{slug(meta.get('prompt'))}-{digest}"


def build_index(results_dir=RESULTS, output_path=None):
    entries = []
    problems = []
    root_dir = os.path.dirname(results_dir)
    for mpath in sorted(glob.glob(os.path.join(results_dir, "*.json"))):
        if os.path.basename(mpath) == "index.json":
            continue
        name = os.path.basename(mpath).removesuffix(".json")
        try:
            with open(mpath) as f:
                meta = json.load(f)
        except (OSError, ValueError) as exc:
            problems.append(f"{name}: invalid metadata: {exc}")
            continue

        if not isinstance(meta, dict):
            problems.append(f"{name}: metadata must be an object")
            continue

        for key in REQUIRED:
            if key not in meta:
                problems.append(f"{name}: missing '{key}'")
        if meta.get("endpoint") is not None and not isinstance(meta.get("endpoint"), str):
            problems.append(f"{name}: endpoint must be a string or null")
        problems.extend(validate_reasoning(meta, name))
        reasoning = normalize_reasoning(meta)

        svg_path = os.path.join(results_dir, f"{name}.svg")
        svg_file = f"results/{name}.svg"
        if meta.get("file") is not None and not isinstance(meta.get("file"), str):
            problems.append(f"{name}: file must be a string or null")
        elif meta.get("file") and not os.path.exists(os.path.join(root_dir, meta["file"])):
            problems.append(f"{name}: referenced SVG missing: {meta['file']}")
        entry = {
            "id": name,
            "group_id": group_id(meta),
            "reasoning": reasoning,
            **{key: meta.get(key) for key in (
                "benchmark", "prompt", "model", "endpoint", "timestamp", "params",
                "usage", "reasoning_tokens", "finish_reason", "svg_len", "has_animation",
                "uses_js", "error", "seconds", "contributor", "cost_usd", "note",
                "hardware", "engine", "quantization")},
            "svg": svg_file if os.path.exists(svg_path) else None,
        }
        entries.append(entry)

    output_path = output_path or os.path.join(results_dir, "index.json")
    with open(output_path, "w") as f:
        json.dump({"version": 2, "count": len(entries), "entries": entries}, f, indent=2)
    return entries, problems


def main():
    entries, problems = build_index()
    print(f"index.json: {len(entries)} entries, {len(problems)} problems")
    for problem in problems:
        print("  !", problem)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
