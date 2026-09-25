#!/usr/bin/env python3
"""skateboarder benchmark runner.

Sends the skateboard benchmark prompts (minimal / constrained, verbatim) to any
OpenAI-compatible chat/completions endpoint (OpenRouter, llama.cpp, ollama,
vLLM, ...) and saves the extracted SVG plus a JSON metadata sidecar.

Usage:
  python3 bench.py --model xiaomi/mimo-v2.6-flash --variant minimal
  python3 bench.py --model gpt-oss:20b --base http://localhost:11434/v1
  python3 bench.py --model hf/MODEL --base http://localhost:8080/v1 --no-reasoning

Auth: OPENROUTER_API_KEY (or BENCH_API_KEY for any other endpoint).
Results land in results/<slug>-<variant>.svg + .json
"""
import argparse
import datetime
import json
import os
import re
import socket
import sys
import time
import urllib.error
import urllib.request

# --- keep long reasoning phases alive through gateways that reset idle flows ---
def _keepalive():
    orig_create = socket.socket.connect

    def connect_with_keepalive(self, address):
        result = orig_create(self, address)
        try:
            self.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
            self.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPIDLE, 30)
            self.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPINTVL, 10)
            self.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPCNT, 6)
        except OSError:
            pass
        return result

    socket.socket.connect = connect_with_keepalive


_keepalive()

HERE = os.path.dirname(os.path.abspath(__file__))
PROMPTS_DIR = os.path.join(HERE, "prompts")
RESULTS_DIR = os.path.join(HERE, "results")

VARIANTS = {}
for fname in ("minimal.txt", "constrained.txt"):
    with open(os.path.join(PROMPTS_DIR, fname)) as f:
        VARIANTS[fname.removesuffix(".txt")] = f.read()


def slug(name):
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


def extract_svg(text):
    m = re.search(r"<svg[\s\S]*?</svg>", text or "")
    return m.group(0) if m else None


def run(args):
    base = args.base.rstrip("/")
    key = args.api_key or os.environ.get("BENCH_API_KEY") or os.environ.get("OPENROUTER_API_KEY") or ""
    headers = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = f"Bearer {key}"

    body = {"model": args.model, "messages": [{"role": "user", "content": VARIANTS[args.variant]}]}
    if args.max_tokens:
        body["max_tokens"] = args.max_tokens
    if args.temperature is not None:
        body["temperature"] = args.temperature
    if args.reasoning_effort:
        body["reasoning_effort"] = args.reasoning_effort
    if args.no_reasoning:
        # provider-agnostic attempts: OpenRouter style + llama.cpp / ollama styles
        body["reasoning"] = {"exclude": True}
        body["chat_template_kwargs"] = {"enable_thinking": False, "thinking": False}

    req = urllib.request.Request(f"{base}/chat/completions", data=json.dumps(body).encode(), headers=headers)
    t0 = time.time()
    meta = {
        "benchmark": "skateboard",
        "prompt": args.variant,
        "model": args.model,
        "endpoint": base,
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        "params": {k: v for k, v in (
            ("max_tokens", args.max_tokens), ("temperature", args.temperature),
            ("reasoning_effort", args.reasoning_effort), ("no_reasoning", args.no_reasoning)) if v},
    }
    try:
        with urllib.request.urlopen(req, timeout=args.timeout) as r:
            resp = json.load(r)
    except urllib.error.HTTPError as e:
        meta["error"] = f"HTTP {e.code}: {e.read().decode()[:500]}"
        meta["seconds"] = round(time.time() - t0, 1)
        write_meta(args, meta, None)
        sys.exit(1)
    except Exception as e:
        meta["error"] = str(e)
        meta["seconds"] = round(time.time() - t0, 1)
        write_meta(args, meta, None)
        sys.exit(1)

    meta["seconds"] = round(time.time() - t0, 1)
    choice = (resp.get("choices") or [{}])[0]
    content = (choice.get("message") or {}).get("content") or ""
    usage = resp.get("usage") or {}
    details = usage.get("completion_tokens_details") or {}
    meta["usage"] = usage
    meta["reasoning_tokens"] = details.get("reasoning_tokens")
    meta["finish_reason"] = choice.get("finish_reason")
    meta["content_len"] = len(content)

    svg = extract_svg(content)
    meta["svg_len"] = len(svg) if svg else 0
    meta["has_animation"] = bool(re.search(r"<animate|animateTransform|animateMotion|@keyframes", svg)) if svg else False
    meta["uses_js"] = "<script" in svg.lower() if svg else False
    meta["error"] = None if svg else "no SVG in response"

    write_meta(args, meta, svg)
    if not svg:
        sys.exit(1)
    print(f"OK: {meta['file']}  ({meta['svg_len']}B, {usage.get('total_tokens')} tok, {meta['seconds']}s)")


def write_meta(args, meta, svg):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    name = f"{slug(args.model)}-{args.variant}"
    if svg:
        path = os.path.join(RESULTS_DIR, f"{name}.svg")
        with open(path, "w") as f:
            f.write(svg)
        meta["file"] = f"results/{name}.svg"
    mpath = os.path.join(RESULTS_DIR, f"{name}.json")
    with open(mpath, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"meta: results/{name}.json" + ("  ERROR: " + meta["error"] if meta.get("error") else ""))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True)
    p.add_argument("--variant", choices=VARIANTS, default="minimal")
    p.add_argument("--base", default="https://openrouter.ai/api/v1")
    p.add_argument("--api-key", default=None)
    p.add_argument("--max-tokens", type=int, default=128000)
    p.add_argument("--temperature", type=float, default=None)
    p.add_argument("--reasoning-effort", default=None, help="low|medium|high (OpenRouter/OpenAI)")
    p.add_argument("--no-reasoning", action="store_true", help="disable thinking (multi-provider flags)")
    p.add_argument("--timeout", type=int, default=1800)
    run(p.parse_args())


if __name__ == "__main__":
    main()
