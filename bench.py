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
Before the request it asks a few optional run-metadata questions (contributor,
hardware, engine, quantization, notes); local llama.cpp/ollama/vLLM servers are
auto-probed. Pipe stdin or pass --non-interactive to skip the prompts.
Results land in results/<slug>-<variant>[-reasoning_<effort>].svg + .json;
runs that yield no SVG keep the raw response in results/<name>.txt for inspection.
"""
import argparse
import datetime
import glob
import hashlib
import json
import os
import platform
import re
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.parse
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
REASONING_EFFORTS = ("minimal", "low", "medium", "high", "xhigh", "max")

VARIANTS = {}
for fname in ("minimal.txt", "constrained.txt"):
    with open(os.path.join(PROMPTS_DIR, fname)) as f:
        VARIANTS[fname.removesuffix(".txt")] = f.read()


def slug(name):
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


def is_openrouter(base):
    try:
        return urllib.parse.urlparse(base).hostname == "openrouter.ai"
    except ValueError:
        return False


def requested_reasoning(args):
    if args.no_reasoning:
        return {"requested": "disabled", "effort": None}
    if args.reasoning_effort:
        return {"requested": "effort", "effort": args.reasoning_effort}
    return {"requested": "default", "effort": None}


def reasoning_from_meta(meta):
    reasoning = meta.get("reasoning")
    if isinstance(reasoning, dict) and reasoning.get("requested") in ("default", "disabled", "effort", "unknown"):
        return {"requested": reasoning["requested"], "effort": reasoning.get("effort")}
    params = meta.get("params") or {}
    if params.get("no_reasoning"):
        return {"requested": "disabled", "effort": None}
    if params.get("reasoning_effort"):
        return {"requested": "effort", "effort": params["reasoning_effort"]}
    return {"requested": "unknown", "effort": None}


def build_request_body(args):
    body = {"model": args.model, "messages": [{"role": "user", "content": VARIANTS[args.variant]}]}
    if args.max_tokens:
        body["max_tokens"] = args.max_tokens
    if args.temperature is not None:
        body["temperature"] = args.temperature

    reasoning = requested_reasoning(args)
    if is_openrouter(args.base):
        if reasoning["requested"] == "effort":
            body["reasoning"] = {"effort": reasoning["effort"]}
        elif reasoning["requested"] == "disabled":
            body["reasoning"] = {"effort": "none"}
    elif reasoning["requested"] == "effort":
        body["reasoning_effort"] = reasoning["effort"]
    elif reasoning["requested"] == "disabled":
        body["reasoning_effort"] = "none"
        body["chat_template_kwargs"] = {"enable_thinking": False, "thinking": False}
    return body


def fetch_openrouter_reasoning(base, model, headers, timeout=10):
    """Snapshot OpenRouter's advertised reasoning defaults without affecting the request."""
    try:
        query = urllib.parse.urlencode({"q": model})
        req = urllib.request.Request(f"{base.rstrip('/')}/models?{query}", headers=headers)
        with urllib.request.urlopen(req, timeout=timeout) as response:
            models = (json.load(response) or {}).get("data") or []
        info = next((item for item in models if isinstance(item, dict) and item.get("id") == model), None)
        if info is None:
            raise LookupError(f"model {model!r} not found in OpenRouter catalog")
        advertised = info.get("reasoning")
        snapshot = {
            "source": "openrouter-models",
            "checked_at": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
            "supported": isinstance(advertised, dict),
        }
        if isinstance(advertised, dict):
            snapshot.update({
                "enabled": advertised.get("default_enabled"),
                "effort": advertised.get("default_effort"),
                "mandatory": advertised.get("mandatory"),
                "supported_efforts": advertised.get("supported_efforts"),
                "supports_max_tokens": bool(advertised.get("supports_max_tokens")),
            })
        return snapshot
    except Exception as exc:
        print(f"warning: could not read OpenRouter reasoning defaults: {exc}", file=sys.stderr)
        return None


def extract_svg(text):
    m = re.search(r"<svg[\s\S]*?</svg>", text or "")
    return m.group(0) if m else None


# --- optional run metadata: auto-detection + interactive prompts ---

QUANT_RE = re.compile(r"\b(I?Q\d[_A-Za-z0-9]*|F16|BF16|FP16|F32|FP32)\b")


def detect_gpu():
    def run(argv):
        try:
            out = subprocess.run(argv, capture_output=True, text=True, timeout=4)
            return out.stdout if out.returncode == 0 else ""
        except Exception:
            return ""

    out = run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"])
    if out.strip():
        return ", ".join(l.strip() for l in out.splitlines() if l.strip())
    out = run(["rocm-smi", "--showproductname", "--showmeminfo", "vram"])  # AMD ROCm
    names = list(dict.fromkeys(re.findall(r"Card series:\s*(.+)", out)))
    if names:
        v = re.search(r"VRAM Total Memory \(B\):\s*(\d+)", out)
        return ", ".join(n.strip() for n in names) + (f", {round(int(v.group(1)) / 2**30)} GB VRAM" if v else "")
    out = run(["lspci"])  # generic PCI fallback: AMD Vulkan, Intel Arc, ...
    gpus = [l.rsplit(":", 1)[-1].strip() for l in out.splitlines() if re.search(r"\bVGA|Display|3D controller", l, re.I)]
    if gpus:
        return ", ".join(gpus)
    try:  # sysfs amdgpu (no pciutils / no rocm-smi installed)
        for vpath in glob.glob("/sys/class/drm/card*/device/vendor"):
            with open(vpath) as f:
                if f.read().strip() == "0x1002":
                    return "AMD GPU (amdgpu)"
    except OSError:
        pass
    return None


def detect_hardware():
    """Best-effort local machine specs; every field is optional."""
    hw = {"os": f"{platform.system()} {platform.release()} {platform.machine()}".strip()}
    cpu = None
    try:
        with open("/proc/cpuinfo") as f:
            cpu = next(l.split(":", 1)[1].strip() for l in f if l.startswith("model name"))
    except (OSError, StopIteration):
        pass
    if not cpu:
        try:
            out = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True, timeout=2)
            cpu = out.stdout.strip()
        except Exception:
            pass
    hw["cpu"] = cpu or platform.processor() or None
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        hw["gpu"] = "Apple Silicon (unified memory)"
    else:
        hw["gpu"] = detect_gpu()
    try:
        with open("/proc/meminfo") as f:
            hw["ram_gb"] = round(int(next(l for l in f if l.startswith("MemTotal")).split()[1]) / 1048576)
    except (OSError, StopIteration, ValueError, IndexError):
        try:
            out = subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True, timeout=2)
            hw["ram_gb"] = round(int(out.stdout.strip()) / 2**30)
        except Exception:
            pass
    return {k: v for k, v in hw.items() if v} or None


def probe_engine(base, model):
    """Guess engine/version/quantization from well-known local inference-server endpoints."""
    root = re.sub(r"/v1/?$", "", base.rstrip("/"))

    def get(path, data=None):
        headers = {"Content-Type": "application/json"} if data else {}
        req = urllib.request.Request(root + path, data=json.dumps(data).encode() if data else None, headers=headers)
        with urllib.request.urlopen(req, timeout=3) as r:
            return json.load(r)

    try:  # llama.cpp server
        p = get("/props")
        eng = {"name": "llama.cpp"}
        bi = p.get("build_info")
        if isinstance(bi, dict):
            bi = bi.get("commit") or bi.get("number")
        if bi:
            eng["version"] = str(bi)
        m = QUANT_RE.search(str(p.get("model_path") or ""))
        return eng, (m.group(1) if m else None)
    except Exception:
        pass
    try:  # vLLM
        v = get("/version")
        if isinstance(v, dict) and v.get("version"):
            return {"name": "vllm", "version": str(v["version"])}, None
    except Exception:
        pass
    try:  # ollama
        v = get("/api/version")
        eng = {"name": "ollama"}
        if isinstance(v, dict) and v.get("version"):
            eng["version"] = str(v["version"])
        try:
            info = get("/api/show", {"model": model})
            return eng, (info.get("details") or {}).get("quantization_level") or None
        except Exception:
            return eng, None
    except Exception:
        pass
    return None, None


def ask(label, default=None, hint=None):
    suffix = f" ({hint})" if hint else ""
    shown = default if isinstance(default, str) else json.dumps(default, separators=(",", ":")) if default else None
    try:
        v = input(f"{label}{suffix}" + (f" [{shown}]" if shown else "") + ": ").strip()
    except (EOFError, KeyboardInterrupt):
        print()
        sys.exit(130)
    return v or default


def parse_hardware(text):
    out = {}
    for part in text.split(","):
        k, _, v = part.partition("=")
        if k.strip() and v.strip():
            out[k.strip()] = int(v) if v.strip().isdigit() else v.strip()
    return out or {"desc": text.strip()}


def parse_engine(text, det_name=None):
    parts = re.split(r"[/ ]", text.strip(), maxsplit=1)
    if len(parts) == 2 and parts[1].strip():
        return {"name": parts[0].strip(), "version": parts[1].strip()}
    name = parts[0]
    if det_name and name.lower() != det_name.lower() and re.match(r"^[bv]\d", name):
        return {"name": det_name, "version": name}  # bare version typed for the detected engine
    return {"name": name}


def collect_meta(args):
    """Optional run metadata: CLI flags > interactive answers > auto-detection."""
    remote = is_openrouter(args.base)
    det_hw = None if remote else detect_hardware()
    det_eng, det_quant = (None, None) if remote else probe_engine(args.base, args.model)
    if isinstance(args.engine, str):  # CLI: parse now that detection may know the engine name
        args.engine = parse_engine(args.engine, (det_eng or {}).get("name"))

    if args.non_interactive or not sys.stdin.isatty():
        if not remote:  # scripted local run: keep what was auto-detected
            args.hardware = args.hardware or det_hw
            args.engine = args.engine or det_eng
            args.quantization = args.quantization or det_quant
        filled = {k: getattr(args, k) for k in ("contributor", "hardware", "engine", "quantization", "note") if getattr(args, k)}
        if filled:
            print("run metadata: " + json.dumps(filled, separators=(",", ":")))
        return

    print(f"run metadata (all optional: Enter accepts [default] or skips) -- {args.model} @ {args.base}")
    if not args.contributor:
        args.contributor = ask("contributor (GitHub username)")
    if args.hardware is None:
        hint = "remote API: this info is not needed, Enter to skip" if remote else None
        if det_hw:
            v = ask("hardware", default=det_hw, hint="Enter = detected, n = skip, text = override")
            args.hardware = None if isinstance(v, str) and v.lower() in ("n", "no", "none") else (parse_hardware(v) if isinstance(v, str) else v)
        else:
            v = ask("hardware (free text or key=value pairs)", hint=hint)
            args.hardware = parse_hardware(v) if v else None
    if args.engine is None:
        v = ask("engine / version / branch (e.g. llama.cpp/b6420)", default=det_eng)
        if isinstance(v, str):
            args.engine = None if v.lower() in ("n", "no", "none") else parse_engine(v, (det_eng or {}).get("name"))
        else:
            args.engine = v
    if not args.quantization:
        v = ask("quantization (e.g. Q4_K_M)", default=det_quant)
        args.quantization = None if isinstance(v, str) and v.lower() in ("n", "no", "none") else v
    if not args.note:
        args.note = ask("notes / comments")


def run(args):
    base = args.base.rstrip("/")
    key = args.api_key or os.environ.get("BENCH_API_KEY") or os.environ.get("OPENROUTER_API_KEY") or ""
    headers = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = f"Bearer {key}"

    body = build_request_body(args)

    req = urllib.request.Request(f"{base}/chat/completions", data=json.dumps(body).encode(), headers=headers)
    meta = {
        "benchmark": "skateboard",
        "prompt": args.variant,
        "model": args.model,
        "endpoint": base,
        "timestamp": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        "params": {k: v for k, v in (
            ("max_tokens", args.max_tokens), ("temperature", args.temperature),
            ("reasoning_effort", args.reasoning_effort),
            ("no_reasoning", True if args.no_reasoning else None)) if v is not None},
        "reasoning": requested_reasoning(args),
    }
    if is_openrouter(base):
        provider_default = fetch_openrouter_reasoning(base, args.model, headers)
        if provider_default is not None:
            meta["reasoning"]["provider_default"] = provider_default
    for k in ("contributor", "hardware", "engine", "quantization", "note"):
        if getattr(args, k, None):
            meta[k] = getattr(args, k)
    t0 = time.time()
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
    if svg:
        meta["error"] = None
    else:
        meta["error"] = f"no SVG in response (finish_reason={meta.get('finish_reason')!r}, content_len={len(content)})"
        provider_error = resp.get("error")
        if provider_error:
            meta["provider_error"] = provider_error
            message = provider_error.get("message") if isinstance(provider_error, dict) else provider_error
            if message:
                meta["error"] += f": {message}"

    write_meta(args, meta, svg, raw=content)
    if not svg:
        sys.exit(1)
    print(f"OK: {meta['file']}  ({meta['svg_len']}B, {usage.get('total_tokens')} tok, {meta['seconds']}s)")


def request_identity(meta):
    params = dict(meta.get("params") or {})
    params.pop("reasoning_effort", None)
    params.pop("no_reasoning", None)
    return {
        "benchmark": meta.get("benchmark"),
        "prompt": meta.get("prompt"),
        "model": meta.get("model"),
        "endpoint": (meta.get("endpoint") or "").rstrip("/"),
        "params": params,
        "reasoning": reasoning_from_meta(meta),
        "contributor": meta.get("contributor"),
        "hardware": meta.get("hardware"),
        "engine": meta.get("engine"),
        "quantization": meta.get("quantization"),
    }


def preferred_output_name(args):
    name = f"{slug(args.model)}-{args.variant}"
    if args.no_reasoning:
        return f"{name}-no_reasoning"
    if args.reasoning_effort:
        return f"{name}-reasoning_{slug(args.reasoning_effort)}"
    return name


def output_name(args, meta, results_dir=None):
    results_dir = results_dir or RESULTS_DIR
    preferred = preferred_output_name(args)
    identity = request_identity(meta)

    def availability(name):
        json_path = os.path.join(results_dir, f"{name}.json")
        svg_path = os.path.join(results_dir, f"{name}.svg")
        if not os.path.exists(json_path):
            return "occupied" if os.path.exists(svg_path) else "free"
        try:
            with open(json_path) as f:
                existing = json.load(f)
            return "same" if isinstance(existing, dict) and request_identity(existing) == identity else "occupied"
        except (OSError, ValueError, TypeError, AttributeError):
            return "occupied"

    preferred_state = availability(preferred)
    if preferred_state in ("free", "same"):
        return preferred

    encoded = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    digest = hashlib.sha256(encoded).hexdigest()
    for length in (10, 16, 32, 64):
        candidate = f"{preferred}-{digest[:length]}"
        if availability(candidate) in ("free", "same"):
            return candidate
    raise RuntimeError(f"could not allocate a safe output name for {preferred}")


def write_meta(args, meta, svg, raw=None):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    name = output_name(args, meta)
    path = os.path.join(RESULTS_DIR, f"{name}.svg")
    raw_path = os.path.join(RESULTS_DIR, f"{name}.txt")
    if svg:
        with open(path, "w") as f:
            f.write(svg)
        meta["file"] = f"results/{name}.svg"
    else:
        meta.pop("file", None)
        try:
            os.remove(path)
        except FileNotFoundError:
            pass
    if not svg and raw:
        with open(raw_path, "w") as f:
            f.write(raw)
        meta["raw_output"] = f"results/{name}.txt"
    else:  # success (or no content at all): drop the stale raw dump of a previous failed run
        meta.pop("raw_output", None)
        try:
            os.remove(raw_path)
        except FileNotFoundError:
            pass
    mpath = os.path.join(RESULTS_DIR, f"{name}.json")
    with open(mpath, "w") as f:
        json.dump(meta, f, indent=2)
    detail = f"  ERROR: {meta['error']}" if meta.get("error") else ""
    if meta.get("raw_output"):
        detail += f" (raw response kept: {meta['raw_output']})"
    print(f"meta: results/{name}.json{detail}")


def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True)
    p.add_argument("--variant", choices=VARIANTS, default="minimal")
    p.add_argument("--base", default="https://openrouter.ai/api/v1")
    p.add_argument("--api-key", default=None)
    p.add_argument("--max-tokens", type=int, default=128000)
    p.add_argument("--temperature", type=float, default=None)
    reasoning = p.add_mutually_exclusive_group()
    reasoning.add_argument("--reasoning-effort", choices=REASONING_EFFORTS, default=None,
                           help="minimal|low|medium|high|xhigh|max")
    reasoning.add_argument("--no-reasoning", action="store_true", help="disable thinking")
    p.add_argument("--timeout", type=int, default=1800)
    p.add_argument("--contributor", default=None, help="GitHub username stored with the result")
    p.add_argument("--hardware", default=None, help='run hardware, "key=value, ..." or free text')
    p.add_argument("--engine", default=None, help='inference engine "name/version", e.g. llama.cpp/b6420')
    p.add_argument("--quantization", default=None, help="model quantization, e.g. Q4_K_M")
    p.add_argument("--note", default=None, help="free-form notes about this run")
    p.add_argument("--non-interactive", action="store_true", help="skip prompts; auto-fill engine/hardware/quant for local endpoints")
    return p


def main():
    args = build_parser().parse_args()
    if isinstance(args.hardware, str):
        args.hardware = parse_hardware(args.hardware) if args.hardware.strip() else None
    collect_meta(args)
    run(args)


if __name__ == "__main__":
    main()
