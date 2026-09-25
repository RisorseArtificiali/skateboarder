# 🛹 skateboarder

Animated-SVG benchmark for LLMs, from [Risorse Artificiali](https://risorseartificiali.com/skateboard/):
send a fixed prompt to any model (OpenRouter or a local llama.cpp / ollama / vLLM endpoint) and collect the animated SVG it produces. Physics-aware: can the model encode gravity, momentum and pendulum motion in code?

Contributions welcome: run the benchmark against your model and open a PR with the SVG + metadata.

## The prompts

Two variants, always verbatim (typos included), in [`prompts/`](prompts/):

- **minimal** — the original one-liner: *"generate an animated svg of a man doing skatevoard tricks on a pipe. Be mindful of the real phisics"*
- **constrained** — fully specified: half-pipe scene, skater + board anatomy, pendulum motion, parabolic aerials, board spin, seamless loop, SMIL/CSS only, no JS.

## Run it

Any OpenAI-compatible `chat/completions` endpoint works.

**OpenRouter:**

```bash
export OPENROUTER_API_KEY=sk-or-...
python3 bench.py --model openai/gpt-6-luna --variant constrained
```

**Local (ollama):**

```bash
python3 bench.py --model gpt-oss:20b --base http://localhost:11434/v1 --variant minimal
```

**Local (llama.cpp server):**

```bash
python3 bench.py --model my-model --base http://localhost:8080/v1 --variant constrained --no-reasoning
```

Useful flags:

| flag | meaning |
|---|---|
| `--max-tokens N` | output cap (default 128000; reasoning models may need it) |
| `--no-reasoning` | disable thinking (OpenRouter + llama.cpp/ollama style flags) |
| `--reasoning-effort low` | cap reasoning on OpenAI/OpenRouter |
| `--timeout S` | request timeout (default 1800s) |

The script extracts the `<svg>`, writes `results/<model>-<variant>.svg` plus a JSON sidecar
with timestamp, endpoint, params, token usage, reasoning tokens, finish reason and animation detection.
TCP keepalive is patched in so long reasoning phases survive gateway idle resets.

### Run metadata

Before the request goes out, the script asks a few optional questions and stores the answers
in the sidecar JSON: contributor (GitHub username), hardware, engine name + version/branch,
model quantization and free-form notes. All optional — Enter accepts the suggestion or skips;
for hosted APIs the hardware question just reminds you it's not needed.

Local endpoints are auto-probed and offered as defaults: llama.cpp `/props`, vLLM `/version`,
ollama `/api/version` + `/api/show` (quantization level), plus CPU/RAM/GPU detection via
`/proc`, `sysctl`, and `nvidia-smi` / `rocm-smi` / `lspci` (AMD ROCm and Vulkan setups included).

To skip the prompts (CI, scripted runs): pipe stdin or pass `--non-interactive` — local runs
then auto-fill engine/hardware/quantization from detection. The same fields can be set
explicitly: `--contributor NAME`, `--hardware "gpu=RTX 4090, ram_gb=24"`,
`--engine llama.cpp/b6420`, `--quantization Q4_K_M`, `--note "free text"`.

## Submit your results

1. Fork, run the benchmark (one or both variants).
2. Run the benchmark (one or both variants), check the sidecar JSON in `results/` has model, timestamp, usage filled in — the metadata prompts (contributor, hardware, engine, quantization, notes) fill the rest.
3. Open a PR — that's it. A GitHub Action rebuilds `results/index.json` for you on every PR (validation) and again at merge. Failed runs are welcome too: errors with metadata are data.

Naming: `results/<model-slug>-<variant>.svg` (handled automatically).
Multiple runs of the same model are fine: append a suffix (`-v2`, `-lowtemp`, `-m2`, …).

## Browse the gallery

GitHub Pages serves `index.html`: sidebar with search + filters (minimal / constrained / ok / errors),
live SVG preview, raw metadata per run.

## License

MIT
