# 🛹 skateboarder

Animated-SVG benchmark for LLMs, from [Risorse Artificiali](https://risorseartificiali.com/skateboarder/):
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
| `--no-reasoning` | disable thinking (OpenRouter and supported local engines) |
| `--reasoning-effort low` | request `minimal`, `low`, `medium`, `high`, `xhigh`, or `max` reasoning |
| `--timeout S` | request timeout (default 1800s) |

The reasoning flags are mutually exclusive. On OpenRouter the runner uses the unified `reasoning.effort`
request field; on llama.cpp it uses the request-level `reasoning_effort` supported by the chat template.
With neither flag, the request leaves reasoning at the model/provider default. For OpenRouter runs the
sidecar snapshots the model catalog's advertised default without forcing that value in the request.

The script extracts the `<svg>` and writes it with a JSON sidecar containing timestamp, endpoint, params,
requested reasoning, provider defaults when available, token usage, reasoning tokens, finish reason and
animation detection. Explicit reasoning modes get automatic filename suffixes:

```text
results/<model>-<variant>-no_reasoning.svg
results/<model>-<variant>-reasoning_low.svg
results/<model>-<variant>-reasoning_xhigh.svg
```

Default reasoning keeps the original `results/<model>-<variant>.svg` form. Repeating the same request
overwrites that reasoning variant; a different configuration receives an automatic short hash rather than
overwriting unrelated data.
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

Naming is handled automatically, including reasoning modes and configuration collisions. Do not rename the
generated SVG and JSON sidecar manually.

## Browse the gallery

GitHub Pages serves `index.html`: on first load (no model picked yet) an intro page explains the benchmark
with a link to the repo; the sidebar has search + filters (minimal / constrained / ok / errors),
live SVG preview and raw metadata. Runs that differ only by reasoning are grouped into one row. A selector
switches the visible reasoning result, and every model in comparison mode has an independent selector.
`max_tokens` is deliberately ignored when forming these groups so larger reasoning budgets remain comparable;
the exact value is still available in each run's technical details.

## License

MIT
