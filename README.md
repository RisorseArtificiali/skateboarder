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

## Submit your results

1. Fork, run the benchmark (one or both variants).
2. Run the benchmark (one or both variants), check the sidecar JSON in `results/` has model, timestamp, usage filled in.
3. Open a PR — that's it. A GitHub Action rebuilds `results/index.json` for you on every PR (validation) and again at merge. Failed runs are welcome too: errors with metadata are data.

Naming: `results/<model-slug>-<variant>.svg` (handled automatically).
Multiple runs of the same model are fine: append a suffix (`-v2`, `-lowtemp`, `-m2`, …).

## Browse the gallery

GitHub Pages serves `index.html`: sidebar with search + filters (minimal / constrained / ok / errors),
live SVG preview, raw metadata per run.

## License

MIT
