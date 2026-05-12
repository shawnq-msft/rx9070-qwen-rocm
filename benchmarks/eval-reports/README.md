# Eval reports — HumanEval & PinchBench

Real, GPU-on-the-card numbers from this RX 9070 16 GB / WSL ROCm box.
Heterogeneous harnesses (lm-eval-harness, bespoke vLLM HumanEval, llama.cpp REPORT.md,
vLLM TurboQuant, PinchBench) — schema notes are in the
`aggregating-local-eval-reports` skill if you want to reproduce the aggregation.

## Files

- `qwen27b-vs-qwen35b-tq3-agent-bench-2026-05.md` — latest Qwen3.6-27B TQ3_4S vs Qwen3.6-35B-A3B TQ3_4S agent/code comparison; includes the 27B rerun failure diagnosis and OpenClaw config fix
- `qwen36-35b-tq3-4s-64k-humaneval-2026-05.md` — Qwen3.6-35B-A3B TQ3_4S 64k deep-KV HumanEval fullset: 97/164, pass@1 0.5915
- `pinchbench-humaneval-summary.md` — consolidated table (2026-05-02)
  - HumanEval 1a–1d: lm-eval, vLLM bespoke, llama.cpp KV sweep, vLLM TurboQuant
  - PinchBench 2a–2d: smoke, 3-suite, automated-only, all-suite
  - Section 3: highlight reel
- `model-eval-report-2026-04.md` — earlier qualitative report (2026-04-17),
  long-context viability + Hermes-daily-driver picks. Conclusions:
  - Daily driver: `Qwen3.5-9B-Q8_0` (128k, full GPU offload, Q8 quality)
  - Big model long-ctx: `Gemma 26B Q3_K_M` (112k usable; 128k OOMs)

## Highlights (TL;DR)

HumanEval pass@1 (164 tasks unless marked):
- llama.cpp **Qwen3.6-27B TQ3_4S**, ctx 64k, KV K q4_0 / V tq3_0 + FA → **0.7805** (128/164) @ 16.8 aggregate completion tps
- llama.cpp **Qwen3.6-35B-A3B TQ3_4S**, ctx 64k, KV K q4_0 / V tq3_0 + FA → **0.5915** (97/164) @ 38.0 aggregate completion tps
- llama.cpp Qwen3.5-9B **UD-Q6_K_XL + KV q8_0** → **0.7073** @ 32 tps, ctx 16k ★ best small-local
- llama.cpp Qwen3.5-9B Q8_0 + KV q8_0 → 0.6951 @ 29.5 tps
- llama.cpp Qwen3.5-9B UD-Q4_K_XL + KV f16 → 0.6707 @ 39.8 tps (fastest small)
- vLLM TurboQuant **Qwopus3.5-9B tq4_nc, ctx 131k** → 0.6768 @ **58.3 tps** ★ best long-ctx tps
- vLLM TurboQuant Qwopus3.5-9B tq3_nc, ctx 65k → 0.6402 @ 36.6 tps
- lm-eval Qwen3.6-35B-A3B TQ3_1S → 0.80 (20-task short)
- lm-eval Gemma 4-26B TQ3_1S → 0.45 (20-task short) — under-trained for code

PinchBench smoke / agent:
- llama.cpp **Qwen3.6-35B-A3B TQ3_4S**, 5-task full-agent smoke → **91.6%** (4.5786/5), 1092.7 s
- llama.cpp **Qwen3.6-27B TQ3_4S**, reasoning-on 5-task full-agent smoke → **89.1%** (4.4571/5), 1622.1 s; current rerun exposed OpenClaw config/streaming issues, not a model-quality regression

PinchBench full-automated (26 tasks):
- **Qwen3.6-35B TQ3_4S (hermes)**: PROD 91 / RES 100 / CODING 89 / ANALYSIS 26 / MEM 100 / SKILLS 100, 4175 s ★
- Gemma 4-26B Q4_K_S 128k ncmoe=10: PROD 20 / RES 50 / CODING 56 / ANALYSIS 14 / MEM 100 / SKILLS 43, 4074 s
- Qwopus 9B v3.5 Q6_K (hermes 3-suite): MEMORY 17 / PROD 13 / WRITING 22 — 9B can't carry agent suites
- Qwen3.5-9B Q6_K_XL all-automated: mostly 0% outside PROD — same story

## Caveats

- llama.cpp's OpenAI endpoint doesn't return a `usage` block, so PinchBench rows
  show `total_tokens=0`. That's a harness/endpoint gap, not a model bug.
- vLLM TurboQuant `tps` is per-request decode rate, not aggregate throughput; not
  apples-to-apples with llama.cpp single-stream tps.
- Some early Qwopus / Qwen3.6 PinchBench `writing=0%` rows are OpenClaw 1006 judge
  failures. The hermes-suffixed reruns are the real numbers.
- Qwen3.6-27B TQ3_4S HumanEval was scored from the saved `predictions.jsonl` with the OpenAI `human-eval` evaluator; a failed-36 rerun did not improve pass@1.
- PinchBench smoke (1 task / 5 task) vs full (26 / 123) — counts are baked into every row;
  don't average across them.
- Qwen3.6-27B TQ3_4S has a valid 5-task smoke, but the 2026-05-12 rerun first failed due to an invalid OpenClaw `models.aliases` config and then still showed OpenClaw `Connection error` with direct HTTP healthy. Treat that rerun as infrastructure/adapter evidence, not model score.
