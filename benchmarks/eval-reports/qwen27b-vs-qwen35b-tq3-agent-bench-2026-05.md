# Qwen3.6 27B TQ3_4S vs 35B-A3B TQ3_4S — RX 9070 agent/code benchmarks

Date: 2026-05-12
Host: WSL2 + ROCm, AMD Radeon RX 9070 16 GB
Runtime: llama.cpp TurboQuant fork (`/home/qiushuo/src/llama.cpp-tq3/build-gfx1201-tq3/bin/llama-server`)

This note records the latest on-card benchmark evidence for the two practical large Qwen3.6 TurboQuant routes on this 16 GB card.

## Serving configs compared

### Qwen3.6-35B-A3B TQ3_4S — current preferred local-agent route

Model:
`/home/qiushuo/models/qwen/YTan2000-Qwen3.6-35B-A3B-TQ3_4S/Qwen3.6-35B-A3B-TQ3_4S.gguf`

Command shape:

```bash
llama-server \
  -m Qwen3.6-35B-A3B-TQ3_4S.gguf \
  --host 127.0.0.1 --port 8810 --alias qwen36-35b-tq3_4s \
  -ngl 99 \
  -c 65536 -np 1 \
  -b 256 -ub 64 \
  -ctk q4_0 -ctv tq3_0 -fa on \
  --jinja \
  --reasoning auto --reasoning-budget 512 --reasoning-format deepseek
```

Verified `/props`: `n_ctx=65536`, `total_slots=1`.

### Qwen3.6-27B TQ3_4S — reasoning-on smoke route

Model:
`/home/qiushuo/models/qwen/YTan2000-Qwen3.6-27B-TQ3_4S/Qwen3.6-27B-TQ3_4S.gguf`

Command shape:

```bash
llama-server \
  -m Qwen3.6-27B-TQ3_4S.gguf \
  --host 127.0.0.1 --port 8250 \
  -ngl 99 -fa on \
  -ctk q4_0 -ctv tq3_0 \
  -c 65536 -ub 64 -b 256 --parallel 1 \
  --jinja
```

The 27B route intentionally keeps reasoning on: no `--reasoning off`, no `--reasoning-format none`, and no `--skip-chat-parsing`.

Live readiness probe on 2026-05-12:

- `/health` became OK after about 41 s.
- Direct `/v1/chat/completions` worked.
- Probe timing: prompt about 70.7 tok/s, decode about 25.7 tok/s.
- Response exposed `reasoning_content`, confirming reasoning is active.

## PinchBench 5-task smoke comparison

Standard smoke tasks:

- `task_sanity`
- `task_calendar`
- `task_summary`
- `task_weather`
- `task_csv_iris_summary`

| Model / run | Score | Runtime | Requests | Notes |
|---|---:|---:|---:|---|
| Qwen3.6-35B-A3B TQ3_4S, full OpenClaw agent | 4.5786 / 5 = **91.6%** | 1092.7 s | 18 | best current large-agent smoke baseline |
| Qwen3.6-27B TQ3_4S, reasoning-on, full OpenClaw agent, 30 s rest | 4.4571 / 5 = **89.1%** | 1622.1 s | 23 | close score, slower end-to-end; one hybrid judge issue |
| Qwen3.6-27B TQ3_4S, 2026-05-12 rerun after OpenClaw config regression | 0 / 5 = **invalid** | 85.5 s | 0 | not a model result: global OpenClaw config had invalid `models.aliases`, so every task failed before agent execution |
| Qwen3.6-27B TQ3_4S, direct llama.cpp fallback sanity after config fix | 1 / 1 = **100%** | 8.2 s | 1 | confirms endpoint/model still responds; not comparable to full-agent PinchBench |

### 35B per-task scores

Source: `/home/qiushuo/reports/pinchbench/qwen36-tq3-4s-smoke-metrics/0007_qwen36-35b-tq3_4s.json`

| Task | Score | Runtime note |
|---|---:|---|
| task_sanity | 1.0 | included in 1092.7 s total |
| task_calendar | 1.0 | included |
| task_weather | 1.0 | included |
| task_summary | 1.0 | included |
| task_csv_iris_summary | 0.5786 | `LLM judge failed: no parseable response after all attempts` |

### 27B per-task scores from the valid reasoning-on smoke

Source root: `/home/qiushuo/reports/pinchbench/qwen36-27b-tq3-4s-5task-reasoning-on-30srest/`

| Task | Score | Runtime | Requests | Source JSON |
|---|---:|---:|---:|---|
| task_calendar | 1.0 | 294.18 s | 4 | `1_task_calendar_20260511-144342/0004_llama-cpp-qwen3-6-27b-tq3_4s-gguf.json` |
| task_summary | 1.0 | 277.98 s | 3 | `2_task_summary_20260511-144938/0005_llama-cpp-qwen3-6-27b-tq3_4s-gguf.json` |
| task_weather | 0.8571 | 423.22 s | 11 | `3_task_weather_20260511-145520/0006_llama-cpp-qwen3-6-27b-tq3_4s-gguf.json` |
| task_csv_iris_summary | 0.6000 | 387.95 s | 4 | `4_task_csv_iris_summary_20260511-150328/0007_llama-cpp-qwen3-6-27b-tq3_4s-gguf.json` |
| task_sanity | 1.0 | 238.77 s | 1 | `5_task_sanity_20260511-151334/0008_llama-cpp-qwen3-6-27b-tq3_4s-gguf.json` |

Interpretation:

- 27B TQ3_4S can run the 5-task OpenClaw full-agent smoke with reasoning on.
- Quality is close to 35B in this small smoke: 89.1% vs 91.6%.
- End-to-end runtime is worse: 1622 s vs 1093 s, partly due to OpenClaw embedded/gateway overhead and reasoning-first streaming behavior.
- The CSV score is polluted by the hybrid judge path (`LLM judge failed`) and should not be read as a clean model-analysis failure.

## 2026-05-12 rerun failure and fix

The requested rerun initially produced five 0% task results. That was not a model-quality result.

Observed failure:

```text
Config invalid
File: ~/.openclaw/openclaw.json
Problem:
  - models: Unrecognized key: "aliases"
```

This made the benchmark fail before any agent transcript or model request was produced (`total_requests=0`).

Fix applied locally:

- Backed up config to `/home/qiushuo/.openclaw/openclaw.json.bak.fix-aliases-20260512-162806`.
- Removed the schema-invalid top-level `models.aliases` key.
- Preserved the alias through the schema-valid `agents.defaults.models[<provider/model>].alias` shape.
- `openclaw config validate` now passes.

After the config fix:

- Direct llama.cpp fallback sanity passed: 1/1, 8.19 s, 171 tokens.
- Full OpenClaw `task_sanity` still failed with repeated `Connection error.` despite direct HTTP success. Transcript shows `modelId=Qwen3.6-27B-TQ3_4S.gguf`, thinking off in OpenClaw, then four empty assistant error messages. This is an OpenClaw/stream-consumption compatibility issue, not endpoint readiness.

## HumanEval code benchmarks

Sources:

- 35B: `qwen36-35b-tq3-4s-64k-humaneval-2026-05.md`
- 27B: `/home/qiushuo/reports/vllm-rocm-eval/results/qwen36-27b-tq3-4s-humaneval-full164/`

| Model | ctx | KV | FA | pass@1 | Passed | Runtime | Avg completion toks | Aggregate completion tps |
|---|---:|---|---|---:|---:|---:|---:|---:|
| Qwen3.6-27B TQ3_4S | 65,536 | K q4_0 / V tq3_0 | on | **0.7805** | **128 / 164** | 754.3 s | 91.4 | 16.8 |
| Qwen3.6-35B-A3B TQ3_4S | 65,536 | K q4_0 / V tq3_0 | on | **0.5915** | **97 / 164** | 434.4 s | 100.6 | 38.0 |

The 27B HumanEval fullset was scored locally with `human-eval` against the saved `predictions.jsonl` and wrote:

- `/home/qiushuo/reports/vllm-rocm-eval/results/qwen36-27b-tq3-4s-humaneval-full164/predictions.jsonl_results.jsonl`

The 27B pass/fail split is 128 passed / 36 failed. A follow-up failed-36 rerun exists, but replacing the failed completions did not change pass@1, so the canonical fullset score remains **128/164 = 0.7805**.

For 35B, the previous comparable full run for the same model alias scored 90/164 = 0.5488, so the 64k deep-KV run solved 7 more tasks (+4.27 percentage points).

## vLLM / FP8 note for 35B

The current practical 35B route on this RX 9070 16 GB box is the llama.cpp TQ3_4S GGUF route above. A 35B FP8 + vLLM route is not treated as a usable baseline on this card: 35B FP8/vLLM does not fit the 16 GB VRAM budget in the current local experiments, whereas the GGUF TQ3_4S path runs at 64k context with full GPU offload.

## Operational recommendation

1. Use **Qwen3.6-35B-A3B TQ3_4S on port 8810** as the current local large-agent default.
   - It has the better 5-task PinchBench result.
   - It has a fresh full HumanEval score.
   - It is the canonical local service port for this machine.
2. Treat **Qwen3.6-27B TQ3_4S** as a strong code model but still only a smoke/research agent route.
   - HumanEval is much stronger than 35B here: 0.7805 vs 0.5915.
   - Valid 5-task agent smoke is close to 35B, but slower: 89.1% vs 91.6%, 1622 s vs 1093 s.
   - Current OpenClaw full-agent path can regress into `Connection error` despite direct HTTP success.
3. For a stable 27B daily agent route, prefer the mainline **Qwen3.6-27B Q4_K_S** configs documented in `skills/qwen-27b/` (`q8_0/q8_0 @ 24k` or `q4_0/q4_0 @ 40k`) rather than the 27B TQ3_4S reasoning-on branch.

## Artifacts

- 35B PinchBench smoke: `/home/qiushuo/reports/pinchbench/qwen36-tq3-4s-smoke-metrics/0007_qwen36-35b-tq3_4s.json`
- 35B HumanEval: `/home/qiushuo/reports/vllm-rocm-eval/results/humaneval-qwen36-35b-tq3-4s-64k-kq4-vtq3-20260512-153158/`
- 27B HumanEval fullset: `/home/qiushuo/reports/vllm-rocm-eval/results/qwen36-27b-tq3-4s-humaneval-full164/`
- 27B valid 5-task smoke: `/home/qiushuo/reports/pinchbench/qwen36-27b-tq3-4s-5task-reasoning-on-30srest/`
- 27B invalid 2026-05-12 rerun log: `/home/qiushuo/reports/pinchbench/qwen36-27b-tq3-4s-5task-reasoning-on-30srest.run.log`
- 27B direct fallback sanity after config fix: `/home/qiushuo/reports/pinchbench/qwen36-27b-tq3-4s-direct-fallback-sanity-20260512/0040_llama-cpp-qwen3-6-27b-tq3_4s-gguf.json`
- 27B full-agent sanity after config fix, showing OpenClaw `Connection error`: `/home/qiushuo/reports/pinchbench/qwen36-27b-tq3-4s-fullagent-sanity-after-configfix-20260512/0041_transcripts/task_sanity.jsonl`

