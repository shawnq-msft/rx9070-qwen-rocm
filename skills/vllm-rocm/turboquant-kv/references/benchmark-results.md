# vLLM TurboQuant benchmark — RX 9070 16GB / WSL / ROCm 7.2.2

Source runs under `/home/qiushuo/reports/vllm-rocm-eval/results/`.
All on `caiovicentino1/Qwopus3.5-9B-v3-HLWQ-v7-GPTQ` (9B hybrid Mamba+Attn, GPTQ 4bit weights).
vLLM HEAD `5cdddddd4` + local hybrid-TurboQuant patches.

## HumanEval full 164 tasks

| KV dtype | ctx | KV tokens | GPU @ ready | pass@1 | avg tok/s | run dir |
|---|---|---|---|---|---|---|
| TQ3 (`turboquant_3bit_nc`) | 65536 | 217,152 | 13.99 GiB | 0.640 | 36.6 | `qwopus-tq3-tq4-full-humaneval-20260420-082305/tq3` |
| **TQ4 (`turboquant_4bit_nc`)** | **131072** | 133,056 | 13.12 GiB | **0.677** | **58.3** | `qwopus-tq3-tq4-full-humaneval-20260420-082305/tq4` |
| TQ K8V4 plugin | 65536 | — | — | 0.671 | 36.1 | `humaneval-qwopus-tqk8v4-64k-20260428-224336` |
| FP8 baseline | 65536 | — | — | 0.689 | 52.5 | `humaneval-qwopus-fp8kv-64k-20260428-222822` |

Best vLLM TurboQuant config: **TQ4 + ctx 131072 + mbt 16384 + seqs 1** (`tq4_ctx131072_mbt16384_seq1`). Quality within ~1pt of FP8 while doubling context.

## Continuous batching (TQ4, 4 concurrent)

| candidate | ready_s | gpu_used | kv_tokens | agg tok/s | p95 lat |
|---|---|---|---|---|---|
| ctx131072 mbt4096 seq4 | 308.6 | 13.28 GiB | 139,392 | 96.27 | 4.89 s |
| ctx131072 mbt8192 seq4 | 322.0 | 13.61 GiB | 136,224 | 96.52 | 4.84 s |

Source: `qwopus-tq4-continuous-b4-20260420-151219`.

## Long-context boundaries (single-prompt)

- TQ3: 24k–65k stable. Higher fails on KV alloc.
- TQ4: 49k single-long-prompt failed in one run, but full HumanEval 128k path works at mbt=16384/seq=1.
- FP8: verified up to **196,608** prompt tokens (`qwopus-fp8-max-real-ctx`). 229k stalls.
- This makes FP8 the right choice when ctx >128k matters and TurboQuant unnecessary.

## Short-prompt token rate (128 completion tokens)

| KV | tok/s | run |
|---|---|---|
| TQ3 | 40.3 | `kv-compare-patched/tq3_kv` |
| TQ3 (other run) | 56.5 | `tq3-64k-128k/ctx_65536` |
| FP8 | 8.9 | `kv-compare-patched/fp8_kv` (anomalously slow, recheck if relied upon) |
| auto fp16 | 16.2 | `kv-compare-patched/auto_kv` |

## Untested but plausible model candidates (require GPTQ/AWQ/FP8 weights)
- Qwen3.6-A3B GPTQ — would need to find/produce; 35B-A3B too large for 16 GiB without offload.
- Qwen3.6-27B GPTQ — borderline fit; depends on group size and KV.
- Anything in `~/models/qwen/` ending in `.gguf` is **NOT loadable here** — route to llama.cpp.

## Decision tree
1. Need >128k ctx? → FP8 KV.
2. Need 128k ctx + tight VRAM? → TQ4 KV.
3. Short ctx (≤64k), max quality? → FP8 KV.
4. KV size dominates and quality loss acceptable? → TQ3 KV.
5. Model only available as GGUF? → wrong runtime, use llama.cpp fork.
