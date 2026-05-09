---
name: rx9070-llamacpp-hip-fattn-headdim-traps
description: Diagnose llama.cpp HIP build crashes / OOMs caused by missing FlashAttention kernel specializations on RX 9070 (RDNA4 gfx1201) when serving models with non-standard attention head_dim (notably head_dim=256 on Qwen3.5/3.6 hybrid Mamba+Attn arch). Covers the chain "FA on → invalid device function → must keep V=f16 → can't shrink KV → ctx ceiling drops" and the working fallback configs. Load before quoting any KV-quant / long-ctx plan for a Qwen3.5/3.6 27B–class GGUF on this user's box.
---

# RDNA4 llama.cpp HIP — fattn head_dim trap

This is a recurring class on this user's WSL + ROCm 7.2 + RX 9070 (gfx1201) + mainline `~/src/llama.cpp` HIP build. The HIP fattn template instances that ship in mainline llama.cpp do not cover every `(head_dim, kv_dtype)` combo for RDNA4. Hitting an uninstantiated combo crashes at first decode, not at load time — so the model looks fine until the first token.

## Trigger this skill when

- User asks about KV quantization, long context, FlashAttention, TurboQuant TQ4/K4V3, or KV-to-RAM offload on a Qwen3.5 / Qwen3.6 27B-class GGUF (or anything with `arch = qwen35` in loader log, including 35B-A3B).
- A llama-server run aborts with `ROCm error: invalid device function` inside `launch_fattn` / `fattn-common.cuh`.
- A llama-server run is rejected at init with `V cache quantization requires flash_attn`.

## Symptom signatures

1. Crash on first decode, FA on, head_dim=256:
   ```
   ROCm error: invalid device function
     in function launch_fattn at ggml/src/ggml-cuda/template-instances/../fattn-common.cuh
     hipOccupancyMaxActiveBlocksPerMultiprocessor(...)
   ```
   Aborted (core dumped). Means: no compiled fattn instance for this (head_dim, dtype, block) on gfx1201.

2. Refused at context init, FA off + V quantized:
   ```
   llama_init_from_model: V cache quantization requires flash_attn
   ```
   Means: llama.cpp policy — V-quant only via the FA path. Combined with #1 this forces V=f16.

3. compute pp buffers OOM with KV f16 at high ctx:
   ```
   ggml_backend_cuda_buffer_type_alloc_buffer: allocating ~500 MiB ... cudaMalloc failed: out of memory
   graph_reserve: failed to allocate compute buffers
   ```

## Hard-won decision tree (Qwen3.5/3.6 qwen35-arch class)

1. Confirm arch from loader log: `arch = qwen35`. If yes, it is hybrid (Mamba2 / Gated Delta Net + Attn). Only ~25% of layers carry KV; KV cache is small and **KV-quant savings are nearly worthless** here (real VRAM hog is weights + compute buffer + RS cache).
2. Do **not** route the user to TurboQuant fork for these models unless TurboQuant-quantized weights exist on HF (verified 2026-05: 35B-A3B yes, 27B no).
3. Do **not** suggest `--no-kv-offload` — speed collapses, RS cache stays on GPU anyway.
4. Force V=f16. Choose K from {q8_0, f16}. K=q8_0 saves ~80–180 MiB.
5. Default `--parallel` is 4 → RS cache ×4 → OOM. Always pin `--parallel 1` for single-user roundtable use.
6. compute buffer is the wall. Drop ctx, then drop `-ub`/`-b`, then accept the resulting ceiling. On 27B Q4_K_S the ceiling is ctx≈12k.

## Verified working baseline (Qwen3.6-27B Q4_K_S, 14.76 GiB weights)

```bash
export HSA_ENABLE_DXG_DETECTION=1 ROC_ENABLE_PRE_VEGA=0
~/src/llama.cpp/build/bin/llama-server \
  -m ~/models/qwen/unsloth-Qwen3.6-27B-GGUF/Qwen3.6-27B-Q4_K_S.gguf \
  --host 127.0.0.1 --port 8820 \
  -ngl 999 -fa off \
  -ctk q8_0 -ctv f16 \
  -c 12288 -ub 256 -b 1024 --parallel 1 \
  --reasoning off --reasoning-format none --skip-chat-parsing
```
- VRAM ~15.7 GB (~300 MiB headroom)
- TG ≈ 17 tok/s
- ctx 12k is the ceiling on this build until upstream adds head_dim=256 fattn instances for RDNA4

## Verified failing configs (do not re-test before checking upstream changes)

| config | failure |
| --- | --- |
| `-fa on -ctk q8_0 -ctv q8_0` | invalid device function (fattn h=256) |
| `-fa on -ctk f16 -ctv f16`, ctx 16k | compute buffer OOM (~500 MiB) |
| `-fa on -ctk f16 -ctv f16`, ctx 16k, `-ub 128 -b 512` | still compute buf OOM (~124 MiB) |
| `-fa off -ctk q8_0 -ctv q8_0` | refused: V quant needs FA |
| ctx ≥ 16k (any KV combo) | OOM |
| `--parallel 4` (default) | RS cache ×4, OOM at sched_reserve |

## When to retest / lift the ceiling

Retest the FA-on path whenever a llama.cpp HIP rebuild touches:
- `ggml/src/ggml-cuda/template-instances/fattn-*` (new head_dim instances)
- anything mentioning `gfx1201`, `RDNA4`, `head_dim=256`, or `qwen35`

If FA-on starts working with q8_0 KV, ctx jumps from 12k toward 24–32k for Qwen3.6-27B and similarly for the 35B-A3B sibling.

## Related skills

- `rx9070-qwen3.6-35b-a3b-128k-hybrid` — sibling MoE-hybrid; different VRAM dynamics, same arch family.
- `rx9070-gguf-hermes-eval` — class-level GGUF eval harness for this box.
- `rx9070-moe-cpu-offload-sweep` — for MoE only; does not apply to dense-hybrid 27B.
- `rx9070-vllm-turboquant` / `rx9070-agustin-llamacpp-turboquant` — TurboQuant routing; explicitly NOT recommended for Qwen3.6-27B (no TQ weights exist).

## Reference data

- `references/qwen3.6-27b-q4ks-validation.md` — session log: HF model inventory (unsloth/froggeric/havenoammo/RDson/Lorbus sizes), full failing-config matrix with stderr, TQ4/K4V3 verdict.
