---
name: rx9070-qwen3.6-35b-a3b-128k-hybrid
description: Run unsloth Qwen3.6-35B-A3B (Qwen3-Next hybrid Mamba2+Attn MoE, 256 experts/8 active) at 128k context on this user's WSL + ROCm + RX 9070 16GB box via llama.cpp HIP build with `-ncmoe` hybrid CPU/GPU offload. Optimal point and pitfalls verified end-to-end.
---

# Qwen3.6-35B-A3B Q4_K_M @ 128k hybrid on RX 9070 16GB

## When to use
- Need Qwen3.6-35B-A3B (the Qwen3-Next variant) at long ctx on this single-GPU box
- Already on llama.cpp HIP build at /home/qiushuo/src/llama.cpp/build/bin
- Box has 23 GB RAM and 16 GB VRAM (RX 9070); a small NVIDIA Quadro P620 is also present but only used for Xwayland — ignore its nvidia-smi output

## Architecture facts (verified, helps reason about VRAM)
- arch tag in GGUF: `qwen35moe`
- 40 blocks total, **only 10 are full-attention** (indices 3,7,11,15,19,23,27,31,35,39 — every 4th). The other 30 are **SSM/Mamba2** (gated delta net).
- 256 experts, 8 used per token; expert FFN dim 768; shared expert present
- n_embd 2048, n_head 16, n_head_kv 2 (GQA 8x), head_dim 256 → K=V dim 512 per attn layer
- n_ctx_train (yarn) = 262144, so 128k is well within native; no rope scaling needed
- ssm: d_state 128, d_inner 4096, n_group 16; recurrent state buffer is tiny (~63 MiB total at 128k)

KV cache @ 128k q8_0: **only 1.36 GiB** (10 attn layers × 128k × (512K+512V) × 1 byte). This is the unlock — pure-attention 35B at 128k would need ~10 GB just for KV.

## Recommended commands (pick by ctx)

### 128k ctx (long-context inference, ncmoe=16 ub=256)
```
HSA_ENABLE_DXG_DETECTION=1 ROC_ENABLE_PRE_VEGA=0 \
/home/qiushuo/src/llama.cpp/build/bin/llama-server \
  -m /home/qiushuo/models/qwen/unsloth-Qwen3.6-35B-A3B-GGUF/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf \
  -ngl 999 -ncmoe 16 -c 131072 \
  -ub 256 -b 2048 \
  -ctk q8_0 -ctv q8_0 --kv-unified --parallel 1 --no-mmap \
  --host 127.0.0.1 --port 8765 \
  --jinja --reasoning auto
```
PP **94.8** / TG **26.1** tok/s. VRAM ~14.6 / 16 GB.

### 64k ctx (agent / tool-calling workloads, ncmoe=16 ub=1024 b=1024) ★ recommended for PinchBench
```
HSA_ENABLE_DXG_DETECTION=1 ROC_ENABLE_PRE_VEGA=0 \
/home/qiushuo/src/llama.cpp/build/bin/llama-server \
  -m /home/qiushuo/models/qwen/unsloth-Qwen3.6-35B-A3B-GGUF/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf \
  -ngl 999 -ncmoe 16 -c 65536 \
  -ub 1024 -b 1024 \
  -ctk q8_0 -ctv q8_0 --kv-unified --parallel 1 --no-mmap \
  --host 127.0.0.1 --port 8765 \
  --jinja --reasoning auto
```
PP **822** / TG **26.5** tok/s — **8.6× faster prefill** than the 128k config because larger ub fits in the freed KV budget. VRAM: model 13.16 + KV 0.68 + compute 0.99 = ~14.8 / 16 GB.

### CRITICAL: chat parsing flags for agent / tool-calling use
**Do NOT use `--reasoning off --reasoning-format none --skip-chat-parsing` for agent benchmarks (PinchBench, OpenClaw, function-calling).** That combo silently disables tool-call parsing in llama-server's Jinja layer, so the model emits valid `<tool_call>` text but the API returns it as raw `content` with no `tool_calls[]`. Symptom: agent runs complete without errors but score 0% on all tasks needing structured output (we hit 0/3 → 1.8/3 just by switching).

Correct flags for agent work:
- `--jinja` (enabled by default but be explicit)
- `--reasoning auto` — keeps thinking ON, exposes via `message.reasoning_content`
- DO NOT pass `--skip-chat-parsing`

Verify with two probes before benchmarking (see `scripts/verify_chat.sh`):
1. `reasoning_content` field length > 0 on a math prompt
2. `finish_reason: tool_calls` on a tool-required prompt

## Sweep table (Q4_K_M, KV q8_0, ncmoe=16)

### 128k ctx
| ncmoe | ub | GPU model | RSS | PP | TG | notes |
|------:|---:|----------:|----:|---:|---:|:------|
| 40    | 512 | 1.92 GB | 20.2 GB | 34.5 | 11.1 | safe fallback |
| 30    | 512 | 6.66 GB | 15.1 GB | 56.1 | 17.9 | conservative |
| 20    | 512 | 11.30 GB | 10.5 GB | 74.4 | 23.1 | prior winner |
| 18    | 512 | 11.94 GB |  9.5 GB | 79.7 | 24.5 | |
| 17    | 512 | 12.40 GB |  9.1 GB | 89.3 | 24.5 | |
| 16    | 512 | — | — | — | — | params_fit OOM |
| **16** | **256** | **12.85 GB** | **8.5 GB** | **94.8** | **26.1** | **★ 128k optimal** |
| 15    | 256 | — | — | — | — | params_fit OOM (hard cap) |

### 64k ctx (much smaller KV → bigger ub fits)
| ncmoe | ub | b | compute buf | PP | TG | notes |
|------:|---:|--:|------------:|---:|---:|:------|
| 16 | 512  | 512  | 493 MiB | 666 | 25.3 | old default |
| **16** | **1024** | **1024** | **986 MiB** | **822** | **26.5** | **★ 64k optimal** |
| 16 | 1024 | 2048 | 986 MiB | 845 | 25.3 | TG slightly drops |
| 16 | 2048 | 2048 | OOM | — | — | compute buf doesn't fit |
| 18 | 1536 | 1536 | 1479 MiB | 854 | 24.4 | TG loss not worth it |
| 20 | 512  | 2048 | crash on prefill | — | — | dxgkio_make_resident -12 |

Going below ncmoe=16 even at ub=256 hits a hard wall (params_fit refuses to abort). At 128k, reducing ub from 512→256 buys exactly enough compute-buffer headroom (493→361 MiB) to fit the extra 4 expert layers on GPU at ncmoe=16. At 64k, the KV savings (1.36 → 0.68 GiB) free up ~700 MiB which lets ub double to 1024, giving a ~9× PP speedup vs the 128k config and ~23% PP vs the prior 64k default.

## Pitfalls

- **Quadro P620 nvidia-smi readings are unrelated** — that's the secondary card running Xwayland. ROCm work happens on RX 9070; rocm-smi is unavailable on WSL; rely on llama.cpp's log-reported buffer sizes for VRAM accounting.
- **`-ngl 999` blocks llama.cpp's auto-fit fallback**: when params don't fit, llama.cpp aborts with "n_gpu_layers already set by user to 999, abort" instead of a graceful retry. Read this as VRAM OOM, not a bug. If hit, lower ncmoe by 1 or reduce `-ub`.
- **`-ub 256` is the trick at the knee**: shrinks compute buffer enough to fit one more block of experts on GPU. Throughput is *higher* at ub=256 here because it lets more MoE layers stay GPU-resident (no PCIe expert traffic per token).
- **Always pass `--no-mmap`** with `-ncmoe`; mmap+partial-offload is inconsistent on ROCm.
- **`--kv-unified --parallel 1`** keeps KV layout predictable for measurement; for serving with multiple slots you'll need to re-test budget.
- **First load reads 22 GB from disk** (~3 min cold). Subsequent loads are fast (page cache). Quote cold and warm separately when reporting.
- **Reasoning flags**: `--reasoning off --reasoning-format none --skip-chat-parsing` matches the user's preferred llama-server config (think tag handled out-of-band, not stripped silently).
- **23 GB system RAM is the hard ceiling for raising ncmoe further** — at ncmoe=40 RSS is already 20.2 GB. Don't push experts further to CPU; instead push to GPU until VRAM saturates.
- **PinchBench / OpenClaw integration**:
  - The openclaw CLI lives at `/home/qiushuo/.hermes/node/bin/openclaw`. PinchBench's benchmark.py spawns it as `openclaw` (no path), so any launcher script must `export PATH="/home/qiushuo/.hermes/node/bin:$PATH"` or tasks fail instantly with `openclaw CLI not found while listing agents`.
  - Use `--parallel 1` only. `--parallel 2` triggered OpenClaw's `[llm-idle-timeout]` watchdog mid-task, killing the run.
  - Combined with the chat-parsing fix above, smoke went from 0/3 (all errored or 0% scored) to 1.0 + 0.8 + ... within the first two tasks.

## Quick bench helper
See `scripts/bench.py` — sends a deterministic 50-line counting prompt to /v1/chat/completions and prints PP/TG from the timings field.

## Source files
- Reports: `/home/qiushuo/reports/qwen3.6-35b-128k/`
  - `sweep_summary.txt`, `sweep2_summary.txt`, `run_ncmoe{40,30,20,19,18,17,16,15,10}*.log`
- Model file: `/home/qiushuo/models/qwen/unsloth-Qwen3.6-35B-A3B-GGUF/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf` (22.13 GB)
- HF repo: `unsloth/Qwen3.6-35B-A3B-GGUF`, file `Qwen3.6-35B-A3B-UD-Q4_K_M.gguf`
