---
name: rx9070-qwen3.6-27b-llamacpp-hip
description: Run Qwen3.6-27B GGUF on this user's WSL + ROCm + RX 9070 16GB box via llama.cpp HIP build. Covers the (surprising) hybrid Mamba+Attn architecture, the rocWMMA-FATTN build flag that unlocks RDNA4 flash attention, and the verified 24k-context serve config.
---

# Qwen3.6-27B on RX 9070 (WSL + ROCm + llama.cpp HIP)

## When to load
- Picking serve config for Qwen3.6-27B GGUF on this 16 GB box.
- User asks about KV quantization, TurboQuant TQ4/K4V3, KV offload to RAM, OOM, or "FA invalid device function" on RDNA4.

## Critical architecture fact (do NOT treat as dense)
Qwen3.6-27B reports `arch=qwen35` in llama.cpp, but it is a **hybrid Mamba (Gated Delta Net) + Attention** model:
- n_layer=64, but only 16 layers carry attention/KV. The other 48 are SSM with recurrent-state (RS) cache.
- n_head=24, n_head_kv=4, **head_dim=256**.
- Per-seq RS cache ≈ 150 MiB at any ctx. Multiplied by `--parallel N` (default 4) → easy OOM.
- KV is small (~3% of VRAM). Real budget killer is the 14.4 GB weight buffer + RS + compute.

## RDNA4 flash-attention: rocWMMA build flag is mandatory
On RX 9070 (gfx1201), llama.cpp's default fattn path lacks head_dim=256 kernels — `-fa on` aborts with `invalid device function`. The fix is build-side, not runtime:

```bash
# rocWMMA 2.2.0 must be installed (apt: rocwmma-dev)
cd ~/src/llama.cpp/build
cmake .. -DGGML_HIP=ON \
  -DGGML_HIP_ROCWMMA_FATTN=ON \
  -DAMDGPU_TARGETS=gfx1201 \
  -DGGML_HIP_NO_VMM=ON \
  -DCMAKE_BUILD_TYPE=Release
cmake --build . --target llama-server -j$(nproc)
```

This activates the RDNA4 + AMD_WMMA path in `fattn-mma-f16.cuh`, which covers head_dim=256 with K/V quantization.

Verify the cache after rebuild:
```
grep ROCWMMA_FATTN ~/src/llama.cpp/build/CMakeCache.txt
# expect: GGML_HIP_ROCWMMA_FATTN:BOOL=ON
```

## Verified working config (single-user serve, FA on)
```bash
HSA_ENABLE_DXG_DETECTION=1 ROC_ENABLE_PRE_VEGA=0 \
~/src/llama.cpp/build/bin/llama-server \
  -m ~/models/qwen/unsloth-Qwen3.6-27B-GGUF/Qwen3.6-27B-Q4_K_S.gguf \
  --host 127.0.0.1 --port 8820 \
  -ngl 999 -fa on \
  -ctk q8_0 -ctv q8_0 \
  -c 24576 -ub 256 -b 1024 --parallel 1 \
  --reasoning off --reasoning-format none --skip-chat-parsing
```
Measured (Q4_K_S, ctx 24k):
- 300-tok gen: 22.6 tok/s
- 600-tok gen: 21.2 tok/s (no dequant decay vs. FA-off path)
- 16k ctx alt: ~22 tok/s steady state

## Context limits with FA on
With KV `q8_0/q8_0`:
| ctx | result |
|-----|--------|
| 16384 | ✅ ~22 tok/s |
| 24576 | ✅ ~21 tok/s |
| 32768 | ❌ OOM (compute pp buffer 247 MiB short) |
| 65536 | ❌ OOM (KV buffer 2176 MiB) |

With KV `q4_0/q4_0` you can push to **40960** at ~21 tok/s (see
`references/kv-quant-sweep-rx9070.md`). 45056+ hangs at hipblas init; 49152+ OOMs.

## KV quantization rule for this card
Only two KV combos are worth using on RX 9070 with this model:
- `-ctk q8_0 -ctv q8_0 -c 24576` → max quality, ~22 tok/s
- `-ctk q4_0 -ctv q4_0 -c 40960` → max ctx, ~21 tok/s — see `scripts/serve-q4ks-q4kv-40k.sh`

**Avoid `q4_1`, `q5_1`, `iq4_nl` for KV.** RDNA4 fattn-mma has no fast kernel for `_1`/non-linear
KV types — they cost 30-40% throughput with no quality upside. There is no `q6_0/q6_K` for KV.
Full sweep table in `references/kv-quant-sweep-rx9070.md`.

## Mandatory runtime constraints
- `--parallel 1` — default 4 multiplies RS cache and OOMs even at 16k.
- `-fa on` requires the rocWMMA build above. Without it, decode aborts on first token.
- KV q8_0/q8_0 is fine with FA on; saves ~200 MiB vs. f16 KV.
- Don't use `--no-kv-offload` — hybrid model still computes RS on GPU; speed collapses.

## Things NOT to do on this model/machine
- Don't reach for TurboQuant TQ4 / K4V3 KV: TurboQuant llama.cpp forks need TurboQuant weights (none on HF for 27B), and KV is not the bottleneck.
- Don't reach for vLLM: only Qwopus 9B GPTQ verified here; 27B GPTQ weights ~18-20 GB > 16 GB.
- Don't try ctx ≥ 32768 with KV q8_0 — compute pp buffer doesn't fit.
- Don't pick MTP variants (froggeric / havenoammo / RDson): mainline llama.cpp does not decode MTP heads → no speed gain, just bigger files.

## Model variant guidance (HF, sizes verified)
Best for this box: `unsloth/Qwen3.6-27B-GGUF` Q4_K_S (14.76 GiB) or IQ4_XS (15.44 GiB).
Higher quality (Q4_K_M 16.82 GiB, UD-Q4_K_XL 17.61 GiB) require partial -ngl offload to CPU; speed drops sharply, not worth it on a single 9070.

## KV cache quant sweep (Q4_K_S, FA on, ub=256 b=1024 parallel=1)

| -ctk / -ctv | max ctx | 300tok | 800tok | 备注 |
|---|---|---|---|---|
| **q8_0 / q8_0** | **24k** | **22.6** | **~22** | **★默认 daily driver** |
| q4_0 / q4_0 | 40k | 19.1 | 21.4 | ★长 ctx fallback (>20k) |
| q5_1 / q4_1 | 32k | 13.4 | 12.6 | 慢 ~40%，不要用 |
| q4_1 / q4_1 | 40k | 14.4 | 13.4 | 慢，不要用 |
| iq4_nl/iq4_nl | 40k | 13.1 | 12.9 | 最慢，不要用 |

注意：llama.cpp HIP 没有 q6 KV 选项；K 最高量化是 q5_1。
RDNA4 fattn-mma 对 block-only 量化（q8_0、q4_0）有快路径；带 mins 的 q4_1/q5_1 和非线性 iq4_nl 走慢路径，性能赔 30-40%。
49k+ ctx 全部在 hipblas init 时 OOM；40k 是稳定天花板。

启动脚本：
- `~/.local/bin/qwen27b-default.sh` — q8_0/q8_0 24k（推荐默认）
- `~/.local/bin/qwen27b-longctx.sh` — q4_0/q4_0 40k（仅长上下文用）

## Performance reference
Same machine, same Q4_K_S, before/after rocWMMA fix:

| build | KV | ctx | short gen | long gen |
|---|---|---|---|---|
| FA off (no rocwmma) | q8_0/f16 | 12k | 17 | 17 |
| FA off (no rocwmma) | f16/f16 | 8k | 22.7 | 18 |
| **FA on (rocwmma)** | **q8_0/q8_0** | **24k** | **22.6** | **21.2** |

Net: long-gen +30%, ctx ×2, no quality regression.

## Related skills
- `rx9070-qwen3.6-35b-a3b-128k-hybrid` — MoE sibling (35B-A3B), uses `-ncmoe`. Doesn't apply here (27B is not MoE) but shares the rocWMMA build prereq.
- `rx9070-gguf-hermes-eval` — evaluation harness.
- `rx9070-vllm-turboquant` — explains why the vLLM TQ4/K4V3 path is closed for this model.
