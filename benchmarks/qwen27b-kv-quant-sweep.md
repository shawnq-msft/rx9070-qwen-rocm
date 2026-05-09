# Qwen3.6-27B Q4_K_S KV-quant sweep on RX 9070 16 GB

Date: 2026-05-09
Model: `unsloth/Qwen3.6-27B-GGUF` Q4_K_S variant (14.76 GiB, 4.71 BPW)
Build: llama.cpp mainline + `-DGGML_HIP_ROCWMMA_FATTN=ON`, gfx1201
Server flags: `-ngl 999 -fa on -ub 256 -b 1024 --parallel 1`
Prompt: short essay-style ("Write a detailed essay on the history of computing")

| `-ctk` / `-ctv` | Stable max ctx | 300-tok gen | 800-tok gen | Verdict |
|---|---:|---:|---:|---|
| **q8_0 / q8_0** | **24k** | **22.6** | **~22** | ★ default |
| q4_0 / q4_0 | 40k | 19.1 | 21.4 | ★ long-ctx fallback |
| q5_1 / q4_1 | 32k | 13.4 | 12.6 | slow, skip |
| q4_1 / q4_1 | 40k | 14.4 | 13.4 | slow, skip |
| iq4_nl / iq4_nl | 40k | 13.1 | 12.9 | slowest, skip |

Failures observed:
- All KV variants OOM at ctx ≥ 49k during `hipblasCreate` (compute buffer
  cannot be allocated alongside weights+KV+RS).
- 45k with `q4_0/q4_0` hangs ≥4 min in hipblas init — VRAM is critically
  tight; treat 40k as the practical ceiling.
- 32k with `q5_1/q4_1` (heavier KV-K) failed compute-buffer alloc at 247
  MiB before final pre-run; 24k worked.

Why `q4_0`/`q8_0` are fast and `q4_1`/`q5_1`/`iq4_nl` aren't:
the RDNA4 fattn-mma kernels have fast paths only for block-only quant
formats (no `mins` field, no non-linear codebook). Anything with mins
(`q4_1`, `q5_1`) or non-linear (`iq4_nl`) falls back to a generic path
that costs ~30–40 % throughput.
