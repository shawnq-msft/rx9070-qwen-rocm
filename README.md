# RX 9070 + Qwen on ROCm/HIP — Field Notes

Reproducible serve configs, build flags, KV-quant benchmarks, and pitfalls for
running **Qwen 3.5 / 3.6 (27B & 35B-A3B)** GGUF models on a single
**AMD Radeon RX 9070 16 GB** (RDNA4, gfx1201) via **llama.cpp HIP** under
**WSL2 + ROCm 7.x**.

Everything here is verified end-to-end on one box, not theoretical.

## TL;DR

| Model | Best config | tok/s | Max ctx | Notes |
|---|---|---|---|---|
| Qwen3.6-27B Q4_K_S | FA on, K=q8_0 V=q8_0, ctx=24k | **~22** | 24k | daily driver |
| Qwen3.6-27B Q4_K_S | FA on, K=q4_0 V=q4_0, ctx=40k | ~21 | 40k | long-ctx fallback |
| Qwen3.6-35B-A3B Q4_K_M | hybrid `-ncmoe`, ctx=128k | see skill | 128k | MoE, partial CPU offload |

## The non-obvious gotchas

1. **Qwen 3.5 / 3.6 27B is NOT dense** — `arch=qwen35` reports as hybrid
   Mamba (Gated Delta Net) + Attention. Only 16 of 64 layers carry KV.
   This makes `--parallel`, KV-offload, and ctx-budget reasoning very
   different from dense 27B models.
2. **RDNA4 + flash attention requires a build flag**. Stock llama.cpp HIP
   build aborts with `invalid device function` on `head_dim=256` the moment
   you enable `-fa on`. Fix: rebuild with
   `-DGGML_HIP_ROCWMMA_FATTN=ON` (needs `rocwmma-dev` ≥ 2.2). Without FA,
   you can't quantize V cache, and ctx ceiling collapses.
3. **KV quant hierarchy on RDNA4 is not what you'd expect**. `q4_0` and
   `q8_0` are fast (block-only, fattn-mma fast path). `q4_1`, `q5_1`,
   `iq4_nl` are 30–40% slower. There is **no `q6_0` KV** in mainline
   llama.cpp — only `q5_1` (~6bpw).
4. **35B-A3B at 128k fits** only with `-ncmoe` hybrid CPU offload of MoE
   experts. See `skills/qwen-35b-a3b/SKILL.md` for the optimal split.

## Layout

```
skills/
  llamacpp-build/   # rocWMMA FATTN flag, head_dim=256 trap, V-cache rules
  qwen-27b/         # serve configs + KV-quant sweep results
  qwen-35b-a3b/     # 128k hybrid MoE config
scripts/            # ready-to-run launcher shell scripts
benchmarks/         # raw numbers
```

The `skills/*/SKILL.md` files are written in [Hermes Agent skill format](https://hermes-agent.nousresearch.com/docs)
but they're plain markdown — readable standalone.

## Environment

- AMD Radeon RX 9070 16 GB (RDNA4, gfx1201)
- 23 GB system RAM
- WSL2 (Ubuntu) on Windows
- ROCm 7.2 (radeon WSL build)
- llama.cpp mainline at `~/src/llama.cpp`, built with
  `-DGGML_HIP=ON -DGGML_HIP_ROCWMMA_FATTN=ON -DAMDGPU_TARGETS=gfx1201
   -DCMAKE_BUILD_TYPE=Release -DGGML_HIP_NO_VMM=ON`
- rocWMMA 2.2.0 from `/opt/rocm/include/rocwmma`

The small NVIDIA Quadro P620 visible in `nvidia-smi` is Xwayland-only.
Ignore it.

## License

MIT.
