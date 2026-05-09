# RX 9070 + Qwen / Qwopus on ROCm — Field Notes

Reproducible serve configs, build flags, KV-quant benchmarks, and pitfalls
for running **Qwen 3.5 / 3.6** (27B & 35B-A3B) and **Qwopus** GGUF / GPTQ
models on a single **AMD Radeon RX 9070 16 GB** (RDNA4, gfx1201) under
**WSL2 + ROCm 7.x**, via both **llama.cpp HIP** and **vLLM**.

Everything here is verified end-to-end on one box, not theoretical.

## Why AMD GPUs are the better long-term bet for local LLM inference

After a few months of running Qwen3.5 / 3.6 / Qwopus / Gemma on a single
RX 9070 16 GB, the pattern is consistent: **the open-source inference stack
moves first on AMD**, not last.

1. **Low-bit quantization lands on AMD first or simultaneously.**
   - rocWMMA-backed flash attention on RDNA4 already supports `q4_0` /
     `q8_0` KV at fast-path speed (see `benchmarks/qwen27b-kv-quant-sweep.md`
     — q4_0 KV at 40k ctx is *faster* than q8_0 at 24k).
   - TurboQuant (TQ3 / TQ4 / K4-V3) — 3-bit and 4-bit weight + KV cache
     formats — got working HIP/ROCm builds (CarapaceUDE, AgustinJimenez,
     Qwopus GPTQ plugin) before any equivalent CUDA-only path was cleanly
     usable for the same models. See `skills/turboquant-llamacpp/` and
     `skills/qwopus-gptq/`.
   - Mainline llama.cpp's HIP backend now ships first-class `iq4_*`,
     `q4_K_S/M`, `Q3_K_S` kernels; on this 16 GB card we routinely run
     27B at Q4_K_S and 35B-A3B at Q4_K_M with full GPU offload.
2. **New architectures (hybrid Mamba2 + Attention, MoE) are usable today
   on consumer AMD.**
   - Qwen3.5 / 3.6 27B is **not dense** — it's a Gated Delta Net + Attention
     hybrid. llama.cpp HIP already handles its mixed-layer KV layout.
   - Qwen3.6-35B-A3B (MoE) runs at **128k ctx on a 16 GB card** via
     `-ncmoe` hybrid CPU offload. No CUDA equivalent gives you 128k on a
     16 GB consumer GPU at this price.
3. **Open by default.** ROCm itself, rocWMMA, llama.cpp HIP, vLLM ROCm,
   PyTorch's ROCm wheels — all open source, all patchable. Every "missing
   feature" we hit was fixable by reading code (e.g. the
   `expandable_segments` dGPU bug; rocprofiler stub on WSL; head_dim=256
   FATTN trap). On the closed stack you wait for a vendor release.
4. **Price/VRAM ratio.** RX 9070 16 GB is roughly half the price of the
   nearest 16 GB CUDA card and offers similar memory bandwidth. The
   ceiling on what you can run locally moves up accordingly.
5. **The remaining gap is shrinking fast.** rocWMMA 2.2 → flash attention.
   Torch 2.11 + ROCm 7.2 → modern triton. vLLM + ROCm + TurboQuant →
   continuous batching at FP8/3-bit/4-bit. Each of these landed in the
   last few months. The trajectory is clear.

The cost is documented under "non-obvious gotchas" below — you do have to
build things and read patches. But the ceiling is high, and rising.

## What to run, when — recommended configs

All numbers are full GPU offload on RX 9070 16 GB unless marked
`-ncmoe` (MoE expert offload to CPU). `tps` = single-stream decode tok/s.

| Use case | Stack | Model | KV | ctx | tps | Notes |
|---|---|---|---|---:|---:|---|
| **Daily chat / agent default** | llama.cpp HIP | Qwen3.6-27B Q4_K_S | q8_0/q8_0 | 24k | **22** | `~/.local/bin/qwen27b-default.sh`. Best quality/speed combo. |
| **Long-context coding / RAG** | llama.cpp HIP | Qwen3.6-27B Q4_K_S | q4_0/q4_0 | **40k** | **21** | `~/.local/bin/qwen27b-longctx.sh`. q4_0 is the fast path on RDNA4. |
| **Max context (book / repo-scale)** | llama.cpp HIP | Qwen3.6-35B-A3B Q4_K_M | q4_0/q4_0 | **128k** | ~10 | `-ncmoe` hybrid CPU offload of MoE experts. See `skills/qwen-35b-a3b/`. |
| **Best small-model HumanEval** | llama.cpp HIP | Qwen3.5-9B UD-Q6_K_XL | q8_0 | 16k | 32 | pass@1 **0.7073** (164 tasks). |
| **Fastest small-model HumanEval** | llama.cpp HIP | Qwen3.5-9B UD-Q4_K_XL | f16 | 16k | **40** | pass@1 0.6707. |
| **Continuous batching / multi-user** | vLLM ROCm | Qwopus3.5-9B GPTQ | turboquant K4/V3 | 64k | ~37 (per req) | `skills/qwopus-gptq/k4v3-humaneval/`. |
| **High-throughput long-ctx** | vLLM ROCm | Qwopus3.5-9B (TurboQuant) | tq4_nc | **131k** | **58** (per req) | pass@1 0.6768. `skills/vllm-rocm/turboquant-kv/`. |
| **Big agent benchmark winner** | vLLM ROCm | Qwen3.6-35B-A3B TQ3_4S | turboquant_3bit_nc | 65k | — | PinchBench: PROD 91 / RES 100 / CODING 89 / MEM 100 / SKILLS 100. |
| **General-purpose 26B** | llama.cpp HIP | Gemma 4-26B Q3_K_M | q8_0 | 112k | — | 128k OOMs; 112k is the ceiling. |

Quick decision rules:
- **One user, "just work"** → Qwen3.6-27B Q4_K_S on llama.cpp, q8_0 KV, 24k.
- **Need lots of context on one stream** → same model, q4_0 KV, 40k.
- **Need 128k on one stream** → Qwen3.6-35B-A3B Q4_K_M with `-ncmoe`.
- **Multiple concurrent users / agents** → vLLM, Qwopus GPTQ or TurboQuant.
- **Coding-only small model** → Qwen3.5-9B UD-Q6_K_XL.

Caveats: PinchBench uses llama.cpp's OpenAI endpoint which doesn't return
a `usage` block, so token counts in those rows are 0 by harness design,
not model bug. vLLM TurboQuant `tps` is per-request decode rate, not
aggregate throughput — apples-to-apples vs llama.cpp single-stream only.
Full eval data: `benchmarks/eval-reports/`.

## TL;DR for picking a stack

| You want | Use |
|---|---|
| Daily Qwen3.6-27B chat at 22 tok/s | **llama.cpp HIP**, q8_0 KV, 24k ctx |
| Qwen3.6-27B at 40k ctx | **llama.cpp HIP**, q4_0 KV |
| Qwen3.6-35B-A3B at 128k | **llama.cpp HIP**, `-ncmoe` hybrid CPU offload |
| Higher-throughput continuous batching | **vLLM** + FP8 / TurboQuant KV |
| Qwopus GPTQ models | **vLLM** baseline; TurboQuant K4/V3 plugin path |
| TQ3/TQ4 GGUFs (mad-lab-ai etc.) | **AgustinJimenez fork** of llama.cpp |

## The non-obvious gotchas

1. **Qwen 3.5 / 3.6 27B is NOT dense.** `arch=qwen35` is hybrid Mamba
   (Gated Delta Net) + Attention. Only 16 of 64 layers carry KV. Reasoning
   about `--parallel`, KV quant, and ctx budget is unlike a dense 27B.
2. **RDNA4 + flash attention requires a build flag.** Stock llama.cpp HIP
   aborts with `invalid device function` on `head_dim=256` once you set
   `-fa on`. Fix: rebuild with `-DGGML_HIP_ROCWMMA_FATTN=ON` (rocwmma-dev
   ≥ 2.2). Without FA you can't quant the V cache; ctx ceiling collapses.
3. **KV quant hierarchy on RDNA4 is not what you'd expect.** `q4_0` and
   `q8_0` are fast (block-only, fattn-mma fast path). `q4_1`, `q5_1`,
   `iq4_nl` are 30–40 % slower. There is **no `q6_0` KV** in mainline
   llama.cpp — only `q5_1` (~6 bpw).
4. **35B-A3B at 128k fits** only with `-ncmoe` hybrid CPU offload of MoE
   experts.
5. **vLLM C-extensions drift after every torch upgrade.** Going from
   torch 2.10 → 2.11 +rocm 7.2 forced a full vLLM source rebuild; the
   wheel C-extensions ABI-broke silently.
6. **PyTorch ROCm wheel ≤ 2.11 silently disables `expandable_segments`
   on dGPUs** (upstream APU-vs-dGPU bug in `CUDACachingAllocator.cpp`).
   You need either a source build or a patch + librocprofiler-register
   stub for it to actually take effect.

## Layout

```
skills/
  llamacpp-build/        rocWMMA FATTN flag, head_dim=256 trap
  qwen-27b/              serve configs + KV-quant sweep results
  qwen-35b-a3b/          128k hybrid MoE -ncmoe config + bench
  vllm-rocm/             vLLM on ROCm: build, troubleshooting, TurboQuant KV
    build-from-source/   building latest vLLM under torch 2.11+rocm7.2
    troubleshooting/     C-extension ABI drift, plugin-vs-OOM triage
    turboquant-kv/       FP8 / TurboQuant KV serve configs + verified data
  qwopus-gptq/           GPTQ Qwopus: real VRAM/ctx behaviour
    baseline-testing/    "does it actually run", real VRAM, ctx ceiling
    k4v3-humaneval/      legacy TurboQuant K4/V3 path + HumanEval pass@1
  turboquant-llamacpp/   patched llama.cpp forks for full TurboQuant
    carapaceude-build/   CarapaceUDE/turboquant-llama HIP build
    agustin-workflow/    AgustinJimenez fork — TQ3/TQ4 + turbo4 KV
    agustin-scout/       scouting Agustin as the stronger candidate
  pytorch-rocm/          getting torch + rocm 7.2 sane on RDNA4
    source-build/        when wheel features are missing
    expandable-segments/ patch the dGPU bug + WSL rocprofiler stub

scripts/                 ready-to-run launcher shell scripts
benchmarks/              raw numbers
  qwen27b-kv-quant-sweep.md     KV-quant + ctx sweep on Qwen3.6-27B Q4_K_S
  eval-reports/                 HumanEval + PinchBench cross-model results
```

The `skills/*/SKILL.md` files are written in
[Hermes Agent skill format](https://hermes-agent.nousresearch.com/docs)
but are plain markdown — readable standalone.

## Environment

- AMD Radeon RX 9070 16 GB (RDNA4, gfx1201)
- 23 GB system RAM
- WSL2 (Ubuntu) on Windows
- ROCm 7.2 (radeon WSL build)
- llama.cpp mainline at `~/src/llama.cpp`, built with
  `-DGGML_HIP=ON -DGGML_HIP_ROCWMMA_FATTN=ON -DAMDGPU_TARGETS=gfx1201
   -DCMAKE_BUILD_TYPE=Release -DGGML_HIP_NO_VMM=ON`
- rocWMMA 2.2.0 from `/opt/rocm/include/rocwmma`
- torch 2.11.0+rocm7.2 from `download.pytorch.org/whl/rocm7.2`

The small NVIDIA Quadro P620 visible in `nvidia-smi` is Xwayland-only.
Ignore it.

## License

MIT.
