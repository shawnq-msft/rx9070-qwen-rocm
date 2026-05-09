# RX 9070 + Qwen / Qwopus on ROCm — Field Notes

Reproducible serve configs, build flags, KV-quant benchmarks, and pitfalls
for running **Qwen 3.5 / 3.6** (27B & 35B-A3B) and **Qwopus** GGUF / GPTQ
models on a single **AMD Radeon RX 9070 16 GB** (RDNA4, gfx1201) under
**WSL2 + ROCm 7.x**, via both **llama.cpp HIP** and **vLLM**.

Everything here is verified end-to-end on one box, not theoretical.

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
