---
name: rx9070-vllm-turboquant
description: Run vLLM with TurboQuant KV cache (and FP8 baseline) on this user's WSL + ROCm 7.2.2 + RX 9070 16GB box. Use when picking a serving configuration, comparing KV cache dtypes, sizing context length, or answering "what's the best model/config for vLLM here." Covers the local hybrid-TurboQuant patch state, accepted model formats, and verified pass@1 / tok/s / ctx data.
---

# vLLM TurboQuant on RX 9070 (WSL + ROCm 7.2.2)

## When to load
- User asks about vLLM TurboQuant performance, KV dtype tradeoffs, max context, or which model to serve.
- User wants to add a new model to this runtime and asks if format X works.
- Comparing this path against the llama.cpp TurboQuant fork path.

## Runtime facts (verified, do not rebuild from scratch)
- Build: `/home/qiushuo/src/vllm`, working HEAD `5cdddddd4` with local hybrid-TurboQuant patches applied to:
  - `vllm/engine/arg_utils.py`
  - `vllm/model_executor/layers/quantization/turboquant/config.py`
  - `vllm/platforms/{interface.py, rocm.py, __init__.py}`
  - `vllm/v1/attention/backends/turboquant_attn.py`
- venv: `/home/qiushuo/.venvs/vllm-rocm-latest`, torch `2.11.0+rocm7.2`.
- ROCm 7.2.2 is **current latest stable**; nothing to upgrade for TurboQuant.
- ROCm itself does **not** ship TurboQuant — it's a vLLM-side feature. AMD's official quant is Quark, unrelated.
- Required env: `HSA_ENABLE_DXG_DETECTION=1 ROC_ENABLE_PRE_VEGA=0 VLLM_TARGET_DEVICE=rocm VLLM_USE_V1=1`.

## Accepted model formats (CRITICAL pitfall)
vLLM does NOT load GGUF on this build. Acceptable formats:
- HF safetensors FP16
- GPTQ (verified working: Qwopus 9B GPTQ)
- AWQ / FP8 (untested locally but supported by vLLM)

If the user has Qwen 3.6 / Gemma / etc. only as GGUF (mad-lab-ai, YTan2000, unsloth GGUF dirs under `~/models/qwen/`), tell them to use the **llama.cpp TurboQuant fork** path instead — see sibling skill `rx9070-agustin-llamacpp-turboquant`. Don't try to convert; route them.

## Models verified on this runtime
Only one so far: `caiovicentino1/Qwopus3.5-9B-v3-HLWQ-v7-GPTQ` (hybrid Mamba+Attn, 9B, GPTQ 4bit). It IS the entire "vLLM TurboQuant has been validated" surface area on this box. Anything else is unverified — say so explicitly.

## KV cache dtype recommendations
- **TQ4 (`turboquant_4bit_nc`)** — best TurboQuant point. 128k ctx works, quality ≈ FP8.
- **FP8 (`fp8`)** — best raw quality + speed at ≤64k. Goes up to ~196k single-request.
- **TQ3 (`turboquant_3bit_nc`)** — measurable quality drop; only pick when KV size is the binding constraint.
- **K4V3 / K8V4 plugin paths** — superseded by TQ4. Don't reach for them by default.

## Canonical serve commands
TQ4 @ 128k (best validated):
```
python -m vllm.entrypoints.openai.api_server \
  --model /home/qiushuo/models/qwen/caiovicentino1-Qwopus3.5-9B-v3-HLWQ-v7-GPTQ \
  --quantization gptq --dtype float16 --language-model-only \
  --kv-cache-dtype turboquant_4bit_nc --mamba-ssm-cache-dtype float16 \
  --gpu-memory-utilization 0.9 --max-model-len 131072 \
  --max-num-batched-tokens 16384 --max-num-seqs 1 \
  --host 127.0.0.1 --port 8533
```
FP8 short-ctx baseline: swap `--kv-cache-dtype fp8` and drop ctx to 65536, mbt 8192.

## Pitfalls
- "TurboQuant is not yet compatible with FlashAttention >= 3" — known warning, harmless.
- "gptq_gemm kernel for GPTQ is buggy" — known, vLLM auto-picks alternative; harmless.
- amdsmi init fails in WSL — ignore, use the `gpu_before/after` numbers logged by the bench harness.
- Hybrid (Mamba+Attn) models hit `NotImplementedError` in upstream vLLM main — required the local patch listed above. If a fresh vLLM checkout regresses, re-port PR #39931 ideas.
- Single 9070 (16 GiB) cannot fit Qwen 3.6 35B-A3B in GPTQ 4bit (≈18–20 GiB). Don't try without CPU offload, and CPU-offload on ROCm vLLM is rough — prefer llama.cpp `-ncmoe` path.

## Reference data
- `references/benchmark-results.md` — full HumanEval pass@1 / tok/s / max-ctx table from the 2026-04 vLLM TurboQuant runs (Qwopus 9B, all KV dtypes).
- `references/runtime-compatibility-matrix.md` — what does NOT work across (TQ3 GGUF × Vulkan × SGLang × vLLM × HIP llama.cpp) and which cross-runtime comparisons are physically possible on this box. Read first when a "compare backend X vs Y on TQ3 model" request comes in.

## Related skills
- `rx9070-agustin-llamacpp-turboquant` — llama.cpp side (handles GGUF TurboQuant models).
- `rx9070-local-humaneval-benchmark` — harness that produced the numbers in references/.
- `rx9070-pytorch-rocm-expandable-segments` — base ROCm/PyTorch fixes this runtime depends on.
