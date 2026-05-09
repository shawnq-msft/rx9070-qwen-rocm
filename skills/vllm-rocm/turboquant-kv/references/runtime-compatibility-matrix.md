# RX 9070 quant-runtime compatibility matrix

When the user asks to compare a TurboQuant GGUF model across "Vulkan / SGLang / vLLM / HIP llama.cpp" or any cross-product of (model format × backend × KV dtype), check this matrix first. Several common-sounding requests are physically impossible with what's installed.

## Hard incompatibilities (don't promise these without saying why)

| Request | Why it fails |
|---|---|
| **TQ3 / TQ4 GGUF on mainline llama.cpp Vulkan backend** | `tq3_*` / `turbo3` / `turbo4` ggml types are private to turbo-tan/llama.cpp-tq3 and AgustinJimenez forks. Mainline + Vulkan have no dequant kernel → `unknown ggml type` at load. |
| **TQ3 GGUF in SGLang** | SGLang loads HF safetensors only (FP16 / AWQ / GPTQ / FP8). It does not parse GGUF at all. |
| **TQ KV cache in SGLang or transformers** | "TQ KV" is a llama.cpp / vLLM (patched) concept. SGLang attention is FlashInfer/Triton — no `turboquant_*_nc` dtype exists there. |
| **SGLang on RX 9070 WSL (gfx1201, RDNA4)** | No verified working build on this box. vLLM ROCm is the validated alternative. |
| **Qwen 3.6-35B-A3B in vLLM 4bit on single 9070** | GPTQ 4bit ≈ 18–20 GiB, doesn't fit in 16 GiB VRAM. Use llama.cpp `-ncmoe` hybrid offload instead. |

## What "TQ3_4S" actually means
`Qwen3.6-35B-A3B-TQ3_4S.gguf` (YTan2000) is built for the **turbo-tan/llama.cpp-tq3** fork (build at `/home/qiushuo/src/llama.cpp-tq3/build-gfx1201-tq3`). The `4S` is weight-side TurboQuant variant; pair it with `-ctk q4_0 -ctv tq3_0 -fa on`. Only this fork's HIP build runs it. AgustinJimenez fork supports a slightly different TQ matrix (TQ3_1S/TQ4_1S + turbo4 KV) — do not mix model & fork.

## Real cross-runtime comparisons that DO work on this box

1. **KV-dtype ablation (same fork, same weights)** — HIP llama.cpp-tq3 + Qwen3.6-35B-A3B-TQ3_4S, sweep `-ctv tq3_0 / q4_0 / f16` at ctx 16k/32k/64k. Cleanest.
2. **HIP vs Vulkan llama.cpp backend** — requires switching to a *mainline-compatible* Q4_K_M / Q5_K_M GGUF (not TQ3) and building the Vulkan backend (`vulkan-tools` not installed by default).
3. **llama.cpp TurboQuant vs vLLM TurboQuant** — different model required on each side: GGUF TQ3 on llama.cpp side, GPTQ/AWQ + `--kv-cache-dtype turboquant_4bit_nc` on vLLM side. Not the same weights, but the closest "is my TQ pipeline competitive" answer.

## When the user asks the impossible combo
Say so up front and offer the three concrete alternatives above. Don't silently substitute.
