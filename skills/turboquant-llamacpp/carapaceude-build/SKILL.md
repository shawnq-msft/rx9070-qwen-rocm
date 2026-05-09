---
name: rx9070-llamacpp-full-turboquant-hip-build
description: Build and patch CarapaceUDE/turboquant-llama on this user's WSL + ROCm + RX 9070 (gfx1201) setup so llama-server recognizes full TurboQuant KV types including turbo4.
---

# RX 9070 full TurboQuant llama.cpp HIP build

Use this when the user wants a **full TurboQuant** llama.cpp fork on this exact machine, not just `turbo-tan/llama.cpp-tq3`.

## When to use
- User wants **all TQ types** / a fork that supports more than TQ3 runtime
- User specifically wants **KV turbo4 / TQ4-style KV**
- User says standard llama.cpp doesn't recognize TurboQuant quants and wants the correct fork installed

## Proven fork on this machine
- Repo: `https://github.com/CarapaceUDE/turboquant-llama.git`
- Local clone used successfully: `/home/qiushuo/src/llama.cpp-turboquant-full`
- Build dir used successfully: `/home/qiushuo/src/llama.cpp-turboquant-full/build-gfx1201-turboquant-full`
- Binary: `/home/qiushuo/src/llama.cpp-turboquant-full/build-gfx1201-turboquant-full/bin/llama-server`

Keep this fork separate from:
- `/home/qiushuo/src/llama.cpp`
- `/home/qiushuo/src/llama.cpp-tq3`
- `/home/qiushuo/src/llama.cpp-latest-rocm`

## Important findings
1. This fork can be built on the user's **WSL + ROCm + RX 9070 gfx1201** machine.
2. To get full TurboQuant FlashAttention template coverage linked correctly, **`GGML_CUDA_FA_ALL_QUANTS=ON` was required**.
3. The fork also needed **manual HIP compatibility patches** similar to the TQ3 fork.
4. The KV cache type names accepted by the built binary are:
   - `turbo2`
   - `turbo3`
   - `turbo4`
   **Not** `turbo2_0` / `turbo3_0` / `turbo4_0`.
5. Verified CLI behavior on this machine:
   - `--cache-type-k turbo4 --cache-type-v turbo4` works
   - `--cache-type-k turbo4_0 --cache-type-v turbo4_0` fails with `Unsupported cache type: turbo4_0`

## Configure command
```bash
repo=/home/qiushuo/src/llama.cpp-turboquant-full
if [ ! -d "$repo/.git" ]; then
  git clone https://github.com/CarapaceUDE/turboquant-llama.git "$repo"
fi

cd "$repo"
mkdir -p build-gfx1201-turboquant-full
HIPCXX="$(hipconfig -l)/clang" HIP_PATH="$(hipconfig -R)" cmake -S . -B build-gfx1201-turboquant-full \
  -DGGML_HIP=ON \
  -DGGML_HIP_GRAPHS=ON \
  -DGGML_FLASH_ATTN=ON \
  -DGGML_CUDA_FA_ALL_QUANTS=ON \
  -DAMDGPU_TARGETS=gfx1201 \
  -DCMAKE_BUILD_TYPE=Release
```

## Build command
```bash
cmake --build /home/qiushuo/src/llama.cpp-turboquant-full/build-gfx1201-turboquant-full \
  --config Release --target llama-server -j 8
```

## Required HIP compatibility patches
These were needed to get this fork building on ROCm/HIP.

### 1) Use HIP header instead of raw CUDA headers in TurboQuant files
Patch `ggml/src/ggml-cuda/turbo-quant-cuda.cuh`:
```c
#pragma once
#if defined(GGML_USE_HIP)
#include "vendors/hip.h"
#else
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#endif
#include "ggml-common.h"
```

Patch `ggml/src/ggml-cuda/turbo-sink.cuh`:
```c
#pragma once
#if defined(GGML_USE_HIP)
#include "vendors/hip.h"
#else
#include <cuda_fp16.h>
#endif
#include <cstdint>
```

Patch `ggml/src/ggml-cuda/turbo-sink.cu`:
```c
#include "turbo-sink.cuh"
#if defined(GGML_USE_HIP)
#include "vendors/hip.h"
#else
#include <cuda_runtime.h>
#endif
#include <cstdlib>
```

### 2) Add missing HIP aliases in `vendors/hip.h`
On this fork, add missing mappings if absent:
```c
#define cudaMemcpyToSymbol hipMemcpyToSymbol
#define cudaMemcpyFromSymbol hipMemcpyFromSymbol
#define cudaEventCreate hipEventCreate
#define cudaEventCreateWithFlags hipEventCreateWithFlags
#define cudaEventDisableTiming hipEventDisableTiming
#define cudaEventRecord hipEventRecord
#define cudaEventSynchronize hipEventSynchronize
#define cudaEventElapsedTime hipEventElapsedTime
```

## Critical build fix: include all TurboQuant FA template instances
### Symptom
Without the full-quants setting and source inclusion, `llama-server` link can fail with many undefined references like:
- `ggml_cuda_flash_attn_ext_vec_case<128, (ggml_type)43, (ggml_type)8>`
- `ggml_cuda_flash_attn_ext_vec_case<64, (ggml_type)42, (ggml_type)42>`
- many combinations involving TurboQuant types and `q8_0`

### Why
The binary references TurboQuant FlashAttention vector cases for `turbo2/3/4`, but not all template instances were getting built/linked in the HIP build.

### What fixed it on this machine
1. Configure with:
```bash
-DGGML_CUDA_FA_ALL_QUANTS=ON
```
2. Ensure `ggml/src/ggml-cuda/CMakeLists.txt` explicitly appends the missing TurboQuant template instances inside the `if (GGML_CUDA_FA_ALL_QUANTS)` block:
```cmake
list(APPEND GGML_SOURCES_CUDA
    template-instances/fattn-vec-instance-turbo2_0-turbo2_0.cu
    template-instances/fattn-vec-instance-turbo2_0-turbo3_0.cu
    template-instances/fattn-vec-instance-turbo2_0-q8_0.cu
    template-instances/fattn-vec-instance-turbo3_0-turbo2_0.cu
    template-instances/fattn-vec-instance-turbo3_0-turbo3_0.cu
    template-instances/fattn-vec-instance-turbo3_0-turbo4_0.cu
    template-instances/fattn-vec-instance-turbo3_0-q8_0.cu
    template-instances/fattn-vec-instance-turbo4_0-turbo3_0.cu
    template-instances/fattn-vec-instance-turbo4_0-turbo4_0.cu
    template-instances/fattn-vec-instance-turbo4_0-q8_0.cu
    template-instances/fattn-vec-instance-q8_0-turbo2_0.cu
    template-instances/fattn-vec-instance-q8_0-turbo3_0.cu
    template-instances/fattn-vec-instance-q8_0-turbo4_0.cu)
```

Then rerun CMake configure and rebuild.

## Verification steps
### 1) Verify help output exposes TurboQuant KV types
```bash
/home/qiushuo/src/llama.cpp-turboquant-full/build-gfx1201-turboquant-full/bin/llama-server --help | sed -n '/cache-type-k/,+8p'
```
Expected allowed values include:
- `turbo2`
- `turbo3`
- `turbo4`

### 2) Verify parser accepts the TurboQuant KV names
```bash
/home/qiushuo/src/llama.cpp-turboquant-full/build-gfx1201-turboquant-full/bin/llama-server \
  --cache-type-k turbo4 \
  --cache-type-v turbo4 \
  --version
```
This succeeded on this machine.

### 3) Confirm the old name fails
```bash
/home/qiushuo/src/llama.cpp-turboquant-full/build-gfx1201-turboquant-full/bin/llama-server \
  --cache-type-k turbo4_0 \
  --cache-type-v turbo4_0 \
  --version
```
Expected error:
- `Unsupported cache type: turbo4_0`

## Known-good conclusion from this session
This fork is the correct installed base when the user asks for a **full TurboQuant llama.cpp build** and wants **KV turbo4** support on this RX 9070 machine.

## Practical next step after install
The next real runtime test should use:
```bash
-ctk turbo4 -ctv turbo4
```
with an actual model load and real API request validation.
Do not claim success from help text alone; after install, the next job is always:
1. model loads
2. server becomes healthy
3. one real request succeeds
4. record VRAM + ctx behavior

## Pitfalls
- Do not use `turbo4_0` in runtime commands; use `turbo4`.
- Do not overwrite older validated llama.cpp builds.
- If link errors mention missing `ggml_cuda_flash_attn_ext_vec_case<...>` TurboQuant instantiations, check both:
  - `GGML_CUDA_FA_ALL_QUANTS=ON`
  - explicit TurboQuant template entries in `ggml/src/ggml-cuda/CMakeLists.txt`
- This skill only establishes the build/install path and CLI support for `turbo4`; separate runtime validation is still required per model.
