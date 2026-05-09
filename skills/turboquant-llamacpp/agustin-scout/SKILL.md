---
name: rx9070-agustin-llamacpp-turboquant-scout
description: Evaluate AgustinJimenez/llama.cpp on this user's WSL + ROCm + RX 9070 as a stronger candidate when full TurboQuant support must include both TQ3_1S/TQ4_1S weight loading and turbo4 KV cache.
---

# RX 9070 Agustin llama.cpp TurboQuant scout

Use this when the user questions whether a previously chosen TurboQuant fork is the right one and wants a fork that supports **both**:
- TurboQuant weight formats such as `TQ3_1S` / `TQ4_1S`
- TurboQuant KV cache such as `turbo4`

## Why this skill exists
A prior fork (`CarapaceUDE/turboquant-llama`) was successfully built on this machine and exposed `turbo4` KV cache types, but real validation showed that was **not enough** to claim it met the user's goal. The user expects separate verification of:
1. **weight-format support** (can it load `TQ3_1S` / `TQ4_1S` GGUFs?)
2. **KV-cache support** (does it accept `turbo4` / related KV types?)

Do not conflate these.

## Proven scouting workflow
### 1) Search alternative forks first
Candidate forks surfaced in-session:
- `AgustinJimenez/llama.cpp`
- `atomicmilkshake/llama-cpp-turboquant`
- `test1111111111111112/llama-cpp-turboquant-gemma4`
- `smurz/turboquant-llama`
- `flamme-demon/llama.cpp-hip-turboquant-tq3`

### 2) Clone candidates locally and inspect source, not just README
For each candidate, search for these markers:
- weight support markers:
  - `TQ3_1S`
  - `TQ4_1S`
  - `GGML_TYPE_TQ3_1S`
  - `GGML_TYPE_TQ4_1S`
  - `LLAMA_FTYPE_MOSTLY_TQ3_1S`
  - `LLAMA_FTYPE_MOSTLY_TQ4_1S`
- KV support markers:
  - `turbo4`
  - `TURBO4_0`
  - `cache-type-k`
  - `cache-type-v`

### 3) Rank by source-level evidence
In this session, the strongest candidate was:
- `AgustinJimenez/llama.cpp`

Because source inspection found all of the following together:
- `include/llama.h`
  - `LLAMA_FTYPE_MOSTLY_TQ3_1S = 43`
  - `LLAMA_FTYPE_MOSTLY_TQ4_1S = 44`
- `ggml/include/ggml.h`
  - `GGML_TYPE_TQ3_1S = 44`
  - `GGML_TYPE_TQ4_1S = 45`
- `src/llama-model-loader.cpp`
  - loader mappings for `TQ3_1S` and `TQ4_1S`
- `src/llama-quant.cpp`
  - quant mappings for `TQ3_1S` and `TQ4_1S`
- `common/arg.cpp`
  - `cache-type-k`
- KV-related `turbo4` hits in runtime code such as:
  - `src/llama-kv-cache.cpp`
  - `src/llama-context.cpp`
  - `src/llama-graph.cpp`

This is stronger evidence than README-only claims.

## Candidate classification learned in-session
### Likely full candidate
- `AgustinJimenez/llama.cpp`
  - strongest source-level evidence for both weight + KV support

### Likely KV/runtime-only branches
- `atomicmilkshake/llama-cpp-turboquant`
- `test1111111111111112/llama-cpp-turboquant-gemma4`
  - rich `turbo4`/KV/runtime references, but not clear source evidence for `TQ3_1S` / `TQ4_1S` weight-loader support

### TQ3-family / diverged design branches
- `flamme-demon/llama.cpp-hip-turboquant-tq3`
  - README explicitly says it supports `TQ3_1S`, `TQ3_4S`, `TQ3_0` KV
  - and explicitly says `TURBO2_0 / TURBO3_0 / TURBO4_0 / TQ4_1S` are not supported there
- `smurz/turboquant-llama`
  - looked more like an experimental/research branch than a clearly validated full-support runtime

## Local paths used
- Candidate clone used for the best current target:
  - `/home/qiushuo/src/llama.cpp-agustin-turboquant`
- Build dir created:
  - `/home/qiushuo/src/llama.cpp-agustin-turboquant/build-gfx1201`

## Configure command used successfully
```bash
repo=/home/qiushuo/src/llama.cpp-agustin-turboquant
if [ ! -d "$repo/.git" ]; then
  git clone https://github.com/AgustinJimenez/llama.cpp.git "$repo"
fi
cd "$repo"
mkdir -p build-gfx1201
HIPCXX="$(hipconfig -l)/clang" HIP_PATH="$(hipconfig -R)" cmake -S . -B build-gfx1201 \
  -DGGML_HIP=ON \
  -DGGML_HIP_GRAPHS=ON \
  -DGGML_FLASH_ATTN=ON \
  -DAMDGPU_TARGETS=gfx1201 \
  -DCMAKE_BUILD_TYPE=Release
```

## Build result on this machine
The Agustin fork did compile successfully on this user's WSL + ROCm + RX 9070 machine.

Built binary:
- `/home/qiushuo/src/llama.cpp-agustin-turboquant/build-gfx1201/bin/llama-server`

Observed version after build:
- `version: 8814 (8590cbff9)`

`--help` and direct parser checks confirmed KV cache accepts:
- `turbo2`
- `turbo3`
- `turbo4`

Example parser validation that succeeded:
```bash
/home/qiushuo/src/llama.cpp-agustin-turboquant/build-gfx1201/bin/llama-server \
  --cache-type-k turbo4 --cache-type-v turbo4 --version
```

## Critical runtime finding: stock Agustin build initially failed on mad-lab-ai TQ3_1S GGUF, but a narrow loader patch fixed it
Initial real load attempt against:
- `/home/qiushuo/models/qwen/mad-lab-ai-Qwen3.6-35B-A3B-tq-gguf/qwen3.6-35b-a3b-instruct-TQ3_1S.gguf`

failed before server readiness with:
```text
gguf_init_from_file_ptr: tensor 'blk.0.attn_gate.weight' has offset 671465472, expected 735035392
gguf_init_from_file_ptr: failed to read tensor data
```

This established that the stock fork was not immediately compatible with released `mad-lab-ai` TQ3_1S GGUFs, and that the failure lived in GGUF tensor metadata / offset interpretation rather than ROCm build or `turbo4` CLI parsing.

## Minimal compatibility patch that mattered
The successful fix on this fork was:
- read `general.file_type` from GGUF metadata using the literal key string:
  - `"general.file_type"`
- in `ggml/src/gguf.cpp`, after reading each tensor's raw `info.t.type`, add a compatibility shim:
  - if `general_file_type == 43` and raw tensor type is `45`
  - reinterpret it as `GGML_TYPE_TQ3_1S`

Important nuance learned on this fork:
- this tree does **not** define a `GGUF_KEY_GENERAL_FILE_TYPE` macro in `ggml/include/gguf.h`
- so use the literal key string instead of assuming a macro exists

Reasoning:
- released `mad-lab-ai` TQ3_1S GGUFs advertise `general.file_type == 43`
- some quantized tensors are stored with raw type `45`
- if that raw type is interpreted as this repo's native `TQ4_1S` layout, tensor byte-size math drifts and offset validation fails
- reinterpreting those tensors as `GGML_TYPE_TQ3_1S` before type validation / size / offset calculation is the minimal first fix to try

## What was actually proven after patching
After rebuilding with that loader patch, the fork achieved all of the following on this machine:
1. **Real model load success** for the local mad-lab-ai Qwen `TQ3_1S` GGUF
2. **Real server readiness** (`/health` and `/v1/models`)
3. **Real chat request success**
4. **Real turbo4 KV runtime activation**, not just CLI parsing

A low-risk proof command that worked was:
```bash
HSA_ENABLE_DXG_DETECTION=1 ROC_ENABLE_PRE_VEGA=0 \
/home/qiushuo/src/llama.cpp-agustin-turboquant/build-gfx1201/bin/llama-server \
  -m /home/qiushuo/models/qwen/mad-lab-ai-Qwen3.6-35B-A3B-tq-gguf/qwen3.6-35b-a3b-instruct-TQ3_1S.gguf \
  -ngl 1 \
  -fa on \
  -c 1024 \
  -np 4 \
  -ctk turbo4 \
  -ctv turbo4 \
  --host 127.0.0.1 \
  --port 8248 \
  --reasoning off \
  --reasoning-format none \
  --skip-chat-parsing
```

The proof points in logs / API were:
- API returned `OK`
- log showed:
  - `llama_kv_cache: TurboQuant rotation matrices initialized (128x128)`
  - `K (turbo4)`
  - `V (turbo4)`

Do not call the fork validated until you see both the API success **and** the runtime KV lines above.

## Deployment envelope learned from real tuning on RX 9070 16GB
### Offload ceiling
For the local Qwen `TQ3_1S` + `turbo4` path on this machine:
- `-ngl 37` at `ctx 8192`, `-np 1` is the highest validated stable offload point tested
- `-ngl 38` fails during compute-buffer reservation with OOM
- `-ngl 40+` also fails

### Long-context envelope
Real validated long-context points with `-ctk turbo4 -ctv turbo4`:
- `-ngl 35 -c 131072 -np 1` → **success**
- `-ngl 34 -c 131072 -np 1` → **success**
- `-ngl 36 -c 131072 -np 1` → failed allocating compute buffers
- `-ngl 36 -c 131072 -np 4` → also failed allocating compute buffers
- `-ngl 36 -c 98304 -np 1` → success
- `-ngl 36 -c 65536 -np 4` → success in earlier probe

Operational takeaway:
- for **128k**, use `-np 1` and back off to `-ngl 35` or `34`
- keeping `-ngl 36+` at `128k` leaves too little room for compute buffers on this card

## Performance reality on this route
This route is good for proving **TQ3_1S weights + turbo4 KV + long context** on 16 GB VRAM, but it is **not** a high-throughput deployment.

Measured examples:
- `-ngl 37 -c 8192 -np 1` short response:
  - prompt ≈ `4.32 tok/s`
  - decode ≈ `6.65 tok/s`
- `-ngl 37 -c 8192 -np 1` longer decode run (~70 completion tokens actually produced):
  - prompt ≈ `5.08 tok/s`
  - sustained decode ≈ `3.43 tok/s`
- `-ngl 35 -c 131072 -np 1`:
  - prompt ≈ `2.82 tok/s`
  - decode ≈ `3.63 tok/s`
- `-ngl 34 -c 131072 -np 1`:
  - prompt ≈ `3.13 tok/s`
  - decode ≈ `3.29 tok/s`
- `-ngl 36 -c 98304 -np 1`:
  - prompt ≈ `0.09 tok/s`
  - decode ≈ `4.01 tok/s`

Practical conclusion from real tests:
- **128k context is achievable**
- **>20 tok/s is not jointly achievable on this machine on this 35B TQ3_1S + turbo4 route**
- this path should be treated as a long-context feasibility / deployment path, not a fast-serving path

## Recommended runtime presets from this session
### Qwen 35B long-context preset
Use when the priority is getting to 128k reliably on the local mad-lab-ai Qwen `TQ3_1S` GGUF:
```bash
-ngl 35 -c 131072 -np 1 -fa on --fit-target 128 -ctk turbo4 -ctv turbo4
```

### Qwen 35B faster small-context preset
Use when the priority is the best speed seen on this Qwen route:
```bash
-ngl 37 -c 8192 -np 1 -fa on --fit-target 128 -ctk turbo4 -ctv turbo4
```

### Gemma 26B TQ3_1S faster small-context preset
For the local model:
- `/home/qiushuo/models/gemma/mad-lab-ai-google-gemma-4-26b-a4b-tq3_1s/google-gemma-4-26b-a4b-tq3_1s.gguf`

the best speed-first preset found on this machine was:
```bash
-ngl 32 -c 8192 -np 1 -fa on --fit-target 128 -ctk turbo4 -ctv turbo4
```
Measured at this preset:
- prompt ≈ `25.12 tok/s`
- decode ≈ `14.80 tok/s`

### Gemma 26B TQ3_1S long-context preset
The best validated 128k-class preset found on this machine was:
```bash
-ngl 24 -c 131072 -np 1 -fa on --fit-target 128 -ctk turbo4 -ctv turbo4
```
Measured at this preset:
- prompt ≈ `2.27 tok/s`
- decode ≈ `2.01 tok/s`

## Reusable launcher scripts created in-session
### Qwen 35B TQ3_1S + turbo4 128k
- start: `/home/qiushuo/.local/bin/start-agustin-qwen36-tq3-turbo4-128k`
- stop: `/home/qiushuo/.local/bin/stop-agustin-qwen36-tq3-turbo4-128k`
- status: `/home/qiushuo/.local/bin/status-agustin-qwen36-tq3-turbo4-128k`

Defaults baked into the start script:
- `PORT=8248`
- `CTX=131072`
- `NGL=35`
- `NPARALLEL=1`
- `BIN=/home/qiushuo/src/llama.cpp-agustin-turboquant/build-gfx1201/bin/llama-server`

These scripts were validated with:
1. `/health` success
2. `/v1/models` success
3. `/v1/chat/completions` returning `OK`
4. stop/status working correctly

### Gemma 26B TQ3_1S + turbo4 speed profile
- start: `/home/qiushuo/.local/bin/start-agustin-gemma26-tq3-turbo4-speed`
- stop: `/home/qiushuo/.local/bin/stop-agustin-gemma26-tq3-turbo4`
- status: `/home/qiushuo/.local/bin/status-agustin-gemma26-tq3-turbo4`

Defaults baked into the speed launcher:
- `PORT=8249`
- `CTX=8192`
- `NGL=32`
- `NPARALLEL=1`

### Gemma 26B TQ3_1S + turbo4 128k profile
- start: `/home/qiushuo/.local/bin/start-agustin-gemma26-tq3-turbo4-128k`
- stop: `/home/qiushuo/.local/bin/stop-agustin-gemma26-tq3-turbo4`
- status: `/home/qiushuo/.local/bin/status-agustin-gemma26-tq3-turbo4`

Defaults baked into the 128k launcher:
- `PORT=8250`
- `CTX=131072`
- `NGL=24`
- `NPARALLEL=1`

## Gemma-specific caveat
The Gemma `TQ3_1S` route was healthy and benchmarkable, but API responses through the current chat path sometimes include chat-template artifacts such as:
- `<|im_`
- repeated template fragments / trailing conversation markers

Interpretation:
- throughput numbers are still useful and the deployment path is real
- but for day-to-day assistant serving, this Gemma route likely needs template / stop-token cleanup before being treated as a polished chat service

## Critical operational lesson
Before claiming success on a new fork, always validate in this order:
1. **Source evidence** for weight types and KV types
2. **Build/configure succeeds** on this RX 9070 + ROCm machine
3. **CLI/help or parser evidence** for KV type acceptance
4. **Real model-load test** with the user's local GGUF, e.g.
   - `/home/qiushuo/models/qwen/mad-lab-ai-Qwen3.6-35B-A3B-tq-gguf/qwen3.6-35b-a3b-instruct-TQ3_1S.gguf`
5. **Real request test** after the server becomes healthy
6. **Runtime log verification** that KV really used `turbo4`
7. **VRAM / ctx / offload sweep** before promising a deployment target

Do not stop at step 3.

## Environment caveat
This user prefers only one active llama.cpp server at a time. Before runtime validation, check for an existing server and stop it if needed. In-session, an old server was found still running:
- `/home/qiushuo/src/llama.cpp-tq3/build-gfx1201-tq3/bin/llama-server`
- model: local Qwen `TQ3_1S`
- port `8234`
- VRAM at that moment: about `15.348 GiB` used / `0.525 GiB` free

## Reusable takeaway
When evaluating TurboQuant forks on this machine, do **fork triage by source markers first**, then ROCm configure/build, then real model-load validation. README claims and `--help` output alone are insufficient.
