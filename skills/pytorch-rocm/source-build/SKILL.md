---
name: rx9070-pytorch-rocm-source-build
description: Build PyTorch from source on this user's WSL + ROCm 7.2 + RX 9070 (gfx1201) setup. Use when the official rocm wheel is missing a feature (e.g. expandable_segments, recent ROCm-only allocator paths) or when patching torch internals. Tells you which branch to pick and what the v2.11 wheel actually has compiled out.
---

# Building PyTorch from source for RX 9070 / ROCm 7.2 (WSL)

## When to use this skill

- User wants `PYTORCH_HIP_ALLOC_CONF=expandable_segments:True` to actually take effect (not just be silently rejected with a `TORCH_WARN_ONCE`).
- User needs a ROCm-only allocator/driver-API path that landed after v2.11 (e.g. `ROCM_VERSION >= 70100`/`70200` gated code in `c10/cuda/CUDACachingAllocator.cpp`).
- User wants to hipify-port or patch torch internals and a wheel install isn't enough.

## Critical facts about the installed wheel (`torch==2.11.0+rocm7.2` from `pytorch.org/whl/rocm7.2`)

The wheel **cannot** enable expandable_segments by flipping a flag, and patching it is not cheap. Don't waste time on it:

1. The warning `expandable_segments not supported on this platform` lives in `c10/cuda/CUDAAllocatorConfig.h`. In v2.11 the guard is purely `#ifndef PYTORCH_C10_DRIVER_API_SUPPORTED`, but **every implementation site** (`driver_api.cpp`, `CUDAAllocatorConfig.cpp`, `CUDACachingAllocator.cpp`, `PeerToPeerAccess.cpp`) is wrapped in `#if !defined(USE_ROCM) && defined(PYTORCH_C10_DRIVER_API_SUPPORTED)` — i.e. `USE_ROCM` is a hard veto, not just the macro.
2. So even if you rebuild v2.11 with `-DPYTORCH_C10_DRIVER_API_SUPPORTED`, the ROCm path stays dead. You'd have to port `ExpandableSegment` + `DriverAPI` to HIP yourself (cuMem* → hipMem*, libcuda → libamdhip64), across 5+ files. That is a real multi-day port, not a one-line patch.
3. `libamdhip64.so` (ROCm 7.2.2) **does** export the VMM API (`hipMemAddressReserve`, `hipMemCreate`, `hipMemMap`, `hipMemSetAccess`, `hipMemRelease`) — runtime support is fine. The gap is purely in PyTorch's source tree.

## The right move: build PyTorch `main`

PyTorch `main` already has ROCm-native expandable_segments. The guard is now:

```cpp
// c10/cuda/CUDAAllocatorConfig.h on main
#if !defined(PYTORCH_C10_DRIVER_API_SUPPORTED) && \
    (!defined(USE_ROCM) || (ROCM_VERSION < 70000))
   TORCH_WARN_ONCE("expandable_segments not supported on this platform")
```

ROCm 7.2.2 satisfies `ROCM_VERSION >= 70000`, so the warning path is skipped and the real allocator is compiled in. There are also ROCm-version-gated improvements at `>= 70100` and `>= 70200` in `CUDACachingAllocator.cpp` — we get those too.

So: **don't try to backport into v2.11. Build `main` (or the most recent release tag that has the ROCm allocator land).** Do not promise a "small patch" up front; explain the whole-subsystem cost first.

## Environment (this user's machine)

- WSL2, ROCm 7.2.2, `gfx1201` (RX 9070), 20 cores, **17 GiB RAM (tight!)**, ~575 GiB free disk
- Existing venv `~/.venvs/qwen-rocm` with the official wheel — **don't overwrite it**. Make a sibling like `~/.venvs/qwen-rocm-src` so the working setup keeps running if the build breaks.
- No `ccache` installed and `sudo` requires a password. `apt install ccache` will block; ask the user once if they want to install it (second build is much faster with it). Otherwise proceed without.

## Workflow

1. Source layout: `~/src/pytorch-build/pytorch` (already cloned at v2.11.0 if previous session ran). `git fetch origin main && git checkout main && git submodule sync && git submodule update --init --recursive --depth 1 --jobs 6`. Submodules total ~2.8 GiB; reuse what's already there.
2. **Reuse the existing `~/.venvs/torch-build` venv** (Python 3.12) — it is already stocked with all the build-side deps: `cmake 4.3.2`, `ninja 1.13.0`, `numpy 2.4.4`, `pyyaml`, `setuptools`, `wheel`, `typing_extensions`, `astunparse`, `expecttest`, `sympy`, `networkx`, `Jinja2`, `fsspec`, `requests`. Do NOT `cp -r ~/.venvs/qwen-rocm` — that venv is 15 GiB and a recursive copy will hang/timeout. If `torch-build` is somehow gone, create a fresh `python -m venv` and `pip install -r requirements.txt -r requirements-build.txt` from the source tree.
3. **Run hipify FIRST** (this is mandatory and was missed in earlier attempts — cmake configure will fail with "File c10/hip/impl/hip_cmake_macros.h.in does not exist" otherwise):
   ```bash
   ~/.venvs/torch-build/bin/python tools/amd_build/build_amd.py
   ```
   This generates the ROCm sources from CUDA originals (e.g. `c10/hip/`, `aten/src/ATen/hip/`). Re-run any time you switch branches or `git clean` the tree.
4. Build env vars (paste these together):
   ```
   USE_ROCM=1 USE_CUDA=0
   PYTORCH_ROCM_ARCH=gfx1201
   ROCM_PATH=/opt/rocm  ROCM_HOME=/opt/rocm  HIP_PATH=/opt/rocm
   USE_KINETO=0 USE_FBGEMM=0 USE_NNPACK=0 USE_QNNPACK=0 USE_XNNPACK=0
   USE_FLASH_ATTENTION=0 USE_MEM_EFF_ATTENTION=0
   USE_MKLDNN=1 USE_DISTRIBUTED=1 USE_GLOO=1 USE_MPI=0 USE_TENSORPIPE=0
   BUILD_TEST=0
   MAX_JOBS=6                  # 17 GiB RAM cap; drop to 4 if linker OOMs
   CMAKE_CXX_COMPILER_LAUNCHER=ccache  # only if ccache is installed
   PATH=$VENV/bin:/opt/rocm/bin:/opt/rocm/llvm/bin:$PATH
   CMAKE_PREFIX_PATH=$VENV
   ```
5. Build with `pip install -e . -v --no-build-isolation` (per upstream AGENTS.md — do **not** invoke `setup.py` directly).
6. Always run as `terminal(background=True, notify_on_complete=True)`. First build is ~2–3 h with single-arch hipcc; do not try to wait synchronously.

A ready-to-run build script lives at `scripts/build_pytorch_rocm.sh` in this skill — copy it to `~/src/pytorch-build/build_pytorch_rocm.sh` and run.

## Pitfalls / gotchas

- **Forgetting hipify**: cmake will configure most of the way through, print a happy summary block (`USE_ROCM : ON`, `USE_NCCL : ON`, etc.) and THEN error with `File /home/qiushuo/src/pytorch-build/pytorch/c10/hip/impl/hip_cmake_macros.h.in does not exist` plus a sibling error for `aten/src/ATen/hip/HIPConfig.h.in`. The fix is always: run `tools/amd_build/build_amd.py` first. Do not try to `touch` the missing files.
- **Recursive copy of large venvs**: `cp -r ~/.venvs/qwen-rocm ~/.venvs/qwen-rocm-src` will time out (15 GiB, no reflink on the user's filesystem). Always reuse `~/.venvs/torch-build` or build a fresh thin venv.
- **`apt install ccache` needs sudo password** which the agent doesn't have. Ask the user once if they want it; otherwise just skip — first build runs fine without it, just slower on rebuilds.

- The version string becomes `2.12.0a0+gitXXXX`. Anything pinning `torch==2.11.*` (e.g. some vLLM/aiter combos in this user's other skills) will refuse to install. Keep this venv separate.
- `amdsmi` init fails under WSL (`Error code: 34`). It is unrelated to expandable_segments — that error is independent and only blocks NVML-style telemetry, not the allocator. Don't conflate them when diagnosing.
- The `Quadro P620` showing up in `nvidia-smi` is Xwayland passthrough; ignore it. Real GPU = `gfx1201`.
- `rocm-smi` is not available under WSL. Use llama.cpp / torch logs for VRAM accounting.
- Probing the public main branch for code changes is best done with `git clone --depth 1 --filter=blob:none --sparse` then `git sparse-checkout set c10/cuda c10/hip` — full clone is wasteful when you only need to grep for guards.

## Verification after install

```python
import torch, os
os.environ["PYTORCH_HIP_ALLOC_CONF"] = "expandable_segments:True"
print(torch.__version__, torch.version.hip)
x = torch.empty(1024, device="cuda")          # trigger allocator init
# Success criterion: NO "expandable_segments not supported on this platform"
# warning, and torch.cuda.memory_stats()['active_bytes.all.current'] grows
# without large reservation jumps when allocating.
```

If the warning still shows up, the build picked the v2.11 guard path — verify `git rev-parse HEAD` is on main (not the v2.11.0 tag) and re-run the cmake configure step.

## Cross-refs

- `rx9070-gguf-hermes-eval`, `rx9070-vllm-rocm-turboquant` — they share the venv conventions and ROCm assumptions; mention this skill if expandable_segments gets brought up.
- Memory entry "WSL ROCm venv /home/qiushuo/.venvs/qwen-rocm" documents the production venv that must NOT be clobbered.
