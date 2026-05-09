---
name: rx9070-pytorch-rocm-expandable-segments
description: Diagnose and patch PyTorch ROCm wheels so `PYTORCH_HIP_ALLOC_CONF=expandable_segments:True` actually works on this user's WSL + RX 9070 (RDNA4 gfx1201, dGPU) box. Covers the upstream APU-vs-dGPU bug in `c10/cuda/CUDACachingAllocator.cpp`, WSL-specific librocprofiler-sdk dlopen abort, and a verified librocprofiler-register stub.
---

# RX 9070 PyTorch ROCm expandable_segments

## When to use
- User on RX 9070 16GB (RDNA4 gfx1201) WSL2 + ROCm 7.2.x reports OOM at vLLM/llama.cpp boundaries and asks whether `expandable_segments` can recover headroom.
- User asks why `PYTORCH_HIP_ALLOC_CONF=expandable_segments:True` prints "expandable_segments not supported on this platform" or crashes on alloc.
- Considering upgrading to PyTorch ROCm nightly (B) vs source-rebuild PyTorch (C).

## TL;DR — answer
**Stable `pip install torch --index-url .../rocm7.2` (2.11.0) wheels:** wheel was built without `PYTORCH_C10_DRIVER_API_SUPPORTED` → `libc10_hip.so` has stub branch that just warns. Ship is half-broken.

**Nightly `rocm7.2` wheels (2.13.0.dev*):** driver API IS compiled in (`hipMemAddressReserve` / `hipMemCreate` linked, `ExpandableSegment` symbols present), BUT:

1. Nightly hard-links `librocprofiler-sdk.so` from `libtorch_cpu.so`. On WSL `/sys/class/kfd` is missing → fatal abort at `import torch`.
2. Even after stubbing rocprofiler, `expandable_segments:True` fails at `hipMemSetAccess` with `hipErrorInvalidValue`.

**Root cause of #2 (verified):** `pytorch/c10/cuda/CUDACachingAllocator.cpp::ExpandableSegment::setAccess` does:
```cpp
#if defined(USE_ROCM) && (ROCM_VERSION >= 70200)
  constexpr int num_desc = 2;
  desc[1].location.type = CU_MEM_LOCATION_TYPE_HOST;  // assumes APU coherent host mem
  ...
#else
  constexpr int num_desc = 1;
#endif
```
This unconditionally turns on host-coherent access for ROCm ≥ 7.2 — only valid on APUs (MI300A / Strix Halo / Ryzen AI MAX). RX 9070 is dGPU → runtime rejects.

So **B (nightly wheel) cannot work as-is**. Must go **C (source rebuild)** with a 6-line patch, OR write a runtime LD_PRELOAD shim that intercepts `hipMemSetAccess` and forces count=1.

## Verification commands

### 1. Check whether installed PyTorch wheel has driver API compiled
```bash
nm -DC <venv>/lib/python*/site-packages/torch/lib/libc10_hip.so | grep -iE "ExpandableSegment|getExpandableSegmentSizes" | head
strings <...>/libc10_hip.so | grep "expandable_segments not supported"
```
- Old wheel: stub branch, warning string present, no ExpandableSegment symbols
- Nightly: no warning, ExpandableSegment symbols present

### 2. Confirm runtime (not PyTorch) supports VMM
Save to `/tmp/vmm_probe.cpp`:
```cpp
#include <stdio.h>
#include <hip/hip_runtime.h>
int main(){
  hipMemAllocationProp prop = {};
  prop.type = hipMemAllocationTypePinned;
  prop.location.type = hipMemLocationTypeDevice;
  prop.location.id = 0;
  hipMemGenericAllocationHandle_t h;
  hipError_t e = hipMemCreate(&h, 256*1024*1024, &prop, 0);
  printf("hipMemCreate: %d (%s)\n", e, hipGetErrorString(e));
  void* ptr=nullptr;
  hipMemAddressReserve((hipDeviceptr_t*)&ptr, 256*1024*1024, 0, 0, 0);
  hipMemMap((hipDeviceptr_t)ptr, 256*1024*1024, 0, h, 0);
  hipMemAccessDesc d{};
  d.location.type = hipMemLocationTypeDevice;
  d.location.id = 0;
  d.flags = hipMemAccessFlagsProtReadWrite;
  // num_desc = 1 — DO NOT add HOST entry on dGPU
  e = hipMemSetAccess((hipDeviceptr_t)ptr, 256*1024*1024, &d, 1);
  printf("hipMemSetAccess(num=1): %d (%s)\n", e, hipGetErrorString(e));
  return 0;
}
```
```bash
/opt/rocm-7.2.2/bin/hipcc /tmp/vmm_probe.cpp -o /tmp/vmm_probe
HSA_ENABLE_DXG_DETECTION=1 /tmp/vmm_probe
```
Both calls return `0 (no error)` on this user's WSL — VMM works at the runtime level. If they fail, ROCm runtime itself is the blocker (then expandable_segments is impossible).

### 3. Capture which `hipMemSetAccess` call PyTorch actually emits
```bash
HSA_ENABLE_DXG_DETECTION=1 PYTORCH_HIP_ALLOC_CONF=expandable_segments:True \
  AMD_LOG_LEVEL=3 python -c "import torch; torch.zeros(int(256*1024*1024/4),device='cuda')" 2>&1 \
  | grep -E "MemSetAccess|hipMemSetAccess"
```
If you see `hipMemSetAccess ( ..., 2 )` followed by `hipErrorInvalidValue`, you've hit the APU-vs-dGPU bug.

## WSL-specific: stubbing librocprofiler

Nightly wheel adds `DT_NEEDED librocprofiler-sdk.so`. On WSL `/sys/class/kfd` doesn't exist → `agent.cpp:608` fatal. `librocprofiler-register.so` is the gatekeeper that dlopens sdk; stubbing register skips the whole chain.

Find what `libtorch_cpu.so`/`libamdhip64.so`/`libhsa-runtime64.so`/`librccl.so` actually need from register:
```bash
for f in <torch>/lib/*.so; do
  syms=$(nm -D --undefined-only "$f" 2>/dev/null | awk '/U rocprofiler_register/ {print $2}' | sort -u)
  [ -n "$syms" ] && echo "$f: $syms"
done
```
Today (nightly 2.13.0.dev20260428): only 2 symbols — `rocprofiler_register_library_api_table` and `rocprofiler_register_error_string`.

Stub:
```c
/* /tmp/rocprof_register_stub.c */
#include <stddef.h>
int rocprofiler_register_library_api_table(const char* a, void* b, unsigned c, void* d) { return 0; }
const char* rocprofiler_register_error_string(int e) { return "stubbed"; }
```
```bash
gcc -shared -fPIC -Wl,-soname,librocprofiler-register.so.0 \
    -o /tmp/librocprofiler-register.so /tmp/rocprof_register_stub.c
cp <torch>/lib/librocprofiler-register.so <torch>/lib/librocprofiler-register.so.real
cp /tmp/librocprofiler-register.so <torch>/lib/librocprofiler-register.so
```
DO NOT stub `librocprofiler-sdk.so` — register dlopens it and fails the symbol check anyway. Stubbing register is the clean cut.

## C-plan patch (source rebuild)

Patch `c10/cuda/CUDACachingAllocator.cpp`:
```diff
-#if defined(USE_ROCM) && (ROCM_VERSION >= 70200)
-    constexpr int num_desc = 2;
-    CUmemAccessDesc desc[num_desc];
-    desc[1].location.type = CU_MEM_LOCATION_TYPE_HOST;
-    desc[1].location.id = 0;
-    desc[1].flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
-#else
-    constexpr int num_desc = 1;
-    CUmemAccessDesc desc[num_desc];
-#endif
+    // RX 9070 is RDNA4 dGPU; HOST access desc only valid on APUs.
+    constexpr int num_desc = 1;
+    CUmemAccessDesc desc[num_desc];
```
(Also consider removing rocprofiler-sdk DT_NEEDED via `cmake/Dependencies.cmake` if WSL-target build, or keep stub workflow.)

Build env (verified working on this 17 GB RAM / 20 core / RX 9070 / WSL box):
```bash
# Toolchain
export ROCM_HOME=/opt/rocm-7.2.2 ROCM_PATH=/opt/rocm-7.2.2
export PATH=/opt/rocm-7.2.2/bin:$PATH
export USE_ROCM=1 USE_CUDA=0
export PYTORCH_ROCM_ARCH=gfx1201   # gfx1201 ONLY — saves ~70% wall time

# Skip WSL-broken / unneeded components
export USE_KINETO=0 USE_ITT=0 USE_NUMA=0     # kineto pulls rocprofiler, breaks on WSL
export USE_FBGEMM=0 USE_XNNPACK=0 USE_QNNPACK=0 USE_PYTORCH_QNNPACK=0
export BUILD_CAFFE2=0 BUILD_TEST=0

# vLLM needs distributed but NOT NCCL
export USE_DISTRIBUTED=1 USE_NCCL=0 USE_GLOO=1

# Memory cap — MAX_JOBS=$(nproc) WILL OOM at 17 GB RAM (hipcc heap ~2.5 GB/proc on heavy ROCm .cu)
export MAX_JOBS=6 NVCC_THREADS=1

export USE_FLASH_ATTENTION=1 USE_MEM_EFF_ATTENTION=1
export HSA_ENABLE_DXG_DETECTION=1
```

Build sequence (order matters):
```bash
cd pytorch
git checkout <nightly-commit>          # use torch.version.git_version from the nightly wheel
git submodule update --init --recursive --depth 1
# 1. Apply patch to c10/cuda/CUDACachingAllocator.cpp FIRST
patch -p1 < 0001-force-num_desc-1-rdna4.patch
# 2. THEN run hipify — it propagates the patch into c10/hip/HIPCachingAllocator.cpp
python tools/amd_build/build_amd.py
# 3. Build (4–6 h on this box)
python setup.py bdist_wheel
```
**Critical:** if you hipify before patching, the actual build target `c10/hip/HIPCachingAllocator.cpp` keeps the upstream `num_desc=2` and the wheel still breaks. Patch → hipify → build, in that order. Stop all GPU consumers (vLLM/llama.cpp) before launching — RAM contention swap-thrashes the build to a halt.

## LD_PRELOAD alternative (faster than rebuild)
Wrap `hipMemSetAccess` to clamp count to 1:
```c
#define _GNU_SOURCE
#include <dlfcn.h>
#include <hip/hip_runtime.h>
static int (*real)(void*, size_t, const hipMemAccessDesc*, size_t) = NULL;
hipError_t hipMemSetAccess(void* p, size_t s, const hipMemAccessDesc* d, size_t n){
  if(!real) real = dlsym(RTLD_NEXT, "hipMemSetAccess");
  if (n > 1) n = 1;  // RDNA4 dGPU: drop HOST desc
  return real(p, s, d, n);
}
```
Compile + use:
```bash
gcc -shared -fPIC -ldl -o /tmp/libfix_setaccess.so fix.c
LD_PRELOAD=/tmp/libfix_setaccess.so python ...
```
Worth trying before committing to a 4–6h source rebuild.

## Pitfalls
- Do NOT trust the warning "expandable_segments not supported on this platform" — that's a wheel-build-time stub, not a runtime check. Always inspect symbols + run the C VMM probe before concluding ROCm doesn't support VMM.
- `hipcc` requires `.cpp` extension to enable HIP language; `.c` files emit "fatal error: 'hip/hip_runtime.h' file not found".
- `amdsmi` returns Error 34 in WSL — irrelevant to VMM, ignore.
- Don't disable rocprofiler via env vars (`ROCPROFILER_DISABLE`, `HSA_DISABLE_TOOLS`, etc.) — none of them work; only the stub library does.
- Stable wheel users: don't waste time hacking 2.11.0 — it lacks the entire `ExpandableSegment` machinery; you must move to nightly OR build from source.

## Status
- ROCm runtime VMM: ✅ works on WSL (verified 2026-04-29 with C probe up to 2 GiB)
- 2.11.0 stable wheel: ❌ ships stub, no driver API
- 2.13.0.dev20260428 nightly: ⚠️ has driver API but APU-only setAccess + WSL rocprofiler abort
- LD_PRELOAD shim on nightly: 🟡 untested, likely path of least resistance
- Source rebuild w/ 6-line patch from nightly commit: ✅ wheel works, VMM 8 GiB smoke passes — **but breaks vLLM ABI** (see below)

## Verified results (2026-04-29 build)
- Wall time on this box: **113 min** (`MAX_JOBS=6 NVCC_THREADS=1 PYTORCH_ROCM_ARCH=gfx1201`), not 4–6h. Earlier estimate was conservative.
- Wheel: `torch-2.13.0a0+gitf144290-cp312-cp312-linux_x86_64.whl` (~295 MB)
- **Self-built wheel does NOT need the librocprofiler-register stub** — `USE_KINETO=0 USE_ITT=0 USE_NUMA=0` build env drops the `DT_NEEDED librocprofiler-sdk.so` from `libtorch_cpu.so`. Stub is only needed for upstream nightly wheels.
- VMM smoke (`/tmp/vmm_smoke.py` allocating 16×512MiB then realloc 2GiB): ✅ pass.

## ⚠️ ABI gotcha: vLLM 0.19.x C extensions
vLLM `_C.abi3.so` / `_rocm_C.abi3.so` / `cumem_allocator.abi3.so` / `_moe_C.abi3.so` are NOT actually `abi3` w.r.t. PyTorch's C++ API — they link against torch's libstdc++ ABI symbols. Installing patched torch 2.13 over a vllm-rocm built against torch 2.11 fails with e.g.:
```
ImportError: vllm/_C.abi3.so: undefined symbol: _ZN2at4cuda24getCurrentCUDABlasHandleEv
```
(2.13 changed the signature/namespace of `at::cuda::getCurrentCUDABlasHandle`.)

**Two options:**
1. **Rebuild vLLM C extensions against torch 2.13** — `cd /home/qiushuo/src/vllm && pip install -e . --no-build-isolation` in the patched venv. Risk: triton 3.5.1 < 3.6.* warning from torch 2.13, plus other ROCm 7.2 + new torch ABI surfaces (flash-attn, custom kernels). 30–60 min compile.
2. **Backport patch to torch 2.11.0+rocm7.2** ← **preferred when keeping vLLM untouched.** Same 6-line `num_desc=1` change, but on the `release/2.11` branch — wheel ABI matches `vllm-rocm-latest` exactly, drop-in replacement. Same ~2h compile budget. Caveat: 2.11 may not even have the driver-API code path; check `c10/cuda/CUDACachingAllocator.cpp` for `ExpandableSegment` symbols on that branch first. If 2.11 is the stub-only version, only option 1 remains.

**Decision rule:** before doing the C source rebuild, decide which torch version to target by checking what the existing vLLM venv was linked against (`pip show torch`). Build torch from THAT exact tag, not from nightly main.

## Faster validation flow (use this next time)
1. Clone target torch source at the exact commit matching the vLLM venv's torch.
2. Apply patch → `python tools/amd_build/build_amd.py` → check both `c10/cuda/...` and `c10/hip/...` show `num_desc=1`.
3. Build into `dist/`.
4. Install into a fresh throwaway venv with **only** numpy: `pip install --no-deps <wheel> numpy`.
5. Run `/tmp/vmm_smoke.py` (16×512MiB + 2GiB realloc) with `PYTORCH_HIP_ALLOC_CONF=expandable_segments:True`. Must print `VMM SMOKE PASS`.
6. Only then: `rsync -a` clone the prod vLLM venv, sed-fix `pyvenv.cfg` + `bin/*` shebangs, `pip install --no-deps --force-reinstall <wheel>`.
7. `python -c "from vllm import _C"` — if `undefined symbol`, you targeted the wrong torch version; rebuild against the matching tag.

## Related skills
- `rx9070-vllm-rocm-turboquant` — vLLM build on this same box, the original consumer of this allocator
- `rx9070-gptq-qwopus-testing` — Qwopus TQ4 model that surfaced the OOM driving this investigation
- `rocm-wsl-radeon` — base ROCm install on WSL
