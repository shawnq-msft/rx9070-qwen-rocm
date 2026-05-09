---
name: rx9070-vllm-rocm-troubleshooting
description: Class-level troubleshooting umbrella for running vLLM + ROCm + custom CUDA-style plugins (TurboQuant, etc.) on this user's WSL2 + RX 9070 (gfx1201) + ROCm 7.2 + torch 2.11+rocm7.2 setup. Covers vLLM C-extension ABI drift after torch upgrades, and CUDA-source plugins that don't compile cleanly under hipcc / RDNA4 wave64. Load before assuming "it's the quant plugin" or "the model is OOM" — the failure is often elsewhere.
version: 1.0.0
author: Hermes Agent
license: MIT
---

# vLLM + ROCm troubleshooting on RX 9070 (gfx1201)

This skill collects recurring infrastructure-level failures on this user's setup:
- WSL2 Ubuntu (kernel 6.6.x microsoft-standard)
- AMD RX 9070, 16 GiB VRAM, gfx1201 (RDNA4, wave64)
- ROCm 7.2.2 (latest upstream as of session)
- torch 2.11.0+rocm7.2 (from `download.pytorch.org/whl/rocm7.2`)
- local vLLM checkout at `~/src/vllm`
- venv `~/.venvs/vllm-rocm-latest`

Use it whenever a vLLM server fails to start or a quant/attention plugin fails to build/load. **Triage in this order**: vLLM C-ext ABI → plugin build flags → wave64 mask widths → quant config / OOM.

## 1. vLLM `_C.abi3.so` ABI drift after torch upgrade

**Symptom** (often appears in EngineCore log, sometimes after `pip install -U torch`, `hermes update`, or a fresh venv):
```
ImportError: vllm/_C.abi3.so: undefined symbol: _ZN2at4cuda24getCurrentCUDABlasHandleEb
…
AttributeError: '_OpNamespace' '_C' object has no attribute 'silu_and_mul'
```
The `_C` C-extension fails to load silently → every `torch.ops._C.*` access raises `AttributeError`. Model load chokes on `SiluAndMul.__init__` long before any quant or attention kernel runs.

**Cause**: prebuilt `vllm/_C.abi3.so` was linked against an older torch where `at::cuda::getCurrentCUDABlasHandle` took a `bool` (`...HandleEb`). Current torch exports the void overload (`...HandleEv`).

**Don't blame the quant plugin** when this fires. It's pure ABI drift, unrelated to TurboQuant / GPTQ / KV-cache config.

**Verify**:
```bash
nm -D ~/.venvs/vllm-rocm-latest/lib/python3.12/site-packages/torch/lib/libtorch_hip.so \
  | grep getCurrentCUDABlasHandle
nm -D ~/src/vllm/vllm/_C.abi3.so | grep getCurrentCUDABlasHandle
```
Mismatch (`Ev` exported by torch vs. `Eb` undefined-required by vllm `_C`) confirms drift.

**Fix** — rebuild vLLM C extensions in-place against current torch:
```bash
cd ~/src/vllm
PYTORCH_ROCM_ARCH=gfx1201 \
CMAKE_BUILD_PARALLEL_LEVEL=$(( $(nproc) / 2 )) \
~/.venvs/vllm-rocm-latest/bin/python setup.py build_ext --inplace
```
Takes ~20–40 min on this machine. After it finishes verify:
```bash
~/.venvs/vllm-rocm-latest/bin/python -c "
import vllm._C, torch
print('silu_and_mul:', hasattr(torch.ops._C, 'silu_and_mul'))
"
```
Should print `silu_and_mul: True`.

Run this rebuild **first** any time EngineCore dies on `_OpNamespace '_C'` / `silu_and_mul` / `getCurrentCUDABlasHandle`. Don't bisect quant configs until ABI is verified clean.

## 2. CUDA-source plugins that need ROCm/HIP gating

Many third-party kernels (TurboQuant `turboquant_vllm`, etc.) ship `.cu` sources written for nvcc and a `build.py` that hardcodes nvcc-only flags. On this ROCm-built torch they misdetect because `torch.version.cuda is None`.

**Pattern A — gate detection by `torch.version.hip`, not `cuda`**:
```python
_is_hip = (torch.version.hip is not None)   # truthy on ROCm builds
_is_cuda = (torch.version.cuda is not None) and not _is_hip
```
The ROCm wheel's `torch.version.cuda` is `None` and `torch.version.hip = '7.2.26015'`. Plugin code that says `if torch.version.cuda:` will skip the GPU path entirely or pass nvcc flags into hipcc.

**Pattern B — strip nvcc-only flags on the HIP path**:
- drop `--use_fast_math` (hipcc rejects, replace with `-ffast-math`)
- drop all `-gencode=arch=compute_*,code=sm_*` (CUDA SM gating; on ROCm use `PYTORCH_ROCM_ARCH=gfx1201` env)
- keep `-O3 -ffast-math` for HIP

The hipify pipeline rewrites `.cu` → `.hip` automatically; you don't need to translate sources manually.

## 3. RDNA4 wave64 + ROCm 7.2 needs 64-bit `__shfl_xor_sync` masks

**Symptom**: hipcc compile fails on warp shuffles with a static_assert about lane-mask width. CUDA sources usually pass `0xFFFFFFFFu` (32-bit) as the active mask.

**Fix** — widen mask conditionally so CUDA stays 32-bit:
```cpp
#ifdef __HIP_PLATFORM_AMD__
  constexpr uint64_t FULL_MASK = 0xFFFFFFFFFFFFFFFFull;
#else
  constexpr uint32_t FULL_MASK = 0xFFFFFFFFu;
#endif
__shfl_xor_sync(FULL_MASK, val, lane);
```
Don't unconditionally widen; CUDA path expects 32-bit.

## 4. WSL HSA blocks `expandable_segments` — don't chase this knob

`PYTORCH_HIP_ALLOC_CONF=expandable_segments:True` is dead in WSL2: the HSA runtime's `hsa_amd_vmem_set_access` is broken, and the official PyTorch ROCm wheel was also built without `PYTORCH_C10_DRIVER_API_SUPPORTED`, so the path is gated off in code anyway. Don't waste time here.

Practical alternative for fragmentation under tight VRAM:
```bash
export PYTORCH_HIP_ALLOC_CONF="garbage_collection_threshold:0.9,max_split_size_mb:512"
```
Buys 0.5–1 GiB headroom; not equivalent to expandable but actually works on WSL.

## 5. amdsmi noise is harmless on WSL

```
UserWarning: Can't initialize amdsmi - Error code: 34
```
Always present on WSL. Ignore. It does not cause downstream failures by itself, though it does cause torch/vllm to fall back away from any code path that probes amdsmi for capacity.

## Triage flow when a vLLM ROCm server fails

1. **Read the EngineCore log, not just the APIServer tail.** APIServer's traceback is `RuntimeError: Engine core initialization failed. See root cause above.` — the actual error is in lines prefixed `(EngineCore pid=…)`.
2. If the root cause mentions `_OpNamespace '_C'` / `silu_and_mul` / `getCurrentCUDABlasHandle` → §1 ABI drift, rebuild `vllm/_C`. **Do this before anything else.**
3. If hipcc fails with `unknown argument: --use_fast_math` or `-gencode=...` → §2, patch plugin `build.py`.
4. If hipcc fails with a `__shfl_xor_sync` / `static_assert` about mask width → §3, widen mask under `__HIP_PLATFORM_AMD__`.
5. If allocator OOMs on prefill at otherwise sane ctx → §4, set `PYTORCH_HIP_ALLOC_CONF` (do not chase `expandable_segments`).
6. Only after the above does it make sense to bisect `--max-model-len`, `--max-num-seqs`, KV bits, etc.

## Sanity check after fixes

Once the server starts, before declaring victory:
```bash
curl -s http://127.0.0.1:$PORT/health -o /dev/null -w "health: %{http_code}\n"
curl -s -X POST http://127.0.0.1:$PORT/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"<MODEL_PATH>","prompt":"def fib(n):\n    \"\"\"Return nth Fibonacci.\"\"\"\n","max_tokens":64,"temperature":0}' \
  | python3 -m json.tool
grep -E "Available KV cache|GPU KV cache size|Application startup" "$SERVER_LOG" | tail
```
Required: `health: 200`, non-empty completion, KV size logged. A `/health 200` alone is not proof — confirm a real generation completes.

## Scouting HF for vLLM-loadable TurboQuant checkpoints

When the user asks for "find a TQ3/TQ4 model X for vLLM", most `*-TQ3_*` repos on HF
are GGUF-only. See `references/scouting-hf-for-vllm-turboquant.md` for the
distinguishing signals (`library_name: gguf` vs `transformers`, must contain both
`*.safetensors` AND `turboquant_config.json`), API recipes (`hf` has no `search`
subcommand — use Hub HTTP API + Python; `jq` not installed), and a session-validated
list of decoy vs. real repos for Qwen3.5-27B TurboQuant.

## Related skills
- `rx9070-qwopus-k4v3-plugin-humaneval` (archived): full Qwopus K4V3 deployment + HumanEval recipe; reference for env vars `TQ_KV_K_BITS=4 / TQ_KV_V_BITS=3 / TQ_KV_NORM_CORRECTION=1 / TQ_KV_BOUNDARY_LAYERS=5` and the `local-completions` lm-eval invocation.
- `rx9070-pytorch-rocm-expandable-segments`: details of why §4 is dead on WSL.
- `rx9070-pytorch-rocm-source-build`: when fixing this at the torch level instead of working around.
