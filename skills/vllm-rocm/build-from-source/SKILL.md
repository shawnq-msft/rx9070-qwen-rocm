---
name: rx9070-vllm-rocm-turboquant
description: Build latest vLLM from source on this user's WSL + ROCm + RX 9070 setup, then test local GPTQ Qwen/Qwopus models with baseline KV cache and TurboQuant KV cache presets.
version: 1.2.0
author: Hermes Agent
license: MIT
---

# RX 9070 WSL vLLM ROCm + TurboQuant workflow

Use this when the user wants to test **latest vLLM main** on this exact machine:
- WSL Ubuntu
- AMD Radeon RX 9070 16GB
- ROCm 7.2.x via `/dev/dxg`
- local GPTQ Qwen/Qwopus model, especially `caiovicentino1/Qwopus3.5-9B-v3-HLWQ-v7-GPTQ`
- compare baseline vs TurboQuant KV cache presets from PR `#38479`

## What was learned on this machine

### Latest vLLM main already contains the needed KV-quant code
On the tested checkout:
- repo: `/home/qiushuo/src/vllm`
- tested HEAD: `5cdddddd4a03b0d1b6fa1940458e432b91457ae1`

Important merged commits found locally:
- `f4b42df04` — `[Attention Backend] TurboQuant: 2-bit KV cache compression with 4x capacity (#38479)`
- `6ef1efd51` — `[ROCm] Fix TurboQuant on ROCm: backend routing, flash-attn compat, int64 overflow (#39953)`

So do **not** waste time hunting for an unmerged PR first; current `main` already includes both the feature and a later ROCm fix.

### Source-build pitfall on this WSL ROCm setup
A naive editable install can fail with errors like:
- `RuntimeError: Unknown runtime environment`
- pyproject validation errors around `project.license`

Practical fix that worked here:
1. create a clean Python 3.12 venv
2. preinstall ROCm PyTorch there
3. ensure `setuptools>=77,<81` in the target venv
4. install vLLM editable with `--no-build-isolation`

Do **not** rely on isolated editable-build env detection for ROCm on this machine.

### Runtime platform-detection pitfall on WSL ROCm
Even after a successful source build, `vllm` may still fail at runtime on this WSL box because ROCm platform detection goes through `amdsmi`, and on WSL that often fails with:
- `AmdSmiLibraryException(34)`
- `AMDSMI_STATUS_DRIVER_NOT_LOADED - Driver not loaded`

Observed failure pattern before patching:
- `torch.version.cuda is None`
- `torch.version.hip` is populated
- but `resolve_current_platform_cls_qualname()` still resolves to `vllm.platforms.cuda.CudaPlatform`
- one failing startup then crashes in import-time logging with:
  - `ImportError: cannot import name 'current_platform' from 'vllm.platforms'`

Practical lesson: on this machine, a successful build is **not** enough. You must verify runtime platform resolution separately.

## Known-good build flow on this machine

### 1) Create the venv
```bash
/home/qiushuo/.local/bin/uv venv --python 3.12 /home/qiushuo/.venvs/vllm-rocm-latest
source /home/qiushuo/.venvs/vllm-rocm-latest/bin/activate
```

### 2) Preinstall build basics + ROCm torch
```bash
/home/qiushuo/.local/bin/uv pip install -U pip setuptools wheel packaging ninja cmake pybind11
/home/qiushuo/.local/bin/uv pip install --index-url https://download.pytorch.org/whl/rocm7.2 torch==2.11.0+rocm7.2 torchvision torchaudio
```

### 3) Make sure setuptools is new enough
This mattered here.
```bash
/home/qiushuo/.local/bin/uv pip install 'setuptools>=77,<81' wheel packaging jinja2 setuptools_scm numpy
```

### 4) Build/install vLLM from source
Use **no build isolation**.
```bash
cd /home/qiushuo/src/vllm
export HSA_ENABLE_DXG_DETECTION=1
export ROC_ENABLE_PRE_VEGA=0
export VLLM_TARGET_DEVICE=rocm
MAX_JOBS=4 PYTORCH_ROCM_ARCH=gfx1201 VLLM_TARGET_DEVICE=rocm \
  /home/qiushuo/.local/bin/uv pip install -e . --torch-backend=auto --no-build-isolation
```

Known-good resulting package on this machine:
- import verification returned `vllm 0.19.1rc1.dev383+g5cdddddd4`

Note: during earlier notes a `.rocm722` suffix was mentioned once, but the final import-verified runtime version string on this machine was:
- `0.19.1rc1.dev383+g5cdddddd4`

### 5) Verify import/runtime
```bash
export HSA_ENABLE_DXG_DETECTION=1 ROC_ENABLE_PRE_VEGA=0
source /home/qiushuo/.venvs/vllm-rocm-latest/bin/activate
python - <<'PY'
import torch, vllm
print('torch', torch.__version__)
print('vllm', vllm.__version__)
print('cuda_available', torch.cuda.is_available())
if torch.cuda.is_available():
    print('device', torch.cuda.get_device_name(0))
PY
```

Expected on this machine:
- `torch 2.11.0+rocm7.2`
- vLLM imports successfully
- `torch.cuda.is_available() == True`
- device `AMD Radeon RX 9070`

## Local Qwopus GPTQ model facts that matter
Model path:
- `/home/qiushuo/models/qwen/caiovicentino1-Qwopus3.5-9B-v3-HLWQ-v7-GPTQ`

Important local findings:
- GPTQ 4-bit, `group_size=64`, `desc_act=true`
- multimodal Qwen3.5 architecture, so for text-only serving use `--language-model-only`
- on ROCm, do **not** assume GPTQ-Marlin is the working path
- practical baseline is:
  - `--quantization gptq`
  - `--dtype float16`
  - `--language-model-only`

## KV cache presets available in current vLLM
Current local source exposes these useful KV-cache dtypes:
- `auto`
- `fp8`
- `fp8_e4m3`
- `turboquant_k8v4`
- `turboquant_4bit_nc`
- `turboquant_k3v4_nc`
- `turboquant_3bit_nc`

For ROCm, the practical progression is:
1. `auto`
2. `fp8` / `fp8_e4m3`
3. `turboquant_k8v4`
4. `turboquant_4bit_nc`
5. `turboquant_k3v4_nc`
6. `turboquant_3bit_nc`

### Updated finding on this exact Qwopus/Qwen3.5 model
Do **not** assume upstream `main` behavior matches the currently working local checkout on this machine.

On this machine, for the local model:
- `/home/qiushuo/models/qwen/caiovicentino1-Qwopus3.5-9B-v3-HLWQ-v7-GPTQ`

The important historical finding was correct:
- unpatched local `main` blocked TurboQuant on hybrid (attention + Mamba) models at config-build time with:
  - `NotImplementedError: TurboQuant KV cache is not supported for hybrid (attention + Mamba) models. Boundary layer protection requires uniform attention layers.`

However, that is **no longer the effective state of the local repo**.

A local patch based on upstream PR `#39931` was applied successfully in `/home/qiushuo/src/vllm`, and after that patch this exact hybrid Qwen3.5/Qwopus model **does run `turboquant_3bit_nc` successfully** on this WSL + ROCm + RX 9070 setup.

Key local patch areas:
- `vllm/model_executor/layers/quantization/turboquant/config.py`
  - add hybrid full-attention-layer discovery from `layer_types`, `layers_block_type`, or `attn_type_list`
  - disable dense-style boundary skip auto-insertion for hybrid models
- `vllm/engine/arg_utils.py`
  - remove the hard hybrid-model TQ rejection
  - call `TurboQuantConfig.get_boundary_skip_layers(model_config)`
- `vllm/platforms/interface.py`
  - use `TQFullAttentionSpec` for hybrid block/page-size alignment when `cache_dtype.startswith("turboquant_")`
- `vllm/v1/attention/backends/turboquant_attn.py`
  - add ROCm-safe wrapper for `flash_attn_varlen_func(..., out=...)`

Practical verified result on this machine after patching:
- server start with `--kv-cache-dtype turboquant_3bit_nc` succeeds
- logs show:
  - `TQ hybrid: full-attention layers [3, 7, 11, 15, 19, 23, 27, 31]`
  - `Found incompatible backend(s) ... Overriding with TURBOQUANT ...`
  - `GPU KV cache size: 144,768 tokens`
  - `Starting vLLM server on http://127.0.0.1:8024`
  - `Application startup complete.`
- HTTP checks pass:
  - `/health` -> `200`
  - `/version` -> `200`
  - `/v1/models` -> `200`
- a real completion request succeeded, including one 128-token test at about `48.133 tok/s`

Important interpretation:
- For this exact model, **TQ3 is now usable on the patched local checkout**.
- The earlier “hybrid models cannot use TurboQuant here” guidance is now stale for this machine.
- If reproducing on a fresh checkout, first check whether the hybrid-TurboQuant patch is already present. If not, expect the old `NotImplementedError` until a `#39931`-style patch is applied.

Current decision rule for this machine:
- If using the patched `/home/qiushuo/src/vllm`, it is valid to test `turboquant_3bit_nc` directly on this Qwopus/Qwen3.5 hybrid model.
- If using an unpatched upstream checkout and the user requires this exact model's TQ3 to work, apply the `#39931`-style local patch first.
- Keep `#40128` in reserve only if a later run hits hybrid page-size unification failures at different context lengths or on other hybrid architectures.

## Practical serve commands

### Important WSL ROCm startup workflow
On this machine, use the following progression rather than assuming the first startup failure means the build is bad:
1. start from the already built source tree in `/home/qiushuo/src/vllm`
2. export:
   - `HSA_ENABLE_DXG_DETECTION=1`
   - `ROC_ENABLE_PRE_VEGA=0`
   - `VLLM_TARGET_DEVICE=rocm`
3. ignore `VLLM_USE_V1` as a requirement; current builds may warn `Unknown vLLM environment variable detected: VLLM_USE_V1`, and the server can still start successfully without relying on it
4. if startup dies in `rocm.py` with `warning_once` -> `current_platform` import recursion, patch the local source before retrying
5. do not judge success from engine warmup logs alone; wait for all of:
   - `Starting vLLM server on http://127.0.0.1:<port>`
   - successful `GET /health`
   - successful `GET /version`
   - successful `GET /v1/models`

### Baseline
```bash
export HSA_ENABLE_DXG_DETECTION=1 ROC_ENABLE_PRE_VEGA=0 VLLM_TARGET_DEVICE=rocm VLLM_USE_V1=1
source /home/qiushuo/.venvs/vllm-rocm-latest/bin/activate
vllm serve /home/qiushuo/models/qwen/caiovicentino1-Qwopus3.5-9B-v3-HLWQ-v7-GPTQ \
  --quantization gptq \
  --dtype float16 \
  --language-model-only \
  --kv-cache-dtype auto \
  --mamba-ssm-cache-dtype float16 \
  --gpu-memory-utilization 0.90 \
  --max-model-len 8192 \
  --served-model-name qwopus-9b-gptq \
  --host 127.0.0.1 \
  --port 8000
```

Equivalent command that was actually verified in this session:
```bash
export HSA_ENABLE_DXG_DETECTION=1 ROC_ENABLE_PRE_VEGA=0 VLLM_TARGET_DEVICE=rocm VLLM_USE_V1=1
source /home/qiushuo/.venvs/vllm-rocm-latest/bin/activate
python -m vllm.entrypoints.openai.api_server \
  --model /home/qiushuo/models/qwen/caiovicentino1-Qwopus3.5-9B-v3-HLWQ-v7-GPTQ \
  --quantization gptq \
  --dtype float16 \
  --language-model-only \
  --kv-cache-dtype auto \
  --mamba-ssm-cache-dtype float16 \
  --gpu-memory-utilization 0.9 \
  --max-model-len 8192 \
  --host 127.0.0.1 \
  --port 8012
```

### FP8 KV cache
```bash
vllm serve /home/qiushuo/models/qwen/caiovicentino1-Qwopus3.5-9B-v3-HLWQ-v7-GPTQ \
  --quantization gptq \
  --dtype float16 \
  --language-model-only \
  --kv-cache-dtype fp8 \
  --mamba-ssm-cache-dtype float16 \
  --gpu-memory-utilization 0.90 \
  --max-model-len 8192 \
  --served-model-name qwopus-9b-gptq-fp8kv \
  --host 127.0.0.1 \
  --port 8001
```

### TurboQuant starting point
```bash
vllm serve /home/qiushuo/models/qwen/caiovicentino1-Qwopus3.5-9B-v3-HLWQ-v7-GPTQ \
  --quantization gptq \
  --dtype float16 \
  --language-model-only \
  --kv-cache-dtype turboquant_k8v4 \
  --mamba-ssm-cache-dtype float16 \
  --gpu-memory-utilization 0.90 \
  --max-model-len 8192 \
  --served-model-name qwopus-9b-gptq-tq-k8v4 \
  --host 127.0.0.1 \
  --port 8002
```

Then repeat for:
- `turboquant_4bit_nc`
- `turboquant_k3v4_nc`
- `turboquant_3bit_nc`

## Benchmark scripts already created on this machine
Use these instead of rewriting ad hoc scripts each time:
- `/home/qiushuo/reports/vllm-rocm-eval/bench_gptq_inprocess.py`
- `/home/qiushuo/reports/vllm-rocm-eval/bench_openai_server.py`
- `/home/qiushuo/reports/vllm-rocm-eval/summarize_runs.py`
- `/home/qiushuo/reports/vllm-rocm-eval/run_humaneval_gptq.py`
- `/home/qiushuo/reports/vllm-rocm-eval/run_humaneval_openai_api.py`
- `/home/qiushuo/reports/vllm-rocm-eval/probe_tq3_ctx_usage.py`
- `/home/qiushuo/reports/vllm-rocm-eval/run_tq3_128k_eval.py`
- continuous-batching notes/scripts: see `references/qwopus-continuous-batching.md`
- high-ctx startup-vs-real-prefill lesson and OOM interpretation: see `references/high-ctx-prefill.md`

Results directory:
- `/home/qiushuo/reports/vllm-rocm-eval/results`

## Reusable 128k TQ3 mini-HumanEval workflow
When the user specifically wants **this exact GPTQ Qwopus model + `turboquant_3bit_nc` + `ctx=131072` + a minimal HumanEval with token rate / GPU VRAM / CPU-load evidence**, use:
- `/home/qiushuo/reports/vllm-rocm-eval/run_tq3_128k_eval.py`

What it automates:
1. kills residual `VLLM::EngineCore`, API server, and `resource_tracker`
2. launches the patched local vLLM server with the exact 128k TQ3 config
3. verifies `/health`, `/version`, and `/v1/models`
4. records GPU memory using `torch.cuda.mem_get_info()`
5. records CPU/load and RAM snapshots from:
   - `ps -eo pid,ppid,pcpu,pmem,rss,comm,args`
   - `/proc/loadavg`
   - `/proc/meminfo`
6. runs token-rate benchmark with `bench_openai_server.py`
7. runs a minimal HumanEval subset with `run_humaneval_openai_api.py`
8. scores pass@1 with `evaluate_humaneval_passk.py`
9. saves all outputs in a timestamped directory under `results/tq3-128k-minihe/`

Important observed baseline on this machine for the 128k TQ3 run:
- server is healthy and usable at 128k on the patched checkout
- model load memory: about `8.55 GiB`
- available KV cache memory: about `4.46 GiB`
- GPU KV cache size: about `187,920 tokens`
- engine init time: about `26.6 s`
- token-rate benchmark: about `57.9 tok/s`
- minimal HumanEval (`limit=10`) pass@1: `0.8`
- ready/benchmark VRAM usage: about `15.16-15.21 GiB`
- CPU snapshot during benchmark:
  - `VLLM::EngineCore` roughly `300%+`
  - API server process roughly `220%+`

Interpretation:
- 128k + TQ3 is genuinely usable on this machine for this exact model
- but it is already very close to the RX 9070 16GB VRAM ceiling
- system RAM is not the bottleneck here; GPU headroom is

## Optimization finding from real 128k TQ3 retest
A follow-up retest tried:
- `--safetensors-load-strategy prefetch`
- `--attention-config '{"flash_attn_version":2}'`

Observed result on this machine:
- no quality gain (`pass@1` stayed `0.8` on the 10-task mini subset)
- no memory improvement
- token rate was slightly worse (about `56.0 tok/s` vs `57.9 tok/s` baseline)
- engine init was slightly slower as well

Practical recommendation:
- for this exact 128k TQ3 setup, keep the simpler baseline launch
- do **not** assume `prefetch` or explicitly forcing FA2 improves performance here; direct measurement says it does not

## Varjosoft Gemma TQ3 / native-TQ3 findings on this ROCm RX 9070 machine

Two different Hugging Face checkpoints matter here:

1. `varjosoft/gemma-4-26B-A4B-it-TQ3`
- This is still a standard FP16 checkpoint on disk (~52 GB).
- The model card suggests runtime re-compression via:
  - `from turboquant_vllm import enable_weight_quantization`
  - `enable_weight_quantization(bits=3)`
  - then `vllm serve varjosoft/gemma-4-26B-A4B-it-TQ3`
- But the card also states the full checkpoint must fit in GPU memory during loading before compression and recommends an A100 80GB or larger GPU.
- Therefore on this RX 9070 16GB setup, do **not** waste time trying the page's runtime-recompression path for this checkpoint. It is not a realistic fit.

2. `varjosoft/gemma-4-26B-A4B-it-TQ3-native`
- This is the only plausible route on this machine.
- It downloads at about 12 GB and must be loaded with:
  - `from turboquant_vllm import load_tq3_model`
- It is **not** a normal `AutoModelForCausalLM.from_pretrained()` checkpoint.

### Critical ROCm-specific finding for `turboquant-plus-vllm`
On this WSL + ROCm machine, the plugin's optional CUDA extension build path is not ROCm-safe.

### New finding: exact varjoranta-style asymmetric K4/V3 is **not** the same as upstream vLLM presets
When the goal is specifically **"exact asymmetric K4/V3"** for the local Qwopus/Qwen3.5 GPTQ model, do **not** assume it maps to `turboquant_k3v4_nc` or any other upstream preset.

What was verified from `varjoranta/turboquant-vllm` on this machine:
- the repo's **exact asymmetric K4/V3** naming belongs to the **legacy monkey-patch/plugin KV path**
- activation path is via:
  - `patch_vllm_attention(k_bits=4, v_bits=3, norm_correction=True, ...)`
  - or env vars like:
    - `TQ_KV_K_BITS=4`
    - `TQ_KV_V_BITS=3`
- upstream vLLM presets are different names and semantics:
  - `turboquant_k8v4`
  - `turboquant_4bit_nc`
  - `turboquant_k3v4_nc`
  - `turboquant_3bit_nc`
- the plugin README explicitly says standard GQA/MHA families (Qwen/Llama/Mistral/Gemma) should use upstream KV compression, while the legacy monkey-patch path is mainly retained for MLA cases

Practical decision rule for this machine:
- for **Qwopus/Qwen3.5**, treat varjoranta **exact K4/V3** as a separate legacy route, **not** as an alias for upstream presets
- if the user requests exact K4/V3 specifically, first verify the plugin path itself before doing any benchmark comparisons
- if the user only wants the nearest practical local route, prefer upstream patched-local-vLLM presets instead of the plugin path

### New finding: exact K4/V3 plugin route is not currently a credible deployment path for Qwopus on this ROCm box
On this machine, for the local model:
- `/home/qiushuo/models/qwen/caiovicentino1-Qwopus3.5-9B-v3-HLWQ-v7-GPTQ`

Practical observations from real server probes:
- the plugin entry point exists in the venv under `vllm.general_plugins` as `turboquant_weight -> turboquant_vllm._vllm_plugin:register`
- if `VLLM_PLUGINS=turboquant_weight` is set and `load_general_plugins()` is called manually, `FlashAttentionImpl` methods do get wrapped, proving the monkey-patch code path is reachable
- however, real server runs on this machine still failed to produce reliable plugin-activation evidence in logs (`TurboQuant plugin activated`, `TurboQuant K4/V3 KV monkey-patch registered`, `TurboQuant+ patched vLLM` were absent in the probe logs)
- runtime still showed backend routing like:
  - `Found incompatible backend(s) [TURBOQUANT] with AttentionType.DECODER. Overriding with ROCM_ATTN ...`
- therefore the exact K4/V3 path was **not** considered successfully validated for Qwopus on this machine

Interpretation:
- exact K4/V3 via varjoranta's legacy path is **not** currently a trustworthy deployment target here for Qwopus/Qwen3.5 on ROCm
- do **not** spend a long benchmark run or full HumanEval on that route unless you first obtain explicit patch-activation evidence in the actual server logs
- the practical fallback for this machine remains the patched local upstream presets (`turboquant_4bit_nc`, `turboquant_k3v4_nc`, `turboquant_3bit_nc`), not the exact legacy K4/V3 route

### New Gemma4 native-TQ3 + vLLM findings from real 64k FP8 KV debugging
When pushing `varjosoft/gemma-4-26B-A4B-it-TQ3-native` through vLLM with:
- `--quantization turboquant`
- `--kv-cache-dtype fp8`
- `--max-model-len 65536`
- `--attention-backend TRITON_ATTN`

there were **three distinct blockers in sequence** on this machine.

1. **Checkpoint-name remapping bug for Gemma4 MoE native TQ3 tensors**
   - Initial failure was:
     - `KeyError: 'layers.0.moe.down_proj.tq_norms'`
   - Root cause:
     - local `turboquant_vllm.vllm_quant._decompress_get_all_weights()` only recognized
       - `*.weight.tq_packed`
       - `*.weight.tq_norms`
     - but Gemma4 native TQ3 MoE expert tensors are stored as e.g.
       - `model.language_model.layers.0.experts.down_proj.tq_packed`
       - `model.language_model.layers.0.experts.gate_up_proj.tq_norms`
       - i.e. **without an intermediate `.weight` segment**.
   - Practical fix that worked:
     - patch `.../site-packages/turboquant_vllm/vllm_quant.py`
     - extend `_decompress_get_all_weights()` to also pair bare suffixes:
       - `name.endswith('.tq_packed')`
       - `name.endswith('.tq_norms')`

2. **Gemma4 MoE tensors must stay 3D during decompression**
   - After fixing the suffix handling, the next failure became:
     - `KeyError: 'layers.0.moe.down_proj'`
   - Root cause:
     - the same hook always reconstructed packed tensors with shape `(1, n_rows, in_dim)` and then `.squeeze(0)`.
     - that is fine for ordinary linear weights, but **wrong for Gemma4 MoE expert tensors**.
     - Gemma4's loader expects MoE tensors like `moe.down_proj` / `moe.gate_up_proj` to remain 3D so it can explode them into per-expert 2D weights.
   - Practical fix that worked:
     - in `_decompress_get_all_weights()`, if the base name contains:
       - `.experts.down_proj`
       - `.experts.gate_up_proj`
     - recover `num_experts` from `model_config.hf_config.text_config.num_experts`
     - decode to `(num_experts, n_rows // num_experts, in_dim)`
     - only squeeze when the target leading dimension is actually `1`

3. **After both compatibility fixes, the real blocker is VRAM peak during MoE recompression**
   - Once the two loader/remapping bugs were fixed, startup advanced much further and the explicit Triton KV-cache dtype blocker was gone.
   - The real failure then became:
     - `torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 1.89 GiB`
   - Important interpretation:
     - `fp8` KV cache **does successfully bypass** the earlier Gemma4 + `TRITON_ATTN` rejection of `turboquant_3bit_nc` KV cache.
     - the remaining blocker is now **weight-side**, not KV-side.
   - Exact failing path in the later startup:
     - `turboquant_vllm.vllm_quant._buffering_loader`
     - `_materialize_and_process(...)`
     - `TurboQuantOnlineMoEMethod._do_compress(...)`
     - `_compress_3d_param(layer, 'w13_weight', ...)`
     - `Compressed3D(...)`
     - `quantizer.quantize(grouped)`
     - `torch_ops._rotate()`
     - `padded = x.clone()`
   - Observed memory state at failure:
     - total VRAM: about `15.87 GiB`
     - free VRAM: about `1.46 GiB`
     - PyTorch allocated: about `13.65 GiB`
     - attempted extra allocation: about `1.89 GiB`
   - Conclusion for this exact machine:
     - **64k Gemma4 native TQ3 + FP8 KV is no longer blocked by unsupported Triton KV dtype once the loader is patched**.
     - it is now blocked by **peak VRAM during TurboQuant MoE online recompression**, especially for `w13_weight`.

### Follow-up finding: chunking `Compressed3D` by expert does mitigate the original MoE OOM
A later local patch to:
- `.../site-packages/turboquant_vllm/weight_quant.py`
- class `Compressed3D.__init__`

changed 3D MoE compression from a one-shot quantization of the full `(num_experts, out_dim, in_dim)` tensor to an **expert-chunked** loop:
- compute `chunk_experts` from env var `TQ_COMPRESS_3D_CHUNK_MB` (default used locally: `192`)
- for each chunk:
  - reshape only that chunk to 2D
  - pad if needed
  - call `quantizer.quantize(...)`
  - append packed/norm chunks
- concatenate at the end

Why this mattered:
- the original failure came from `PolarQuantTorch._rotate()` doing `x.clone()` on a very large grouped tensor during `_compress_3d_param(layer, "w13_weight", ...)`
- chunking reduces that transient working set substantially

Observed result on this machine after patching:
- reran `Gemma native TQ3 + FP8 KV + 32k ctx`
- result dir:
  - `/home/qiushuo/reports/vllm-rocm-eval/results/gemma32-fp8/20260419-032143/`
- the old immediate OOM stack **did not recur**
- startup progressed much further, including:
  - `Loading safetensors checkpoint shards: 33% Completed | 1/3 [05:05<10:11, 305.94s/it]`
- but the server still did **not** become healthy before the harness timeout

Important interpretation:
- this patch is a **real mitigation**, not a no-op
- it appears to convert the previous failure mode from:
  - **hard OOM during MoE online compression**
  into
  - **very slow / not-ready-before-timeout startup**
- therefore the next debugging target is no longer just peak VRAM; it is also total load latency and whether a later-stage blocker still exists after shard 1/3

Practical next-step rule:
- if Gemma native TQ3 + FP8 KV still times out after adding chunked `Compressed3D`, do **not** assume the OOM is unchanged
- first inspect whether the run now advances into shard loading / later model-load phases
- if yes, increase startup timeout and add richer progress logging before changing quantization logic again

### Important negative result: `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` does not help here
Tried as a fragmentation mitigation, but on this HIP/ROCm path the log warned:
- `expandable_segments not supported on this platform`

So do not expect that env var to solve the Gemma4 native-TQ3 MoE OOM on this machine.

### New finding: chunking `Compressed3D` mitigates the early Gemma MoE OOM, but makes startup extremely slow
A local patch to:
- `/home/qiushuo/.venvs/vllm-rocm-latest/lib/python3.12/site-packages/turboquant_vllm/weight_quant.py`

changed `Compressed3D.__init__()` from one-shot 3D MoE quantization to **expert-chunked compression** controlled by:
- `TQ_COMPRESS_3D_CHUNK_MB`

Practical patch behavior:
- estimate per-expert working-set size as roughly `out_dim * in_dim * max(dtype_size, 4)`
- choose `chunk_experts` from the chunk budget
- loop over experts in chunks
- quantize each chunk separately
- `torch.cat(...)` the packed/norm tensors at the end

Why this mattered on this machine:
- the earlier failure path was:
  - `_compress_3d_param(layer, "w13_weight", ...)`
  - `Compressed3D(...)`
  - `quantizer.quantize(grouped)`
  - `torch_ops._rotate()`
  - `padded = x.clone()`
  - `torch.OutOfMemoryError: Tried to allocate 1.89 GiB`
- after chunking, that **early OOM no longer reproduced** on the 32k Gemma FP8-KV retry

Observed result after the patch:
- run progressed past the old early OOM point
- log advanced to:
  - `Loading safetensors checkpoint shards: 33% Completed | 1/3 [05:01<10:03, 301.64s/it]`
- `VLLM::EngineCore` remained alive and CPU-heavy instead of crashing immediately
- but the server still did **not** become healthy within the original timeout window

Interpretation:
- chunking is a **real mitigation for peak VRAM** during load-time MoE compression
- but it converts the failure mode from:
  - **hard early OOM**
  to:
  - **very slow startup / timeout before readiness**

Important practical sizing heuristic on this exact Gemma checkpoint:
- one expert's rough fp32 working set is about `45.38 MiB`
- therefore:
  - `TQ_COMPRESS_3D_CHUNK_MB=192` → about `4` experts per pass → `32` passes for `128` experts
  - `TQ_COMPRESS_3D_CHUNK_MB=96` → about `2` experts per pass → `64` passes
  - `TQ_COMPRESS_3D_CHUNK_MB=64` or `32` → `1` expert per pass → `128` passes

Decision rule learned here:
- smaller chunk size = safer VRAM, slower startup
- larger chunk size = faster startup, higher risk of returning to the old OOM
- on this machine, `96` or `128` is the practical range to explore first; going too small makes startup painfully slow

### New finding: slow startup is explained by real WSL/ROCm degradations plus the heavier chunked MoE path
During the chunked 32k Gemma rerun, startup slowness was not explained by a new fatal stack trace. Instead, logs showed several real degradations/warnings:
- `Using 'pin_memory=False' as WSL is detected. This may slow down the performance.`
- `Auto-prefetch is disabled because the filesystem (EXT4) is not a recognized network FS ... If you want to force prefetching, start vLLM with --safetensors-load-strategy=prefetch.`
- `AITER is not found or QuarkOCP_MX is not supported on the current platform. QuarkOCP_MX quantization will not be available.`
- `Using TRITON Unquantized MoE backend out of potential backends: ['ROCm AITER', 'TRITON', 'BATCHED_TRITON']`

Interpretation on this machine:
- startup is slow because load-time work is genuinely heavy:
  - materialize meta params on GPU
  - replay buffered weight loaders
  - compress MoE weights chunk-by-chunk
- and that work is happening with multiple platform-side degradations rather than the best-case path

Practical next optimization to try first:
- add `--safetensors-load-strategy prefetch`
- keep the chunk size conservative but not tiny (`96` or `128` MB)
- increase the launcher health timeout so `slow but progressing` is not mistaken for a true startup failure

This is the current best debugging/optimization playbook for Gemma native-TQ3 + FP8-KV on this RX 9070 WSL ROCm setup.

### Reuse strategy once a local deployment finally works
On this machine, if a Gemma/TurboQuant serving configuration finally becomes stable, the practical way to avoid redoing the debugging is:
- keep using the existing native-TQ3 checkpoint directory (for example `/home/qiushuo/models/gemma-tq3-native/varjosoft-gemma-4-26B-A4B-it-TQ3-native`)
- freeze the exact working launch parameters into reusable `start/stop/status` scripts
- save the exact env vars too (for example chunk size, ROCm env, port, ctx, KV dtype, safetensors load strategy)

Important limitation learned here:
- there is **not** currently a known built-in path in `turboquant_vllm` / vLLM on this machine to serialize the fully materialized / post-processed in-memory serving state into a new "fast reload" checkpoint
- in practice, the reusable artifact is the native-TQ3 checkpoint plus the exact launch script, **not** a dumped warmed vLLM runtime state

### Additional WSL spawn / pickling pitfall discovered later
When trying to use `quantization='turboquant'` through vLLM `LLM(...)` / engine startup on this machine, worker startup can fail before model load with:
- `TypeError: cannot pickle '_Ops' object`

Root cause found here:
- `turboquant_vllm.vllm_quant.register()` defines `TurboQuantConfig` as a **local class** inside the function
- on WSL, vLLM forces multiprocessing `spawn`
- vLLM serializes the full `VllmConfig` with `cloudpickle`
- the local `TurboQuantConfig` class is therefore pickled by value and drags in an unpicklable `_Ops` object

Practical local fix that worked here:
- patch `.../site-packages/turboquant_vllm/vllm_quant.py`
- add a module-level rebuild helper, e.g. `_rebuild_turboquant_config(bits, group_size, sensitive_bits)` that calls `register()` and reconstructs the config via `get_quantization_config("turboquant")`
- add `TurboQuantConfig.__reduce__()` to return that helper plus `(self.bits, self.group_size, self.sensitive_bits)`

After this patch on this machine:
- `cloudpickle.dumps(vllm_config)` succeeds
- vLLM worker startup proceeds past the previous pickling failure
- the next blocker becomes backend compatibility, not multiprocessing serialization

Observed failure:
- `turboquant_vllm.build.py` passes NVIDIA-only flags into `hipcc`, including:
  - `--use_fast_math`
  - `-gencode=arch=compute_75,code=sm_75`
  - `-gencode=arch=compute_80,code=sm_80`
  - etc.
- ROCm clang then fails with errors like:
  - `clang++: error: unknown argument: '--use_fast_math'`
  - `clang++: error: unknown argument: '-gencode=arch=compute_80,code=sm_80'`

Practical workaround that allowed the native checkpoint to load:
```python
import turboquant_vllm.weight_quant as wq
wq._cuda_available = False
wq._cuda_mod = None
wq._get_cuda_module = lambda: None
```
Do this **before** calling `load_tq3_model(...)` so the plugin skips the broken CUDA-extension probe and falls back to Triton / PyTorch paths.

### Actual result on this machine
Using:
- checkpoint: `/home/qiushuo/models/gemma-tq3-native/varjosoft-gemma-4-26B-A4B-it-TQ3-native`
- plugin: `turboquant-plus-vllm` in `/home/qiushuo/.venvs/vllm-rocm-latest`
- loader: `load_tq3_model()` with the CUDA-extension probe disabled as above

The model did run end-to-end:
- load time: about `24.6 s`
- VRAM after load: about `12.8 GiB`
- VRAM during tiny generation: about `14.1 GiB`
- sample completion was correct (`The capital of Finland is Helsinki.`)

But throughput was extremely poor:
- `8` completion tokens in about `204 s`
- about `0.039 tok/s`

Interpretation:
- `...-TQ3-native` is **technically runnable** on this RX 9070 16GB machine.
- It is **not practically useful** in its current ROCm fallback state.
- Treat it as a feasibility demo, not a recommended everyday local model route.
- If the user asks for a usable local Gemma option on this machine, prefer the already-validated GGUF / llama.cpp HIP path instead.

## TQ3 context-limit probing workflow discovered on this machine
When the user changes goal from quality benchmarks to **"find the maximum TQ3 context and measure memory usage"**, do not keep running HumanEval/PinchBench. Cancel or stop those runs, clear all vLLM leftovers, and switch to a fresh per-context probing workflow.

### New GGUF Qwen3.5 limitation for this vLLM environment
If the user asks to test a locally downloaded **Qwen3.5 GGUF** model (for example `/home/qiushuo/models/qwen/unsloth-Qwen3.5-9B-GGUF/Qwen3.5-9B-Q8_0.gguf`) with vLLM KV-cache dtypes like `fp8` or `turboquant_3bit_nc`, do **not** assume the experiment is meaningful on this machine's current vLLM stack.

What was verified here:
- launching either
  - `--kv-cache-dtype fp8`
  - `--kv-cache-dtype turboquant_3bit_nc`
- against the local GGUF file failed before engine startup
- both runs exited with the same config-load error:
  - `ValueError: GGUF model with architecture qwen35 is not supported yet.`

Practical interpretation:
- on this machine's current `vllm 0.19.1rc1.dev383+g5cdddddd4` + current `transformers`, **Qwen3.5 GGUF is not a valid vLLM KV-cache test target**
- this is not a VRAM or context-length result
- it is an upstream/model-loader compatibility limitation

Decision rule:
- if the user wants to test **GGUF Qwen3.5**, prefer `llama.cpp`
- if the user wants to test **vLLM fp8/turboquant KV cache**, switch to a non-GGUF checkpoint (GPTQ/AWQ/safetensors family)

### New full-context fp8 startup finding for the local Qwopus GPTQ model
For the local non-GGUF Q4-equivalent checkpoint:
- `/home/qiushuo/models/qwen/caiovicentino1-Qwopus3.5-9B-v3-HLWQ-v7-GPTQ`

A new full-context startup test with:
```bash
python -m vllm.entrypoints.openai.api_server \
  --model /home/qiushuo/models/qwen/caiovicentino1-Qwopus3.5-9B-v3-HLWQ-v7-GPTQ \
  --quantization gptq \
  --dtype float16 \
  --language-model-only \
  --kv-cache-dtype fp8 \
  --mamba-ssm-cache-dtype float16 \
  --gpu-memory-utilization 0.9 \
  --max-model-len 262144 \
  --host 127.0.0.1 \
  --port 8120
```
was verified to reach full API readiness on this machine.

Observed metrics:
- `Starting vLLM server on http://127.0.0.1:8120`
- `/health` -> `200`
- `/version` -> `200`
- `/v1/models` -> `200`
- `Model loading took 7.53 GiB memory and 16.22 s`
- `Available KV cache memory: 4.47 GiB`
- `GPU KV cache size: 72,896 tokens`
- `Maximum concurrency for 262,144 tokens per request: 1.11x`
- `init engine ... took 199.01 s`
- live ready-state VRAM observed from `torch.cuda.mem_get_info()` was about `14.28 GiB` used / `1.59 GiB` free

Important interpretation:
- on this patched local vLLM checkout, **Qwopus GPTQ + fp8 KV can at least start at full 262144 context** on the RX 9070 16GB machine
- do **not** collapse this into "real full-length request is proven" unless a near-full long request also succeeds
- but it is strong evidence that `fp8` is materially more viable than some earlier TQ3/TQ4 startup-only paths at extreme context

### New full-context real-request result for local Qwopus GPTQ Q4-equivalent path
A later direct comparison on this machine used the local non-GGUF GPTQ checkpoint:
- `/home/qiushuo/models/qwen/caiovicentino1-Qwopus3.5-9B-v3-HLWQ-v7-GPTQ`

with full serving context:
- `--max-model-len 262144`

and compared:
1. `--kv-cache-dtype fp8`
2. `--kv-cache-dtype turboquant_3bit_nc`

using the same practical local baseline:
```bash
python -m vllm.entrypoints.openai.api_server \
  --model /home/qiushuo/models/qwen/caiovicentino1-Qwopus3.5-9B-v3-HLWQ-v7-GPTQ \
  --quantization gptq \
  --dtype float16 \
  --language-model-only \
  --kv-cache-dtype <fp8-or-tq3> \
  --mamba-ssm-cache-dtype float16 \
  --gpu-memory-utilization 0.9 \
  --max-model-len 262144 \
  --host 127.0.0.1 \
  --port <port>
```

Observed result for **fp8**:
- server became healthy (`/health`, `/version`, `/v1/models` all `200`)
- log metrics:
  - `GPU KV cache size: 72,896 tokens`
  - `Available KV cache memory: 4.47 GiB`
  - `Model loading took 7.53 GiB memory and 16.22 s`
  - `init engine ... took 199.01 s`
  - `Maximum concurrency for 262,144 tokens per request: 1.11x`
- ready-state VRAM from `torch.cuda.mem_get_info()`:
  - about `14.222 GiB` used
- a **real long request succeeded**:
  - HTTP `200`
  - prompt tokens about `131,006`
  - completion tokens `14`
  - total tokens `131,020`
  - elapsed time about `298.25 s`
  - VRAM after request about `14.281 GiB`

Observed result for **turboquant_3bit_nc** at the same full serving context:
- server also became healthy (`/health`, `/version`, `/v1/models` all `200`)
- log metrics:
  - `GPU KV cache size: 144,768 tokens`
  - `Available KV cache memory: 3.42 GiB`
  - `Model loading took 8.57 GiB memory and 11.81 s`
  - `init engine ... took 187.31 s`
  - `Maximum concurrency for 262,144 tokens per request: 2.17x`
- ready-state VRAM:
  - about `14.128 GiB` used
- but the first real long request failed:
  - HTTP `500`
  - API error: `EngineCore encountered an issue`
  - service died after the request
  - VRAM dropped back to almost idle (~`0.008 GiB` used)
- root cause from server log:
  - `torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 1024.00 MiB`
  - failure again occurred in `vllm/v1/attention/backends/turboquant_attn.py` during continuation-prefill attention

Practical interpretation on this machine:
- for **full-context serving config at 262144**, `fp8` is currently the only one of these two that has been shown to survive a real long request
- `turboquant_3bit_nc` may look better on static KV-cache capacity / theoretical concurrency, but still loses on dynamic real-prefill memory headroom
- for this exact Qwopus GPTQ deployment, prefer `fp8` over `tq3` when the user's goal is the highest practically usable long context rather than theoretical cache density

### Reusable max-real-context probing workflow for the working fp8 path
When the user asks for the **maximum real single-request context** on this machine for the working Qwopus GPTQ + fp8 path, use a server-side token-budgeted binary-search probe rather than character-count heuristics.

Reusable script created for this purpose:
- `/home/qiushuo/reports/vllm-rocm-eval/probe_qwopus_fp8_max_real_ctx.py`

Key method that worked here:
1. launch one server at a candidate `--max-model-len <ctx>` using the local patched vLLM checkout
2. wait for `/health`, `/version`, and `/v1/models`
3. use the server's `/tokenize` endpoint to build the longest prompt satisfying:
   - `prompt_tokens + max_tokens <= max_model_len - safety_margin`
4. send one real `/v1/completions` request with tiny `max_tokens` (for example `16`)
5. treat the ctx as a success only if the request returns `200`
6. use coarse points first, then binary-search between highest success and first failure

Why this workflow matters:
- it avoids false negatives from naive character-based prompt sizing
- it avoids false positives from startup-only readiness
- it gives a directly reusable `verified_max_ctx` for later continuous-batching sizing discussions

### Practical rule for sizing `max-num-batched-tokens`
For this user's continuous-batching experiments, do not choose `max-num-batched-tokens` from startup-only KV-cache metrics alone.

Instead, once a **real** single-request maximum context (or a stable near-maximum prompt budget) is measured, estimate:
```text
max_num_batched_tokens ≈ concurrent_requests × real_prompt_budget_per_request
```
then add only a modest decode/scheduler margin.

For `batch=2`, the practical recommendation format should be:
- **conservative**: about `2 ×` the verified real prompt budget
- **aggressive ceiling**: about `2 × (verified real prompt budget + small decode margin)`

This is more reliable on this machine than using theoretical KV-cache-token capacity, because the real bottleneck repeatedly came from dynamic prefill working-set growth rather than static cache size alone.

### Updated fp8 max-real-context result on this machine
A later continuation of the fp8 investigation pushed the working Qwopus GPTQ + fp8 path beyond the earlier ~131k long-request proof.

Using the same patched local vLLM checkout and the same real-request method:
- launch one server per candidate context
- wait for `/health`, `/version`, and `/v1/models`
- size the prompt with the **server-compatible tokenizer budget** and keep:
  - `prompt_tokens + max_tokens <= max_model_len - safety_margin`
- send one real `/v1/completions` request with `max_tokens=16`

Observed verified successes:
- `ctx=131072`
  - real request `200`
  - `prompt_tokens=130992`
  - `elapsed≈299.1 s`
  - `gpu_ready≈14.251 GiB`
  - `gpu_after_request≈14.308 GiB`
- `ctx=163840`
  - real request `200`
  - `prompt_tokens=163760`
  - `elapsed≈447.3 s`
  - `gpu_ready≈14.262 GiB`
  - `gpu_after_request≈14.318 GiB`
- `ctx=196608`
  - real request `200`
  - `prompt_tokens=196528`
  - `elapsed≈631.1 s`
  - `gpu_ready≈14.221 GiB`
  - `gpu_after_request≈14.280 GiB`
  - `available_kv_cache_gib≈4.47`
  - `gpu_kv_cache_tokens≈72896`

Current practical boundary from this machine:
- **verified_max_ctx = 196608**
- **first_non_success_ctx = 229376**

Important nuance for `ctx=229376`:
- server did become healthy (`/health`, `/version`, `/v1/models` all `200`)
- client-side debug logging confirmed:
  - prompt built to about `229296` tokens
  - `/tokenize` succeeded with `200`
  - request entered `completion_start`
- but `/v1/completions` did not produce a completion/HTTP result in a practical time window, and the run was manually terminated
- therefore treat `229376` as a **non-success boundary**, not a verified success

Practical reporting rule:
- when a long-context request reaches `completion_start` but never returns in a reasonable window, do **not** silently count it as success just because the server stayed healthy
- report it as:
  - first non-success / failure boundary
  - or "hang / no completion observed before manual termination"
- for this user's decision-making, that is enough to keep the safe answer at the highest fully completed real request (`196608`)

### Concrete batch=2 token-budget recommendations from the measured fp8 result
From the verified `ctx=196608` run:
- `prompt_tokens=196528`
- `max_tokens=16`
- verified real single-request total budget:
  - `196544`

Therefore for `batch=2` on this machine:
- exact 2-request baseline:
  - `2 × 196544 = 393088`
- **conservative recommendation**:
  - `max-num-batched-tokens = 384000`
- **aggressive ceiling**:
  - `max-num-batched-tokens ≈ 397184`
  - derived as `2 × (196544 + 2048)`

Interpretation:
- `384000` leaves modest scheduler/decode headroom below the empirically verified dual-request-equivalent budget
- `397184` is a practical ceiling, not a comfort setting
- do **not** keep increasing this just because the static KV-cache metrics look unchanged; the next coarse single-request point (`229376`) was already non-successful in real use

### New throughput-interpretation rule for ultra-long-context runs
When the user asks whether performance shows a **generation-throughput cliff** at long context, do **not** over-interpret runs that only generate a tiny completion (for example `max_tokens=14` or `16`).

What was learned on this machine:
- with very long prompts and only about `14` completion tokens, vLLM log lines such as:
  - `Avg generation throughput: 0.6 tok/s`
  - `Avg generation throughput: 0.5 tok/s`
  - `Avg generation throughput: 0.2–0.3 tok/s`
  are too noisy to diagnose a real decode cliff by themselves
- the more stable signal in those runs was the **end-to-end effective throughput** computed from:
  - `prompt_tokens / request_elapsed_s`
  or
  - `total_tokens / request_elapsed_s`
- on this machine, longer-context fp8 runs showed clear overall slowdown, but not enough evidence of a hard decode-only cliff from the tiny-completion runs alone

Practical decision rule:
- if completion length is tiny, analyze:
  1. end-to-end wall-clock throughput
  2. prompt/prefill cost growth
  3. request success/failure and health after request
- only claim a true **generation-throughput cliff** after running a dedicated decode test with:
  - fixed long prompt length(s)
  - materially larger `max_tokens` (for example `256` or `512`)
  - the same serving config across contexts

Recommended workflow for decode-cliff checks on this machine:
1. choose one working config first (for example Qwopus GPTQ + fp8)
2. choose several prompt-length checkpoints (for example `131k`, `163k`, `196k` if they are all real-request viable)
3. keep the prompt fixed per checkpoint
4. set `max_tokens=256` or `512`
5. compare:
   - vLLM's logged `Avg generation throughput`
   - end-to-end completion-token rate
   - total request latency
6. if only total latency worsens while decode rate stays roughly similar, treat the bottleneck as prefill/context scaling rather than a decode cliff

### Reusable decode-focused long-context benchmark workflow
When the user asks a question like:
- "正常 16k 以上输出的时候 output token rate 怎么样"
- "generation throughput 有没有断崖"
- "不要再用 14-token completion 看 decode 速度"

use a **decode-focused** benchmark rather than the max-real-context probe.

Reusable script created on this machine:
- `/home/qiushuo/reports/vllm-rocm-eval/bench_qwopus_fp8_decode_longctx.py`

What it does:
1. uses the working local Qwopus GPTQ + `fp8` path on the patched local vLLM checkout
2. tests a ladder of prompt contexts (currently `16384`, `65536`, `131072`, `163840`, `196608`)
3. sets `max_tokens=256` so decode time is materially represented
4. uses the server `/tokenize` endpoint to build the longest prompt satisfying:
   - `prompt_tokens + max_tokens <= max_model_len - safety_margin`
5. records for each prompt context:
   - `prompt_tokens`
   - `completion_tokens`
   - `effective_generation_tok_s`
   - `effective_total_tok_s`
   - logged `Avg generation throughput`
   - ready / after-request VRAM
6. runs one server at a time and cleans residual `api_server`, `VLLM::EngineCore`, and `resource_tracker` between points

Practical prep check discovered later:
- before launching the long cron run, verify the script constants are actually valid Python and not half-templated placeholders
- a real local copy of `bench_qwopus_fp8_decode_longctx.py` still had:
  - `MAX_TOKENS=***`
- it had to be patched to:
  - `MAX_TOKENS = 256`
- so for future reuse, do not assume the prepared benchmark script is launch-ready just because the filename already exists

Measured decode-heavy result from this machine (valid samples only):
- `ctx=16384`
  - request succeeded
  - `prompt_tokens=16064`
  - `completion_tokens=256`
  - `effective_generation_tok_s≈3.2537`
  - `effective_total_tok_s≈207.42`
  - logged `Avg generation throughput` stabilized around `4.3 tok/s`
  - ready / after-request VRAM: about `14.221 -> 14.278 GiB`
- `ctx=65536`
  - request succeeded
  - `prompt_tokens=65216`
  - `completion_tokens=256`
  - `effective_generation_tok_s≈0.8253`
  - `effective_total_tok_s≈211.07`
  - logged `Avg generation throughput` stabilized around `1.1-1.2 tok/s`
  - ready / after-request VRAM: about `14.225 -> 14.282 GiB`

Interpretation from those valid samples:
- decode speed already dropped sharply by `65k` vs `16k`
- roughly:
  - `3.25 tok/s -> 0.83 tok/s`
- but total end-to-end token throughput stayed near `~207-211 tok/s`, which means the long request remains heavily dominated by prompt/prefill work
- therefore on this machine, for fp8 long-context decode, the useful summary is:
  - **output token rate degrades sharply by 65k**
  - while **overall total throughput stays prompt-dominated**
- do not over-claim behavior beyond `65k` unless `131k+` also has valid decode samples

Important validation pitfall discovered while trying to accelerate the higher-context part:
- a simplified retry that skipped tokenizer-budgeted prompt construction and just sent a giant repeated-text prompt can produce a misleading result pattern:
  - HTTP status `200`
  - response body literally `null`
  - `prompt_tokens=null`
  - `completion_tokens=0`
  - service becomes unhealthy immediately after the request
- this happened on a `131072` retry on this machine
- treat that as **invalid / crashed request**, not as success
- equivalently: for long-context decode benchmarks, **HTTP 200 is not sufficient**
- require all of:
  1. parseable non-null JSON body
  2. non-null `usage.prompt_tokens`
  3. sane completion / finish fields
  4. post-request `/health` still alive

Important interpretation rule:
- for decode-cliff questions, prefer this script over the max-real-context probe because the latter keeps `max_tokens` tiny and is dominated by prefill
- if the 256-token run still shows weak signal, increase to `512` before drawing strong conclusions about decode cliffs
- if `131k+` runs start failing or returning malformed bodies, do **not** replace tokenizer-budgeted prompt sizing with naive character-count shortcuts unless you re-validate the full response schema and post-request health

### Paged-attention optimization rule on this exact ROCm/WSL setup
For this machine, first distinguish **scheduler/serving knobs** from **backend/kernel limits**.

Local source evidence:
- `/home/qiushuo/src/vllm/vllm/v1/attention/ops/chunked_prefill_paged_decode.py`
- warning around lines ~398-400:
  - `Cannot use ROCm custom paged attention kernel, falling back to Triton implementation.`

Interpretation:
- if this warning appears, the current run is **not** using the ROCm custom paged-attention kernel
- therefore, performance limits seen in long-context decode may be backend-related, not just bad scheduler settings

Knobs that are still worth trying without source/kernel changes:
- `--enable-chunked-prefill`
- `--max-num-batched-tokens`
- `--max-num-seqs`
- async scheduling
- prefix caching (only when prompts actually share reusable prefixes)
- optimization level `-O1/-O2/-O3`

What these knobs can and cannot do:
- they can improve admission, chunking, overlap, and end-to-end serving behavior
- they **cannot** by themselves turn the Triton fallback back into the ROCm custom paged-attention kernel
- if the real bottleneck is the backend fallback itself, meaningful improvement likely requires upstream/local source changes or a newer backend path, not just flag tuning

Useful docs evidence already verified on this machine:
- vLLM engine args docs state:
  - `--max-num-batched-tokens` = maximum tokens processed in a single iteration
  - `--enable-chunked-prefill` allows prefills to be chunked based on remaining `max_num_batched_tokens`

Practical decision rule:
- if the user asks whether paged attention can be optimized *right now*, answer in two layers:
  1. yes, serving knobs can still be tuned for scheduling/chunking behavior
  2. no, the ROCm paged-attention backend itself is not likely to improve materially without backend/kernel/source work if the run is already falling back to Triton

### New exact reason this Qwen/Qwopus 9B fp8 path misses ROCm custom paged attention
A later source-level check on this machine pinned down the exact gating conditions in:
- `/home/qiushuo/src/vllm/vllm/platforms/rocm.py`
- function: `use_rocm_custom_paged_attention(...)`

For the RDNA / `_ON_GFX1X` path, custom ROCm paged attention currently requires all of:
- `head_size == 128`
- `block_size == 16`
- `gqa_ratio` in `[3, 16]`
- `max_seq_len <= 128 * 1024`
- `alibi_slopes is None`
- `kv_cache_dtype == "auto"`
- `sinks is None`

For this exact local model:
- `/home/qiushuo/models/qwen/caiovicentino1-Qwopus3.5-9B-v3-HLWQ-v7-GPTQ`

local config inspection showed:
- `num_attention_heads = 16`
- `num_key_value_heads = 4`
- `gqa_ratio = 4`
- `head_dim = 256`

Practical interpretation on this machine:
- the model already misses the custom-kernel gate because `head_dim = 256`, not `128`
- the user's common long-context fp8 runs miss it again because:
  - `kv_cache_dtype = fp8`, not `auto`
  - `max_seq_len` is often `> 128k`
- therefore, for this exact Qwen/Qwopus 9B + fp8 + ultra-long-context path, failure to use ROCm custom paged attention is **structural**, not just a missing runtime flag

Decision rule:
- if the user wants the best chance of hitting the custom ROCm paged-attention kernel, test a separate control run with:
  - `kv_cache_dtype=auto`
  - `ctx <= 128k`
- but do **not** promise that the local Qwen/Qwopus 9B path can ever hit that kernel, because `head_dim=256` is already outside the current `_ON_GFX1X` custom-kernel gate

### Follow-up control run: 128k + auto still falls back on this machine
A later controlled verification on this machine explicitly checked whether lowering serving context to 128k changes the structural `head_dim` or is only enough to satisfy the `max_seq_len <= 128k` gate.

What was verified:
- model config remains:
  - `head_dim = 256`
  - `num_attention_heads = 16`
  - `num_key_value_heads = 4`
- and this did **not** change when evaluating hypothetical serve contexts like:
  - `32768`
  - `65536`
  - `131072`

Interpretation:
- lowering `--max-model-len` does **not** reduce head size
- it only affects runtime/backend gating conditions
- therefore `ctx=128k` can remove the `max_seq_len <= 128k` blocker, but cannot solve the structural `head_dim=256` mismatch

A later practical control script used:
- `--max-model-len 131072`
- `--kv-cache-dtype auto`
- once with baseline env
- once with `FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE`

Observed result on this machine:
- both runs reached full API readiness (`/health`, `/version`, `/v1/models` all `200`)
- both runs still emitted / matched the effective condition:
  - `Cannot use ROCm custom paged attention kernel, falling back to Triton implementation.`

Practical rule:
- `128k + auto` is a useful control run because it removes two easy blockers (`ctx > 128k` and non-`auto` KV cache)
- but for this exact Qwen/Qwopus 9B model it still does **not** unlock ROCm custom paged attention, because `head_dim=256` remains unchanged

### Updated flash-attention ROCm finding on this machine
A later runtime check verified that the local vLLM ROCm stack *can* look for a Triton-AMD FlashAttention route, and the practical state on this machine is now split into **two different gates**.

Relevant local source clue:
- `vllm.platforms.rocm.flash_attn_triton_available()` checks for:
  - Python package `flash_attn`
  - module `flash_attn.flash_attn_triton_amd`
  - env var `FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE`
- separately, `vllm/v1/attention/backends/fa_utils.py` on ROCm imports upstream `flash_attn.flash_attn_varlen_func`

Observed runtime result after installing ROCm flash-attention into `/home/qiushuo/.venvs/vllm-rocm-latest`:
- install command that succeeded:
  - `cd /home/qiushuo/src/flash-attention`
  - `MAX_JOBS=4 FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE /home/qiushuo/.venvs/vllm-rocm-latest/bin/pip install --no-build-isolation .`
- installed packages included:
  - `flash_attn-2.8.4`
  - `triton-3.5.1`
- `is_flash_attn_varlen_func_available()` became:
  - `True`
- but `flash_attn_triton_available()` still remained:
  - `False`
- even with `FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE`

Important reason:
- the installed package exposes upstream flash-attn functionality used by `fa_utils.py`
- but it still does **not** expose `flash_attn.flash_attn_triton_amd`, which is the module name the vLLM ROCm RDNA gate is checking for

Practical interpretation:
- ROCm upstream `flash_attn_varlen_func` is now available on this machine
- therefore TurboQuant prefill / continuation paths that rely on `_HAS_FLASH_ATTN` can potentially use flash-attn instead of SDPA fallback
- but the **special vLLM ROCm `flash_attn_triton_available()` gate is still not active**
- and this still does **not** imply that the current Qwen/Qwopus 9B path can use the custom ROCm paged-attention kernel, because that remains separately blocked by the model's `head_dim=256`

Decision rule:
- if the user asks whether flash-attn installation helped, answer:
  1. **yes** for upstream `flash_attn_varlen_func` availability used by TurboQuant attention paths
  2. **no** for the dedicated RDNA `flash_attn_triton_amd` vLLM gate, which is still unmet
  3. **no** for custom ROCm paged attention on this model, which remains structurally unavailable

### Important decoder/TurboQuant clarification discovered later
When the user asks to enable **FA2 + paged attention + TurboQuant simultaneously** on this exact ROCm Qwopus/Qwen3.5 setup, do **not** collapse all three into one backend gate.

What the local source showed:
- `vllm.platforms.rocm.use_rocm_custom_paged_attention(...)` is the gate for the **ROCm custom paged-attention kernel**.
- For this exact model on gfx1x/RDNA, it requires all of:
  - `head_size == 128`
  - `block_size == 16`
  - `gqa_ratio in [3, 16]`
  - `max_seq_len <= 128k`
  - `kv_cache_dtype == "auto"`
  - `alibi_slopes is None`
  - `sinks is None`
- This exact local Qwopus/Qwen3.5 GPTQ model still has:
  - `head_dim = 256`
- Therefore lowering `--max-model-len` to `128k` does **not** make custom ROCm paged attention available; it only removes the max-seq-len blocker, not the structural head-size blocker.

Separate and equally important local source clue:
- `vllm/v1/attention/backends/fa_utils.py` on ROCm imports **upstream** `flash_attn.flash_attn_varlen_func` directly.
- `is_flash_attn_varlen_func_available()` on ROCm is controlled by whether upstream `flash_attn` imported successfully; it is **separate** from the custom paged-attention gate above.
- `vllm/v1/attention/backends/turboquant_attn.py` uses `flash_attn_varlen_func(...)` in its prefill path and its large-continuation path when `_HAS_FLASH_ATTN` is true; otherwise it falls back to SDPA.

Practical interpretation for this machine:
- For this Qwopus/Qwen3.5 9B path, **ROCm custom paged attention is structurally unavailable** because `head_dim=256`.
- But **TurboQuant + FlashAttention** is still a meaningful target, because TQ prefill / continuation can use upstream ROCm flash-attn even when the custom paged-attention kernel is unavailable.
- Therefore the practical optimization target on this machine is:
  - get upstream ROCm `flash_attn` working
  - keep using the working patched TurboQuant backend
  - do not promise that this will also unlock the custom ROCm paged-attention kernel

### Practical install path for ROCm FlashAttention Triton AMD on this machine
When trying to improve Qwopus/Qwen3.5 TurboQuant attention performance on this machine, prefer the official ROCm flash-attention repo rather than ad-hoc packages:
- repo: `/home/qiushuo/src/flash-attention` (cloned from `https://github.com/ROCm/flash-attention`)

Useful upstream README facts verified during this session:
- ROCm flash-attention provides both CK and Triton backends.
- The Triton backend supports RDNA 3/4, fp16/bf16/fp32, arbitrary Q/KV sequence lengths and head sizes, MQA/GQA, and paged attention.
- The Triton backend kernels are provided by the `aiter` submodule and are installed during setup.

Recommended install command in the existing vLLM ROCm venv:
```bash
cd /home/qiushuo/src/flash-attention
MAX_JOBS=4 FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE \
  /home/qiushuo/.venvs/vllm-rocm-latest/bin/pip install --no-build-isolation .
```

Recommended verification after install:
```bash
source /home/qiushuo/.venvs/vllm-rocm-latest/bin/activate
python - <<'PY'
import importlib.util
mods = ['flash_attn', 'flash_attn.flash_attn_triton_amd']
for m in mods:
    print(m, bool(importlib.util.find_spec(m)))
PY
```

Decision rule:
- if `flash_attn` and `flash_attn.flash_attn_triton_amd` both import and `FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE` is set, then rerun a **small controlled TurboQuant startup/bench** to see whether TQ prefill now uses flash-attn-backed paths.
- do **not** interpret successful flash-attn installation as proof that the custom ROCm paged-attention kernel is now active for this model; those are separate mechanisms.

### New HF-aligned max-context rule for this exact Qwopus GPTQ model
If the user explicitly says to **follow the Hugging Face model page startup parameters** and only change context length plus TQ3 KV quantization, first distinguish between:
1. the **model-card example syntax** the user wants respected, and
2. the **minimum extra flags required to make this exact local GPTQ model runnable**.

For this exact model page (`caiovicentino1/Qwopus3.5-9B-v3-HLWQ-v7-GPTQ`), the visible recommended vLLM serve form is:
```bash
vllm serve caiovicentino1/Qwopus3.5-9B-v3-HLWQ-v7-GPTQ \
  --trust-remote-code --language-model-only \
  --max-model-len 16384
```

However, on this machine, that HF-minimal form is **not sufficient** for the local GPTQ checkpoint when TQ3 KV cache is enabled.

#### Important local finding: GPTQ requires explicit `--dtype float16`
A real rerun using the HF-style local equivalent plus only `--kv-cache-dtype turboquant_3bit_nc` and larger `--max-model-len` failed immediately during config validation with:
- `torch.bfloat16 is not supported for quantization method gptq. Supported dtypes: [torch.float16]`

So the **minimum necessary local correction** is:
```bash
vllm serve /home/qiushuo/models/qwen/caiovicentino1-Qwopus3.5-9B-v3-HLWQ-v7-GPTQ \
  --trust-remote-code \
  --language-model-only \
  --dtype float16 \
  --kv-cache-dtype turboquant_3bit_nc \
  --max-model-len <ctx> \
  --host 127.0.0.1 \
  --port <port>
```

Interpretation rule:
- `--host` / `--port` are operational necessities for local probing, not methodological deviations.
- `--dtype float16` is a **minimum GPTQ compatibility fix** on this machine, not an optional tuning flag.
- If the user asked for HF-page alignment, call out this deviation explicitly before treating the run as methodologically aligned.

#### Important follow-up finding: HF-minimal + `--dtype float16` still failed at 98k/131k here
A later rerun with the corrected HF-style command above still failed for:
- `ctx=98304`
- `ctx=131072`

Observed behavior:
- startup progressed much further than the pure HF-minimal failure
- model load and KV-cache metrics were produced (for example around `gpu_kv_cache_tokens=144160`, `available_kv_cache_gib=3.43`, `model_load_mem_gib=8.55`)
- but the server still exited before readiness with:
  - `RuntimeError: Engine core initialization failed. See root cause above. Failed core proc(s): {}`

Practical implication:
- do **not** treat failure of the HF-minimal route as proof that TQ3 itself is impossible on this machine
- it only proves that the HF-style minimal command is **not the right max-context probing baseline** for this exact local GPTQ deployment

#### For real max-context probing, prefer the prior working local qt3 parameter set
When the user’s goal is **"find the real maximum ctx"** rather than reproduce the model-card syntax, switch back to the previously validated local serving config:
```bash
python -m vllm.entrypoints.openai.api_server \
  --model /home/qiushuo/models/qwen/caiovicentino1-Qwopus3.5-9B-v3-HLWQ-v7-GPTQ \
  --quantization gptq \
  --dtype float16 \
  --language-model-only \
  --kv-cache-dtype turboquant_3bit_nc \
  --mamba-ssm-cache-dtype float16 \
  --gpu-memory-utilization 0.9 \
  --max-model-len <ctx> \
  --host 127.0.0.1 \
  --port <port>
```
with env:
```bash
HSA_ENABLE_DXG_DETECTION=1
ROC_ENABLE_PRE_VEGA=0
VLLM_TARGET_DEVICE=rocm
VLLM_USE_V1=1
```

This is the parameter family that previously produced real ready servers on this machine for the local Qwopus TQ3 path, including:
- `ctx=32768` → ready, `/health`/`/version`/`/v1/models` all `200`
- `ctx=65536` → ready, `/health`/`/version`/`/v1/models` all `200`

Representative metrics from those prior working runs:
- `gpu_kv_cache_tokens ≈ 187920`
- `available_kv_cache_gib ≈ 4.44–4.46`
- `model_load_mem_gib ≈ 8.55–8.57`
- ready-state VRAM usage ≈ `15.13–15.15 GiB`

Decision rule:
- use the HF-minimal route only when the user explicitly wants model-card-faithful methodology
- use the **prior working local qt3 params** when the user wants the best answer to "what is the actual maximum usable ctx on this machine?"

### New real-request validation rule for HF-aligned max-context probing
For this user's max-context question, startup-only readiness is not enough.

After `/health`, `/version`, and `/v1/models` succeed, send **one real long prompt** whose token budget is constructed to satisfy:
- `prompt_tokens + max_tokens <= max_model_len`
- with a small safety margin (for example 64 tokens)

Recommended pattern on this machine:
1. build a long filler prompt by tokenizing repeatedly with the real tokenizer
2. append a tiny deterministic suffix like `Reply with exactly: OK`
3. re-tokenize after the suffix is appended
4. keep `max_tokens` tiny (for example `16`) so the probe is mostly testing long-prefill viability, not generation length
5. treat the context as a success only if the request itself returns `200`

This avoids two false conclusions that happened repeatedly in earlier experiments:
- **false success**: server started, but a real long-prefill request still failed
- **false failure**: request exceeded token budget because the final prompt was not re-tokenized after suffix appends

Reusable script created for this:
- `/home/qiushuo/reports/vllm-rocm-eval/probe_tq3_ctx_usage.py`

What it does:
1. forcibly kills residual `VLLM::EngineCore`, `api_server`, and `resource_tracker` processes before each trial
2. starts the GPTQ model with:
   - `--kv-cache-dtype turboquant_3bit_nc`
   - `--mamba-ssm-cache-dtype float16`
   - `--gpu-memory-utilization 0.9`
3. probes a coarse set of candidate contexts first
4. if it finds a working ctx and a failing ctx, it binary-searches with a fine step to estimate the maximum verified ctx
5. records for each ctx:
   - startup success/failure
   - log-derived KV cache metrics
   - GPU memory via `torch.cuda.mem_get_info()`
   - system memory from `/proc/meminfo`
   - descendant process RSS totals

Important limitation discovered later:
- `probe_tq3_ctx_usage.py` verifies **deployment/startup readiness only**.
- It waits for `/health`, `/version`, and `/v1/models`, then samples memory/process state.
- It does **not** send a real long prompt after startup.
- Therefore its `verified_max_ctx` means:
  - the maximum context where the server can initialize and become healthy
  - **not** necessarily the maximum context that survives a real long-prefill request
- If the user asks for the maximum ctx for an actual long-context task, follow the startup probe with at least one real long request near the reported ceiling before treating that number as final.
- Also, if the user explicitly asks what is being tested, report these before launching:
  - model path
  - on-disk model size
  - exact startup command / key serve parameters
  This matched the user's expectation in later max-ctx experiments.

### New real-long-request rule for TQ3 on this machine
A later investigation proved that for this exact Qwopus GPTQ + patched-local-vLLM + `turboquant_3bit_nc` path, **startup success can be very misleading**.

What was observed:
- `ctx=65536`:
  - server became fully healthy (`/health`, `/version`, `/v1/models` all `200`)
  - `GPU KV cache size: 187,920 tokens`
  - `Available KV cache memory: 4.44 GiB`
  - but the first real long `/v1/completions` request returned `500`
- `ctx=98304`:
  - server also became fully healthy
  - `GPU KV cache size: 187,920 tokens`
  - `Available KV cache memory: 4.46 GiB`
  - first real long request again returned `500`
- log inspection showed the real root cause was not generic API failure, but:
  - `EngineCore encountered a fatal error`
  - `torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 512.00 MiB`
  - failure occurs inside:
    - `vllm/v1/attention/backends/turboquant_attn.py`
    - `_prefill_attention()` / `_continuation_prefill()`
    - `F.scaled_dot_product_attention(...)`

Important interpretation:
- the limiting factor for **real** long-context single-request use on this machine is not just static KV-cache capacity
- it is the **dynamic continuation-prefill working set** of TurboQuant attention on top of GPTQ weights
- therefore numbers like `gpu_kv_cache_tokens` and `available_kv_cache_gib` are only upper-bound hints, not proof that a real long prompt is viable

Practical decision rule:
- do **not** start real-longctx probing at `64k+` just because startup succeeded there
- once a startup-only probe shows readiness at high ctx, switch to a second probe that:
  1. starts from a much lower range (for example `8k → 16k → 24k → 28k → 32k`)
  2. sends an actual long prompt whose token budget is re-tokenized after suffix insertion
  3. captures the full HTTP status/body on failure, not just `HTTPError 500`
  4. records post-request `/health` / `/version` / `/v1/models`
  5. inspects the server log for `OutOfMemoryError`, especially `Tried to allocate 512.00 MiB`
  6. only then binary-searches between the highest real success and the first real failure

Reusable probe created for this purpose:
- `/home/qiushuo/reports/vllm-rocm-eval/probe_tq3_real_longctx_targeted.py`

Use that style of probe when the user asks:
- "真实长请求到多少"
- "真正可用的最大 ctx"
- or otherwise makes clear that startup-only readiness is not sufficient.

Output paths:
- log: `/home/qiushuo/reports/vllm-rocm-eval/results/tq3-ctx-probe/run.log`
- summary: `/home/qiushuo/reports/vllm-rocm-eval/results/tq3-ctx-probe/summary.json`

Use this workflow when the user wants:
- maximum working context for TQ3
- VRAM / RAM usage by context length
- a clean answer about context capacity instead of throughput or benchmark quality

## Confirmed baseline server-start findings from this machine
A later verified startup on this machine succeeded end-to-end despite the WSL `amdsmi` issue.

Confirmed facts from the successful run:
- process reached full API startup, not just model-load warmup
- server logged:
  - `Starting vLLM server on http://127.0.0.1:8012`
- HTTP checks succeeded:
  - `GET /health` -> `200`
  - `GET /version` -> `200`
  - `GET /v1/models` -> `200`

Important interpretation:
- the noisy `amdsmi` failure on WSL does **not** automatically mean startup will fail
- the real success criterion is API reachability, not absence of warnings

Important runtime warnings/quirks from the successful baseline startup:
- `Unknown vLLM environment variable detected: VLLM_USE_V1`
- engine config still reported `device_config=cuda` even on HIP/ROCm torch
- startup warned:
  - `Currently, the 4-bit gptq_gemm kernel for GPTQ is buggy. Please switch to gptq_marlin.`
- backend selection also logged:
  - `Found incompatible backend(s) [TURBOQUANT] with AttentionType.DECODER. Overriding with ROCM_ATTN ...`

Treat these as signs that the runtime path is usable but still not perfectly clean. For TurboQuant experiments, verify with logs and endpoint health; do not assume the requested backend was truly honored.

## Known GPTQ baseline datapoint on this machine
From a real ROCm in-process GPTQ smoke run on the local Qwopus model:
- tokenizer load: `2.609 s`
- model load: `21.418 s`
- VRAM after load: `8.148 GiB`
- VRAM peak load: `8.242 GiB`
- prompt tokens: `17`
- generation: `128` tokens in `19.962 s`
- generation throughput: `6.412 tok/s`
- VRAM peak during generation: `8.346 GiB`

Important qualitative finding:
- this path fell back to conservative `TorchQuantLinear`
- so it proves compatibility, but not best-case performance

## HumanEval workflow on this machine

### GPTQ baseline
Generate:
```bash
export HSA_ENABLE_DXG_DETECTION=1 ROC_ENABLE_PRE_VEGA=0
source /home/qiushuo/.venvs/qwopus-gptq/bin/activate
python /home/qiushuo/reports/vllm-rocm-eval/run_humaneval_gptq.py \
  --model /home/qiushuo/models/qwen/caiovicentino1-Qwopus3.5-9B-v3-HLWQ-v7-GPTQ \
  --output-dir /home/qiushuo/reports/vllm-rocm-eval/results/humaneval-gptq \
  --max-new-tokens 256
```

### vLLM server
Generate:
```bash
source /home/qiushuo/.venvs/vllm-rocm-latest/bin/activate
python /home/qiushuo/reports/vllm-rocm-eval/run_humaneval_openai_api.py \
  --base-url http://127.0.0.1:8000 \
  --model /home/qiushuo/models/qwen/caiovicentino1-Qwopus3.5-9B-v3-HLWQ-v7-GPTQ \
  --output-dir /home/qiushuo/reports/vllm-rocm-eval/results/humaneval-vllm \
  --max-tokens 256
```

### Important follow-up: generation is not enough — score pass@1 explicitly
The existing HumanEval scripts only generate `predictions.jsonl` plus timing metadata. They do **not** compute functional correctness by themselves.

On this machine, the practical scoring flow is:
```bash
source /home/qiushuo/.venvs/vllm-rocm-latest/bin/activate
python -m pip install human-eval
python /home/qiushuo/reports/vllm-rocm-eval/evaluate_humaneval_passk.py \
  --predictions /home/qiushuo/reports/vllm-rocm-eval/results/humaneval-vllm/predictions.jsonl \
  --output /home/qiushuo/reports/vllm-rocm-eval/results/humaneval-vllm/pass_at_1.json \
  --timeout 5.0 \
  --workers 4 \
  --ignore-incomplete
```

Helper script created for this exact purpose:
- `/home/qiushuo/reports/vllm-rocm-eval/evaluate_humaneval_passk.py`

Practical note:
- For quick iteration, run a subset first (for example `--limit 20`) to compare KV-cache configs cheaply.
- For final numbers, rerun full HumanEval once the serving config is stable.

### Important HumanEval pitfall discovered on this machine
The current helper script:
- `/home/qiushuo/reports/vllm-rocm-eval/run_humaneval_openai_api.py`

currently appends:
- `Complete the following Python function. Return only the function body completion without markdown fences.`

to each HumanEval prompt, while also using stop sequences including:
- `\ndef`

On this Qwopus/Qwen3.5 GPTQ model served by vLLM, that combination can produce **false-empty completions**:
- the model often starts by re-emitting a function signature like `def ...`
- the `\ndef` stop sequence fires immediately
- the raw completion becomes just `"\n"`
- after `clean_completion(...).rstrip()`, the saved completion becomes `""`
- downstream HumanEval scoring then reports many/all failures even though the model is generating normally

This was reproduced directly against the local vLLM endpoint:
- with the helper script's current prompt+stop logic, the API returned `finish_reason='stop'`, `stop_reason='\\ndef'`, `completion_tokens=2`, and effectively empty output
- removing the stop sequences allowed the model to continue generating
- using the **original HumanEval prompt without the extra suffix** produced a normal function-body continuation even with stop sequences enabled

Practical rule for this machine:
- if HumanEval results show many examples with `completion_tokens ~= 2` and empty completions, do **not** conclude the model or TQ3 path is broken
- first inspect `predictions.jsonl` and the raw completion response to see whether `\ndef` is truncating generations immediately

Recommended fix path:
1. prefer the original HumanEval prompt without the extra instructional suffix
2. do not assume empty completions mean the model is broken; first inspect whether `stop_reason='\\ndef'` is truncating a regenerated function signature
3. add a cleanup step in `clean_completion()` that truncates at the first dedented non-blank line so top-level tails like `if __name__ == ...`, `import doctest`, and `print(...)` do not contaminate the function body
4. validate the fix on a small subset before trusting pass@1 numbers, then rerun a larger subset or full HumanEval
5. only compare KV-cache configs after confirming completions are non-empty and structurally valid Python

Implemented fix on this machine:
- patched `/home/qiushuo/reports/vllm-rocm-eval/run_humaneval_openai_api.py`
- patched `/home/qiushuo/reports/vllm-rocm-eval/run_humaneval_gptq.py`
- set `PROMPT_SUFFIX = ""`
- kept `clean_completion()` but extended it to stop at the first dedented non-blank line

Observed result after the fix on this exact setup:
- config: patched local vLLM + TQ3 KV cache + `max_model_len=65536`
- model: `/home/qiushuo/models/qwen/caiovicentino1-Qwopus3.5-9B-v3-HLWQ-v7-GPTQ`
- HumanEval subset: 50 tasks
- pass@1 improved from the bogus empty-output result (`0.0` on the earlier broken 10-task run) to a real measured `0.78` on 50 tasks
- average completion throughput during the 50-task retest was about `54.588 tok/s`

## PinchBench on this machine

### Key finding
PinchBench is **not** a plain API benchmark. It runs the model as an **OpenClaw agent**. A local OpenAI-compatible vLLM endpoint works, but only through OpenClaw.

### Practical setup that worked here
OpenClaw can be installed directly on this WSL box:
```bash
npm install -g openclaw@latest
openclaw --version
openclaw agents list
```

This machine ended up with:
- `openclaw` available on PATH
- default `main` agent present

### Minimal local PinchBench run against vLLM
After the vLLM server is healthy on `http://127.0.0.1:PORT`, run:
```bash
export OPENAI_API_KEY=EMPTY
cd /tmp/pinchbench-skill
uv run scripts/benchmark.py \
  --model '/home/qiushuo/models/qwen/caiovicentino1-Qwopus3.5-9B-v3-HLWQ-v7-GPTQ' \
  --base-url 'http://127.0.0.1:PORT/v1' \
  --suite 'task_sanity,task_weather' \
  --no-upload \
  --output-dir /tmp/pinchbench-results
```

Why this matters:
- `--base-url` is the crucial option; it bypasses OpenRouter validation and builds a custom OpenAI-compatible provider in the temporary OpenClaw agent config.
- `--model` should match the model ID returned by the local vLLM `/v1/models` endpoint. On this machine that ID is the full local model path.

### Recommended local benchmarking strategy
For local model comparisons on this machine:
1. start with `task_sanity,task_weather` to verify end-to-end plumbing
2. if stable, try `--suite automated-only` before any judge-heavy suite
3. use `--no-upload` for local experiments

### Important PinchBench caveats
- PinchBench requires the `openclaw` CLI; without it, the benchmark cannot execute tasks.
- Judge-less / automated tasks are much easier to run locally than suites requiring external LLM judging.
- PinchBench writes JSON summaries to the chosen `--output-dir`; parse the newest JSON file for aggregate scores and efficiency stats.

### Helper orchestration script created in this session
A reusable matrix runner was created to automate server startup, token-rate benchmark, HumanEval subset generation/scoring, and a small PinchBench suite for KV-cache comparisons:
- `/home/qiushuo/reports/vllm-rocm-eval/run_kv_quant_matrix.py`

Current baked-in configs in that helper:
- `bfloat16` KV cache
- `fp8` KV cache
- `turboquant_3bit_nc` KV cache

Treat it as the current quick-comparison harness for this machine; extend it rather than rewriting ad hoc benchmark glue each time.

## Practical pitfalls
- Do not benchmark fairly with other resident model servers running; free the GPU first.
- On this WSL ROCm path, always export:
  - `HSA_ENABLE_DXG_DETECTION=1`
  - `ROC_ENABLE_PRE_VEGA=0`
- If editable install fails in weird ways, retry with:
  - ROCm torch already installed in target venv
  - `--no-build-isolation`
  - correct setuptools range
- For this model, `--language-model-only` is the right default for text benchmarks.
- If memory is tight, lower context first and prefer `--mamba-ssm-cache-dtype float16`.
- On this machine, treat `process` notifications about earlier failed startups carefully; confirm the currently active server process separately with `/health` and `/v1/models` before concluding the server is down.
- Failed or interrupted vLLM launches can leave behind an orphaned `VLLM::EngineCore` plus a Python `multiprocessing.resource_tracker` process even after the API server port is closed. That orphan can keep `/dev/dxg` open and consume most GPU memory, causing the next launch to fail with errors like `Free memory on device cuda:0 ... is less than desired GPU memory utilization`.
- When a restart unexpectedly fails for memory reasons, explicitly check for leftovers before retrying:
  - `ps -ef | grep -E 'vllm|api_server|VLLM::EngineCore' | grep -v grep`
  - `lsof /dev/dxg` (or `fuser -v /dev/dxg`)
  - `ss -ltnp | grep -E ':(PORT)'`
  - if only `VLLM::EngineCore` remains, kill that orphan and re-check `torch.cuda.mem_get_info()` before starting the next benchmark
- If a local patch changed `rocm.py` from `logger.warning_once(...)` to `logger.warning(...)`, remember that the repo is no longer pristine upstream; future debugging should account for local source diffs first.
- Do **not** include your own harness script name in broad cleanup/kill regexes. On this machine, both `probe_tq3_ctx_usage.py` and `run_tq3_64k_128k_eval.py` hit self-termination patterns where cleanup logic matched the orchestration script itself, leading to empty logs or exit code `143` with no summary output.
- For cleanup, prefer matching only actual serving/runtime processes:
  - `VLLM::EngineCore`
  - `python -m vllm`
  - `api_server`
  - `resource_tracker`
  and exclude the benchmark wrapper/harness script.
- **Important new lesson for Qwopus GPTQ batch-2 testing:** a healthy startup is not enough. On this machine, `turboquant_3bit_nc` and `turboquant_k3v4_nc` both reached full API readiness at `ctx=16384` and `ctx=8192` (`/health`, `/version`, `/v1/models` all 200), but both failed real batch-2 validation when two long prompts were actually sent concurrently. The failures were fatal `EngineCore` crashes in the `gptq_gemm` path with `torch.OutOfMemoryError`, so do **not** treat startup-only probes as evidence that batch=2 is stable.
- For this exact Qwopus GPTQ model, the practical bottleneck in batch-2 tests is not just KV-cache capacity; GPTQ prefill working-set growth during dual long-prefill requests can still OOM after startup. In the observed run:
  - `turboquant_3bit_nc`
    - `ctx=16384`: startup OK, batch-2 validation failed; `torch.OutOfMemoryError`, tried to allocate about `762 MiB`
    - `ctx=8192`: startup OK, batch-2 validation still failed; both requests returned `500`, then service became unhealthy
  - `turboquant_k3v4_nc`
    - `ctx=16384`: startup OK, batch-2 validation failed; `torch.OutOfMemoryError`, tried to allocate about `762 MiB`
    - `ctx=8192`: startup OK, batch-2 validation still failed; both requests returned `500`, then service died
  - `turboquant_4bit_nc`
    - `ctx=32768`: failed even earlier during engine init / profile run with `torch.OutOfMemoryError`, tried to allocate about `192 MiB` with only about `531 MiB` free
- Therefore, when the user asks for **max stable context at batch=2**, the answer may legitimately be **no stable context found** even if multiple startup-only contexts look healthy.

### New 128k TQ4 startup-fix finding on this machine
A later targeted startup-debugging pass showed that `turboquant_4bit_nc` at:
- `--max-model-len 131072`
- single-request mode (`--max-num-seqs 1`)

is **not** inherently impossible on this RX 9070 16GB WSL/ROCm machine.

What was verified:
- the earlier failing launch used:
  - `--max-num-batched-tokens 131072`
- that run failed before readiness during startup/profile/compile with:
  - `torch._inductor.exc.InductorError`
  - `Failed to run autotuning code block`
  - `CUDA out of memory`
  - attempted extra allocation about `6.00 GiB`
- model weight load itself had already succeeded (`model_load_mem_gib ≈ 7.51`), so the failure was **not** checkpoint loading; it was startup-stage compile/autotune peak VRAM.

A follow-up single-variable scan kept:
- `--max-model-len 131072`
- `--kv-cache-dtype turboquant_4bit_nc`
- `--max-num-seqs 1`

and only changed `--max-num-batched-tokens`.

Observed results:
- `--max-num-batched-tokens 16384`
  - **startup succeeded**
  - `gpu_kv_cache_tokens ≈ 130,944`
  - `available_kv_cache_gib ≈ 4.10`
  - `engine_init_s ≈ 185.29`
- `--max-num-batched-tokens 32768`
  - **startup succeeded**
  - `gpu_kv_cache_tokens ≈ 63,360`
  - `available_kv_cache_gib ≈ 2.01`
  - `engine_init_s ≈ 188.16`
- `--max-num-batched-tokens 65536`
  - **startup failed**
  - `torch.OutOfMemoryError`
  - attempted extra allocation about `3.00 GiB` with only about `3.19 GiB` free

Practical interpretation:
- for this exact TQ4 128k path, `max-num-batched-tokens` is not a minor scheduler tweak; it materially changes startup/profile/compile peak VRAM
- the 128k TQ4 startup problem can be mitigated by **lowering `max-num-batched-tokens` aggressively**
- the best currently verified default on this machine is:
  - `--max-num-batched-tokens 16384`
- `32768` is a weaker fallback that can still start but leaves much less KV headroom
- `65536` and above are currently unsafe on this machine for TQ4 @ 128k

Practical decision rule:
- if the goal is simply to get `turboquant_4bit_nc` to **start** at `128k`, first try:
  - `--max-num-seqs 1`
  - `--max-num-batched-tokens 16384`
- if that works and the next goal is real long-request benchmarking, keep the low `max-num-batched-tokens` while validating request stability and token rate
- do **not** assume the earlier TQ4 128k failure means the config is impossible; it was specifically a startup-stage peak-VRAM failure caused by too-large batched-token settings

### New tq4 single-request long-context estimation rule
When the user switches from **batch-2 continuous batching** to **"what is the longest context for tq4 on a real long prompt"**, do **not** reuse the batch-2 conclusion as the final bound.

What was learned on this machine:
- `turboquant_4bit_nc` previously showed:
  - `ctx=32768`: startup failed during engine init with `torch.OutOfMemoryError`, trying to allocate about `192 MiB`, with only about `531 MiB` free
  - `ctx=16384`: startup succeeded, but real batch-2 long-prefill validation crashed later with `torch.OutOfMemoryError`, trying to allocate about `1.49 GiB`
- That means **batch-2 failure does not directly bound single-request maximum context**.
- For single-request long-context estimation, the better starting point is the successful `qt4 @ 16384` startup metrics:
  - `GPU KV cache size: 64,416 tokens`
  - `Available KV cache memory: 2.03 GiB`

Practical estimation heuristic from those numbers:
- if you reserve roughly `4k` tokens of dynamic headroom, the KV-only upper bound is about **60,320** tokens
- if you reserve roughly `8k` tokens of dynamic headroom, the KV-only upper bound is about **56,224** tokens
- because GPTQ prefill working-set growth is real on this machine, treat the **practical single-request test window** for tq4 as roughly:
  - **49k → 57k → 61k** on success path
  - or **49k → 41k/53k/55k** on fallback path

Recommended workflow for tq4 single-request longest-context probing:
1. clear all residual `api_server`, `VLLM::EngineCore`, and `resource_tracker` processes
2. use **one server** with:
   - `--kv-cache-dtype turboquant_4bit_nc`
   - `--max-num-seqs 1`
   - `--max-num-batched-tokens = ctx`
3. build a **real long prompt**, not a short smoke prompt
4. append a tiny verification suffix and require a deterministic short answer, so the request proves real long-prefill behavior rather than just startup
5. always re-tokenize after appending the suffix and verify:
   - `prompt_tokens + max_tokens <= max_model_len`
6. probe a narrow band around the estimate first instead of brute-forcing many points

New hard negative result from a dedicated 128k tq4 decode attempt on this machine:
- script used:
  - `/home/qiushuo/reports/vllm-rocm-eval/bench_qwopus_tq4_128k_decode.py`
- config used:
  - `--kv-cache-dtype turboquant_4bit_nc`
  - `--max-model-len 131072`
  - `--max-num-seqs 1`
  - `--max-num-batched-tokens 131072`
  - `max_tokens=256`
- result:
  - `server_ready = false`
  - server exited before any real request was sent
  - no token-rate sample was produced
- important log evidence:
  - model load itself succeeded (`Model loading took 7.51 GiB memory and 15.15 s`)
  - failure happened later during startup/profile/compile
  - `torch._inductor.exc.InductorError: RuntimeError: Failed to run autotuning code block: CUDA out of memory. Tried to allocate 6.00 GiB`
  - at failure time the log reported only about `5.31 GiB` free

Interpretation update:
- on this machine, **tq4 @ 128k is currently blocked already at startup/profile/autotune**, not merely at real long-request time
- this means the immediate bottleneck is not just static KV-cache capacity
- it is also the peak compile/autotune working set for the `(1, 131072)` range under this backend

Practical decision rule:
- do **not** expect `turboquant_4bit_nc` to be a credible 128k long-output configuration on this RX 9070 16GB setup without changing the startup regime
- if the user still wants to try to rescue 128k tq4, the first knob to try is **lowering `--max-num-batched-tokens` while keeping `--max-model-len=131072`**, because the failed run used `max_num_batched_tokens = ctx` and the compile/profile shape likely contributed materially to the 6 GiB autotune spike
- otherwise, treat tq4 as a **mid-context** candidate on this machine and prefer `fp8` or `turboquant_3bit_nc` for truly long-context work

## Reusable batch-2 KV-matrix workflow discovered on this machine
When the user asks to compare multiple TurboQuant KV-cache dtypes for the local Qwopus GPTQ model under **real batch=2** conditions, reuse:
- `/home/qiushuo/reports/vllm-rocm-eval/run_qwopus_kv_matrix_b2.py`

What this harness does:
1. creates a fresh timestamped result dir under `results/qwopus-kv-matrix-b2-<timestamp>/`
2. cleans residual `VLLM::EngineCore` / `api_server` / `resource_tracker` processes between probes
3. launches one config at a time with:
   - `--max-num-seqs 2`
   - `--max-num-batched-tokens = 2 * ctx`
4. waits for all of:
   - `/health`
   - `/version`
   - `/v1/models`
5. performs **real** batch-2 validation by sending two concurrent `/v1/completions` requests, each with prompt length approximately `ctx - 64` tokens and `max_tokens=16`
6. records per-probe:
   - exact serve command
   - startup health
   - GPU / RAM snapshots
   - key log data (`GPU KV cache size`, `Available KV cache memory`, `Maximum concurrency`, `Application startup complete`)
   - batch-2 validation results in `batch2_validation.json`
7. if a config ever truly passes, runs token-rate benchmark and HumanEval subset scoring; otherwise records the negative result without pretending startup was enough

Practical interpretation:
- This harness is the right tool when the user explicitly cares about **stable dual-concurrency**, not just single-request serving or startup viability.
- If `config_summary.json` shows `max_stable_ctx: null` with startup-success probes underneath, that means the startup looked healthy but **real batch-2 validation disproved stability**.

## Stepwise real-benchmark workflow for this user
When the user asks for **"each step finished, report progress"** or explicitly wants a **real test rather than just startup/load**, do **not** hide the run inside one long black-box script that does 64k + 128k + HumanEval end-to-end.

Use this phased workflow instead:
1. clean residual vLLM/EngineCore/resource_tracker processes and confirm GPU memory is free enough
2. start one server for a single context length (for example 64k on one port)
3. wait for all three checks to pass:
   - `/health`
   - `/version`
   - `/v1/models`
4. immediately capture runtime evidence before running benchmarks:
   - GPU memory via `torch.cuda.mem_get_info()`
   - system memory from `/proc/meminfo`
   - process snapshot / RSS for API server, EngineCore, resource_tracker
   - key log lines like `GPU KV cache size`, `Maximum concurrency`, `Application startup complete`
5. run token-rate benchmark separately and save `token_rate.json`
6. report progress to the user
7. run HumanEval subset separately and then score pass@1 separately
8. report progress again
9. only then stop the server, clean up, and move to the next context length (for example 128k)

Why this matters on this machine:
- long wrapper scripts can self-kill or terminate with `143` and leave little/no useful top-level logging
- per-step execution makes it obvious whether failure happened during startup, throughput, or HumanEval generation/scoring
- it matches this user's explicit preference for incremental status updates

## Matrix-comparison workflow for this user's KV-cache experiments
When the user wants a **head-to-head KV-cache comparison** (for example `turboquant_3bit_nc` vs `turboquant_k3v4_nc` vs `turboquant_4bit_nc`) rather than a single-config validation, use this exact structure:

1. Test configs **sequentially**, not interleaved.
   - Preferred order on this machine:
     1. `turboquant_3bit_nc`
     2. `turboquant_k3v4_nc`
     3. `turboquant_4bit_nc`
   - Clean up residual `api_server`, `VLLM::EngineCore`, and `resource_tracker` between configs.

2. For **each config**, find the **maximum stable context at batch size 2** before running any quality benchmark.
   - Do not treat "batch 2" as implied by a config flag alone.
   - If applicable, set `--max-num-seqs=2`, but also run **two real concurrent requests** to verify the config is truly stable at that context.
   - Use stepwise probing plus binary search/coarse-to-fine refinement rather than testing only `64k` and `128k`.

3. Only after the max stable `batch=2` context is identified, run **HumanEval with `--limit 20`** on that same serving config.
   - Use the already patched HumanEval helper scripts (empty `PROMPT_SUFFIX`, improved `clean_completion()`), not any stale copy.
   - Generate predictions first, then run explicit pass@1 scoring.

4. **Report after every config**, not only at the end.
   - This user explicitly prefers incremental updates once one config finishes.
   - Each per-config report should include:
     - KV dtype
     - exact serve command
     - max stable ctx at batch 2
     - whether real dual-concurrency verification succeeded
     - HumanEval-20 pass@1
     - token rate
     - VRAM / RAM snapshot
     - key warnings / backend fallbacks / failure points

5. After all configs finish, send one final cross-config recommendation.
   - Explicitly identify which config is best on this RX 9070 WSL ROCm machine for the user's goals.
   - If one config has better context but worse HumanEval, call out that trade-off directly.

This matrix workflow is the preferred comparison method for this user; it is better than a single black-box harness that runs everything silently and reports only once.

### Continuous batching batch=2 definition for this user
When the user says **"continuous batching batch2"**, use this exact interpretation on this machine:
- start one vLLM server instance for the target config
- set `--max-num-seqs 2`
- set `--max-num-batched-tokens` high enough for the real two-request token budget at that target context
- send **two independent concurrent requests** to the same `/v1/completions` or equivalent endpoint
- let **vLLM** do the scheduling / continuous batching internally
- do **not** reinterpret this as a client-side static batch payload unless the user explicitly asks for that

Important practical rule:
- for large-context comparisons (for example `32k / 64k / 128k`), do **not** leave a stale small-token cap like an `8k`-scale `--max-num-batched-tokens` in place
- scale `--max-num-batched-tokens` to the actual test regime, otherwise the experiment is not a valid continuous-batching `batch=2` comparison

### Fixed-context comparison workflow discovered in this session
If the user narrows scope from "find the maximum context" to **"test only a few fixed context points"**, switch to a simpler matrix workflow instead of binary search:
- launch/test each KV config sequentially
- for each config, test the requested fixed contexts exactly (for example `32k`, `64k`, `128k`)
- at each context, record:
  - exact serve command
  - `kv-cache-dtype`
  - `max_model_len`
  - `max-num-seqs`
  - `max-num-batched-tokens`
  - `/health`, `/version`, `/v1/models`
  - whether real dual-concurrency succeeded
  - GPU/RAM snapshots and key log lines
- if at least one context works, run HumanEval on the **largest successful fixed context** for that config
- report after **each config**, then send a final comparison summary

### New fixed-point result: upstream Qwopus batch-2 matrix at 32k / 64k / 128k all failed on this machine
A later rerun followed the user's stricter fixed-point continuous-batching workflow exactly:
- model: `/home/qiushuo/models/qwen/caiovicentino1-Qwopus3.5-9B-v3-HLWQ-v7-GPTQ`
- local patched vLLM source: `/home/qiushuo/src/vllm`
- configs tested in order:
  1. `turboquant_3bit_nc`
  2. `turboquant_k3v4_nc`
  3. `turboquant_4bit_nc`
- fixed contexts only:
  - `32768`
  - `65536`
  - `131072`
- continuous batching definition enforced literally:
  - `--max-num-seqs 2`
  - `--max-num-batched-tokens = 2 * ctx`
  - two independent concurrent requests to one vLLM server
- prompt budget was explicitly guarded so this was **not** the old `400 Bad Request` token-budget mistake

Practical outcome on this machine:
- **none of the three configs reached a passing fixed point**
- because no config had even one successful fixed context, **HumanEval-20 was skipped for all three**

Per-config result:
- `turboquant_3bit_nc`
  - `32k`: failed before readiness during engine/profile init with `torch.OutOfMemoryError`, tried to allocate about `192 MiB`, only about `532 MiB` free
  - `64k`: failed before readiness during compile/init; no healthy API
  - `128k`: failed before readiness with `torch._inductor.exc.InductorError: OutOfMemoryError`, tried to allocate about `2.00 GiB`, only about `2.40 GiB` free
- `turboquant_k3v4_nc`
  - `32k / 64k / 128k`: all failed before real batch-2 validation; observed unstable exits like `rc=-9` / `rc=-15`, with no healthy `/health` + `/version` + `/v1/models` window to validate
- `turboquant_4bit_nc`
  - `32k / 64k / 128k`: all failed before readiness as well; some runs loaded weights and enabled async scheduling, but still died during later init/compile rather than reaching usable API state

Important interpretation:
- prior single-request or startup-only successes do **not** transfer to this stricter fixed-point batch-2 matrix
- for these exact fixed points on this RX 9070 16GB WSL/ROCm machine, the practical answer is currently:
  - `qt3`: no working point
  - `k3v4`: no working point
  - `qt4`: no working point
- if the user asks specifically for this exact fixed-point matrix again, the expected answer is now **all fail, no HumanEval stage** unless the local runtime/build situation changes materially

Artifact directory from the fixed-point rerun:
- `/home/qiushuo/reports/vllm-rocm-eval/results/qwopus-kv-matrix-continuous-b2-20260419-075706`
- report:
  - `/home/qiushuo/reports/vllm-rocm-eval/results/qwopus-kv-matrix-continuous-b2-20260419-075706/REPORT.md`

### Probe-script token-budget pitfall discovered in this session
A `400 Bad Request` during concurrency probing does **not** automatically mean the model/config failed.

A real failure seen on this machine came from the probe script constructing prompts too close to `max_model_len`:
- server `max_model_len = 1024`
- request asked for `32` output tokens
- prompt tokenization drifted to about `993` input tokens
- total exceeded the context limit (`1025 > 1024`)
- vLLM raised `VLLMValidationError` and returned `400 Bad Request`

Practical rule:
- after **all** prompt edits/appends (including request-specific suffixes), re-tokenize and verify:
  - `prompt_tokens + max_tokens <= max_model_len`
- keep a real safety margin
- do not count this class of 400 as evidence that continuous batching or the KV-cache config failed

## PinchBench / tool-using benchmarks on RX 9070 16GB (learned 2026-04)

When using the Qwopus-TQ4 vLLM server (`~/scripts/qwopus-tq4/start-server.sh`,
port 8533) for **PinchBench** or any multi-turn tool-calling benchmark, two
things are mandatory or you will waste a 90-min run:

### 1. Tool-call parser is required
HumanEval works fine without it. PinchBench does not — the first task
(`task_sanity`) returns 400 and scores 0.0/1.0, triggering fail-fast. Add
to the vLLM serve command:

```
--enable-auto-tool-choice --tool-call-parser hermes
```

These flags are now baked into `start-server.sh`. Do not remove them.

### 2. 131k ctx + max-num-seqs >= 4 OOMs during PinchBench
HumanEval (short single-turn) survives on 131072 ctx, but PinchBench's
multi-turn tool transcripts grow KV cache fast and hit
`torch.OutOfMemoryError: Tried to allocate 384 MiB ... 710 MiB free` on
the very first task. EngineCore dies but the benchmark runner keeps
hammering 400s and reports all 0s for the remaining ~25 tasks. You will
not notice unless you tail the server log.

Working defaults for **PinchBench on this 16GB card**:

```
CTX=65536       # was 131072
SEQS=2          # was 4 (or 1)
GMU=0.9         # leave headroom; do not push to 0.95
```

PinchBench single sessions never need >64k context. `max-num-seqs=2`
keeps batching benefit while halving activation peak.

### 3. PinchBench runner does not detect server death
Always `tail -f` the vLLM server log in parallel, or grep run.log for
consecutive `0.0/1.0` scores at the same ~95s interval — that is the
classic "server is dead, runner is firing into a void" signature.
Watch patterns to set on the runner: `EngineDeadError`, `OutOfMemory`,
`FAIL FAST`, `Aborting`.

## Good final summary format for future runs
Report each tested config with:
1. exact serve command
2. whether server started successfully
3. prompt tokens / completion tokens
4. completion tok/s
5. VRAM after load / peak VRAM
6. HumanEval pass@1
7. notes on failures, fallback kernels, or obvious quality regressions
