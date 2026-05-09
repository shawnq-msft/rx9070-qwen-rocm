---
name: rx9070-gptq-qwopus-testing
description: Test GPTQ Qwen/Qwopus-style models on this user's WSL + ROCm + RX 9070 setup, determine whether they truly run, measure real VRAM usage, and check whether long context (especially 64k/128k) is actually viable.
---

# RX 9070 WSL GPTQ Qwopus Testing Workflow

Use this when the user asks whether a **GPTQ** model (especially Qwopus/Qwen-family repos with `quantize_config.json`) can run on their local:
- WSL Ubuntu
- ROCm via `/dev/dxg`
- AMD Radeon RX 9070 16GB
- PyTorch/transformers/gptqmodel stack

This is specifically for cases where the model is **not GGUF / llama.cpp**, and the question is whether the GPTQ repo itself is usable on this machine and whether long context is realistic.

## Core lessons from the Qwopus3.5-9B-v3-HLWQ-v7-GPTQ test

Empirical result on this user's machine for:
- repo: `caiovicentino1/Qwopus3.5-9B-v3-HLWQ-v7-GPTQ`
- local path: `/home/qiushuo/models/qwen/caiovicentino1-Qwopus3.5-9B-v3-HLWQ-v7-GPTQ`

Findings:
- The model **does load and generate successfully** with:
  - `torch 2.11.0+rocm7.2`
  - `transformers 5.5.4`
  - `gptqmodel 6.0.3`
  - GPU visible as `AMD Radeon RX 9070`
- Basic smoke test succeeded:
  - tokenizer load about `2.77 s`
  - model load about `58.93 s`
  - VRAM after load about `8.148 GiB`
  - VRAM peak on load about `8.242 GiB`
  - one short generation took about `30.25 s`
- However, long-context viability was poor on this stack:
  - `64k` prompt test failed with GPU OOM
  - `128k` prompt test failed with GPU OOM
- Therefore, on this exact machine, treat this GPTQ model as:
  - **loadable / runnable**
  - **not a viable 64k/128k long-context candidate**

Important comparison lesson:
- Do **not** assume a 9B GPTQ model that fits at load time will match the long-context behavior of the user's proven `GGUF + llama.cpp` setups.
- On this machine, `GGUF + llama.cpp` is materially stronger for long-context than `GPTQ + transformers/gptqmodel`.

## Environment preparation

Preferred test venv on this machine:
- `/home/qiushuo/.venvs/qwopus-gptq`

Known-good installs for this venv:
```bash
python3 -m venv /home/qiushuo/.venvs/qwopus-gptq
source /home/qiushuo/.venvs/qwopus-gptq/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install --index-url https://download.pytorch.org/whl/rocm7.2 torch torchvision torchaudio
python -m pip install transformers accelerate sentencepiece gptqmodel
```

Critical environment variables for this WSL ROCm path:
```bash
export HSA_ENABLE_DXG_DETECTION=1
export ROC_ENABLE_PRE_VEGA=0
export PYTORCH_ALLOC_CONF='expandable_segments:True,max_split_size_mb:256,garbage_collection_threshold:0.7'
```

## Preconditions before testing

1. Free the GPU first.
- Stop any resident `llama-server` processes before GPTQ testing.
- On this user's machine, ports `8004` and `8006` are often occupied by long-running local servers.

Check:
```bash
ps -C llama-server -o pid=,etimes=,cmd=
ss -ltnp | grep -E ':8004|:8006|:8010|:8011|:8012' || true
```

2. Verify GPU visibility in the target venv.
```bash
export HSA_ENABLE_DXG_DETECTION=1 ROC_ENABLE_PRE_VEGA=0
source /home/qiushuo/.venvs/qwopus-gptq/bin/activate
python - <<'PY'
import torch
print('cuda_available', torch.cuda.is_available())
if torch.cuda.is_available():
    print('device', torch.cuda.get_device_name(0))
    p = torch.cuda.get_device_properties(0)
    print('total_memory_gib', round(p.total_memory/1024**3, 2))
PY
```

## Smoke test method

Do not start with a server. Start with an in-process smoke test that proves:
- tokenizer loads
- model loads
- one real generation succeeds
- VRAM usage is captured

Recommended script pattern:
```python
import json, time, traceback, torch
from gptqmodel import GPTQModel
from transformers import AutoTokenizer

model_path = "/path/to/model"
result = {}

torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats(0)

start = time.time()
tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
result["tokenizer_load_s"] = round(time.time() - start, 2)

start = time.time()
model = GPTQModel.from_quantized(
    model_path,
    trust_remote_code=True,
    device='cuda:0',
    backend='auto',
)
result["model_load_s"] = round(time.time() - start, 2)
result["vram_after_load_gib"] = round(torch.cuda.memory_allocated(0)/1024**3, 3)
result["vram_peak_load_gib"] = round(torch.cuda.max_memory_allocated(0)/1024**3, 3)

prompt = "请用一句中文介绍你自己。"
inputs = tok(prompt, return_tensors='pt')
inputs = {k: v.to('cuda:0') for k, v in inputs.items()}

start = time.time()
out_ids = model.generate(**inputs, max_new_tokens=32)
result["generate_s"] = round(time.time() - start, 2)
result["output_preview"] = tok.decode(out_ids[0], skip_special_tokens=True)[:500]
```

## Long-context testing method

Important: for this stack, measure context by **actual tokenized prompt length**, not just `max_position_embeddings` from config.

### Token budgeting pitfall
A repeated string like `"你好 " * N` may tokenize into about **2x as many tokens** as expected. Always inspect:
```python
seq_len = int(inputs['input_ids'].shape[1])
```

### Recommended context ladder
Test in this order:
- `16k`
- `32k`
- `64k`
- `128k`

But do not waste time once the trend is obvious:
- if `64k` OOMs badly, `128k` is not a real candidate
- still run one `128k` confirmation test if the user explicitly asked

### Capture per test
For each step, save a JSON file with:
- `target_ctx`
- `prompt_tokens`
- `success`
- `error` if any
- `vram_after_load_gib`
- `vram_peak_load_gib`
- `vram_on_error_gib` and `vram_peak_error_gib` on failure
- generation time and tokens/sec if success

Recommended filename pattern:
- `/home/qiushuo/qwopus_gptq_smoke.json`
- `/home/qiushuo/qwopus_gptq_ctx_64k.json`
- `/home/qiushuo/qwopus_gptq_ctx_128k.json`

## What failure looks like on this machine

### 64k failure example
Observed for the Qwopus GPTQ model:
- prompt tokens: `64000`
- VRAM after load: `8.148 GiB`
- peak during attempt: `12.645 GiB`
- error:
  - `torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 3.91 GiB`
  - GPU had only about `2.78 GiB` free at the time

### 128k failure example
Observed for the Qwopus GPTQ model:
- VRAM after load: `8.148 GiB`
- peak during attempt: `13.787 GiB`
- error:
  - `torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 3.66 GiB`
  - GPU had only about `1.81 GiB` free at the time

Interpretation:
- weight fit does **not** imply long-context fit
- prefill/intermediate activations can dominate memory on this stack
- on this hardware, this GPTQ route is not suitable for 64k/128k with this model

## Thinking / reasoning caveat

The user prefers thinking enabled and separated when possible. On this GPTQ stack:
- there is no tested, native equivalent of the user's llama.cpp `--reasoning on --reasoning-format deepseek` workflow
- do not promise clean `reasoning_content` separation
- if the user asks for "thinking on", explain that current GPTQ smoke tests validate raw generation, not the same structured reasoning API shape as llama.cpp

## Token-rate measurement guidance

This stack does not automatically return llama.cpp-style `timings.prompt_per_second` / `predicted_per_second` fields.

If generation succeeds, record rough throughput manually:
```python
completion_tokens = out_ids.shape[1] - inputs['input_ids'].shape[1]
dt = time.time() - start
gen_tps = completion_tokens / dt
```

Do not compare this too literally to llama.cpp server metrics; it is a rough in-process number.

## "Turbo quant" / fast-backend probing on this machine

The user may ask whether enabling a faster GPTQ backend (loosely described as a "turbo quant" technique) can improve the maximum working context.

Important finding on this machine:
- there is **no simple `turbo` flag** exposed by `gptqmodel` for this setup
- the practical interpretation is to probe the faster GPTQ backends directly, especially:
  - `gptq_marlin`
  - `gptq_triton`
  - `gptq_hf_kernel`

Empirical result on this RX 9070 + ROCm setup for `caiovicentino1/Qwopus3.5-9B-v3-HLWQ-v7-GPTQ`:
- `gptq_marlin` failed before usable inference with:
  - `No module named 'gptqmodel_marlin_kernels'`
  - result file: `/home/qiushuo/qwopus_gptq_backend_marlin_smoke.json`
- `gptq_triton` failed with explicit ROCm incompatibility:
  - `TritonV2QuantLinear does not support device: DEVICE.ROCM`
  - result file: `/home/qiushuo/qwopus_gptq_backend_triton_smoke.json`
- `gptq_hf_kernel` also failed with explicit ROCm incompatibility:
  - `HFKernelLinear does not support device: DEVICE.ROCM`
  - result file: `/home/qiushuo/qwopus_gptq_backend_hf_kernel_smoke.json`

Practical conclusion:
- on this machine, the hoped-for "turbo" GPTQ backends are **not actually available for ROCm use**
- the test falls back to the more conservative torch-based path (`TorchQuantLinear`)
- therefore, trying to "turn on turbo quant" did **not** unlock a higher max context on this setup
- when the user asks whether a faster GPTQ kernel might save long-context viability, the answer here is effectively **no on current ROCm support**

## Startup/stop/status scripts

The user asked for helper scripts. Important lesson:
- `gptqmodel` on this machine did **not** expose a ready-to-use `python -m gptqmodel serve ...` entrypoint
- `python -m gptqmodel` failed because there is no `gptqmodel.__main__`

Therefore, if you create scripts, label them honestly:
- they can be loader/keeper scripts
- they are **not** OpenAI-compatible API server scripts unless you separately build a proper server around them

Current helper scripts created on this machine:
- `~/.local/bin/start-qwopus35-gptq`
- `~/.local/bin/stop-qwopus35-gptq`
- `~/.local/bin/status-qwopus35-gptq`

These are currently load-and-hold helpers, not a finished inference API.

## How to summarize findings to the user

Use this structure:
1. Can it load at all?
2. Does one real generation work?
3. What is the real VRAM at load?
4. What happens at 64k?
5. What happens at 128k?
6. Is it a serious long-context candidate on this machine?
7. Compare it to the user's existing GGUF/llama.cpp winners

For this Qwopus GPTQ test, the correct summary is:
- **Yes, it runs.**
- **No, it is not a 128k candidate on this RX 9070 16GB setup.**
- **64k also OOMed in this tested stack.**
- For long-context Hermes use, the user should prefer the already-proven `GGUF + llama.cpp` path.
