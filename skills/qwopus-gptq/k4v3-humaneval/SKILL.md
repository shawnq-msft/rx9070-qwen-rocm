---
name: rx9070-qwopus-k4v3-plugin-humaneval
description: Validate the legacy TurboQuant exact K4/V3 plugin path on this user's WSL + ROCm + RX 9070 for Qwopus GPTQ, find max ctx at batch=2, and run HumanEval through the local OpenAI-compatible API.
version: 1.0.0
author: Hermes Agent
license: MIT
---

# RX 9070 Qwopus GPTQ exact K4/V3 plugin + HumanEval

Use this when the user wants to test the **reference / legacy TurboQuant plugin exact K4/V3** route (not stock `--kv-cache-dtype`) on this machine:
- WSL Ubuntu
- AMD RX 9070 16GB
- ROCm 7.2
- local vLLM checkout at `/home/qiushuo/src/vllm`
- model `/home/qiushuo/models/qwen/caiovicentino1-Qwopus3.5-9B-v3-HLWQ-v7-GPTQ`

This skill exists because the workable path on this machine was found only after trying and rejecting several alternatives.

## Main findings to remember

1. **Do not treat native TQ4 and exact K4/V3 as the same path.**
   - Native vLLM `turboquant_4bit_nc` at 128k was not practically deployable on this 16GB ROCm machine.
   - The reference exact K4/V3 route worked via the installed legacy plugin / monkey-patch environment variables.

2. **On this machine, exact K4/V3 worked at `ctx=131072`, `batch=2`, with:**
   - `--max-num-seqs 2`
   - `--max-num-batched-tokens 8192`
   - `--enforce-eager`
   - env vars:
     - `TQ_KV_K_BITS=4`
     - `TQ_KV_V_BITS=3`
     - `TQ_KV_NORM_CORRECTION=1`
     - `TQ_KV_BOUNDARY_LAYERS=5`

3. **The reference plugin's CUDA kernels do not build cleanly on ROCm here**, but serving can still work.
   Expected warnings/errors in logs:
   - `unknown argument: '--use_fast_math'`
   - `unknown argument: '-gencode=arch=compute_75,code=sm_75'`
   These did **not** mean the test failed. They indicated fallback behavior.

4. **Backend expectation:** this route is usable but not a true native TurboQuant fast path on this ROCm setup.
   Logs may also show:
   - `Cannot use ROCm custom paged attention kernel`
   - backend staying on ROCm/Triton attention rather than TURBOQUANT.

5. **For HumanEval with lm-eval on this machine, use the local OpenAI-compatible API path, not lm-eval's in-process `--model vllm`.**
   The stable route was:
   - start local vLLM API server separately
   - run `lm_eval` with `--model local-completions`
   - use `tokenizer_backend=huggingface`
   - point tokenizer to the local model path

6. **Do not use lm-eval `tokenizer_backend=remote` unless `/tokenizer_info` exists.**
   On this server it returned 404, so `remote` tokenizer failed.

7. **HumanEval requires unsafe code opt-in.**
   Set:
   - `HF_ALLOW_CODE_EVAL=1`
   and pass:
   - `--confirm_run_unsafe_code`

8. **lm-eval API mode also needed `tenacity` installed** in this venv.

## Known-good serving command

Run from anywhere after activating the venv:

```bash
source /home/qiushuo/.venvs/vllm-rocm-latest/bin/activate

HSA_ENABLE_DXG_DETECTION=1 \
ROC_ENABLE_PRE_VEGA=0 \
VLLM_TARGET_DEVICE=rocm \
PYTHONUNBUFFERED=1 \
FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE \
TQ_KV_K_BITS=4 \
TQ_KV_V_BITS=3 \
TQ_KV_NORM_CORRECTION=1 \
TQ_KV_BOUNDARY_LAYERS=5 \
python -m vllm.entrypoints.openai.api_server \
  --model /home/qiushuo/models/qwen/caiovicentino1-Qwopus3.5-9B-v3-HLWQ-v7-GPTQ \
  --quantization gptq \
  --dtype float16 \
  --language-model-only \
  --gpu-memory-utilization 0.9 \
  --max-model-len 131072 \
  --max-num-seqs 2 \
  --max-num-batched-tokens 8192 \
  --enforce-eager \
  --host 127.0.0.1 \
  --port 8200
```

## Health checks

Verify before evaluation:

```bash
python - <<'PY'
import urllib.request
for path in ['/health', '/v1/models']:
    with urllib.request.urlopen('http://127.0.0.1:8200' + path, timeout=20) as r:
        print(path, r.status)
        print(r.read().decode('utf-8', 'replace')[:300])
PY
```

Check live VRAM:

```bash
source /home/qiushuo/.venvs/vllm-rocm-latest/bin/activate && python - <<'PY'
import torch
free, total = torch.cuda.mem_get_info()
print({'free_gib': round(free/1024**3, 3), 'used_gib': round((total-free)/1024**3, 3), 'total_gib': round(total/1024**3, 3)})
PY
```

## Max-ctx validation method

Use a real-serving probe, not startup-only success.

### Required success criteria
A test point counts as pass only if all are true:
1. server becomes ready
2. `/v1/completions` returns HTTP 200
3. both concurrent requests succeed for `batch=2`
4. prompts are actually near target context (tokenized length close to ctx budget)
5. completions are non-empty

### Search strategy
- If already proven good up to some ctx, use **binary search** toward target instead of linear stepping.
- On this machine, a binary search from `32768` to `131072` immediately showed `131072` passed.

### Known-good result on this machine
- `last_good_ctx = 131072`
- `first_bad_ctx = null`
- `batch = 2`
- each request used roughly `130943` prompt tokens + `64` completion tokens in the successful 128k case

## HumanEval workflow

### One-time setup
```bash
source /home/qiushuo/.venvs/vllm-rocm-latest/bin/activate
python -m pip install -U 'lm-eval[vllm]'
python -m pip install tenacity
```

### Smoke test first
Use this before the full run:

```bash
source /home/qiushuo/.venvs/vllm-rocm-latest/bin/activate
HF_ALLOW_CODE_EVAL=1 python -m lm_eval run \
  --model local-completions \
  --model_args \
    model=/home/qiushuo/models/qwen/caiovicentino1-Qwopus3.5-9B-v3-HLWQ-v7-GPTQ \
    base_url=http://127.0.0.1:8200/v1/completions \
    tokenizer_backend=huggingface \
    tokenizer=/home/qiushuo/models/qwen/caiovicentino1-Qwopus3.5-9B-v3-HLWQ-v7-GPTQ \
    num_concurrent=2 \
    max_retries=1 \
    tokenized_requests=False \
  --tasks humaneval \
  --batch_size 1 \
  --confirm_run_unsafe_code \
  --limit 2 \
  --log_samples \
  --output_path /home/qiushuo/reports/vllm-rocm-eval/results/qwopus-plugin-k4v3-humaneval-smoke
```

On this machine, the smoke test succeeded with:
- `pass@1 = 1.0` on `limit=2`

### Full run
```bash
source /home/qiushuo/.venvs/vllm-rocm-latest/bin/activate
HF_ALLOW_CODE_EVAL=1 python -m lm_eval run \
  --model local-completions \
  --model_args \
    model=/home/qiushuo/models/qwen/caiovicentino1-Qwopus3.5-9B-v3-HLWQ-v7-GPTQ \
    base_url=http://127.0.0.1:8200/v1/completions \
    tokenizer_backend=huggingface \
    tokenizer=/home/qiushuo/models/qwen/caiovicentino1-Qwopus3.5-9B-v3-HLWQ-v7-GPTQ \
    num_concurrent=2 \
    max_retries=1 \
    tokenized_requests=False \
  --tasks humaneval \
  --batch_size 1 \
  --confirm_run_unsafe_code \
  --log_samples \
  --output_path /home/qiushuo/reports/vllm-rocm-eval/results/qwopus-plugin-k4v3-humaneval-full
```

## Why these settings matter

- `tokenizer_backend=huggingface`:
  avoids the missing `/tokenizer_info` endpoint problem
- `tokenized_requests=False`:
  avoids relying on remote token ids support for API mode
- `num_concurrent=2`:
  matches the deployment target (`batch=2`) better than strictly serial eval
- `--enforce-eager`:
  reduces graph-related surprises and matched the successful exact K4/V3 deployment
- `--max-num-batched-tokens 8192`:
  was the stable budget for the successful 128k batch=2 deployment

## Failure patterns and interpretation

### If server log shows ROCm compile errors from plugin CUDA sources
Do **not** immediately mark the run failed. Check whether:
- `/health` is 200
- requests still complete
- outputs are real

On this machine those compile errors were compatible with a working fallback route.

### If lm-eval fails with missing tenacity
Install it:
```bash
python -m pip install tenacity
```

### If lm-eval fails with RemoteTokenizer 404 on `/tokenizer_info`
Switch to:
- `tokenizer_backend=huggingface`
- `tokenizer=<local model path>`

### If HumanEval import raises the unsafe code warning
Set:
```bash
export HF_ALLOW_CODE_EVAL=1
```
and pass:
```bash
--confirm_run_unsafe_code
```

## Reporting guidance

When reporting results, distinguish clearly between:
1. **deployability**: server starts and handles real requests
2. **max context**: largest real successful context under target batch
3. **quality**: HumanEval pass@1
4. **backend reality**: whether this is a true ROCm-native fast path or a fallback-compatible path

On this machine, the correct framing is:
- exact K4/V3 plugin route is **deployable and real-request verified** at `128k / batch=2`
- but it appears to be running with **fallback behavior**, not ideal native TurboQuant ROCm kernels
- therefore it is a **useful engineering path**, but benchmark/perf claims should be conservative

## Artifacts to keep

Write reports under:
- `/home/qiushuo/reports/vllm-rocm-eval/results/`

Useful prior artifact names:
- `qwopus-plugin-k4v3-batch2-bisect-128k`
- `qwopus-plugin-k4v3-humaneval-smoke`
- `qwopus-plugin-k4v3-humaneval-full`
