---
name: rx9070-agustin-llamacpp-turboquant
description: Use the patched AgustinJimenez/llama.cpp TurboQuant fork on this user's WSL + ROCm + RX 9070 machine for mad-lab-ai TQ3_1S GGUFs, turbo4 KV, validated launchers, and short-context HumanEval code-eval.
---

# RX 9070 Agustin llama.cpp TurboQuant workflow

Use this on **this user's WSL + ROCm + AMD Radeon RX 9070 16GB** machine when you need a llama.cpp fork that can actually run **mad-lab-ai TQ3_1S GGUFs** with **turbo4 KV**.

## When to use
- Need a fork that supports **TQ3_1S/TQ4_1S source-level types** and **turbo4 KV**
- Need to load local mad-lab-ai GGUFs such as:
  - `/home/qiushuo/models/qwen/mad-lab-ai-Qwen3.6-35B-A3B-tq-gguf/qwen3.6-35b-a3b-instruct-TQ3_1S.gguf`
  - `/home/qiushuo/models/gemma/mad-lab-ai-google-gemma-4-26b-a4b-tq3_1s/google-gemma-4-26b-a4b-tq3_1s.gguf`
- Existing forks either load the model but lack turbo4, or support turbo4 but fail on mad-lab-ai type-45 tensors

## Proven repo/build on this machine
- Repo: `/home/qiushuo/src/llama.cpp-agustin-turboquant`
- Build dir: `/home/qiushuo/src/llama.cpp-agustin-turboquant/build-gfx1201`
- Binary: `/home/qiushuo/src/llama.cpp-agustin-turboquant/build-gfx1201/bin/llama-server`
- GPU target: `gfx1201`

## Critical patch for mad-lab-ai TQ3_1S GGUFs
Without this patch, the fork misreads mad-lab-ai files because the file says `general.file_type == 43` (TQ3_1S) while raw tensor type `45` would normally map to `TQ4_1S` in this fork.

### Patch location
- File: `/home/qiushuo/src/llama.cpp-agustin-turboquant/ggml/src/gguf.cpp`

### Patch behavior
1. Read `general.file_type` early via literal key `"general.file_type"`
2. During tensor type parsing:
   - if `general.file_type == 43`
   - and raw tensor type is `45`
   - reinterpret as `GGML_TYPE_TQ3_1S`

This is a narrow compatibility shim for mad-lab-ai TQ3_1S files, not a global remap.

## Build notes
Configure/build was already proven on this machine with ROCm/HIP. If rebuilding:

```bash
cd /home/qiushuo/src/llama.cpp-agustin-turboquant
HIPCXX="$(hipconfig -l)/clang" HIP_PATH="$(hipconfig -R)" cmake -S . -B build-gfx1201 \
  -DGGML_HIP=ON \
  -DGGML_HIP_GRAPHS=ON \
  -DGGML_FLASH_ATTN=ON \
  -DAMDGPU_TARGETS=gfx1201 \
  -DCMAKE_BUILD_TYPE=Release

cmake --build /home/qiushuo/src/llama.cpp-agustin-turboquant/build-gfx1201 \
  --config Release --target llama-server -j 8
```

## Quick validation checklist
Always validate in this order:
1. binary loads model
2. `/health` returns 200
3. `/v1/models` returns model list
4. real request succeeds
5. confirm KV type in logs (`K (turbo4)` / `V (turbo4)`)

## Proven Qwen deployment
### Long-context stable config
This is the practical 128k deployment for the Qwen TQ3_1S model:

- Model: `/home/qiushuo/models/qwen/mad-lab-ai-Qwen3.6-35B-A3B-tq-gguf/qwen3.6-35b-a3b-instruct-TQ3_1S.gguf`
- Args:
  - `-ngl 35`
  - `-c 131072`
  - `-np 1`
  - `-fa on`
  - `--fit-target 128`
  - `-ctk turbo4 -ctv turbo4`
- Launcher scripts:
  - `~/.local/bin/start-agustin-qwen36-tq3-turbo4-128k`
  - `~/.local/bin/stop-agustin-qwen36-tq3-turbo4-128k`
  - `~/.local/bin/status-agustin-qwen36-tq3-turbo4-128k`
- Default port: `8248`

### Performance reality for Qwen
- `128k` works, but throughput is limited
- Measured at 128k (`ngl35`): roughly `2.8 tok/s` prompt and `3.6 tok/s` decode on a tiny request
- Best short-context speed tested originally was around `ngl37`, `ctx=8192`, but a later MoE/offload sweep found a better configuration than the plain `ngl35/37` baseline.
- Current best validated short-context config for `Qwen3.6-35B-A3B TQ3_1S` on this machine is:
  - `-ngl 999`
  - `-ot "blk\.(0|1|...|31)\.=ROCm0,exps=CPU"`
  - `-b 2048 -ub 2048`
  - `--no-mmap`
  - `--kv-unified`
  - `-fa on`
- That 32-block override + `--kv-unified` reached about `3.76 tok/s` generation in server logs on a short Chinese prompt, versus about `2.76 tok/s` for the earlier `-ngl 35` baseline.
- `--n-cpu-moe` is **not** the same as this override pattern and should not be treated as a throughput optimization on this RX 9070 setup; it is mainly a VRAM-saving / runability knob here.
- Do **not** promise `128k + 20 tok/s` on this machine for Qwen 35B TQ3_1S

### Offload ceiling for Qwen
- `ngl37` short-context works
- `ngl38` fails during compute buffer allocation
- For 128k, use `ngl35` or `ngl34`

## Proven Gemma deployment
### Speed-first config
- Model: `/home/qiushuo/models/gemma/mad-lab-ai-google-gemma-4-26b-a4b-tq3_1s/google-gemma-4-26b-a4b-tq3_1s.gguf`
- Args:
  - `-ngl 32`
  - `-c 8192`
  - `-np 1`
  - `-fa on`
  - `--fit-target 128`
  - `-ctk turbo4 -ctv turbo4`
- Launcher: `~/.local/bin/start-agustin-gemma26-tq3-turbo4-speed`
- Default port: `8249`
- Measured roughly `25.1 tok/s` prompt and `14.8 tok/s` decode on code-style short requests

### Long-context config
- Args:
  - `-ngl 24`
  - `-c 131072`
  - `-np 1`
  - `-fa on`
  - `--fit-target 128`
  - `-ctk turbo4 -ctv turbo4`
- Launcher: `~/.local/bin/start-agustin-gemma26-tq3-turbo4-128k`
- Default port: `8250`
- Measured around `2.27 tok/s` prompt and `2.01 tok/s` decode at 128k

### Shared Gemma control scripts
- `~/.local/bin/stop-agustin-gemma26-tq3-turbo4`
- `~/.local/bin/status-agustin-gemma26-tq3-turbo4`

## Important HumanEval / code-eval lesson
For the Gemma TQ3_1S line, **code evaluation must use `/v1/completions`, not `/v1/chat/completions`**.

### Why
- `/v1/completions` gives clean code continuation, e.g. it correctly continues:
  ```python
  def add(a, b):
      """Return sum."""
  ```
  with:
  ```python
      return a + b
  ```
- `/v1/chat/completions` produced template garbage or unrelated content on this setup, even after trying `--chat-template gemma`

### Code-eval launcher
Use the dedicated launcher for Gemma code tasks:
- `~/.local/bin/start-agustin-gemma26-tq3-turbo4-codeeval`
- `~/.local/bin/stop-agustin-gemma26-tq3-turbo4-codeeval`
- `~/.local/bin/status-agustin-gemma26-tq3-turbo4-codeeval`
- Default port: `8251`

This launcher intentionally keeps `--skip-chat-parsing` and is meant for `/v1/completions` only.

## HumanEval guidance on this machine
Use `lm_eval` with `local-completions` against `/v1/completions`.

### Recommended model args pattern
```bash
HF_ALLOW_CODE_EVAL=1 /home/qiushuo/.venvs/vllm-rocm-latest/bin/python -m lm_eval run \
  --model local-completions \
  --model_args model=<GGUF_PATH> base_url=http://127.0.0.1:<PORT>/v1/completions tokenizer_backend=none tokenized_requests=False num_concurrent=1 max_retries=1 \
  --tasks humaneval \
  --batch_size 1 \
  --confirm_run_unsafe_code \
  --limit 20 \
  --log_samples \
  --output_path <OUTDIR>
```

### CoT guidance
For HumanEval on this setup, **do not enable CoT/reasoning**.
- Goal is executable code completion, not explanation
- CoT is likely to pollute outputs with text/reasoning and hurt `pass@1`
- Best practice here is: pure function prefix + direct code continuation + stop sequences

## Measured HumanEval smoke results
### Qwen TQ3_1S (short smoke)
- Route: `/v1/completions`
- `limit=10`
- `pass@1 = 0.8`
- Result root: `/home/qiushuo/reports/vllm-rocm-eval/results/qwen36-tq3-turbo4-humaneval-short`

### Gemma TQ3_1S
- First quick smoke (`limit=10`): `pass@1 = 0.4`
- After clean code-eval routing (`limit=20`): `pass@1 = 0.45`
- Result root: `/home/qiushuo/reports/vllm-rocm-eval/results/gemma26-tq3-turbo4-humaneval-codeeval-short`

Interpretation: Gemma formatting was improved, but the main limiter is still model capability; Qwen remains the stronger code model here.

## Pitfalls
- Keep only one llama.cpp server running at a time on this machine
- For Gemma code evaluation, do **not** judge the model by `/v1/chat/completions`
- Do not enable CoT for HumanEval here
- For Qwen 35B, long context is viable but throughput stays low
- For GPU state checks, prefer `/home/qiushuo/.venvs/vllm-rocm-latest/bin/python` with `torch.cuda.mem_get_info()`
