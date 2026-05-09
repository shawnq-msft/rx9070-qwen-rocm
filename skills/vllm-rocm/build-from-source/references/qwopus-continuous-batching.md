# Qwopus continuous batching on RX 9070 WSL vLLM

## Key lesson
For the local Qwopus GPTQ model on this machine, do **not** set:
- `max-num-batched-tokens = ctx * batch`

That earlier batch=2 harness design caused startup/compile failures across all tested contexts and should be treated as a **harness-design failure**, not proof that continuous batching is impossible.

## Why
Earlier failed b2 matrix used:
- `ctx=32768/65536/131072`
- `max-num-seqs=2`
- `max-num-batched-tokens=65536/131072/262144`

Outcome:
- all candidates failed before a meaningful multi-request throughput run
- no trustworthy batching conclusion could be drawn

Interpretation:
- on this patched local vLLM checkout, startup and compile behavior is highly sensitive to MBT
- scaling MBT linearly with context is a bad heuristic here

## Better methodology
Anchor from the best known single-request TQ4 config:
- `--kv-cache-dtype turboquant_4bit_nc`
- `--max-model-len 131072`
- `--max-num-batched-tokens 16384`
- `--max-num-seqs 1`

Observed on this machine:
- smoke avg completion throughput: about `58.432 tok/s`
- GPU KV cache size: about `134112` tokens
- available KV cache: about `4.19 GiB`
- logged maximum concurrency for `131072` tokens/request: about `3.97x`
- logs included `Chunked prefill is enabled` and `Asynchronous scheduling is enabled`

For target concurrency like **b4**:
1. keep the proven `ctx=131072` TQ4 baseline
2. set `--max-num-seqs` to the target concurrency (`4` for b4)
3. sweep **moderate** `--max-num-batched-tokens` values around the proven single-request point
4. send **true concurrent requests to one server**
5. judge both throughput and latency, not just fit

## Recommended first-pass b4 grid
Use:
- `ctx=131072`
- `max-num-seqs=4`
- sweep `max-num-batched-tokens` over:
  - `4096`
  - `8192`
  - `12288`
  - `16384`
  - `24576`

Reasoning:
- `16384` is the known good anchor from the best `seq1` 128k TQ4 run
- `8192` and `12288` test whether lower prefill budgets help tail latency or stability
- `24576` tests whether a modest increase helps throughput without jumping to unrealistic `ctx * batch`
- `4096` checks whether a smaller compile/prefill budget materially improves stability

## Benchmark shape
Recommended evaluation:
- one warmup request
- at least 5 rounds
- 4 simultaneous requests per round for b4
- use short realistic prompts for throughput tuning

Success criteria:
- server healthy after startup
- warmup succeeds
- all concurrent rounds succeed
- server still healthy after the benchmark

Metrics to report:
- aggregate completion tok/s
- aggregate total tok/s
- mean / median / p95 latency
- GPU used GiB at ready
- KV cache tokens
- available KV cache GiB
- compile / init time
- post-run health endpoints

## Relevant local scripts
- `/home/qiushuo/reports/vllm-rocm-eval/run_qwopus_kv_matrix_continuous_b2_fixed.py`
- `/home/qiushuo/reports/vllm-rocm-eval/run_qwopus_continuous_b4_sweep.py`
