# Qwopus GPTQ long-context and util-matrix notes on RX 9070

## Current vLLM CLI reality on this machine
- The current local vLLM CLI does **not** accept legacy `--swap-space`.
- Attempting to pass `--swap-space 12` fails immediately with:
  - `api_server.py: error: unrecognized arguments: --swap-space`
- Relevant currently exposed knobs include:
  - `--cpu-offload-gb`
  - `--kv-offloading-size`
  - `--kv-offloading-backend`
  - related offload settings
- Treat them as different mechanisms, not guaranteed equivalents of old swap-space behavior.

## Qwopus GPTQ 384k attempt
Model:
- `/home/qiushuo/models/qwen/caiovicentino1-Qwopus3.5-9B-v3-HLWQ-v7-GPTQ`

Important model limit:
- config still advertises `max_position_embeddings = 262144`

What is required to even try 384k:
- `VLLM_ALLOW_LONG_MAX_MODEL_LEN=1`
- otherwise `--max-model-len 393216` is rejected during config validation

Tested launch pattern that could start:
- `VLLM_ALLOW_LONG_MAX_MODEL_LEN=1`
- `--max-model-len 393216`
- `--cpu-offload-gb 12`
- `--gpu-memory-utilization 0.88`
- `--max-num-batched-tokens 1024`
- `--max-num-seqs 1`
- `--enforce-eager`

Observed startup facts:
- server started successfully
- `Available KV cache memory: 8.53 GiB`
- `GPU KV cache size: 272,448 tokens`
- `Maximum concurrency for 393,216 tokens per request: 2.75x`

Observed execution facts:
- short request (~1k prompt, 128 output) succeeded but was extremely slow
  - measured completion throughput was only about `0.575 tok/s`
- a ~65k-token prompt failed at execution with HTTP 500
- engine log showed TurboQuant attention continuation-prefill OOM:
  - `torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 78.00 MiB`
- after that engine failure, a larger >262k prompt test could not proceed because the server was dead

Practical takeaway:
- On this stack, `ctx 384k` can be made to **start** only by overriding model-length safeguards.
- That does **not** make it a usable deployment.
- For this Qwopus GPTQ path on RX 9070 16GB, treat 384k as non-practical unless the runtime stack changes materially.

## Realistic smoke util-matrix finding at 131072 ctx
Settings family:
- `--max-model-len 131072`
- `--max-num-seqs 1`
- eager mode
- realistic short/mid/long smoke requests

Observed matrix summary:
- `gpu-memory-utilization=0.88`, `mbt=1024` -> pass
- `gpu-memory-utilization=0.88`, `mbt=2048` -> pass
- `gpu-memory-utilization=0.92`, `mbt=1024` -> long failed
- `gpu-memory-utilization=0.92`, `mbt=2048` -> long failed
- `gpu-memory-utilization=0.96`, `mbt=1024` -> mid failed
- `gpu-memory-utilization=0.96`, `mbt=2048` -> mid failed in the earlier direct smoke run

Interpretation:
- `0.88` is the current real-world sweet spot on this machine.
- Higher utilization inflates static KV capacity but starves real request execution headroom.
- Do not optimize this path by startup KV size alone; always validate with real requests.