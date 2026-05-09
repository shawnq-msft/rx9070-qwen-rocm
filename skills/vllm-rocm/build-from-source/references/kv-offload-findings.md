# KV offloading findings on this RX 9070 vLLM checkout

## `--swap-space` is obsolete here
On this user's current vLLM main checkout, `api_server.py` rejects `--swap-space` as an unrecognized argument.

Use these instead for KV swapping/offloading experiments:
- `--kv-offloading-size <GiB>`
- `--kv-offloading-backend native|lmcache`

Do not confuse them with:
- `--cpu-offload-gb` → weight offload, not KV offload.

## Native KV offloading requires disabling the hybrid KV cache manager
On this machine, enabling native KV offloading for Qwopus/Qwen3.5 without disabling the hybrid KV cache manager causes EngineCore init failure before any request runs.

Observed error:
- `ValueError: Connector OffloadingConnector does not support HMA but HMA is enabled. Please set --disable-hybrid-kv-cache-manager`.

Practical rule:
- When testing native KV offload here, include `--disable-hybrid-kv-cache-manager`.
- Otherwise the server can fail during engine initialization and produce no long-prefill/token-rate result.

## Important interpretation rule
If native KV offload fails before `/health` and `/v1/models` become ready, do **not** summarize it as a long-prefill failure.
It is a startup/config-compatibility failure, not evidence that near-limit prefill itself failed.

## Capacity vs real viability
In a failed native offload startup for:
- Qwopus GPTQ
- `--max-model-len 196608`
- `--kv-offloading-size 12`
- `--kv-offloading-backend native`
- `--max-num-batched-tokens 8192`

logs still reported:
- `Available KV cache memory: 4.29 GiB`
- `GPU KV cache size: 136,224 tokens`
- `Maximum concurrency for 196,608 tokens per request: 2.73x`

So KV offload appears to improve theoretical cache capacity, but that does **not** prove real near-limit long-prefill success on this RX 9070 setup.

## Recommended next test sequence
1. Keep the same high-ctx target.
2. Add:
   - `--kv-offloading-size 12`
   - `--kv-offloading-backend native`
   - `--disable-hybrid-kv-cache-manager`
3. Run at least two MBT settings:
   - `8192`
   - `2048`
4. Only claim success if a real near-limit prefill request completes.
5. Report both:
   - startup/theoretical KV capacity
   - real request success and token rates
