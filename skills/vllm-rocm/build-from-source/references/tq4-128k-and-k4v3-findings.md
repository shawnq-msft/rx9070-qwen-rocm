# TQ4 128k and exact K4/V3 findings on this RX 9070 WSL ROCm setup

## Stock vLLM dtype naming
The user may say "K4/V3", but stock local vLLM does **not** expose a literal `turboquant_k4v3_*` dtype.

Confirmed local stock dtypes:
- `turboquant_k8v4`
- `turboquant_4bit_nc`
- `turboquant_k3v4_nc`
- `turboquant_3bit_nc`

Meaning:
- `turboquant_k3v4_nc` = 3-bit keys + 4-bit values + norm correction
- there is no stock upstream exact `k4v3` kv-cache dtype on this machine

## Where exact K4/V3 comes from
The exact asymmetric K4/V3 path referenced by the user maps to the legacy plugin API in `varjoranta/turboquant-vllm`:

```python
patch_vllm_attention(k_bits=4, v_bits=3, norm_correction=True, sink_tokens=4, boundary_layers=5)
```

Important feasibility warnings from the reference repo:
- hybrid models (Qwen3.5, gpt-oss) are **not fully supported yet** for the KV path
- the plugin build path is CUDA/nvcc-oriented, not a proven ROCm/HIP path
- with `norm_correction=True`, the plugin intentionally falls back to a PyTorch path rather than its CUDA kernel

Therefore on this user's ROCm + Qwopus3.5 hybrid GPTQ setup, exact plugin-style K4/V3 must be treated as a separate feasibility experiment, not as a drop-in equivalent of stock vLLM KV presets.

## TQ4 @ 128k findings for Qwopus GPTQ
Model:
- `/home/qiushuo/models/qwen/caiovicentino1-Qwopus3.5-9B-v3-HLWQ-v7-GPTQ`

Machine:
- WSL
- ROCm
- AMD Radeon RX 9070 16GB
- local vLLM checkout `/home/qiushuo/src/vllm`

### Startup can be rescued, but real 128k serving still fails
Baseline `turboquant_4bit_nc` at `--max-model-len 131072` failed during startup compile/autotune OOM.

Startup rescue results:
- `--max-num-batched-tokens 16384` -> startup succeeds
- `--max-num-batched-tokens 32768` -> startup succeeds
- `--max-num-batched-tokens 65536` -> startup OOM

But real ~128k requests still fail:
- with `16384`: startup OK, real request HTTP 500, EngineCore crash
- with `8192`: startup OK, failure happens later than 16k case, but still HTTP 500 / crash

Interpretation:
- lowering `max-num-batched-tokens` can repair startup
- it does **not** make TQ4 a usable 128k deployment route here

## `--enforce-eager` result
`--enforce-eager` was explicitly tested after the chunk-size startup rescue.

Observed behavior:
- eager mode correctly disabled torch.compile and CUDAGraphs
- startup succeeded
- KV-cache headroom improved materially at ready state
- but the real ~130k prompt still failed with request-time OOM

Key failure signature:
- HTTP 500 from `/v1/completions`
- EngineCore crash inside `ops.gptq_gemm(...)`
- `torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 384.00 MiB`
- only about `685.95 MiB` free at failure time
- about `13.76 GiB` allocated by PyTorch and `1.04 GiB` reserved but unallocated

Interpretation:
- once startup is rescued, the blocker is execution-time memory during real long-context inference
- it is **not** merely compile/CUDAGraph overhead
- so `--enforce-eager` should not be presented as the expected fix for TQ4 128k on this machine

## Native KV offloading caveat on this machine
`--kv-offloading-size` is the modern replacement for the old `--swap-space` idea, but on this user's current local vLLM checkout the naive native-offload path is not plug-and-play for Qwopus/Qwen3.5 hybrid runs.

Observed failure:
- enabling native KV offload could fail during EngineCore initialization before any long-prefill request runs
- representative error:
  - `ValueError: Connector OffloadingConnector does not support HMA but HMA is enabled. Please set --disable-hybrid-kv-cache-manager`

Practical implication:
- do **not** summarize a failed `--kv-offloading-size` test as proof that offloading cannot help
- first account for the hybrid-KV-manager / HMA incompatibility on this machine
- if re-testing native KV offload, include a branch that explicitly tries:
  - `--disable-hybrid-kv-cache-manager`

## Q4 KV @ 131k: startup works, real long request still unstable
A later focused probe corrected an earlier measurement mistake where prompt + generation exceeded `max_model_len` and produced an HTTP 400.

### Important testing rule
For long-context probes near the cap, always verify:
- `prepared_prompt_tokens + max_tokens <= max_model_len`

Otherwise a 400 can be a simple budget overflow rather than a true long-prefill/runtime failure.

### Verified 131072 startup metrics on this machine
With:
- model: `/home/qiushuo/models/qwen/caiovicentino1-Qwopus3.5-9B-v3-HLWQ-v7-GPTQ`
- `kv-cache-dtype=turboquant_4bit_nc`
- `max-model-len=131072`
- `max-num-batched-tokens=2048`
- corrected safe request budget: `prepared_prompt_tokens=130700`, `max_tokens=128`

Observed startup/ready metrics:
- `server_ready=true`
- `model_load_mem_gib=7.53`
- `gpu_ready_gib=14.676`
- `available_kv_cache_gib=5.88`
- `gpu_kv_cache_tokens=187968`
- `maximum concurrency for 131072 tokens/request = 5.56x`
- `engine_init_s=12.97`
- `compile_s=6.21`

### But the corrected real request still failed
Even after avoiding prompt-window overflow, the first real 131k-class request still returned:
- `HTTP 500 Internal Server Error`

Meaning:
- Q4 KV can be reported as **startup-capable** at 131072 here
- but it still should **not** be reported as a proven stable real-long-request path on this machine
- the blocker is no longer just context-budget bookkeeping

## Practical decision rule
For this exact Qwopus GPTQ hybrid model on this RX 9070 16GB ROCm setup:
- do **not** present `turboquant_4bit_nc` as a viable practical 128k+ deployment option just because startup succeeds
- distinguish clearly between:
  - startup success
  - non-overflowing request construction
  - real long-request success
- prefer `turboquant_3bit_nc`, FP8, or another serving stack when the user needs a realistic long-context path
- if the user insists on exact K4/V3, first run a small-context feasibility probe on the legacy plugin path before spending time on max-context or HumanEval work
