# High-ctx startup vs real long-prefill viability on this RX 9070 Qwopus vLLM setup

## Core lesson
Do not equate:
1. server startup success,
2. short-request success after startup, and
3. real near-limit long-prefill execution.

On this machine, they are materially different.

## What was verified
### Startup probes
Short-request startup probes with `seqs=1` succeeded at:
- `ctx=147456`
- `ctx=163840`
- `ctx=196608`

So the service can be configured above 128k and still reach healthy startup.

### Real high-prefill execution probes
With true long prompts near the ctx ceiling, using:
- `seqs=1`
- `kv-cache-dtype=turboquant_4bit_nc`
- small decode target (`max_new_tokens` about `1024`)
- `max-num-batched-tokens=8192`

all of the following were `server_ready=true` but the first real request failed with HTTP 500 / EngineCore fatal error:
- `ctx=196608`, `prefill≈192512`
- `ctx=212992`, `prefill≈208896`
- `ctx=229376`, `prefill≈225280`
- `ctx=245760`, `prefill≈241664`
- `ctx=262144`, `prefill≈258048`

## Real failure mode
The surface API error was HTTP 500, but the backend cause was execution-time OOM during the first prefill step.

Representative failure pattern:
- scheduler dump showed `total_num_scheduled_tokens=8192`
- fatal path involved:
  - `gdn_attention_core`
  - `fused_post_conv_prep`
- final exception:
  - `torch.OutOfMemoryError: CUDA out of memory`

Representative 262144-ctx crash state:
- free VRAM about `391.50 MiB`
- PyTorch allocated about `13.73 GiB`
- private pools about `290 MiB`
- reserved but unallocated about `1.28 GiB`
- failing allocation only about `2.00 MiB`
- crash site was a temporary tensor like:
  - `beta = torch.empty(L, HV, dtype=torch.float32, device=device)`

## Interpretation
The limiting factor here is not only KV cache capacity.
For ultra-long prefills, temporary working tensors in the GDN/Mamba-style prefill path become the real limit.

So high ctx on this machine must be reported as two limits:
1. **startup/configurability limit**
2. **real long-prefill execution limit at a given MBT**

## Practical debugging workflow
When the user asks whether theory ctx can be higher:
1. Do a safe startup probe with `seqs=1`, small request, modest MBT.
2. Then run a real high-prefill execution probe with an actually tokenized prompt near target ctx.
3. If the API returns 500, inspect EngineCore logs before concluding anything.
4. If the crash is prefill-time OOM in `gdn_attention_core` / `fused_post_conv_prep`, do not summarize it as simply “ctx unsupported”.
5. Instead report that startup ctx works, but real near-limit prefill is not viable at the tested chunk size.

## Next lever to test
If theory suggests ctx should still be higher, next sweep:
- keep `seqs=1`
- keep the same high ctx + prefill target
- vary smaller `max-num-batched-tokens`, e.g.:
  - `1024`
  - `2048`
  - `4096`
  - `8192`

Reason:
- smaller MBT means smaller chunked-prefill steps
- that may lower per-step temporary activation/workspace peaks
- it may improve viability even if it hurts throughput
