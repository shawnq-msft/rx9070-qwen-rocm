# Qwopus TQ4 prefill-time OOM — root cause and fix

## Symptom
Qwopus 9B GPTQ + `kv-cache-dtype=turboquant_4bit_nc` server starts fine and
reports healthy KV cache (e.g. `GPU KV cache size: 92,928 tokens`, `Maximum
concurrency 5.35x` at ctx=65536), but **dies mid-request** when a real prompt of
~30k+ tokens hits the first chunked-prefill block.

Error in `~/reports/vllm-rocm-eval/results/qwopus-tq4-server-logs/server-*.log`:
```
torch.OutOfMemoryError: HIP out of memory. Tried to allocate 768.00 MiB.
GPU 0 has a total capacity of 15.87 GiB of which 860.50 MiB is free.
12.82 GiB is allocated by PyTorch, and 1.78 GiB is reserved by PyTorch
but unallocated.
```
Triggered inside `gptq_gemm` (vLLM warns "fallback to a buggy 4-bit gptq_gemm").

## Root cause (confirmed 2026-04-29 on RX 9070 / ROCm WSL)
1. `--max-num-batched-tokens 16384` makes the chunked-prefill chunk huge — the
   buggy 4-bit `gptq_gemm` allocates a ~768 MiB intermediate per chunk.
2. `--gpu-memory-utilization 0.82` only leaves ~2.5 GiB headroom for activations.
3. ROCm has **no `expandable_segments` support**, so reserved-but-unallocated
   fragmentation (~1.78 GiB observed) cannot be reclaimed.
4. At ~36k-token prompts the prefill peak collides with the headroom ceiling
   and the next gptq_gemm output cannot be allocated → request fails.

## Fix (validated with 45k-token prompt, 37s, no OOM)
In `~/scripts/qwopus-tq4/start-server.sh`:
- `MBT 16384 → 4096` — shrinks per-chunk activation peak ~4x. KV cache drops
  92,928 → 79,200 tokens but still 4.55x ctx concurrency at 65k, fine for
  single-user long-context.
- `GMU 0.82 → 0.78` — gives ~3.3 GiB headroom for activations + fragmentation.
- Keep `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (ROCm ignores it but
  harmless).

## Quick repro — must run after any config change
**Don't trust startup KV cache numbers alone**; the buggy 4-bit `gptq_gemm`
prefill peak is what blows up, not the steady-state KV math.

```bash
python -c "
import urllib.request, json, time
prompt = ('The quick brown fox jumps over the lazy dog. ' * 4500)  # ~45k tokens
body = json.dumps({
    'model':'qwopus-tq4',
    'messages':[{'role':'user','content':prompt+'\nReply OK.'}],
    'max_tokens':16, 'temperature':0
}).encode()
t = time.time()
r = urllib.request.urlopen(urllib.request.Request(
    'http://127.0.0.1:8533/v1/chat/completions', body,
    {'Content-Type':'application/json'}), timeout=600)
print(round(time.time()-t,1),'s', json.loads(r.read())['usage'])
"
```

Expected: ~37s, no error, `prompt_tokens` ~45000.

## General rule
On this RX 9070 setup, if you raise `--max-model-len`, `--gpu-memory-utilization`,
or `--max-num-batched-tokens` for 4-bit GPTQ + TurboQuant, you **must** re-run
the long-prompt repro above. Startup-time `Available KV cache memory` is not a
sufficient signal.

## Files
- Live config: `~/scripts/qwopus-tq4/start-server.sh`
- Stop helper: `~/scripts/qwopus-tq4/stop-server.sh`
- Server logs: `~/reports/vllm-rocm-eval/results/qwopus-tq4-server-logs/`
