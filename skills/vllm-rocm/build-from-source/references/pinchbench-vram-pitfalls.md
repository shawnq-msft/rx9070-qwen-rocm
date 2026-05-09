# VRAM pitfalls when running PinchBench on TurboQuant vLLM (RX 9070 16GB)

## Symptom: HumanEval-passing config silently fails PinchBench

A TQ3/TQ4 server that scores well on HumanEval (e.g. Qwopus-9B-GPTQ TQ4 → pass@1 0.677 at ctx=131072 / seqs=1 / gmu=0.9) **can OOM-crash on the first long-prompt PinchBench task** while the benchmark runner happily keeps firing requests for ~95 minutes, returning empty assistant messages.

### The fake-score tell

Final summary looks like:
```
Overall Score: 7.4% (1.9 / 26.0)
Total tokens used: 0 (input: 0, output: 0)
Total API requests: 103
```

Score is **fake** — only task 1 (sanity) actually ran on a live engine. Tasks 2–26 hit a dead server. Each transcript shows:
```json
{"role":"assistant","content":[],"stopReason":"error",
 "errorMessage":"EngineCore encountered an issue. See stack trace (above) for the root cause."}
```

## Root cause

PinchBench prompts are 5–10× longer than HumanEval (system prompt + tool definitions + multi-turn history). At 64k–131k ctx with `--gpu-memory-utilization 0.9`, KV cache reservation leaves only ~1–1.5 GiB for activations. A single `gptq_gemm` forward on a long prompt wants ~768 MiB and OOMs.

Server log signature:
```
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 768.00 MiB.
GPU 0 has total 15.87 GiB of which 1.10 GiB is free.
allocated 13.27 GiB by PyTorch ... 1.05 GiB reserved but unallocated
File ".../torch.ops._C.gptq_gemm.default(buf6, ...)"
EngineCore encountered a fatal error.
```

## `expandable_segments` does NOT work on ROCm

Setting `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` prints:
```
UserWarning: expandable_segments not supported on this platform
(Triggered internally at /pytorch/c10/hip/HIPAllocatorConfig.h:40.)
```
…and is silently ignored. Don't rely on it as an OOM mitigation on ROCm/HIP.

## Tested configs on Qwopus-9B-GPTQ + TQ4 + PinchBench automated-only

| Config | Outcome |
|---|---|
| ctx=131072, seqs=1, gmu=0.9, no tool parser | fail-fast: tool parser missing (400) |
| ctx=131072, seqs=1, gmu=0.9, hermes parser | OOM at task 1 (sanity) |
| ctx=65536, seqs=2, gmu=0.9 | OOM at task 1 (extra seq doubles activation) |
| ctx=65536, seqs=1, gmu=0.9 | OOM at task 1 |
| ctx=65536, seqs=1, gmu=0.82, expandable_segments | sanity ✅ then OOM at task 3 (pdf_to_calendar); 23 zombie tasks → fake 7.4% |

## Recommended starting point for PinchBench

`CTX=32768 SEQS=1 GMU=0.85` (no expandable_segments). KV drops ~2.5 GiB vs 64k, leaving ~4 GiB activation headroom. PinchBench single-session tasks rarely exceed 32k ctx in practice.

If even 32k OOMs: drop to `CTX=16384 GMU=0.78`, accepting that the longest tasks may truncate.

## Detection guard — always run before trusting a PinchBench score

1. `Total tokens used` is **non-zero** in the score summary
2. Server still up: `curl -sf http://127.0.0.1:$PORT/v1/models` returns 200
3. Server log has zero matches for `OutOfMemoryError` / `EngineDeadError` / `EngineCore encountered a fatal`
4. Spot-check 2–3 transcripts beyond sanity: assistant `content` array is non-empty

If any fail, the printed score is bogus — the runner does not abort when the upstream server dies (only fail-fasts on task_sanity == 0).

## Required vLLM flags for PinchBench (separate from VRAM)

PinchBench uses tool calling. Without these flags, every task scores 0:
```
--enable-auto-tool-choice --tool-call-parser hermes
```
This is independent of VRAM tuning but commonly forgotten when porting a HumanEval start script.
