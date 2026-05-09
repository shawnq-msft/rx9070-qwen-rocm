# PinchBench / long-prompt OOM pitfalls on RX 9070 16GB (vLLM ROCm + TurboQuant)

Learned 2026-04-29 while trying to run PinchBench automated-only suite on
`qwopus-tq4` (Qwopus 9B GPTQ + TurboQuant 4-bit KV cache) for comparison
against Qwen3.6-35B-A3B-TQ3_4S llama.cpp baseline (74.9%).

## Headline result

TQ4 PinchBench on RX 9070 **does not work at ctx ≥ 64k**, regardless of
GMU or max-num-seqs. HumanEval works fine on the same config because
prompts are short. Once PinchBench prompts grow past ~8k tokens, a single
`gptq_gemm` forward wants ~768 MiB of activation memory. Combined with
the KV cache reservation, this exceeds the 16 GB budget and EngineCore
dies mid-run.

## What was tried (all OOM'd)

| Attempt | ctx | max-num-seqs | GMU | extra | Outcome |
|---|---|---|---|---|---|
| v1 | 131072 | 4 | default | (no tool parser) | task_sanity 400 → fail-fast |
| v2 | 131072 | 4 | default | hermes parser | OOM on task 14 (k8s_debugging) |
| v3 (b2) | 65536 | 2 | default | hermes parser | OOM on sanity |
| v4 (B3) | 65536 | 1 | 0.9 | hermes parser | OOM on sanity |
| v5 | 65536 | 1 | 0.82 | + expandable_segments | Server crashed task 3, runner blind, finished 7.4% (fake) |

## Key gotchas

1. **`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` is silently ignored on ROCm.**
   It emits `UserWarning: expandable_segments not supported on this platform`
   and provides zero benefit. Don't count on it for fragmentation relief.

2. **vLLM benchmark runner does NOT detect server death.** When EngineCore
   crashes from OOM, the runner keeps POSTing to a dead `/v1/chat/completions`
   endpoint. Every subsequent task scores 0, token counts come back as 0,
   and the final summary looks like a real result. Always:
   - Tail the vLLM server stderr for `OutOfMemoryError` / `EngineCore_DEAD`.
   - Cross-check the final summary's `Total tokens used` field — if it's 0,
     the run is invalid.

3. **PinchBench (any tool-using bench) REQUIRES** these flags on `vllm serve`:
   ```
   --enable-auto-tool-choice --tool-call-parser hermes
   ```
   Without them, the sanity task's tool call returns 400 and the suite
   fails fast at task 1.

4. **Don't conflate HumanEval success with "config works".** HumanEval
   prompts are ~200 tokens; PinchBench system prompt + tool definitions
   alone are several thousand. A config that hits 71% on HumanEval can
   still OOM on PinchBench task 1.

## Memory math (approximate, RX 9070 16 GB)

- Qwopus 9B GPTQ weights: ~6.5 GB
- KV cache @ ctx=64k, 4-bit: ~5 GB
- KV cache @ ctx=32k, 4-bit: ~2.5 GB
- vLLM/PyTorch framework + CUDA graphs: ~1 GB
- Per-forward activation (long prompt, single gptq_gemm): up to 768 MiB
- Fragmentation overhead observed: ~1 GB reserved-but-unallocated

64k config → ~13.5 GB committed, ~2.5 GB headroom — too tight for long prompts.
32k config → ~11 GB committed, ~5 GB headroom — should be safe for PinchBench.

## Recommended next attempts

For TQ4 PinchBench on this hardware, in order of preference:

1. **ctx=32k, max-num-seqs=1, GMU=0.85, hermes tool parser.** Most
   PinchBench tasks are single-session and fit comfortably in 32k.
   `task_iterative_code_refine` is multi-session but each session is only
   a few k tokens.
2. ctx=16k + GMU=0.75 if (1) still OOMs. Risk: long tasks may truncate.
3. Drop to TQ3 (3-bit KV) — saves ~1.5 GB on KV side, leaves more for
   activation. But TQ3 HumanEval was 0.640 vs TQ4's 0.677, so quality
   trade-off.
4. Skip TQ4 for long-prompt benches; only use it for HumanEval-style
   short-context evals.

## Baseline for comparison

Qwen3.6-35B-A3B-TQ3_4S in **llama.cpp** (not vLLM) scored **74.9%** on
PinchBench automated-only (26 tasks, runs_per_task=1). That's the number
TQ4 needs to beat to justify itself for agentic workloads.
