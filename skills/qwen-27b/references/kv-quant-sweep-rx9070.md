# KV cache quantization sweep — Qwen3.6-27B Q4_K_S on RX 9070 16GB

Single-user serve, FA on, `--parallel 1`, `-ub 256 -b 1024`, llama.cpp HIP build with
`-DGGML_HIP_ROCWMMA_FATTN=ON`. Bench prompt: short essay completion, temperature 0.7.

| -ctk / -ctv | max ctx fit | KV @ max ctx | 300-tok gen | 800-tok gen | verdict |
|---|---|---|---|---|---|
| q8_0 / q8_0 | 24576 | ~544 MiB | 22.6 | 21.2 | best speed, short ctx |
| q5_1 / q4_1 | 32768 | 704 MiB | 13.4 | 12.6 | slow |
| q4_1 / q4_1 | 40960 | ~792 MiB | 14.4 | 13.4 | slow |
| iq4_nl / iq4_nl | 40960 | ~792 MiB | 13.1 | 12.9 | slowest |
| **q4_0 / q4_0** | **40960** | **720 MiB** | **19.1** | **21.4** | **★ best long-ctx** |

49152 ctx fails for every variant (hipblas alloc OOM on warmup). 45056 with q4_0 hangs
indefinitely at hipblas init — VRAM is right at the edge, do not push past 40960.

## Why q4_0/q4_0 wins and q4_1/q5_1/iq4_nl lose

`-ctv` legal types in current build: `f32, f16, bf16, q8_0, q4_0, q4_1, iq4_nl, q5_0, q5_1`.
There is **no q6_0/q6_K for KV** — `q5_1` is the closest 6-bpw stand-in for K.

The RDNA4 `fattn-mma-f16` path on gfx1201 has fast block-only kernels for `q8_0` and `q4_0`
(no per-block mins, simple dequant). The `_1` variants (`q4_1`, `q5_1`) carry mins, and
`iq4_nl` is non-linear with a lookup table — both fall to a slower generic path. Net cost
is 30–40% throughput regardless of weight quant.

Practical rule on this card: **for KV, only use `q8_0` (max quality) or `q4_0` (max ctx).
Skip everything else.**

## Recommendations

- Long-context / RAG / code review → `-ctk q4_0 -ctv q4_0 -c 40960` → ~21 tok/s, 1.7× ctx of q8_0.
- Short chat / coding → keep `-ctk q8_0 -ctv q8_0 -c 24576` → ~22 tok/s, lowest KV PPL.
- Never pick `q4_1` / `q5_1` / `iq4_nl` for KV on this card: pure speed loss.

## Reproduction recipe

```bash
# wrapper expects 3 args: ctk ctv ctx
cat > /tmp/test-kv.sh <<'EOF'
#!/usr/bin/env bash
CTK=${1:-q4_0}; CTV=${2:-q4_0}; CTX=${3:-40960}
export HSA_ENABLE_DXG_DETECTION=1 ROC_ENABLE_PRE_VEGA=0
exec ~/src/llama.cpp/build/bin/llama-server \
  -m ~/models/qwen/unsloth-Qwen3.6-27B-GGUF/Qwen3.6-27B-Q4_K_S.gguf \
  --host 127.0.0.1 --port 8820 \
  -ngl 999 -fa on -ctk $CTK -ctv $CTV \
  -c $CTX -ub 256 -b 1024 --parallel 1 \
  --reasoning off --reasoning-format none --skip-chat-parsing
EOF
chmod +x /tmp/test-kv.sh
```

Driving harness pattern (Hermes-shell): start in background, poll `/health`, then POST
warmup + two `/v1/completions` calls (max_tokens=300 and 800), divide
`usage.completion_tokens` by wall time. Always `pkill -9 -f llama-server` between configs;
hipblas hangs on context resize.

Wrapper-script bench loop tip: don't run all 4 KV combos in one Python harness with a
300s budget — startup is 30–80 s per config and blocks. Run one combo per background
session, poll `/health` with curl, bench, kill, next.
