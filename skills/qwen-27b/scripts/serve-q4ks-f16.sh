#!/bin/bash
# Short-reply / latency-optimized serve config for Qwen3.6-27B Q4_K_S on RX 9070 16GB.
# ~22.7 tok/s on 300-tok gens, ~18 tok/s on 800-tok gens. ctx hard-capped at 8192.
# For long ctx (12k) use the q8_0/f16 variant in scripts/serve-q4ks-q8kv.sh.
export HSA_ENABLE_DXG_DETECTION=1
export ROC_ENABLE_PRE_VEGA=0
CTX=${CTX:-8192}
PORT=${PORT:-8820}
~/src/llama.cpp/build/bin/llama-server \
  -m ~/models/qwen/unsloth-Qwen3.6-27B-GGUF/Qwen3.6-27B-Q4_K_S.gguf \
  --host 127.0.0.1 --port "$PORT" \
  -ngl 999 -fa off \
  -ctk f16 -ctv f16 \
  -c "$CTX" -ub 256 -b 1024 --parallel 1 \
  --reasoning off --reasoning-format none --skip-chat-parsing
