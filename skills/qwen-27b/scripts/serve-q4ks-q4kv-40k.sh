#!/usr/bin/env bash
# Long-context serve for Qwen3.6-27B Q4_K_S on RX 9070 16GB.
# 40k ctx via q4_0/q4_0 KV; ~21 tok/s long gen.
# Requires llama.cpp built with -DGGML_HIP_ROCWMMA_FATTN=ON.
set -e
PORT=${PORT:-8820}
MODEL=${MODEL:-$HOME/models/qwen/unsloth-Qwen3.6-27B-GGUF/Qwen3.6-27B-Q4_K_S.gguf}
export HSA_ENABLE_DXG_DETECTION=1
export ROC_ENABLE_PRE_VEGA=0
exec "$HOME/src/llama.cpp/build/bin/llama-server" \
  -m "$MODEL" \
  --host 127.0.0.1 --port "$PORT" \
  -ngl 999 -fa on \
  -ctk q4_0 -ctv q4_0 \
  -c 40960 -ub 256 -b 1024 --parallel 1 \
  --reasoning off --reasoning-format none --skip-chat-parsing
