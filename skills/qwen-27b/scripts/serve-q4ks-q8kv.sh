#!/bin/bash
# Long-ctx serve config for Qwen3.6-27B Q4_K_S on RX 9070 16GB.
# ~17 tok/s sustained. ctx 12288. Use this for chat/RAG/multi-turn workloads.
export HSA_ENABLE_DXG_DETECTION=1
export ROC_ENABLE_PRE_VEGA=0
CTX=${CTX:-12288}
PORT=${PORT:-8820}
~/src/llama.cpp/build/bin/llama-server \
  -m ~/models/qwen/unsloth-Qwen3.6-27B-GGUF/Qwen3.6-27B-Q4_K_S.gguf \
  --host 127.0.0.1 --port "$PORT" \
  -ngl 999 -fa off \
  -ctk q8_0 -ctv f16 \
  -c "$CTX" -ub 256 -b 1024 --parallel 1 \
  --reasoning off --reasoning-format none --skip-chat-parsing
