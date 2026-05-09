#!/usr/bin/env bash
# Default daily-driver: Qwen3.6-27B Q4_K_S, FA on, K/V q8_0, 24k ctx (~22 tok/s)
export HSA_ENABLE_DXG_DETECTION=1
export ROC_ENABLE_PRE_VEGA=0
exec /home/qiushuo/src/llama.cpp/build/bin/llama-server \
  -m /home/qiushuo/models/qwen/unsloth-Qwen3.6-27B-GGUF/Qwen3.6-27B-Q4_K_S.gguf \
  --host 127.0.0.1 --port 8820 \
  -ngl 999 -fa on -ctk q8_0 -ctv q8_0 \
  -c 24576 -ub 256 -b 1024 --parallel 1 \
  --reasoning off --reasoning-format none --skip-chat-parsing
