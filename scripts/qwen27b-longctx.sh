#!/usr/bin/env bash
# Long-context fallback: K/V q4_0, 40k ctx (~21 tok/s)
# Use when input/conversation > 20k tokens
export HSA_ENABLE_DXG_DETECTION=1
export ROC_ENABLE_PRE_VEGA=0
exec /home/qiushuo/src/llama.cpp/build/bin/llama-server \
  -m /home/qiushuo/models/qwen/unsloth-Qwen3.6-27B-GGUF/Qwen3.6-27B-Q4_K_S.gguf \
  --host 127.0.0.1 --port 8820 \
  -ngl 999 -fa on -ctk q4_0 -ctv q4_0 \
  -c 40960 -ub 256 -b 1024 --parallel 1 \
  --reasoning off --reasoning-format none --skip-chat-parsing
