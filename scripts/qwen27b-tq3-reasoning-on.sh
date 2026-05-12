#!/usr/bin/env bash
# Qwen3.6-27B-TQ3_4S reasoning-on launcher for RX 9070 16GB.
# Verified from the live 8250 task_sanity pass: --jinja, no --reasoning off, no --skip-chat-parsing.
set -euo pipefail

export HSA_ENABLE_DXG_DETECTION=1
export ROC_ENABLE_PRE_VEGA=0

exec /home/qiushuo/src/llama.cpp-tq3/build-gfx1201-tq3/bin/llama-server \
  -m /home/qiushuo/models/qwen/YTan2000-Qwen3.6-27B-TQ3_4S/Qwen3.6-27B-TQ3_4S.gguf \
  --host 127.0.0.1 --port 8250 \
  -ngl 99 -fa on \
  -ctk q4_0 -ctv tq3_0 \
  -c 65536 -ub 64 -b 256 --parallel 1 \
  --jinja
