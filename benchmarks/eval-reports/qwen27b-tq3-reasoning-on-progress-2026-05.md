# Qwen3.6-27B-TQ3_4S reasoning-on progress / learnings — 2026-05

Machine: RX 9070 16 GB (WSL + ROCm)
Runtime: `/home/qiushuo/src/llama.cpp-tq3/build-gfx1201-tq3/bin/llama-server`
Model: `/home/qiushuo/models/qwen/YTan2000-Qwen3.6-27B-TQ3_4S/Qwen3.6-27B-TQ3_4S.gguf`

## What was stabilized

Two live server shapes were confirmed on this box:

### 1. reasoning-off reference (port 8247)
```bash
/home/qiushuo/src/llama.cpp-tq3/build-gfx1201-tq3/bin/llama-server \
  -m /home/qiushuo/models/qwen/YTan2000-Qwen3.6-27B-TQ3_4S/Qwen3.6-27B-TQ3_4S.gguf \
  --host 127.0.0.1 --port 8247 \
  -ngl 99 -fa on -ctk q4_0 -ctv tq3_0 \
  -c 65536 -ub 64 -b 256 --parallel 1 \
  --jinja --reasoning off --reasoning-format none
```

### 2. reasoning-on candidate (port 8250)
```bash
/home/qiushuo/src/llama.cpp-tq3/build-gfx1201-tq3/bin/llama-server \
  -m /home/qiushuo/models/qwen/YTan2000-Qwen3.6-27B-TQ3_4S/Qwen3.6-27B-TQ3_4S.gguf \
  --host 127.0.0.1 --port 8250 \
  -ngl 99 -fa on -ctk q4_0 -ctv tq3_0 \
  -c 65536 -ub 64 -b 256 --parallel 1 \
  --jinja
```

The reasoning-on profile intentionally does **not** pass `--reasoning off`, `--reasoning-format none`, or `--skip-chat-parsing`.

## Verified benchmark evidence so far

### task_sanity passed on both shapes

Result files:
- `/home/qiushuo/reports/pinchbench/qwen36-27b-tq3-4s-task-sanity-8247/0002_qwen3-6-27b-tq3_4s-gguf.json`
- `/home/qiushuo/reports/pinchbench/qwen36-27b-tq3-4s-task-sanity-8250-reasoning-on/0003_qwen3-6-27b-tq3_4s-gguf.json`

Observed metrics:
- 8247 / reasoning-off: success, score 1.0, request_count 1, execution_time 236.41 s
- 8250 / reasoning-on: success, score 1.0, request_count 1, execution_time 240.30 s

Interpretation:
- Enabling reasoning did **not** break the minimal OpenClaw agent path.
- The extra cost on sanity was negligible in this first pass (~4 s).

## Current operational recommendation

For Qwen3.6-27B-TQ3_4S on this box, treat the following as the current best-known agent / benchmark profile until a broader smoke proves otherwise:

- `--jinja`
- no `--skip-chat-parsing`
- reasoning ON for the serve profile under test
- `-ctk q4_0 -ctv tq3_0`
- `-c 65536 -ub 64 -b 256 --parallel 1`

## Key learnings

1. For this TQ3 runtime, the real serve shape needs to be captured from the **actual live process**, not from older wrapper scripts. The live 8250 process showed the authoritative reasoning-on form.
2. A task_sanity pass is enough to clear the first gate, but not enough to declare the full 5-task smoke solved. CSV / weather / calendar still need verification.
3. `--jinja` remains the anchor flag for OpenClaw / PinchBench compatibility here.
4. Reasoning-on is now strong enough to justify a dedicated launcher and a separate smoke runner, rather than reusing the reasoning-off script.
5. The next benchmark should insert a **30 s cooldown between tasks** to reduce allocator/thermal cross-talk during a sequential 5-task smoke on one RX 9070.

## Artifacts added to the repo

- `scripts/qwen27b-tq3-reasoning-on.sh`
- `scripts/run_qwen27b_tq3_reasoning_on_5task_smoke.sh`

## Next step

Run the standard 5-task smoke sequentially against `http://127.0.0.1:8250/v1`, with a 30-second sleep between tasks, and record which tasks regress relative to the simpler task_sanity pass.
