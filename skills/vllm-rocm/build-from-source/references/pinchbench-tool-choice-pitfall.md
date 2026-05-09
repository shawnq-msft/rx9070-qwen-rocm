# Pitfall: vLLM serving for PinchBench / OpenClaw needs tool-choice flags

When serving a Qwen/Qwopus GPTQ model for **PinchBench** (or any OpenClaw-style
agent that uses OpenAI `tool_choice="auto"`), the vLLM launch **must** include:

```
--enable-auto-tool-choice \
--tool-call-parser hermes
```

## Symptom

Without these, the very first request fails with:

```
400 "auto" tool choice requires --enable-auto-tool-choice and --tool-call-parser to be set
```

PinchBench's `task_sanity` then scores 0% and triggers fail-fast, aborting the
whole run before any real task executes. The transcript at
`<output>/0013_transcripts/task_sanity.jsonl` will show
`"stopReason":"error"` with this exact message in `errorMessage`.

## Why it's easy to miss

A pure-HumanEval server config does **not** need these flags — HumanEval uses
plain `/v1/completions`-style calls without tools. So a server start-script
that was validated only against HumanEval will silently break the first time
it's reused for PinchBench (e.g. `~/scripts/qwopus-tq4/start-server.sh` did
exactly this on 2026-04-28).

## Fix

Always add both flags to any "primary serving" script that may be reused for
agentic benchmarks. For Qwen3.x / Qwopus chat templates,
`--tool-call-parser hermes` is the correct parser on this setup.

After editing, **restart** the server (these are launch-time flags, not
hot-reloadable) and re-run sanity:

```
bash ~/scripts/<model>/stop-server.sh
setsid bash ~/scripts/<model>/start-server.sh </dev/null >/tmp/start.log 2>&1
curl -sf http://127.0.0.1:<port>/v1/models   # readiness check
```
