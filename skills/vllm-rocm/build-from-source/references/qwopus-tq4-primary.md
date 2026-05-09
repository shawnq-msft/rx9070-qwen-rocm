# Qwopus 9B GPTQ + TurboQuant — primary serving path (TQ4)

For day-to-day Qwopus serving on this RX 9070 / WSL ROCm box, the canonical
preset is **`turboquant_4bit_nc` (TQ4)**. It gives near-best HumanEval quality at
the highest measured throughput AND supports the largest usable context.

## Reusable scripts (already on disk)

Path: `~/scripts/qwopus-tq4/`

| script             | purpose                                                                                                |
|--------------------|--------------------------------------------------------------------------------------------------------|
| `start-server.sh`  | Launch vLLM on `:8533`, KV=`turboquant_4bit_nc`, ctx 131072, served-model-name `qwopus-tq4`            |
| `stop-server.sh`   | Kill `vllm.entrypoints.openai.api_server` + `VLLM::EngineCore` cleanly                                 |
| `status.sh`        | Probe `/v1/models`, list vllm processes, print free VRAM                                               |
| `run-humaneval.sh` | Run HumanEval 164 against the live server; output under `~/reports/vllm-rocm-eval/results/humaneval-qwopus-tq4-<ts>/` |

All scripts honor env overrides:
`PORT, HOST, CTX, MBT (max-num-batched-tokens), SEQS (max-num-seqs), SERVED, GMU, MODEL, VENV`.

Default model: `/home/qiushuo/models/qwen/caiovicentino1-Qwopus3.5-9B-v3-HLWQ-v7-GPTQ`
Default venv:  `/home/qiushuo/.venvs/vllm-rocm-latest`

## Why TQ4 was chosen

Verified HumanEval leaderboard (164 tasks, temp=0, max_tokens=512) on this machine:

| KV preset                      | ctx  | pass@1     | TG tok/s | total |
|--------------------------------|------|------------|----------|-------|
| `fp8_e4m3` (vLLM "q8" KV)      | 64k  | **0.689**  | 52.5     | 392 s |
| `turboquant_4bit_nc` (TQ4)     | 128k | 0.677      | **58.3** | 227 s |
| `turboquant_k8v4`              | 64k  | 0.671      | 36.1     | 487 s |
| `turboquant_2bit_nc` (TQ3)     | 64k  | 0.640      | 36.6     | 407 s |

`fp8` only beats TQ4 by <1.5 pp pass@1 but runs at ~2/3 the throughput and is
limited to 64k ctx in this build. `k8v4` is slower than both TQ3 and TQ4 — its
dequant path is not as well optimized; do not pick it as default.

## HumanEval runner gotchas (saved trial-and-error)

`~/reports/vllm-rocm-eval/run_humaneval_openai_api.py` quirks:

- It does **not** accept `--temperature`; it hard-codes `temperature=0.0`. Passing `--temperature` will exit with argparse error.
- It auto-appends `/v1` to `--base-url`. Pass `http://host:port`, **NOT** `http://host:port/v1`. Wrong base-url silently produces an empty `predictions.jsonl` and `pass_at_1.json` reporting score 0 — always sanity-check `wc -l predictions.jsonl` after a few minutes.
- It uses `/v1/completions` (raw completion, no chat template). So any `--reasoning` / chat-parsing / thinking flags on the server are inert for HumanEval scoring.

`~/reports/vllm-rocm-eval/evaluate_humaneval_passk.py` runs as a separate step
(takes `--predictions ...jsonl --output pass_at_1.json --timeout 5.0 --workers 4 --ignore-incomplete`).

## Restart pattern when switching KV presets

vLLM/EngineCore subprocesses can linger and hold VRAM after a SIGTERM. Always:

```
~/scripts/qwopus-tq4/stop-server.sh
sleep 5
# verify clean
pgrep -fa "vllm|EngineCore" | grep -v grep || echo clean
# verify free VRAM
python -c "import torch; f,t=torch.cuda.mem_get_info(); print(f'free={f/1e9:.2f} GB')"
~/scripts/qwopus-tq4/start-server.sh
```
