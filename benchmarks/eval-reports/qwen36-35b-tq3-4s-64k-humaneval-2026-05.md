# Qwen3.6-35B-A3B TQ3_4S 64k HumanEval — RX 9070

Date: 2026-05-12
Host: WSL ROCm, AMD Radeon RX 9070 16 GB
Server: llama.cpp TurboQuant fork (`/home/qiushuo/src/llama.cpp-tq3/build-gfx1201-tq3/bin/llama-server`)

## Serving configuration

Model:
`/home/qiushuo/models/qwen/YTan2000-Qwen3.6-35B-A3B-TQ3_4S/Qwen3.6-35B-A3B-TQ3_4S.gguf`

Model alias:
`qwen36-35b-tq3_4s`

Launch path:
`/home/qiushuo/.local/bin/hermes-use-qwen35b` -> `start-qwen35b-a3b-tq3-64k`

Command shape:

```bash
llama-server \
  -m Qwen3.6-35B-A3B-TQ3_4S.gguf \
  --host 127.0.0.1 --port 8810 --alias qwen36-35b-tq3_4s \
  -ngl 99 \
  -c 65536 -np 1 \
  -b 256 -ub 64 \
  -ctk q4_0 -ctv tq3_0 -fa on \
  --jinja \
  --reasoning auto --reasoning-budget 512 --reasoning-format deepseek \
  --temp 0.7 --top-p 0.8 --top-k 20 --min-p 0
```

Verified `/props`:

```json
{
  "model_alias": "qwen36-35b-tq3_4s",
  "n_ctx": 65536,
  "total_slots": 1
}
```

Model metadata from load log:

- architecture: `qwen35moe`
- parameters: 34.66B
- file type: TQ3_4S, 12.38 GiB, 3.07 BPW
- native training context: 262,144
- attention: `n_head=16`, `n_head_kv=2`, `head_dim=256`
- full GPU offload: 41/41 layers
- ROCm model buffer: 12,274 MiB
- estimated memory fit: 12,718 MiB projected, about 3,358 MiB free at load
- flash attention: enabled

## HumanEval fullset result

Harness:
`/home/qiushuo/reports/vllm-rocm-eval/run_humaneval_openai_api_v2.py`

Scoring:
`/home/qiushuo/reports/vllm-rocm-eval/evaluate_humaneval_passk.py`

Parameters:

- dataset: `openai/openai_humaneval`, test split
- tasks: 164 / 164
- max_tokens: 512
- temperature: 0.0
- base_url: `http://127.0.0.1:8810`

Result:

| model | ctx | KV | FA | pass@1 | passed | runtime | avg prompt toks | avg completion toks | aggregate completion tps |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|
| Qwen3.6-35B-A3B TQ3_4S | 65,536 | K q4_0 / V tq3_0 | on | **0.5915** | **97 / 164** | 434.4 s | 141.9 | 100.6 | 38.0 |

Artifacts on the benchmark machine:

- run dir: `/home/qiushuo/reports/vllm-rocm-eval/results/humaneval-qwen36-35b-tq3-4s-64k-kq4-vtq3-20260512-153158`
- scores: `scores.json`
- predictions: `predictions.jsonl`
- graded results: `predictions.jsonl_results.jsonl`

## Comparison with the previous full run

Previous local full run for the same model alias (directory name included `fp16kv`) scored:

- 90 / 164 = 0.5488 pass@1
- 312.7 s generation time
- avg completion tokens: 81.9
- avg completion tok/s: 35.0

This 64k deep-KV run scored:

- 97 / 164 = 0.5915 pass@1
- 434.4 s generation time
- avg completion tokens: 100.6
- aggregate completion tok/s: 38.0

Delta:

- +7 solved tasks
- +4.27 percentage points pass@1
- longer completions on average, so total runtime increased despite similar/better aggregate token throughput

## Notes

This is a llama.cpp OpenAI-compatible endpoint run, not lm-eval-harness. The custom runner records token usage from the completion API and the scorer uses `human_eval.evaluation.evaluate_functional_correctness` with `pass@1`.

The model's native context is 262k, but this serving configuration intentionally uses a real single-slot 64k context (`-c 65536 -np 1`) for fit and stability on a 16 GB RX 9070.
