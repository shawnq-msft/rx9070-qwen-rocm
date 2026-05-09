# Scouting HF Hub for vLLM-loadable TurboQuant checkpoints

When the user asks for "TQ3 / TQ4 / TurboQuant <model>" to run on **vLLM** (not llama.cpp), the
naming on HF is misleading — most "TQ3_*" / "TQ4_*" repos are **GGUF only**. Use this recipe.

## What vLLM needs

A repo loadable by the local TurboQuant vLLM plugin must contain:

- `config.json` (HF transformers config; for Qwen3.5/3.6 hybrid models you'll see
  `attn_output_gate: true` and a `layer_types` list mixing `linear_attention` and
  `full_attention` — that's normal, not a red flag)
- `model.safetensors` (or sharded `.safetensors`)
- `turboquant_config.json` — per-tensor `{type, bit_width, group_size, rotation: "hadamard", ...}`
- `tokenizer.json`, `tokenizer_config.json`, `chat_template.jinja`

If the repo has `*.gguf` instead, it's llama.cpp-only. Don't try to make vLLM load it.

## Repos to AVOID for vLLM

- `YTan2000/Qwen3.6-27B-TQ3_4S`, `YTan2000/Qwen3.5-27B-TQ3_*`,
  `YTan2000/Qwopus3.5-27B-v3-TQ3_4S` — all GGUF (`tags` includes `gguf`,
  `library_name: gguf`). High download counts mislead.
- `majentik/Qwen3.5-27B-TurboQuant`, `majentik/Qwen3.5-27B-TurboQuant-2bit` —
  **README only, no weight files**. The MLX-suffixed siblings
  (`*-MLX-2bit/4bit/8bit`) are Apple-only.
- `unsloth/Qwen3.6-27B-GGUF`, `bartowski/Qwen_Qwen3.6-27B-GGUF`,
  `lmstudio-community/Qwen3.6-27B-GGUF` — GGUF.

## Repos that DO work for vLLM (Qwen3.5-27B family, as of session)

| repo | bit | safetensors size |
|---|---|---|
| `ianleelamb/qwen3.5-27b-turboquant-3bit` | 3 | ~13.3 GB |
| `ianleelamb/qwen35-27b-4bit-turboquant` | 4 | ~14.4 GB |
| `ianleelamb/qwen35-27b-2bit-turboquant` | 2 | ~8.8 GB |

All three: same architecture (Qwen3.5 hybrid, `attn_output_gate=true`), Hadamard
rotation, group_size 1024 (3-bit uses larger groups 5120/6144). Tag `qwen3_5_text`.

**No Qwen3.6-27B TurboQuant safetensors exist publicly yet** (only GGUF). If 3.6 is
mandatory, the user would have to run TurboQuant quantization themselves on the
bf16 base.

## How to scout (commands)

`hf` CLI has **no `search` subcommand**. Use the Hub HTTP API directly. `jq` is
also not installed in the qwen-rocm venv — use Python.

```python
import urllib.request, urllib.parse, json
q = urllib.parse.quote("Qwen3 27B turboquant")  # or any query
url = f"https://huggingface.co/api/models?search={q}&limit=20&sort=downloads&direction=-1"
data = json.loads(urllib.request.urlopen(url, timeout=20).read())
for m in data:
    print(m["modelId"], "dl=", m.get("downloads", 0))
```

To check a specific repo's files + sizes + library_name + tags:

```python
url = f"https://huggingface.co/api/models/{repo}?blobs=true"
d = json.loads(urllib.request.urlopen(url, timeout=20).read())
print(d.get("library_name"), d.get("tags", [])[:10])
for s in d.get("siblings", []):
    print(s["rfilename"], s.get("size"))
```

To peek at `config.json` / `turboquant_config.json` without downloading:

```python
txt = urllib.request.urlopen(f"https://huggingface.co/{repo}/raw/main/turboquant_config.json", timeout=20).read().decode()
```

## Triage cheatsheet

1. Check `library_name`. `gguf` → llama.cpp only. `transformers` → maybe vLLM-able.
2. Check `siblings` for `*.safetensors` AND `turboquant_config.json`. Both must
   exist. README-only repos are common decoys.
3. Check `tags` for `gguf` / `mlx` to filter out.
4. Confirm architecture in `config.json` matches what your local vLLM build supports
   (Qwen3-Next / hybrid Mamba2+Attention requires the patched build referenced in
   the user's previous TurboQuant work).
5. For RX 9070 16 GB, prefer 3-bit (~13 GB) over 4-bit (~14.4 GB) since KV cache
   stays bf16 by default and 4-bit leaves very little headroom.
