# PinchBench & HumanEval 成绩汇总

GPU: AMD RX 9070 16 GB (WSL ROCm). 所有数据来自 `/home/qiushuo/reports/`。
runtime / toks / ctx 都按各跑次原始日志填写；空白处表示该字段日志未提供。

------------------------------------------------------------------------
## 1. HumanEval (164 题；除非另注；pass@1)

### 1a. lm-eval-harness（vLLM 后端，OpenAI completions）
| 模型 / 配置                            | pass@1 | 题数 | runtime | tps | ctx | 路径 |
|---|---:|---:|---:|---:|---:|---|
| Qwopus3.5-9B v3 HLWQ-v7 GPTQ — k4v3 plugin smoke   | 1.00 | 10  | 32.2 s  | -   | -      | results/qwopus-plugin-k4v3-humaneval-smoke/ |
| Qwopus3.5-9B v3 HLWQ-v7 GPTQ — k4v3 postfix smoke  | 1.00 | 10  | 39.4 s  | -   | -      | results/qwopus-k4v3-postfix-humaneval-smoke/ |
| Gemma 4-26B-A4B TQ3_1S — short                     | 0.40 | 10  | 63.0 s  | -   | -      | results/gemma26-tq3-turbo4-humaneval-short/ |
| Gemma 4-26B-A4B TQ3_1S — codeeval-short            | 0.45 | 20  | 112.1 s | -   | -      | results/gemma26-tq3-turbo4-humaneval-codeeval-short/ |
| Qwen3.6-35B-A3B TQ3_1S — short                     | 0.80 | 20  | 293.5 s | -   | -      | results/qwen36-tq3-turbo4-humaneval-short/ |

### 1b. vLLM HumanEval (167-task harness, base_url + max_tokens=256)
| 模型 / 配置                                       | pass@1 | 题数 | runtime (gen) | avg tps | avg comp toks | avg prompt toks | ctx |
|---|---:|---:|---:|---:|---:|---:|---:|
| Qwen3.6-35B-A3B TQ3_1S smoke                      | -    | 5   | 165.6 s  | 2.00  | 74.4   | 111.6 | - |
| Qwen3.6-35B-A3B TQ3_1S smoke (rerun)              | -    | 10  | 269.3 s  | 2.05  | 59.6   | 107.1 | - |
| Qwen3.6-35B-A3B TQ3_1S **full**                   | -    | 164 | **4120 s** | 1.90  | 54.3   | 141.9 | - |
| Gemma 4-26B-A4B TQ3_1S smoke                      | -    | 10  | 53.1 s   | 10.09 | 60.0   | 113.0 | - |
| Gemma 4-26B-A4B TQ3_1S **full**                   | -    | 164 | **923.3 s** | 9.98  | 69.3   | 148.2 | - |
| Qwopus3.5-9B v3-Q6_K (gguf) smoke                 | -    | 5   | 7.9 s    | 34.30 | 60.4   | 111.6 | - |
| Qwopus3.5-9B v3-Q6_K (gguf) **full**              | -    | 164 | **368.1 s** | 31.01 | 75.9   | 141.9 | - |
| Gemma E4B-it UD-Q6_K_XL smoke                     | -    | 5   | 14.4 s   | 39.35 | 113.2  | 116.8 | - |
| Gemma E4B-it UD-Q6_K_XL **full**                  | -    | 164 | **591.5 s** | 40.31 | 148.3  | 148.2 | - |

### 1c. llama.cpp HumanEval（gfx1201, ngl=999, KV q8_0 除非另注；ctx=16384）
所有 full=164 题，max_tokens=256。

| 模型 (Qwen3.5-9B GGUF)            | KV     | pass@1 | runtime (gen) | tps   | avg comp toks | ctx   |
|---|---|---:|---:|---:|---:|---:|
| UD-Q4_K_XL                        | q8_0   | 0.6707 | 316.5 s | 37.98 | 80.98 | 16384 |
| UD-Q4_K_XL                        | f16    | 0.6707 | 293.7 s | 39.80 | 78.48 | 16384 |
| UD-Q4_K_XL                        | bf16   | 0.6646 | 301.8 s | 38.76 | 78.47 | 16384 |
| UD-Q4_K_XL                        | turbo4 | 0.6524 | 278.8 s | 38.95 | 72.68 | 16384 |
| UD-Q5_K_XL                        | q8_0   | 0.6768 | 325.8 s | 35.45 | 81.11 | 16384 |
| UD-Q6_K_XL                        | q8_0   | 0.7073 | 366.5 s | 32.29 | 82.01 | 16384 |
| Q8_0                              | q8_0   | 0.6951 | 392.0 s | 29.55 | 80.46 | 16384 |

(smoke pass@1≈0.8，10 题，约 6–8 s/run。)

### 1d. llama.cpp Qwen3.6-35B-A3B TQ3_4S deep-KV（RX 9070, ctx=65536, full 164题）
| 模型 / 配置 | KV | FA | pass@1 | 通过 | runtime (gen) | aggregate tps | avg comp toks | avg prompt toks | ctx | 路径 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| Qwen3.6-35B-A3B TQ3_4S | K q4_0 / V tq3_0 | on | **0.5915** | **97/164** | 434.4 s | 38.0 | 100.6 | 141.9 | 65536 | results/humaneval-qwen36-35b-tq3-4s-64k-kq4-vtq3-20260512-153158/ |

对比旧 full run（同 alias，目录名含 fp16kv）：90/164 = 0.5488。deep-KV 64k 版本多解出 7 题，+4.27 pct。

### 1e. vLLM TurboQuant KV — Qwopus3.5-9B GPTQ（full 164题）
| 配置 (mbt / seq)                          | KV dtype             | pass@1 | runtime (gen) | tps   | avg comp toks | ctx     |
|---|---|---:|---:|---:|---:|---:|
| tq3_ctx65536_mbt8192_seq1                 | turboquant_3bit_nc   | 0.6402 | 407.0 s  | 36.58 | 88.99 | 65536   |
| tq4_ctx131072_mbt16384_seq1               | turboquant_4bit_nc   | 0.6768 | 227.4 s  | 58.27 | 86.27 | 131072  |

------------------------------------------------------------------------
## 2. PinchBench（基于 `category_scores`/`efficiency.total_execution_time_seconds`）

> 本地 GGUF/vLLM 跑次 efficiency.total_tokens=0（OpenAI-compat usage 字段未填），
> 因此 toks 列基本为 0，仅 requests/runtime 可信。

### 2a. Smoke / sanity 单题
| 跑次 (run_id 路径)                                                 | model            | suite                  | runtime | reqs | sanity 通过 |
|---|---|---|---:|---:|---|
| pinchbench-q6trio-128k/qwen35-9b-q6/smoke                          | Qwen3.5-9B Q6_K_XL | task_sanity         | 90.2 s  | 1  | ✅ |
| pinchbench-full-seq/gemma-e4b-q6/smoke                             | Gemma-E4B Q6_K_XL  | task_sanity         | 87.2 s  | 1  | ✅ |
| pinchbench-local-validation/gemma-sanity-rerun                     | Gemma-E4B Q6_K_XL  | task_sanity         | 72.1 s  | 1  | ✅ |
| pinchbench-local-validation/qwopus-sanity-rerun                    | Qwopus 9B v3.5 Q6_K | task_sanity        | 76.2 s  | 1  | ✅ |
| pinchbench/qwopus-9b-q6-smoke                                      | Qwopus 9B v3.5 Q6_K | task_sanity        | 84.5 s  | 1  | ✅ |
| pinchbench/qwopus-q6-smoke-20260426-135358                         | qwopus 9B v3.5 Q6_K | task_sanity        | 84.5 s  | 1  | ✅ |
| pinchbench/qwen36-35b-a3b-q4km-smoke                               | Qwen3.6-35B Q4_K_M  | task_sanity        | 92.7 s  | 0  | ❌ (无响应) |
| local-validation/qwopus-smoke3 / gemma-smoke3 (sanity+shell+sum)   | Q6_K 模型           | 3-task              | ~236 s  | 6-7| sanity+shell ✅, summary ❌ |

### 2b. 三套件 (memory / productivity / writing)，hermes 框架
| 模型             | suite        | runtime | reqs | category pct | 路径 |
|---|---|---:|---:|---|---|
| qwen36-35b tq3_4s | memory       | 240 s  | 9   | MEMORY 75.0% | qwen36-tq3-4s-3suite/.../memory |
| qwen36-35b tq3_4s | productivity | 946 s  | 38  | PROD 56.7%   | qwen36-tq3-4s-3suite/.../productivity |
| qwen36-35b tq3_4s | writing      | 567 s  | 38  | WRITING 0.0% | qwen36-tq3-4s-3suite/.../writing |
| qwen36-35b tq3_4s | memory (hermes)       | 251 s | 9  | MEMORY 100% | qwen36-tq3-4s-3suite-hermes |
| qwen36-35b tq3_4s | productivity (hermes) | 961 s | 48 | PROD 82.4%  | 同上 |
| qwen36-35b tq3_4s | writing (hermes)      | 573 s | 34 | WRITING 89.8% | 同上 |
| Qwopus 9B v3.5 Q6 | memory       | 249 s  | 4   | MEMORY 16.8% | qwopus-q6-3suite-hermes |
| Qwopus 9B v3.5 Q6 | productivity | 578 s  | 8   | PROD 12.8%   | 同上 |
| Qwopus 9B v3.5 Q6 | writing      | 443 s  | 6   | WRITING 21.9%| 同上 |

### 2c. automated-only 全跑（26 题）
| 跑次                                              | 模型                  | 完成   | runtime  | reqs | 类别得分 (pct) |
|---|---|---|---:|---:|---|
| pinchbench-q6/full-automated                       | Qwen3.5-9B Q6_K_XL   | 26    | 1719.8 s | 27  | PROD 20 / RESEARCH 0 / CODING 11.7 / ANALYSIS 0 / MEMORY 0 / SKILLS 0 |
| pinchbench-q6-qwopus/full-automated                | Qwopus 9B v3.5 Q6_K  | 26    | 1921.8 s | 27  | PROD 20 / RES 0 / CODING 11.7 / ANALYSIS 0 / MEM 0 / SKILLS 0 |
| pinchbench/qwopus-q6-automated-20260426-135852     | Qwopus 9B v3.5 Q6_K  | 2/26  | 164.6 s  | 5   | PROD 50 (仅 2 题完成) |
| pinchbench/qwen35-9b-q6kxl-128k                    | Qwen3.5-9B Q6_K_XL (ctx=128k) | 4/26 | 468.2 s | 19 | PROD 70.8 |
| pinchbench/gemma4-26b-q4ks-128k-ncmoe10 (run 1)    | Gemma 4-26B Q4_K_S (128k, ncmoe=10) | - | 2027.7 s | 27 | PROD 0 / RES 0 / CODING 11.7 / ANALYSIS 0 / MEM 0 / SKILLS 0 |
| pinchbench/gemma4-26b-q4ks-128k-ncmoe10 fixed      | 同上                | -     | 4073.9 s | 96  | **PROD 20 / RES 50 / CODING 56.5 / ANALYSIS 14.3 / MEM 100 / SKILLS 42.9** |
| pinchbench/qwen36-35b-a3b-q4km-parallel/shardA     | Qwen3.6-35B Q4_K_M  | 5/13  | 609.9 s | 11  | PROD 20 |
| pinchbench/qwen36-tq3-4s-full-hermes               | Qwen3.6-35B TQ3_4S  | -     | **4175.0 s** | 117 | **PROD 91.3 / RES 100 / CODING 88.8 / ANALYSIS 25.9 / MEM 100 / SKILLS 100** |
| pinchbench/qwopus-tq4-full-20260428-234413         | Qwopus TQ4          | 14/26 | 1331.4 s | 52  | PROD 0 / RES 0 / CODING 13.3 |
| pinchbench/qwopus-tq4-c64k-gmu82-20260429-003309   | Qwopus TQ4 ctx=64k  | -     | 2519.0 s | 103 | PROD 20 / RES 0 / CODING 11.7 / ANALYSIS 0 / MEM 0 / SKILLS 0 |
| kv-matrix/fp8_kv_v2 (qwopus-fp8kv-64k)             | Qwopus FP8 KV ctx=64k | 2/26 | 309.9 s | 4 | PROD 50 |

### 2d. all-suite（123 题，automated+writing 等）
| 跑次                                              | 模型                | 完成    | runtime  | reqs | 主要类别 |
|---|---|---|---:|---:|---|
| pinchbench-q6trio-128k/qwen35-9b-q6/full           | Qwen3.5-9B Q6_K_XL ctx=128k | 2/123  | 151.2 s | 3  | PROD 50 |
| pinchbench-full-seq/gemma-e4b-q6/full              | Gemma-E4B Q6_K_XL          | 13/123 | 1831.8 s | 63 | PROD 48.5 / RES 51.7 |

------------------------------------------------------------------------
## 3. 突出结果速览
- **Qwen3.6-35B-A3B TQ3_4S 64k deep-KV HumanEval**：97/164，pass@1 0.5915，K q4_0 / V tq3_0 + FA，434 s / 38 tps。
- **HumanEval 最佳本地 9B**：Qwen3.5-9B UD-Q6_K_XL (q8_0 KV) 0.7073 / 366 s / 32 tps / ctx 16k。
- **HumanEval 最高 tps**：Qwopus tq4 turboquant_4bit_nc (vLLM, ctx 131k) 58.3 tps，pass@1 0.677。
- **PinchBench 自动套件最强**：Qwen3.6-35B TQ3_4S full hermes（PROD 91 / RES 100 / CODING 89 / MEM 100 / SKILLS 100），4175 s。
- **Pinch 三套件 hermes**：Qwen3.6-35B TQ3_4S 在 writing (89.8) / productivity (82.4) / memory (100) 全部赢过 Qwopus-9B-Q6（21.9 / 12.8 / 16.8）。
- **lm-eval HumanEval**：Qwopus3.5-9B v3 GPTQ smoke 1.0 (10 题)；Qwen3.6-35B-A3B TQ3 0.80 (20 题)；Gemma 4-26B TQ3 0.45 (20 题)。
