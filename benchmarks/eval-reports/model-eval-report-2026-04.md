# 本地 GGUF 模型评估报告

更新时间：2026-04-17
环境：WSL + ROCm/HIP + AMD Radeon RX 9070 16GB + llama.cpp

## 评估目标
- 统一评估可测试模型的最长可用上下文（目标 128k）
- 记录 token 速度
- 评估 Hermes 场景下的实际可用性
- 给出本机常驻模型建议与取舍理由

## 结论先看

### 最终推荐
1. **长上下文主力 / Hermes 默认主力：Qwen3.5-9B-Q8_0**
   - 128k 可用
   - full offload 成功
   - Q8 精度相对之前 Q4 更高
   - 更适合作为 Hermes 常驻实例
   - 风险：thinking 模式下最终答案可能被 reasoning 污染；高 ctx 下 ready latency 存在波动

2. **大模型长上下文主力：Gemma 26B Q3_K_M**
   - 112k 可用
   - 128k 失败
   - 是本机上最强的大模型长上下文候选之一
   - 风险：高 ctx 下 warmup / ready 更慢，启动后短时间可能出现 `503 Loading model`

### 不建议作为长上下文正式候选
- **Qwen3.6-35B-A3B-UD-Q3_K_S**：16k / 32k 可用，64k 失败
- **Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled.Q3_K_S**：32k 可用，64k 失败
- **Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled.Q4_K_S**：16k / 32k 都失败

### 一句话建议
- **如果你要一个常驻 Hermes 模型：用 Qwen3.5-9B-Q8_0。**
- **如果你要更大的模型并尽量保长上下文：用 Gemma 26B Q3_K_M。**

---

## 评估方法
本次评估围绕 Hermes 的实际使用，而不是只看学术 benchmark：
1. 是否能在本机成功部署
2. 最长可用上下文是多少
3. 代表性 token 速度如何
4. 在 Hermes 风格任务里是否存在关键可用性风险
   - 中文输出是否稳
   - 结构化输出是否容易被污染
   - 指令遵循是否可靠
   - reasoning 是否影响最终答案可用性
   - 作为常驻服务时 ready latency 是否可接受

---

## 结果总表

| 模型 | 量化 | 文件大小 | 最大已确认可用 ctx | 128k 目标 | 结论 |
|---|---:|---:|---:|---|---|
| Qwen3.5-9B | Q8_0 | 9.53 GB | 128k | 达成 | 最佳常驻候选 |
| Gemma 26B A4B | Q3_K_M | 12.53 GB | 112k | 未达成 | 最强大模型长上下文候选 |
| Qwen3.6-35B-A3B | Q3_K_S | 15.36 GB | 32k | 未达成 | 中上下文可用，不适合长上下文主力 |
| Qwen3.5-27B Claude distilled | Q3_K_S | 12.07 GB | 32k | 未达成 | 32k 可用，但指令跟随风险明显 |
| Qwen3.5-27B Claude distilled | Q4_K_S | 15.57 GB | <16k | 未达成 | 不可用 |

---

## 各模型详细结果

### 1. Qwen3.5-9B-Q8_0
- 路径：`/home/qiushuo/models/qwen/unsloth-Qwen3.5-9B-GGUF/Qwen3.5-9B-Q8_0.gguf`
- 128k：可启动、/health 正常、API 调用成功
- full offload：33/33 layers
- 关键显存：
  - CPU_Mapped model ≈ 1030.62 MiB
  - ROCm model ≈ 8045.05 MiB
  - KV cache = 4096 MiB
  - RS buffer = 201 MiB
  - compute buffer = 493 MiB
- 代表性速度：
  - 冷启动短测：prompt_tps ≈ 263.65 tok/s，gen_tps ≈ 43.90 tok/s
  - 另一轮带 reasoning budget 的实际返回：prompt_tps ≈ 0.05 tok/s，gen_tps ≈ 39.60 tok/s
- Hermes 视角评价：
  - 优点：128k 能力最强；Q8 精度升级成功；作为常驻模型最平衡
  - 风险：thinking 打开时，`message.content` 可能不够干净，需要关注最终答案可用性
  - 工程判断：可作为 Hermes 默认常驻模型，但应把“thinking 与 final answer 分离”作为使用规范

### 2. Gemma 26B A4B Q3_K_M
- 路径：`/home/qiushuo/models/gemma/unsloth-gemma-4-26B-A4B-it-GGUF/gemma-4-26B-A4B-it-UD-Q3_K_M.gguf`
- 已验证：32k / 64k / 96k / 112k 可用
- 128k：失败
- full offload：31/31 layers
- 64k 的代表性短测：
  - prompt_tps ≈ 42.07 tok/s
  - gen_tps ≈ 57.27 tok/s
- 112k 成功链路：
  - offload 成功
  - listening 成功
  - API 成功
- Hermes 视角评价：
  - 优点：在 16GB 卡上把大模型长上下文推到了 112k，非常强
  - 风险：高 ctx 下 warmup/ready 时间长；启动后短时间可能返回 `503 Loading model`
  - 工程判断：适合作为“需要更大模型质量时手动切换”的强力候选，不如 Qwen9 Q8 适合轻快常驻

### 3. Qwen3.6-35B-A3B-UD-Q3_K_S
- 路径：`/home/qiushuo/models/qwen/unsloth-Qwen3.6-35B-A3B-GGUF/Qwen3.6-35B-A3B-UD-Q3_K_S.gguf`
- 16k：可用
- 32k：可用
- 64k：失败（rs cache OOM，缺口约 251 MiB）
- 当前最大已确认 ctx：32k
- 速度：
  - 16k：prompt_tps ≈ 28.14 tok/s，gen_tps ≈ 60.66 tok/s
  - 32k：prompt_tps ≈ 12.44 tok/s，gen_tps ≈ 59.68 tok/s
- Hermes 视角评价：
  - 优点：生成速度不差
  - 缺点：ctx 上限太低，不满足长上下文 Hermes 工作流的核心需求
  - 工程判断：不进入长上下文正式候选组

### 4. Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled.Q3_K_S
- 路径：`/home/qiushuo/models/qwen/mradermacher-Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled-GGUF/Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled.Q3_K_S.gguf`
- 32k：可用
- 64k：失败（compute pp buffers OOM，缺口约 495 MiB）
- 当前最大已确认 ctx：32k
- 32k 代表性速度：
  - prompt_tps ≈ 131.21 tok/s
  - gen_tps ≈ 22.02 tok/s
- 32k 显存分解：
  - model ≈ 10983 MiB
  - context ≈ 2646 MiB
  - compute ≈ 495 MiB
  - self ≈ 14124 MiB
  - free ≈ 2002 MiB
- Hermes 视角评价：
  - 风险：默认模板下未严格服从“exactly”指令，短答被解释性内容污染
  - 工程判断：即使 32k 可用，也不适合作为 Hermes 主力，因为指令跟随与最终答案可用性有明显风险

### 5. Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled.Q4_K_S
- 路径：`/home/qiushuo/models/qwen/mradermacher-Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled-GGUF/Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled.Q4_K_S.gguf`
- 16k：失败（rs cache OOM）
- 32k：失败（KV cache OOM）
- 工程判断：不可用，直接淘汰

---

## Hermes 实际使用视角评分（10 分制）

| 模型 | 长上下文能力 | 工程可服务性 | 指令/输出可控性 | Hermes 适配度 | 备注 |
|---|---:|---:|---:|---:|---|
| Qwen3.5-9B-Q8_0 | 10 | 8 | 7 | 9 | 128k 最强，但要管好 thinking 输出 |
| Gemma 26B Q3_K_M | 9 | 6 | 8 | 8 | 112k 很强，但启动/ready 更重 |
| Qwen3.6-35B-A3B Q3_K_S | 4 | 6 | 7 | 5 | 32k 上限限制太大 |
| 27B distilled Q3_K_S | 4 | 6 | 4 | 4 | 指令跟随和 final answer 可用性偏弱 |
| 27B distilled Q4_K_S | 1 | 1 | 1 | 1 | 不可用 |

说明：
- 长上下文能力：主要看最大已确认可用 ctx
- 工程可服务性：看启动后能否稳定 ready、是否容易出现 loading/预热延迟
- 指令/输出可控性：看是否容易污染 JSON / “exactly” / 最终答案
- Hermes 适配度：综合长上下文、中文任务、代理式总结、常驻使用体验

---

## 最终建议

### 建议 1：默认常驻模型
使用：`Qwen3.5-9B-Q8_0`

原因：
- 是唯一明确达成 128k 的高精度升级方案
- 作为 Hermes 常驻模型，长上下文优势最明显
- Q8 比此前 Q4 更符合你“优先高精度”的偏好

建议的使用原则：
- 默认常驻用它
- 关注 thinking 输出污染 final answer 的问题
- 若是严格结构化输出，可考虑额外约束 prompt 或在业务层做结果校验

### 建议 2：大模型手动切换方案
使用：`Gemma 26B Q3_K_M`

原因：
- 在 16GB 卡上做到了 112k，可视为本机最强大模型长上下文方案之一
- 当你更看重大模型表达/推理风格时，它比 9B 更值得切换

代价：
- 启动更重
- ready latency 更长
- 不如 Qwen9 Q8 适合“轻量、随时响应”的常驻工作流

### 建议 3：不建议继续投入时间的路线
- Qwen3.5-27B distilled Q4_K_S：直接淘汰
- Qwen3.5-27B distilled Q3_K_S：可保留作 curiosity test，但不适合 Hermes 主力
- Qwen3.6-35B-A3B Q3_K_S：如果未来只需要 32k 以内可再考虑；当前不满足长上下文要求

---

## 模型文件路径汇总
- Qwen3.5-9B-Q8_0
  - `/home/qiushuo/models/qwen/unsloth-Qwen3.5-9B-GGUF/Qwen3.5-9B-Q8_0.gguf`
- Gemma 26B A4B Q3_K_M
  - `/home/qiushuo/models/gemma/unsloth-gemma-4-26B-A4B-it-GGUF/gemma-4-26B-A4B-it-UD-Q3_K_M.gguf`
- Qwen3.6-35B-A3B-UD-Q3_K_S
  - `/home/qiushuo/models/qwen/unsloth-Qwen3.6-35B-A3B-GGUF/Qwen3.6-35B-A3B-UD-Q3_K_S.gguf`
- Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled.Q3_K_S
  - `/home/qiushuo/models/qwen/mradermacher-Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled-GGUF/Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled.Q3_K_S.gguf`
- Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled.Q4_K_S
  - `/home/qiushuo/models/qwen/mradermacher-Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled-GGUF/Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled.Q4_K_S.gguf`
