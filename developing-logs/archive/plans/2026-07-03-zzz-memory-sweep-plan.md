# ZipZapZop Memory Sweep 实验（速度 × 正确率 × 记忆窗口 × 温度 × 模型）— 计划（2026-07-03，rev4）

> rev4：加入**模型维度**——7 个开源模型（已逐一验证 HF 可下载性），矩阵变为 模型 × 记忆 × 温度。
> rev5：加入**推理加速对照**——① APC（prefix caching）on/off 显式化（vLLM 默认开启，是测速的隐藏变量）；② client 换 `requests.Session` 连接复用；③ gpt-oss-120b 增测 ngram 投机解码一组。
> rev6（外部评审修正，6 条全采纳）：① APC **cross-run 污染**处理（每 run 前 `POST /reset_prefix_cache`）；② 正式 sweep **顺序平衡**（按 repeat 轮换 + 记录 cell_order_index）；③ **历史自我污染**指标（accuracy_before_first_error / post_error_accuracy + 可选 oracle_history 对照）；④ Todo 0c 明确整条 top_p 调用链（含 run_task/_build_client）；⑤ gpt-oss 延迟拆分字段、标注 reasoning-mode；⑥ strict + normalized 双判分。

> rev2：加入 temperature 维度——`temperature=0`（贪心）测一遍，"正常" temperature（0.7）再测一遍，因为 temperature 预期会影响 accuracy。
> rev3（决策落定）：normal 档 = **temperature 0.7 + top_p 0.8**（Qwen 推荐组合）。top_p 是 per-request 参数、vLLM 直接接受，**server 不用动**；只需给 `LocalChatModelClient` 加可选 top_p 字段（小改动）。temp>0 档**跑 3 次取平均**。

## 目标

一键脚本跑一组对照实验：**同一模型、同一任务（zip_zap_zop）、100 轮**，扫两个维度——"给模型看多少历史" × "采样温度"，测量每格的：

- **速度**：每轮 LLM 响应延迟（latency_ms），汇总 mean / median / p95 / max，以及随轮数变化的曲线；
- **正确率（rev6 双口径）**：
  - `strict_accuracy`（主指标，= 现 validator 语义：strip + 忽略大小写的整串匹配，"Zip." / "The answer is Zip" 都算错——这是游戏规则，但混入了 format-following 能力）；
  - `normalized_accuracy`（分析层重算：去掉句号/引号/首尾空白后再比，把"会算但多嘴"从"不会算"里分离出来）；
  - 辅助：`first_error_turn`、`accuracy_before_first_error`、`post_error_accuracy`（见设计要点 6 的自我污染问题）。
  正式报告两套 accuracy 都给。

实验矩阵 = **4 个记忆档 × 2 个温度档 = 8 格**；greedy 档确定性各跑 1 次，normal 档各跑 **3 次取平均** → 共 4×1 + 4×3 = **16 个 run（约 1600 次请求）**：

| 记忆档 | `--history-limit` | 含义 |
|---|---|---|
| memoryless | 0 | 每轮只看当前数字（run 模式现默认） |
| recent-8 | 8 | 最近 8 轮 |
| recent-32 | 32 | 最近 32 轮 |
| all | 100（= max_turns） | 全部历史（100 轮内等价于无上限） |

| 温度档 | 采样参数 | 重复次数 | 含义 |
|---|---|---|---|
| greedy | temperature=0.0 | 1 | 确定性，同一模型可复现 |
| normal | temperature=0.7 + top_p=0.8 | **3（取平均）** | Qwen 官方推荐组合，有随机性 |

## 模型矩阵（7 个，HF 可下载性已逐一验证，2026-07-03）

| # | HF repo | 参数量 | 精度/量化 | 权重显存 | 卡数(tp) | 下载状态 | 矩阵角色 |
|---|---|---|---|---|---|---|---|
| 1 | `Qwen/Qwen3-8B` | 8B 稠密 | BF16 | ~16 GB | 1 | ✅ 直下 | 小模型基线（注：Qwen3.6 无 8B，此为 8B 档最新） |
| 2 | `Qwen/Qwen3.6-27B` | 27B 稠密 | BF16 | ~54 GB | 1 | ✅ 直下 | 中稠密（有官方 `-FP8` 备选） |
| 3 | `google/gemma-3-27b-it` | 27B 稠密 | BF16 | ~54 GB | 1 | 🔒 需签 license | 同尺寸异家族对照（vs #2） |
| 4 | `Qwen/Qwen3.6-35B-A3B` | 35B-A3B MoE | BF16 | ~70 GB | 1 | ✅ 直下 | MoE vs 稠密（vs #2；激活仅 3B） |
| 5 | `openai/gpt-oss-120b` | 117B-A5B MoE | **MXFP4（原生）** | ~63 GB | 1 | ✅ 直下 | 异家族 + 原生 4bit + 推理型 |
| 6 | `meta-llama/Llama-3.3-70B-Instruct` | 70B 稠密 | BF16 | ~140 GB | **2** | 🔒 需签 license | 大稠密异家族 |
| 7 | `Qwen/Qwen3-235B-A22B-Instruct-2507-FP8` | 235B-A22B MoE | **FP8（官方）** | ~235 GB | **2** | ✅ 直下 | 旗舰 MoE 上限 |

量化选择原则：**能单卡 BF16 就 BF16**（无量化混杂变量）；#5 是模型原生格式别无选择；#6 用 2 卡 BF16（拒绝社区 FP8 版，保持精度维度干净）；#7 用**官方** FP8（BF16 要 4 卡 470GB，官方 FP8 质量有保证且省一半下载）。

> ⚠️ **#7 用 `-Instruct-2507-FP8` 而不是原版 `Qwen3-235B-A22B`**：原版是混合思考模型（默认输出 `<think>`，会撞爆 max_tokens=8 并让判分全错），2507 版是纯非思考、更新更强、且有官方 FP8。

### 每模型 vLLM 启动参数

```bash
# 通用（llm-serving venv + HF 缓存变量）：
export HF_HOME=$HOME/mbzuai-projects/models/huggingface
export HF_HUB_CACHE=$HOME/mbzuai-projects/models/huggingface/hub
VLLM=~/mbzuai-projects/llm-serving/.venv/bin/vllm

# 1 卡模型 (#1 #2 #3 #4 #5)：
CUDA_VISIBLE_DEVICES=<gpu> $VLLM serve <repo> --host 127.0.0.1 --port 8000 \
  --served-model-name <repo> --max-model-len 4096 --gpu-memory-utilization 0.85

# 2 卡模型 (#6 #7)：
CUDA_VISIBLE_DEVICES=<g1>,<g2> $VLLM serve <repo> ... --tensor-parallel-size 2
```

### 思考模式风险（sweep 前每模型必须先冒烟）

- **#1 Qwen3-8B 是混合思考模型**（默认 enable_thinking=True → 输出 `<think>`）：必须禁思考。方案：client 请求里带 `chat_template_kwargs: {"enable_thinking": false}`（vLLM 支持 per-request 传入；`LocalChatModelClient` 需加可选 extra-payload 字段，与 top_p 一起做）。
- **#2/#4 Qwen3.6 系列**默认行为待冒烟确认，如有思考同上处理。
- **#5 gpt-oss-120b 是推理型**（harmony 格式，先输出 analysis 再 final）：`max_tokens=8` 会在 analysis 阶段就被截断、永远到不了答案。处理：该模型单独放大 completion 预算（如 512）并从 final content 取答案（vLLM 的 reasoning 输出与 content 分离时取 content）。
  **rev6 报告口径**：它的延迟是 **"reasoning-mode total latency"**，不可与 8-token 模型同列直接横向比。per-turn 至少拆四个字段：`request_latency_ms`、`completion_tokens`、`tokens_per_second`（= completion/latency，反映纯生成速度）、`answer_extract_status`（final 提取成功/截断/失败）。结论要写成"思维链的推理代价"，不是"这个模型慢"。
- **每模型正式跑前先 3 轮冒烟**：确认回答是裸的 `Zip/Zap/数字` 格式，不是思维链。

### 推理加速对照（rev5 新增）

**① APC（Automatic Prefix Caching）on/off 显式化。** vLLM V1 **默认开启** APC——共享前缀的 KV 跨请求复用。对本实验它是关键隐藏变量：`all` 档历史只追加、前缀每轮全命中（prefill 从 O(全长) 变 O(增量)），而 recent-8/32 滑窗会因最老一行被挤出而大面积 miss。**开着 APC，"全记忆"可能比"32 轮窗口"更快**——不显式控制，"延迟 vs 记忆窗口"的结论就是错的。

**⚠️ cross-run 污染（rev6）**：更隐蔽的问题是同一 server 连续跑多个 cell 时 **APC cache 跨 run 保留**——例如 greedy 的 3 个 normal-repeat prompt 序列完全相同，repeat 2/3 会全程命中 repeat 1 留下的缓存；memoryless 档的 system 前缀也会被所有后续 run 复用。不处理的话 latency 系统性偏快且偏差不均匀。

设计（rev6 修订）：
- **主矩阵**在 APC 默认开启下跑，但**每个 run 开始前调 `POST {base_url}/reset_prefix_cache`** 清空缓存——这样缓存收益只反映**该记忆档自身的 within-run 前缀结构**（这正是要测的），没有跨 run 污染。语义 = "cold-start 后单局游戏的真实缓存行为"。wrapper 启动 server 后先探测该端点是否可用；不可用则降级为每个 config 重启 server（慢但干净），并在 run_config 里记录用了哪种方式；
- **每模型追加一组 APC-off 对照**：server 加 `--no-enable-prefix-caching` 重启，只跑 `all` + `recent-32` 两档、greedy、各 1 run（+200 请求/模型），标 `apc=off`——这是**干净的"延迟 vs prompt 长度"对照**（无任何缓存效应）；
- 汇总表加 `apc` 列；报告单列一节：APC 对各记忆档的加速比（预期 all 档差异巨大、memoryless 档几乎无差）。

**② client 连接复用。** `LocalChatModelClient` 现在每次 `requests.post` 新建 TCP 连接，每轮白付握手开销（~1–3ms，占 50ms 级延迟的 2–6%）。改用 `requests.Session()`（keep-alive），并入 Todo 0c 的 client 改动一起做。这不是实验维度，是**消除测量噪声的前提**。

**③ gpt-oss-120b 增测 ngram 投机解码。** 该模型是唯一长输出（几百 token 思维链）的，投机解码才有肉；ngram 法零草稿模型（从上下文中抄猜测，思维链短语高度重复、命中率预期高）。设计：gpt-oss 跑两遍 sweep——baseline 和 server 加：

```bash
--speculative-config '{"method": "ngram", "num_speculative_tokens": 4, "prompt_lookup_max": 3}'
```

汇总表加 `spec_decode` 列（其余模型恒为 none；输出 1–8 token 的模型配投机无意义，不配）。贪心下投机不改变输出，accuracy 应逐位一致——不一致即 bug，顺带当正确性检查。

### 实验总规模

每模型主矩阵 16 run（4 记忆档 ×（greedy 1 + normal 3））× 100 轮 = 1600 请求 + APC-off 对照 200 请求；gpt-oss 额外一遍 ngram sweep（+1600+200）。合计 **7 × 1800 + 1800 ≈ 14400 请求**。加上每模型 2–3 次 vLLM 启停（主矩阵 / APC-off / gpt-oss 的 ngram，各 1–5 分钟加载），估计总时长 **4–8 小时**——做成整体一键 wrapper 后台挂着跑。

## 基础已就绪（今天刚改完）

- `streammuse-task run --history-limit N` 已实现：`build_turn` 会把 `state.history` 最近 N 轮拼进 prompt，`max_history` 自动顶到 ≥ N（设 100 不会被内部 50 上限截断）。
- 每轮数据已经被记录，脚本只需聚合，不用改生产代码：
  - `trace.jsonl`：每轮 `metrics.latency_ms`、`metrics.prompt_tokens`、`metrics.completion_tokens`、`summary.is_valid`、`summary.expected_output`；
  - `run_summary.json`：总数（turns / valid / invalid）；
  - `response_trace.jsonl`：每轮完整 prompt 和回答（复盘用）。
- token 预算已核算过：100 轮全历史 ≈ 1100 prompt tokens，vLLM `--max-model-len 4096` 富余。

## 实验设计要点

1. **runner 用 `offline_benchmark`**：不定速、请求一个接一个发，latency 就是纯"模型+HTTP"耗时，不被 tick 间隔污染。（`realtime_loop` 是测"定速下能不能跟上"的，不适合本实验。）
2. **温度两档对照**：greedy（temp=0，accuracy 可复现，作 baseline）和 normal（**temp=0.7 + top_p=0.8**，Qwen 推荐组合，测随机性对 accuracy 的伤害）。落地要点：
   - **top_p 是 per-request 参数**：放请求 payload 里，vLLM 直接接受，**server 不用重启不用改**。需要的唯一代码改动：`LocalChatModelClient` 加可选 `top_p`（None 时不发、行为完全不变），并透传到 `run_task`/CLI（`--top-p`）。greedy 档不发 top_p（temp=0 下也无意义）。
   - **temp>0 档跑 3 次取平均**：accuracy 报 3 次均值 ± 标准差（每次也单独留档）；latency 三次合并（300 样本）后取 median/p95。greedy 档确定性只跑 1 次。
   `--max-tokens 8` 两档共用（答案只有一个词/数字；限短防跑题拖时）。速度汇总统一用 median/p95。
3. **all 档的非平稳性**：prompt 随轮数线性变长 → 延迟应随轮数上升。所以除了汇总数，**必须存每轮明细**（config, turn, number, prompt_tokens, latency_ms, is_valid），报告里给"延迟 vs 轮数"分段统计（如前 25 / 中 50 / 后 25 轮），并用 prompt_tokens 佐证相关性。
4. **预热**：vLLM 首个请求慢（编译/缓存）。sweep 开始前先发 1–2 个 warmup 请求（不计入统计）。
5. **顺序平衡（rev6）**：固定 0→8→32→all 会把记忆档和时间漂移（GPU 温度、编译状态、后台负载）耦合——后面的 all 档变慢/变快未必只因为历史长。策略：**冒烟/pilot 可固定顺序；正式 sweep 按 repeat 轮换档位顺序**（如 repeat1: 0,8,32,all；repeat2: all,32,8,0；repeat3: 8,all,0,32），greedy 单 pass 用默认顺序但其延迟结论与 normal 各 repeat 交叉验证；`per_turn.csv` 和汇总都记录 `cell_order_index`。
6. **历史自我污染（rev6）**：`advance_state()` 把**模型实际输出**写进 history——模型第 15 轮答错后，高记忆档后续每轮都看到这个错误，memoryless 档看不到。这把"记忆能力"和"错误污染/自我恢复"混在一起（不算设计错误，但必须显式度量）。处理：
   - 分析层加两个指标（零代码改动，从 per-turn valid 序列直接算）：`accuracy_before_first_error`（首错前正确率，衡量干净上下文下的能力）和 `post_error_accuracy`（首错后正确率，衡量被污染历史下的恢复力）；
   - **可选 oracle 对照**：`ZipZapZopTask` 加 `oracle_history: bool = False` 小 flag——为 True 时 history 写 expected answer 而非模型输出。每模型跑一组 `all + greedy + oracle`（+100 请求），用 "oracle vs 正常 all 档" 的差值分离"上下文长度影响" vs "自我污染影响"。
7. **环境要求**：跑 sweep 时 vLLM 所在 GPU 不要有其他负载；一次只跑一个 sweep。
8. **可选 `--repeats`**：默认 1（每档 100 轮已有 100 个延迟样本）；要更稳可以重复整档取合并统计。第一版实现参数但默认 1。

## 脚本设计：`scripts/run_zzz_memory_sweep.py`

一键入口（默认值即你要的实验）：

```bash
uv run python scripts/run_zzz_memory_sweep.py \
  --model Qwen/Qwen2.5-7B-Instruct \
  --model-url http://localhost:8000/v1 \
  --turns 100 \
  --history-limits 0,8,32,all \
  --temperatures 0,0.7 \
  --top-p 0.8 \
  --repeats-nonzero-temp 3
```

（以上全是默认值，一键 = 不带参数直接跑。`--top-p` 只作用于 temp>0 的档；`--repeats-nonzero-temp` 只让 temp>0 档重复。）

流程：

```
1. Preflight：GET {model-url}/models 确认 server 活着且模型名匹配；
   失败则报错并提示怎么起 vLLM（llm-serving 环境 + HF_HOME），不静默等待。
2. Warmup：发 2 个丢弃请求。
3. 先 build_schedule() 生成正式执行顺序：temp=0 只 1 个 repeat，temp>0 跑 repeats 次；
   pilot 固定顺序，正式 sweep 按 repeat 平衡 history 档位顺序并记录 cell_order_index。
   每个 run 开始前按探测结果调用 reset_prefix_cache（如果 endpoint 可用）。
   in-process 调 cli.run_task(runner="offline_benchmark", history_limit=N,
   temperature=T, top_p=(0.8 if T>0 else None), extra_payload=<per-model>,
   oracle_history=<bool>, max_turns=turns, ...) → 得到 run_dir。
4. 解析该 run 的 trace.jsonl → 每轮 (turn_id, number, is_valid, latency_ms,
   prompt_tokens, completion_tokens)。
5. 全部跑完后聚合输出到 task_runs/zzz_memory_sweep_<时间戳>/：
   ├── per_turn.csv            # 所有 (温度, 档位) 每轮明细（分析/画图用）
   ├── sweep_summary.json      # 每格聚合指标 + 指向各子 run 目录
   ├── sweep_summary.md        # 人读的对照表
   └── (各格的原始 run 目录由 run_task 自己写在同一 sweep 目录下)
6. 终端打印对照表。
```

汇总表字段（单模型 sweep：每 (温度, 档位) 一行共 8 行；全矩阵汇总加 model 列共 56 行；normal 档聚合 3 个 repeat）：

| model | apc | spec_decode | temperature | config | oracle | n_runs | turns | strict_accuracy | normalized_accuracy | first_error_turn | before/post first-error | latency median/p95 (ms) | avg prompt_tokens |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|

（greedy 档 n_runs=1、std 省略；normal 档 accuracy 为 3 次均值±标准差，latency 三次合并统计，first_error_turn 报三次中的最早值并附各次明细。）

设计选择：

- **in-process 调 `run_task()`** 而不是 subprocess：同一 venv、少一层进程；`run_task` 本身就返回结果对象和输出目录。四档串行执行。
- `--history-limits` 接受逗号分隔的 `int` 或字面量 `all`（映射为 `--turns`）；`--temperatures` 接受逗号分隔的 float，默认 `0,0.7`。
- 失败处理：某一档中途异常（如 server 掉了）→ 记录该档 status=error，继续跑不了就整体退出并保留已完成档位的结果。

## 生产代码改动（rev6 明确整条调用链，已于 2026-07-09 实现）

实现路线：选**client-config 持有采样参数**的路线（sweep 每个 run 经 `run_task` 新建 client，top_p 是 run 级常量），不动 `TaskRuntime.generate` 调用签名；默认行为保持不变：

1. `LocalChatModelClientConfig` 加 `top_p: float | None = None`、`extra_payload: dict | None = None`；
2. `LocalChatModelClient`：持有 `requests.Session()`（连接复用）；payload 组装时 top_p 非 None 才并入；extra_payload 非 None 时浅合并，若覆盖已有 payload key 则直接报错；
3. `_build_client()` / `run_task()` 加 `top_p` / `extra_payload` 参数透传到 config；
4. `streammuse-task run` CLI 加 `--top-p`（默认 None）；extra_payload 不上 CLI，只供 sweep in-process 使用；
5. `ZipZapZopTask` 加 `oracle_history: bool = False`（rev6 ③ 的 oracle 对照用）；
6. 单测：top_p / extra_payload 为 None 时不出现在 payload、给值时出现；Session 复用；oracle_history=True 时 history 内容为 expected。

## 整体编排：`scripts/run_zzz_full_matrix.sh`（模型外层循环）

sweep 脚本本身保持"单模型"职责（`--model` 指谁测谁）；7 模型的启停由外层 wrapper 负责：

```
对每个模型（按上表顺序）：
  1. 用对应 CUDA_VISIBLE_DEVICES / tp 启动 vLLM（llm-serving venv 绝对路径 + HF 缓存变量）
  2. 轮询 /v1/models 直到 ready（大模型加载慢，超时 15 分钟）
  3. 跑 3 轮格式冒烟（答案必须是裸词；失败则记录并跳过该模型，不中断整体）
  4. uv run python scripts/run_zzz_memory_sweep.py --model <repo> --sweep-root <总目录>
  5. kill vLLM、等显存释放
汇总：把 7 个模型的 sweep_summary 合并成 matrix_summary.{json,md,csv}（每行 = 模型×温度×记忆档）
```

## 不做的事（本次范围外）
- 不做画图（per_turn.csv 留给你自己画；要的话下次加 matplotlib）；
- 不测多模型对比（脚本参数天然支持换 `--model` 再跑一遍）。

## 执行 Todo List

### Phase 0a：模型授权（🔴 需要你本人网页操作，其余全部可并行推进）

- [ ] 0a.1 登录 HF 账号，到 [meta-llama/Llama-3.3-70B-Instruct](https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct) 点 Accept license（Meta 审批几分钟到几小时）。
- [ ] 0a.2 到 [google/gemma-3-27b-it](https://huggingface.co/google/gemma-3-27b-it) 点 Accept（Google 一般秒批）。
- [ ] 0a.3 验证授权生效（token 已在 `~/.cache/huggingface/token`）：
  ```bash
  curl -s -o /dev/null -w "%{http_code}\n" -H "Authorization: Bearer $(cat ~/.cache/huggingface/token)" \
    https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct/resolve/main/config.json   # 期望 200
  ```

### Phase 0b：模型下载（后台挂，~440GB + gated 2 个 ~194GB）

- [ ] 0b.1 先 export 两个缓存变量（否则会下进默认 `~/.cache`）：`HF_HOME=$HOME/mbzuai-projects/models/huggingface`、`HF_HUB_CACHE=$HF_HOME/hub`。
- [ ] 0b.2 后台逐个 `huggingface-cli download <repo>`（5 个直下：`Qwen/Qwen3-8B`、`Qwen/Qwen3.6-27B`、`Qwen/Qwen3.6-35B-A3B`、`Qwen/Qwen3-235B-A22B-Instruct-2507-FP8`、`openai/gpt-oss-120b`），日志各自落文件。
- [ ] 0b.3 0a 过审后补下 `meta-llama/Llama-3.3-70B-Instruct`、`google/gemma-3-27b-it`。
- [ ] 0b.4 校验：每个 repo 的 snapshots 目录 safetensors 分片数量与 HF 页面一致、`du -sh` 合理；记录实际总占用。

### Phase 0c：生产代码改动（按"生产代码改动"一节的 6 点，一次 PR 级别的小改）

- [x] 0c.1 `LocalChatModelClientConfig` 加 `top_p: float | None = None`、`extra_payload: dict | None = None`。
- [x] 0c.2 `LocalChatModelClient.__init__` 建 `requests.Session()`；`generate()` 改用 session.post；payload 组装：top_p 非 None 才加 `"top_p"`，extra_payload 非 None 则浅合并进 payload；如果覆盖已有 key 则直接 `ValueError`。
- [x] 0c.3 `_build_client()` 与 `run_task()` 加 `top_p=None` / `extra_payload=None` 形参并透传到 config。
- [x] 0c.4 `streammuse-task run` CLI 加 `--top-p`（float，默认 None）→ main() 透传；extra_payload 不上 CLI。
- [x] 0c.5 `ZipZapZopTask.__init__` 加 `oracle_history: bool = False`；`advance_state()` 在 oracle 模式下把 `referee_result.expected_output` 写进 history 的 content（其余字段不变）。
- [x] 0c.6 单测：① top_p/extra_payload 为 None 时 payload 无这两个 key；② 给值时出现且值正确；③ extra_payload 试图覆盖 `model`/`messages` 等已有 key 时会报错；④ 两次 generate 复用同一 Session（mock 断言）；⑤ oracle_history=True 时 history content == expected、False 时 == 模型输出。
- [x] 0c.7 `uv run pytest tests/ -q` 全量绿（本次实测 229 passed + 1 skipped + 1 warning）。

### Phase 1：`scripts/run_zzz_memory_sweep.py`（单模型 sweep）

- [x] 1.1 CLI 参数：`--model`（默认 `Qwen/Qwen2.5-7B-Instruct`，full matrix 显式传）、`--model-url`、`--turns 100`、`--history-limits 0,8,32,all`、`--temperatures 0,0.7`、`--top-p 0.8`、`--repeats-nonzero-temp 3`、`--max-tokens`（默认普通模型 8，gpt-oss 512）、`--sweep-root`（默认 `task_runs/zzz_memory_sweep_<ts>`）、`--pilot`（固定档位顺序 + 不跑附加组，冒烟用）、`--apc-label on|off`（写进结果标签，server 状态由外层控制）、`--spec-label none|ngram`（同上）。
- [x] 1.2 Preflight：GET `{model-url}/models` 确认 server 活 + 模型名匹配，失败即退（打印怎么起 vLLM）；探测 `POST {base}/reset_prefix_cache` 是否 405/404，结果写 `run_config.json`。
- [x] 1.3 Warmup：2 个丢弃请求。
- [x] 1.4 per-model 特殊配置表（脚本内 dict，按 `--model` 匹配）：
  - `Qwen/Qwen3-8B` → extra_payload `{"chat_template_kwargs": {"enable_thinking": false}}`；
  - `Qwen/Qwen3.6-*` → 冒烟确认后决定是否同上（占位注释）；
  - `openai/gpt-oss-120b` → max_tokens=512 + final-content 答案提取 + `answer_extract_status`。
- [x] 1.5 schedule 主循环（pilot 固定顺序；正式 sweep balanced order）：
  - 每个 run 前调 reset_prefix_cache（可用时；endpoint 从 `model-url` 去掉 `/v1` 后得到）；
  - 正式模式档位顺序按 repeat 轮换（r1: 0,8,32,all；r2: all,32,8,0；r3: 8,all,0,32），greedy 用 r1 顺序；`--pilot` 固定顺序；
  - 每 run 记录 `cell_order_index`；
  - in-process 调 `run_task(runner="offline_benchmark", history_limit=N, temperature=T, top_p=(top_p if T>0 else None), extra_payload=<per-model>, max_turns=turns, ...)`。
- [x] 1.6 附加组（非 pilot 时）：`all + greedy + oracle_history=True` 1 run（通过 `run_task/create_task` 透传 `oracle_history`）。
- [x] 1.7 解析每个 run 的 `trace.jsonl` + `response_trace.jsonl` → per-turn 行：`model, apc, spec_decode, temperature, config, repeat, cell_order_index, turn_id, number, response, expected, strict_valid, normalized_valid, latency_ms, prompt_tokens, completion_tokens, tokens_per_second, answer_extract_status, oracle`。
- [x] 1.8 指标聚合（每 (温度,档位) 格）：strict/normalized accuracy（normal 档 3 次均值±std）、`first_error_turn`、`accuracy_before_first_error`、`post_error_accuracy`、latency mean/median/p95/max、前25/中50/后25 轮分段 median、avg prompt_tokens。
- [x] 1.9 输出：`per_turn.csv`、`sweep_summary.json`、`sweep_summary.md`、`run_config.json`（含 reset 方式、顺序表、per-model 配置）+ 终端表格。
- [x] 1.10 失败处理：单 run 异常 → 该格 status=error 继续；server 失联 → 保存已完成部分后退出码非 0。

### Phase 2：单模型冒烟（需 1 张 GPU，~10 分钟）

> 实现状态（2026-07-09）：Phase 0c / Phase 1 / Phase 3 的代码与脚本已完成并通过测试；Phase 0a/0b/2/4 需要 HF 授权、模型下载和正在运行的 vLLM/GPU，本次未实际执行。

- [ ] 2.1 起 Qwen3-8B（下载最快）或已在本地的 Qwen2.5-7B。
- [ ] 2.2 `--pilot --turns 6 --history-limits 0,2,all --temperatures 0,0.7 --repeats-nonzero-temp 2` 跑通。
- [ ] 2.3 核对清单：① 汇总表数字与 trace 手算一致；② all 档 prompt_tokens 逐轮递增；③ temp=0.7 两个 repeat 的 response 序列存在差异；④ 抓一条实际请求（vLLM log 或 response_trace 的 prompt 字段）确认 `top_p` 与 `chat_template_kwargs` 已发出；⑤ Qwen3-8B 输出无 `<think>`；⑥ reset_prefix_cache 探测结果正确记录；⑦ oracle run 的 history 内容是 expected。
- [ ] 2.4 顺手记录 Qwen3.6-27B/35B-A3B 是否需要禁思考（冒烟 3 轮即可），回填 1.4 的占位。

### Phase 3：`scripts/run_zzz_full_matrix.sh`（7 模型编排）

- [x] 3.1 模型清单 + GPU 分配写在脚本头（#1–#5 单卡、#6 #7 两卡 tp=2；`STREAMMUSE_MATRIX_GPUS` 可覆盖）。
- [x] 3.2 每模型生命周期：起 server（llm-serving venv 绝对路径 + HF 变量 + 空闲端口）→ 轮询 `/v1/models`（超时 15 分钟，大模型加载慢）→ 3 轮格式冒烟（失败记录并跳过该模型）→ 主 sweep（apc=on）→ 重启（`--no-enable-prefix-caching`）跑 APC-off 对照（`--history-limits 32,all --temperatures 0` 各 1 run，`--apc-label off`）→ kill + 等显存归零。
- [x] 3.3 gpt-oss 额外一轮：带 `--speculative-config '{"method":"ngram","num_speculative_tokens":4,"prompt_lookup_max":3}'` 重启，整套主 sweep + APC-off 再来一遍（`--spec-label ngram`）；完毕后断言：greedy 档输出与 baseline 逐位一致（不一致即 bug，中止并报告）。
- [x] 3.4 全部完成后合并 `matrix_summary.{json,md,csv}`（行 = 模型×apc×spec×温度×档位）+ 总耗时/失败清单。
- [x] 3.5 中断恢复：每模型完成后落 `done` 标记，重跑 wrapper 跳过已完成模型。

### Phase 4：正式跑（≈15100 请求，4–8 小时，后台）

- [ ] 4.1 前置检查：目标 GPU 空闲（nvidia-smi）、磁盘、7 个模型本地就位、日常 pytest 绿。
- [ ] 4.2 `nohup bash scripts/run_zzz_full_matrix.sh > matrix.log 2>&1 &` 挂后台；中途抽查 matrix.log 与已完成模型的 sweep_summary。
- [ ] 4.3 结束后核对：7 模型全部 done、无 error 格（或 error 已解释）、gpt-oss 投机一致性断言通过。

### Phase 5：报告

- [ ] 5.1 写 `developing-logs/reports/2026-07-03-zzz-memory-sweep-report.md`：matrix 总表 + 分节结论——规模效应（Qwen 8→27→35→235）、家族效应（27B 三方）、MoE vs 稠密、记忆窗口对双口径 accuracy 的影响（含 before/post first-error 与 oracle 差值的"自我污染"分析）、temperature 伤害幅度、延迟 vs prompt 长度斜率（APC-off 组）、APC 对各记忆档加速比、gpt-oss 的 reasoning 代价（completion_tokens/TPS 口径）与 ngram 投机加速比。
- [ ] 5.2 勾掉本文件 todo，附实测总耗时；`per_turn.csv` 位置写进报告供画图。

## 决策（已确认，2026-07-03）

1. "all" 用 100（=turns）实现 ✅
2. normal 档 = temperature 0.7 + **top_p 0.8**（Qwen 推荐组合）。top_p 是 per-request 参数，server 不用动；给 client 加可选字段即可 ✅
3. `--max-tokens 8` ✅
4. 结果放 `task_runs/zzz_memory_sweep_<ts>/` ✅
5. temp>0 档**跑 3 次取平均**（greedy 仍 1 次）；总量 16 run ≈ 1600 请求 ✅
