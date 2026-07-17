# ZipZapZop Memory Sweep 实验报告（2026-07-10）

## 结论摘要

本次按照 `developing-logs/plans/2026-07-03-zzz-memory-sweep-plan.md` 完成了 7 个模型的 ZipZapZop memory sweep。正式实验结果目录：

- run root：`task_runs/zzz_memory_full_matrix_parallel_20260710-090000/`
- 正式汇总（过滤 smoke/pilot）：`matrix_summary_formal.csv` / `matrix_summary_formal.json` / `matrix_summary_formal.md`
- 原始合并汇总（含 7 条 smoke/pilot）：`matrix_summary.csv` / `matrix_summary.json` / `matrix_summary.md`
- 下载日志目录：`task_runs/zzz_memory_download_20260710-080843/`

最终 7 个模型全部完成，正式 summary rows 为 88 条，完整 summary rows 为 95 条（其中 7 条是每模型 3-turn smoke）。所有模型的 smoke 都通过，正式 run cell 没有失败。

## 运行范围

模型矩阵：

| 模型 | vLLM 并行 | 备注 |
|---|---:|---|
| `Qwen/Qwen3-8B` | 1 GPU | Qwen3 thinking 关闭 |
| `Qwen/Qwen3.6-27B` | 1 GPU | Qwen3.6 thinking 关闭 |
| `google/gemma-3-27b-it` | 1 GPU | gated 权重已下载 |
| `Qwen/Qwen3.6-35B-A3B` | 1 GPU | MoE |
| `openai/gpt-oss-120b` | 1 GPU | 原生 MXFP4，`max_tokens=512` |
| `meta-llama/Llama-3.3-70B-Instruct` | 2 GPU | gated 权重已下载 |
| `Qwen/Qwen3-235B-A22B-Instruct-2507-FP8` | 2 GPU | 官方 FP8，见 workaround |

实验 cell：

- 主矩阵：`history={0,8,32,all}` × `temperature={0,0.7}`，`temperature=0.7` 每格 3 repeats。
- Oracle 对照：每模型 `all + temperature=0 + oracle_history=True`。
- APC-off 对照：每模型 `recent-32/all + temperature=0`。
- gpt-oss 额外 ngram speculative decoding 对照：APC on/off 都跑。

## 为完成实验做的关键实现/配置调整

1. 新增 `scripts/run_zzz_parallel_matrix.sh`：把原 full matrix 分成 GPU waves 并行跑，支持 resume/done marker。
2. 修改 `scripts/run_zzz_memory_sweep.py`：Qwen3/Qwen3.6 自动加 `chat_template_kwargs={"enable_thinking": false}`；gpt-oss 使用更大的输出预算并从 final content 判分。
3. 修改 `ZipZapZopTask` prompt：把 user prompt 改成 `Current number: N
Answer:`，避免模型把 `1:` 理解成继续编号。
4. 修改 `scripts/merge_zzz_memory_summaries.py`：保留完整 summary，同时额外输出过滤 `pilot=True` 的 formal summary。
5. Qwen235 FP8 的 vLLM workaround：
   - 第一次失败是 `flashinfer.gemm fp8_blockscale_gemm_sm90` cubin/path assertion。
   - 设置 `VLLM_BLOCKSCALE_FP8_GEMM_FLASHINFER=0` 后可通过加载。
   - 因 235B FP8 每卡加载约 110.25 GiB，`gpu_memory_utilization=0.60` 不足以分配 KV cache；改为 `0.90` 后成功。
   - 实际 Qwen235 retry 使用 GPU `1,2`，`tensor_parallel_size=2`，`max_model_len=4096`，`max_num_seqs=32`。

## 正式结果：最佳准确率配置

以下均来自 `matrix_summary_formal.csv`，只统计 `pilot=False`、`apc=on`、`spec_decode=none`、非 oracle 的主矩阵。

| 模型 | 最佳正式配置 | strict acc | median latency |
|---|---|---:|---:|
| Qwen3-235B-FP8 | recent-8, temp=0 | 1.000 | 146.5 ms |
| Qwen3-8B | recent-32, temp=0/0.7 | 0.760 | 24.8-30.2 ms |
| Qwen3.6-27B | recent-32, temp=0/0.7 | 0.960 | 88.4-89.7 ms |
| Qwen3.6-35B-A3B | memoryless/recent-32 | 0.890 | 79.2-82.5 ms |
| Gemma-3-27B-it | memoryless | 0.820 | 82.0 ms |
| Llama-3.3-70B | recent-8 | 0.730 | 99.9 ms |
| gpt-oss-120b | 多数档位 | 1.000 | 507.8-601.8 ms |

核心观察：

- `gpt-oss-120b` 几乎满分，但因为 reasoning/final 输出路径，延迟显著高于其他短输出模型。
- `Qwen3-235B-FP8` 是非 reasoning 模型里最强，recent-8 greedy 达到 100%。
- `Qwen3.6-27B` 表现很稳，recent-32 达到 96%，比 35B-A3B、Gemma、Llama 都更好。
- `Qwen3-8B`、Gemma、Llama 对历史长度不总是单调受益，说明错误自污染和格式/策略偏差会抵消上下文信息。

## APC 速度影响

下表是 `temperature=0`、非 oracle、spec none 的 median latency 加速比，定义为 `APC-off median / APC-on median`。

| 模型 | all speedup | recent-32 speedup |
|---|---:|---:|
| Qwen3-235B-FP8 | 2.32x | 1.01x |
| Qwen3-8B | 1.43x | 1.04x |
| Qwen3.6-27B | 1.01x | 0.99x |
| Qwen3.6-35B-A3B | 1.06x | 1.03x |
| Gemma-3-27B-it | 1.45x | 1.23x |
| Llama-3.3-70B | 2.06x | 1.22x |
| gpt-oss-120b | 1.14x | 1.08x |

解释：

- APC 对 `all` 档最有用，因为 prompt 只追加历史，prefix 命中高。
- `recent-32` 是滑窗，前缀结构变化更大，收益通常小很多。
- Qwen3.6-27B 的 APC-on/off 基本一样，说明这组请求里 prefill 不是主瓶颈，或者 cache 没有产生明显收益。

## Oracle history 对照

`all + temp=0 + APC-on` 下，把 history 写模型实际回答 vs 写 expected answer 的差值如下：

| 模型 | normal all | oracle all | delta |
|---|---:|---:|---:|
| Qwen3-235B-FP8 | 0.94 | 0.99 | +0.05 |
| Qwen3-8B | 0.68 | 0.77 | +0.09 |
| Qwen3.6-27B | 0.95 | 0.94 | -0.01 |
| Qwen3.6-35B-A3B | 0.82 | 0.90 | +0.08 |
| Gemma-3-27B-it | 0.81 | 0.85 | +0.04 |
| Llama-3.3-70B | 0.58 | 0.74 | +0.16 |
| gpt-oss-120b | 1.00 | 1.00 | +0.00 |

这说明 self-history pollution 对不少模型是实打实的问题，尤其 Llama 和 Qwen3.6-35B-A3B；Qwen3.6-27B 和 gpt-oss 则比较不受影响。

## gpt-oss ngram speculative decoding

ngram speculative decoding 没有改变 greedy 正确率，但显著变慢：

| config | baseline median | ngram median | ngram / baseline |
|---|---:|---:|---:|
| memoryless | 562.2 ms | 754.9 ms | 1.34x |
| recent-8 | 546.3 ms | 792.0 ms | 1.45x |
| recent-32 | 628.2 ms | 955.3 ms | 1.52x |
| all | 601.8 ms | 910.7 ms | 1.51x |

结论：这次 ZipZapZop 的 gpt-oss 输出虽然走 reasoning/final 路径，但 ngram speculative decoding 在当前 vLLM 配置下不划算。后续如果还要测 speculation，应该换更适合长文本连续输出的 task，或者重新扫 speculative 参数。

## 重要限制和注意事项

1. **正式分析必须用 `matrix_summary_formal.*`**。`matrix_summary.*` 包含 7 条 smoke/pilot rows，会把每模型 3-turn 冒烟也算进去。
2. vLLM `/reset_prefix_cache` 在本环境返回 404，所有 run_config 记录为 `available=false`。因此 APC-on 主矩阵没有做到 plan 里理想的“每 run 清空 prefix cache”。APC-on 延迟应解释为同一 server 内连续 run 的 warm/cache 场景；APC-off 对照仍然是干净的无 prefix cache 对照。
3. gpt-oss 的 latency 是 reasoning-mode total latency，不应和 8-token 短回答模型直接做“模型速度”结论；更合理的是单独看 tokens/s、completion_tokens 和 final extraction。
4. Qwen235 可跑，但需要 `VLLM_BLOCKSCALE_FP8_GEMM_FLASHINFER=0` 和 `gpu_memory_utilization=0.90`；否则分别会遇到 FP8 FlashInfer cubin/path assertion 或 KV cache 不足。
5. 本次实验没有画图，但 `per_turn.csv` 已保留每轮 latency、token 数、strict/normalized 判分，可继续做 turn-level 曲线。

## 验证

- `uv run pytest tests/unit/infrastructure/inference/test_local_chat_model_client.py tests/unit/presentation/task/test_task_cli.py tests/unit/domain/tasks/test_zip_zap_zop_task.py tests/unit/application/tasks/test_task_runtime.py tests/unit/presentation/task/test_zzz_memory_sweep_script.py -q`
- 结果：`29 passed`
- `python3 -m py_compile scripts/merge_zzz_memory_summaries.py scripts/run_zzz_memory_sweep.py`
- 结果：通过
