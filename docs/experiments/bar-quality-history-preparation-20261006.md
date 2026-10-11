# 小节质量控制与历史替换：代码准备

> 这是 2026-10-06 的准备阶段记录，不代表当前完成状态。
> 后续 GPU 实验、约束状态与边界修复见 `experiments/bar_quality_history/LIVE_INTEGRATION.md`。

日期：2026-10-06。用户目前没有 H200 GPU 预约；不加载模型、不运行推理。

## 分支及范围

- 分支：`codex/bar-quality-history-experiment`。
- 基线：`a657fe81`，已知 Web 清屏修复版本；是 `f97d8628` 的后继。
- 远端平行目录：`/data/home/yuanxin/StreamMUSE-bar-quality-history-experiment`。
- 不宣称该基线是所有同学分支中最新版本，不合并未知改动。
- 不修改现有 Web、输入量化、scheduler、播放、训练或 offline runner。

## 代码核查

`src/streammuse/infrastructure/inference/lekai_http_backend.py`：

- `_generate_part1_tokens_from_prompt` 在每次调用时设置 `past_key_values = None`；
  本次查到的是调用内解码缓存，不是跨小节常驻 KV cache。
- `_generate_with_interleaved_prompt` 管理伴奏事件与 token 历史。
- `_inject_history_locked` 替换整段 Melody/Accompaniment 历史，清空 token 历史，
  并重置调性追踪、empty guard 和 active pitches。

因此不直接复用 `inject_history` 充当运行中的“小节修复”。后续接模型时需
专门定义局部历史替换、跨界延音、相关 token 失效和约束状态的处理规则。

## 第一阶段交付

`experiments/bar_quality_history` 是独立 CPU 小节事务工具：显式输入已闭合音符、
小节范围、已观察截止位置和外部接受/拒绝标签，输出统计及前后历史。
合成 fixture 只用于逻辑测试，不是模型生成或真实实验结果。

质量阈值尚未校准，不用音符密度等描述量擅自判定“好听”。
Drop 的播放/历史语义尚未由用户最终确定；原型仅测试历史分支，不撤销已播放内容。
尚未闭合的音符及不能安全局部替换的跨界音符应拒绝操作，不能静默裁剪。
离线候选暂由输入提供；尚未接入候选生成、prefill 或 KV 更新。

## 有 GPU 预约之后

1. 明确具体策略，再接同版本模型的小节修正入口。
2. 模型只读取截止当时可见 Melody，修正采样使用独立且已记录的随机流。
3. 同样 Melody、种子、checkpoint 与参数对比 baseline、history drop、history repair。
4. 分别保存真实播放历史、原生成历史、修正后模型历史，不能用修正历史冒充听到的结果。
5. 记录实际修正完成时间；过期修正不可倒填为“在小节边界即时完成”。
6. 验证后续输入 token 一致性、约束状态、跨界延音、覆盖率、听感与实时交付。

本阶段只证明数据处理与边界规则，不证明修正会提高音乐质量或赶得上实时期限。

## 后续本地代码准备

用户随后获得 GPU 1/2/3 的临时使用许可，但最近一次 SSH 连接超时，尚未使用任何 GPU。
继续在本分支补充 `model_probe.py` 与 `probe_runner.py`：隔离 backend 快照、固定小节
重采样、相同种子的下一拍对照。具体边界见 `experiments/bar_quality_history/MODEL_PROBE.md`。
仅做 CPU 假模型验证；尚未同步这批新增代码到远端，也未完成真实 checkpoint 验收。
