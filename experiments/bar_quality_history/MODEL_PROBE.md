# 小节重生成与下一拍对照入口

模型适配与单小节历史替换已通过 H200 GPU2 单曲真模型功能验收。不能报告音乐提升或实时性能。
CPU 假模型测试仅验证调用、种子、历史与输入边界。正式 Web、播放及 scheduler 不变。

## 做什么

输入是一份显式指定小节和已观察截止点的闭合音符快照。对指定小节固定触发：

1. 保留原历史，用 continuation seed 生成边界后的下一拍。
2. 回退到待修正小节之前的伴奏历史，用 repair seed 重生成该小节。
3. 默认仅保存候选；显式开启 `--apply-repair-history` 后，将结构合法候选作为
   固定干预换入实验历史，再用相同 continuation seed 生成下一拍。

候选的“应用”只是指定实验处理，不表示候选被评为更好。质量阈值尚未加入。
修正与下一拍采样各自重置随机流。修正阶段额外消耗的随机数不会推进下一拍种子。

## 技术边界

- 使用现有 ACC-first continuation 接口进行历史小节的重新采样；并非新实现了
  能双向读取整小节 Melody 的 offline 模型。当前 beat 的 ACC 条件如何读取 Melody
  仍由现有 grammar 决定。所有传入 Melody 均限制在已观察截止点，不读取其后的输入。
- 每次是隔离 backend 的历史重建/重新 prefill，不调用运行中 Web 的历史注入 API。
  同样重建 baseline 与 repair 两臂的状态，避免只重置一臂；这仍不是生产会话的
  完整状态快照，不能宣称其输出与历史真人运行 exact-match。
- 只改变后续模型条件，不改写已播放内容。候选、原历史、更新历史、下一拍输出、
  请求/模型 trace 与运行时间分别保存。
- 一个小节完成后才修正，修正耗时不能倒填为“在边界即时生效”。本工具不计算 ISR，
  不模拟播放期限，运行时间也不等于完整系统延迟。
- 初始 MIDI 快照仍要求明确的完整 beat 边界和已闭合音符，不接受原历史跨越替换起点。
  重生成候选改用原生事件表示：保留 `tick < start` 的历史，候选负责 start 及之后的事件；
  核对 trace 中的初始活跃音高，并将候选结尾的真实活跃音符带入下一拍，不伪造 note-off。
  没有对应活跃音的收尾、重复起音或越界事件仍明确拒绝。
- 音符快照转换使用 velocity=100、同 tick off 在 on 前，是显式实验编码，
  不是原始 event 回放。要验证真人 exact-match 必须使用原有原事件入口。
- 自动质量判断、实时 drop、训练 dropout 均未接入。

## 配置与运行

`model_config.template.json` 仅示例配置，checkpoint 路径故意无效；必须替换成已核对
的 checkpoint，并确认全部采样参数、BPM 和约束开关后使用。
`probe.synthetic.json` 是人工合成输入，不是用户录制，不可当成实验结果。

GPU 使用必须先人工确认授权和空闲，仅允许本次授权的 GPU 1/2/3；不要在本机运行。
真实模型加载还要求显式 `--allow-gpu`。不自动查找、抢占或终止其他人的任务。

```bash
cd /data/home/yuanxin/StreamMUSE-bar-quality-history-experiment
export PYTHONPATH="$PWD/src"
# 仅在再次确认 GPU 2 空闲且仍获准使用之后：
export CUDA_VISIBLE_DEVICES=2
python experiments/bar_quality_history/probe_runner.py \
  --input experiments/bar_quality_history/probe.synthetic.json \
  --config /path/to/verified_model_config.json \
  --output /path/to/new_probe_result.json \
  --repair-seed 101 --continuation-seed 202 \
  --allow-gpu --apply-repair-history
```

不用 `--apply-repair-history` 时只生成 baseline 下一拍和修正候选。
已有输出文件不覆盖。模型日志可能写 stdout，因此结果使用独立 JSON 文件而不是重定向 stdout。

2026-10-07：已同步并在 H200 GPU2 跑首轮真实调用，旧闭合音符适配器在边界事件检查处
拒绝替换，原始失败结果保留。新增原生事件拼接只作用于隔离实验，正式 backend/Web 不变。
同输入和种子重测成功：两侧下一拍 Melody 输入相同，完整输入及 raw 输出随伴奏历史改变，
活跃延音状态与下一拍模型日志吻合，无补造 note-off。CPU 测试 43 项通过。
详细证据位于 `F:/repos/StreamMUSE-v1/remote_results/00_bar_history_probe_20261007/README.md`。

后续默认关闭的 backend 接入口与 CPU 验收见 [LIVE_INTEGRATION.md](LIVE_INTEGRATION.md)。
该接入口的实时并发 GPU 验收尚未完成；不要将上面的隔离探针时间当作其上线验证。
