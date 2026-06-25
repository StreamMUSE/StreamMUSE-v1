# 真人实时 session 的 simulator 复现验证（2026-06-25）

## 问题（我的理解）

有一个真人弹的实时 session：`output/2026-6-25/combined.mid`（220 tpb，三轨：Melody=人弹、Accompaniment=机器生成、Metronome=忽略），设定见 `session_config.json`：tempo 75 BPM、ticks_per_beat=4、**count_in_beats=4**、generation_interval_ticks=4、generation_length_frames=4、input=midi_device。

**要验证的命题**：把这条 melody 抽出来、扔掉前 4 拍 count-in，喂给 **simulator（`--input-mode midi_file`）**，采样参数固定成 fix-output（top_k=1, temperature=0），跑出来的 accompaniment **应该和原文件里那条机器 acc 完全一样**。

即：实时（真人 midi_device）与模拟（midi_file）在相同输入 + 确定性参数下，机器伴奏可复现。

## 时间对齐（已核实代码）

实时输入 worker：`tick = seconds_to_tick(now − timeline_start)`，而 `timeline_start` 在 count-in **之后**才设。所以录进文件的 file-tick 直接反映模型域 tick，只差：

- **count-in 偏移**：4 拍 × 220 = **880 file-ticks**（= 模型域 tick 0 的位置，也是 Metronome beat 4 / 真实下拍处）；
- **缩放**：220 file-tpb ÷ 4 model-tpb = **55 file-ticks / model-tick**。

换算公式：**`model_tick(4tpb) = (file_tick − 880) / 55`**。

验证过：Melody 首音 file-tick 1210 → (1210−880)/55 = model tick 6（beat 1.5）；Melody/Acc 在 file-tick<880 内零音符，只有 Metronome 在 beat 0–4 响 —— 与 count-in 设定吻合。

## 两个必须先确认的前提（决定命题成不成立）

1. ⚠️ **原 session 是不是 fix-output（top_k=1, temp=0）跑的？** session_config **不含采样参数**（那是 server 端 `LEKAI_RT_*` env）。若原 session 是默认 temp=0.8 的随机采样，它的 acc 本身不确定，**贪心 simulator 不可能复现**，命题不成立——这不是 bug。**这是头号假设。**
2. ⚠️ **原 session 的条件 BPM token 是多少？** 即原 server 的 `LEKAI_DEFAULT_BPM`（落在 encode_bpm 哪个桶）。simulator 必须用**同一个桶**，否则开头 BPM token 就不同、后面全分叉。session_config 的 `tempo_bpm=75` 是 wall-clock，**不一定**等于条件 BPM。需要你告诉我原来用的条件 BPM（或确认就是默认 120）。

## 实现步骤

### Step 1：抽 melody + 扔 count-in → 干净的输入 MIDI
- 读 `combined.mid`，**只取 Melody 轨**（丢掉 Accompaniment、Metronome）。
- 扔掉 count-in：删除 file-tick < 880 的事件，其余 **整体平移 −880**（真实下拍 → tick 0）。
- 存成单轨 melody-only MIDI（保持 220 tpb）。预期：首音落在 tick 330（= model tick 6 = beat 1.5）。

### Step 2：起 server（确定性 + 对齐条件 BPM）
- env：`LEKAI_RT_TEMPERATURE=0.0 / TOP_K=1 / TOP_P=0.0`、`LEKAI_DEFAULT_BPM=<原 session 的条件 BPM>`、空闲端口、空闲 GPU。

### Step 3：跑 simulator
- `streammuse-cli --input-mode midi_file --midi-file-path <step1 输出> --model-name lekai --inference-type http --server-url ... --generation-interval-ticks 4 --generation-length-frames 4 --count-in-beats 0 --tempo <慢，如 15> --max-ticks <覆盖 melody 长度 + 尾部余量> --output-type session --log-dir ...`
- **count-in=0**：melody 已平移到 tick 0 起，simulator 的 tick 0 即真实下拍，无需再加 count-in。
- tempo 取慢档（如 15）防止丢请求（与生成内容无关）。

### Step 4：对齐 sanity gate（关键，先验证再下结论）
- 把**原文件 Melody**（扔 count-in、平移 −880、缩放到 4tpb）与 **simulator 录的 Melody** 在 (beat,pitch) 层比。
- **必须先 100% 一致**——证明喂进模型的旋律确实对齐。melody 不齐就别看 acc，先修对齐。

### Step 5：对比 accompaniment
- 原文件 Acc（扔 count-in、平移 −880）vs simulator Acc，用 `tests/consistency/midi_pianoroll.py` 的 pianoroll (beat,pitch) 归一化对比（持续音重触发 vs 长音符的差异会被归一化掉），截断到公共窗口。
- 预期 100% 一致。

### Step 6：判读
- melody 齐 + acc 100% → 命题成立，实时可被 simulator 复现。
- melody 齐 + acc 不齐 → 要么原 session 不是贪心（前提 1 不成立），要么有真实分叉，按差异位置深挖（开头差→BPM 桶；尾部差→窗口/max-ticks；中间散→采样不确定）。

## 复用已有产物
- pianoroll 对比：`tests/consistency/midi_pianoroll.py`（已写好，含窗口截断）。
- offline `--bpm` 口子、server 起停/health 轮询/丢弃检测：`tests/consistency/conftest.py` + `runners.py` 里都有，可直接借。
- 这次是**一次性验证脚本**（不是要进 pytest），跑通即可；若你想固化成测试再说。

## 待你确认
1. 原 session 是 fix-output（top_k=1, temp=0）跑的吗？
2. 原 session 的条件 BPM（`LEKAI_DEFAULT_BPM`）是多少 / 是不是默认 120？
3. 这次只要一次性验证（出个对比结论），还是要做成可复跑的脚本/测试？
