# Dataset V2 trimmed Melody MIDI

本目录的准备流程只裁掉 Melody MIDI 第一个正速度 `note_on` 之前的 tick：

```text
source NPZ
  -> existing exporter: scripts.prepare_and_compare_lekai_prompt_alignment.export_npz_melody_midi
  -> source_melody_midi/{id}.mid
  -> MIDI leading trim (mido absolute-tick shift)
  -> trimmed_melody_midi/{id}.mid
```

trim 阶段会重新读取已经落盘的 source MIDI，并把第一个 Melody note-on
移动到 tick 0。音符、velocity、duration、相对 onset、内部休止、尾部内容和
trailing ticks 都保持不变；tempo、time signature、track name、program 等
前缀消息保留在合法的非负 tick。不做重新量化、BPM 修改、tail trim、补静音
或 measure 修复。

远端运行：

```bash
bash experiments/lekai_failcase_dataset_v2/run_prepare_trimmed_melody_midi.sh
```

默认输出位于
`/data/home/yuanxin/data/lekai_failcase_dataset_v2_trimmed_midi`，包含 source
MIDI、trimmed MIDI 与 JSON/CSV manifest。未来 offline 和 StreamMUSE 都应读取
`trimmed_melody_midi`；不存在、也不应读取所谓的 trimmed NPZ。
