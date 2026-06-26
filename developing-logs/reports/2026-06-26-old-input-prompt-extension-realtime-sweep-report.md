# Old Input Prompt-Extension Realtime Sweep Report

日期：2026-06-26

## 1. 运行目标

本次 sweep 使用 realtime simulation，也就是 `streammuse-cli --input-mode midi_file`，对 `prompts/old_input/mel/` 里的 10 首 melody MIDI 逐首运行 prompt+continuation realtime inference。

按 prompt extension 1/2/3/4 beats 分四组运行，每组都跑完整 10 首歌。

## 2. 输出目录

```text
output/prompt_extension_old_input_sweep/20260626-112226
```

关键文件：

- `output/prompt_extension_old_input_sweep/20260626-112226/run_config.json`
- `output/prompt_extension_old_input_sweep/20260626-112226/summary.json`
- `output/prompt_extension_old_input_sweep/20260626-112226/aggregate_summary.json`

## 3. 运行参数

本次没有固定 seed：sweep script 没有 export `LEKAI_PROMPT_SEED` / `LEKAI_SEED`，并且会从 child env 里移除这两个变量。

Prompt 参数：

```text
LEKAI_PROMPT_TEMPERATURE=1.1
LEKAI_PROMPT_TOP_K=0
LEKAI_PROMPT_TOP_P=0.95
LEKAI_PROMPT_REPETITION_PENALTY=1.0
```

Continuation 参数：

```text
LEKAI_RT_TEMPERATURE=0.8
LEKAI_RT_TOP_K=50
LEKAI_RT_TOP_P=0.98
LEKAI_RT_REPETITION_PENALTY=1.2
```

Realtime / scheduling 参数：

```text
tempo=120
ticks_per_beat=4
beats_per_bar=4
prompt_length_ticks=32
generation_interval_ticks=4
LEKAI_PROMPT_CONTINUATION_SCHEDULING_MODE=streaming_events
LEKAI_PROMPT_CONTINUATION_RECOVER_LATE_EVENTS=1
LEKAI_PROMPT_CONTINUATION_BOUND_LATE_RECOVERY=1
LEKAI_PROMPT_CONTINUATION_RECOVER_LATE_MAX_TICKS=4
LEKAI_PROMPT_CONTINUATION_REHYDRATE_ACTIVE_NOTES=1
```

每首歌的 `max_ticks` 按 MIDI 转换后的实际长度加 24 beat tail 自动计算，因此不是只截前 128/432 ticks。

## 4. 总体结果

- total cases: `40`
- success cases: `40`
- failed cases: `0`
- server failures: `0`

| extension | cases | success | scheduled_event_count | paired_future_only_rows | skipped_unpaired | Accompaniment note_on |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 beat | 10 | 10 | 4741 | 0 | 0 | 2107 |
| 2 beat | 10 | 10 | 3240 | 0 | 0 | 1518 |
| 3 beat | 10 | 10 | 4365 | 0 | 0 | 2109 |
| 4 beat | 10 | 10 | 3925 | 0 | 0 | 1887 |

结论：40 个 case 全部成功；所有 extension 的 `paired_future_only_rows=0`、`skipped_unpaired=0`，说明这次全部走了新的 `streaming_events` scheduling path，没有回到旧 pair scheduler。

注意：ext2 / `rush_e` 和 ext3 / `spirited_away` 这两个 case returncode 是 OK，但本次随机采样下 Accompaniment note_on 为 0。因为本次没有固定 seed，这是采样结果层面的空输出，不是 scheduling crash。

## 5. Per-Case 结果

| ext | song | max_ticks | scheduled | rehydrated | unpaired | acc_note_on | combined.mid |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | 001 | 1190 | 296 | 15 | 0 | 134 | `/data/home/bowenzheng/mbzuai-projects/StreamMUSE-v1/output/prompt_extension_old_input_sweep/20260626-112226/ext1_beats/001/cli/2026-06-26/session_112238/combined.mid` |
| 1 | 002 | 998 | 1176 | 228 | 0 | 453 | `/data/home/bowenzheng/mbzuai-projects/StreamMUSE-v1/output/prompt_extension_old_input_sweep/20260626-112226/ext1_beats/002/cli/2026-06-26/session_112507/combined.mid` |
| 1 | 003 | 1251 | 194 | 4 | 0 | 92 | `/data/home/bowenzheng/mbzuai-projects/StreamMUSE-v1/output/prompt_extension_old_input_sweep/20260626-112226/ext1_beats/003/cli/2026-06-26/session_112712/combined.mid` |
| 1 | 004 | 920 | 516 | 30 | 0 | 248 | `/data/home/bowenzheng/mbzuai-projects/StreamMUSE-v1/output/prompt_extension_old_input_sweep/20260626-112226/ext1_beats/004/cli/2026-06-26/session_112949/combined.mid` |
| 1 | 005 | 1272 | 1245 | 181 | 0 | 585 | `/data/home/bowenzheng/mbzuai-projects/StreamMUSE-v1/output/prompt_extension_old_input_sweep/20260626-112226/ext1_beats/005/cli/2026-06-26/session_113144/combined.mid` |
| 1 | nyan_cat | 512 | 166 | 0 | 0 | 78 | `/data/home/bowenzheng/mbzuai-projects/StreamMUSE-v1/output/prompt_extension_old_input_sweep/20260626-112226/ext1_beats/nyan_cat/cli/2026-06-26/session_113423/combined.mid` |
| 1 | princess_mononoke | 532 | 426 | 84 | 0 | 187 | `/data/home/bowenzheng/mbzuai-projects/StreamMUSE-v1/output/prompt_extension_old_input_sweep/20260626-112226/ext1_beats/princess_mononoke/cli/2026-06-26/session_113528/combined.mid` |
| 1 | river_flows | 484 | 219 | 4 | 0 | 105 | `/data/home/bowenzheng/mbzuai-projects/StreamMUSE-v1/output/prompt_extension_old_input_sweep/20260626-112226/ext1_beats/river_flows/cli/2026-06-26/session_113634/combined.mid` |
| 1 | rush_e | 745 | 160 | 4 | 0 | 76 | `/data/home/bowenzheng/mbzuai-projects/StreamMUSE-v1/output/prompt_extension_old_input_sweep/20260626-112226/ext1_beats/rush_e/cli/2026-06-26/session_113735/combined.mid` |
| 1 | spirited_away | 528 | 343 | 35 | 0 | 149 | `/data/home/bowenzheng/mbzuai-projects/StreamMUSE-v1/output/prompt_extension_old_input_sweep/20260626-112226/ext1_beats/spirited_away/cli/2026-06-26/session_113909/combined.mid` |
| 2 | 001 | 1190 | 434 | 66 | 0 | 203 | `/data/home/bowenzheng/mbzuai-projects/StreamMUSE-v1/output/prompt_extension_old_input_sweep/20260626-112226/ext2_beats/001/cli/2026-06-26/session_114027/combined.mid` |
| 2 | 002 | 998 | 182 | 12 | 0 | 84 | `/data/home/bowenzheng/mbzuai-projects/StreamMUSE-v1/output/prompt_extension_old_input_sweep/20260626-112226/ext2_beats/002/cli/2026-06-26/session_114256/combined.mid` |
| 2 | 003 | 1251 | 784 | 103 | 0 | 351 | `/data/home/bowenzheng/mbzuai-projects/StreamMUSE-v1/output/prompt_extension_old_input_sweep/20260626-112226/ext2_beats/003/cli/2026-06-26/session_114501/combined.mid` |
| 2 | 004 | 920 | 660 | 19 | 0 | 319 | `/data/home/bowenzheng/mbzuai-projects/StreamMUSE-v1/output/prompt_extension_old_input_sweep/20260626-112226/ext2_beats/004/cli/2026-06-26/session_114738/combined.mid` |
| 2 | 005 | 1272 | 188 | 6 | 0 | 88 | `/data/home/bowenzheng/mbzuai-projects/StreamMUSE-v1/output/prompt_extension_old_input_sweep/20260626-112226/ext2_beats/005/cli/2026-06-26/session_114933/combined.mid` |
| 2 | nyan_cat | 512 | 147 | 2 | 0 | 70 | `/data/home/bowenzheng/mbzuai-projects/StreamMUSE-v1/output/prompt_extension_old_input_sweep/20260626-112226/ext2_beats/nyan_cat/cli/2026-06-26/session_115212/combined.mid` |
| 2 | princess_mononoke | 532 | 435 | 21 | 0 | 213 | `/data/home/bowenzheng/mbzuai-projects/StreamMUSE-v1/output/prompt_extension_old_input_sweep/20260626-112226/ext2_beats/princess_mononoke/cli/2026-06-26/session_115317/combined.mid` |
| 2 | river_flows | 484 | 286 | 22 | 0 | 135 | `/data/home/bowenzheng/mbzuai-projects/StreamMUSE-v1/output/prompt_extension_old_input_sweep/20260626-112226/ext2_beats/river_flows/cli/2026-06-26/session_115423/combined.mid` |
| 2 | rush_e | 745 | 0 | 0 | 0 | 0 | `/data/home/bowenzheng/mbzuai-projects/StreamMUSE-v1/output/prompt_extension_old_input_sweep/20260626-112226/ext2_beats/rush_e/cli/2026-06-26/session_115524/combined.mid` |
| 2 | spirited_away | 528 | 124 | 2 | 0 | 55 | `/data/home/bowenzheng/mbzuai-projects/StreamMUSE-v1/output/prompt_extension_old_input_sweep/20260626-112226/ext2_beats/spirited_away/cli/2026-06-26/session_115658/combined.mid` |
| 3 | 001 | 1190 | 613 | 75 | 0 | 286 | `/data/home/bowenzheng/mbzuai-projects/StreamMUSE-v1/output/prompt_extension_old_input_sweep/20260626-112226/ext3_beats/001/cli/2026-06-26/session_115816/combined.mid` |
| 3 | 002 | 998 | 595 | 40 | 0 | 292 | `/data/home/bowenzheng/mbzuai-projects/StreamMUSE-v1/output/prompt_extension_old_input_sweep/20260626-112226/ext3_beats/002/cli/2026-06-26/session_120045/combined.mid` |
| 3 | 003 | 1251 | 501 | 82 | 0 | 244 | `/data/home/bowenzheng/mbzuai-projects/StreamMUSE-v1/output/prompt_extension_old_input_sweep/20260626-112226/ext3_beats/003/cli/2026-06-26/session_120250/combined.mid` |
| 3 | 004 | 920 | 660 | 97 | 0 | 323 | `/data/home/bowenzheng/mbzuai-projects/StreamMUSE-v1/output/prompt_extension_old_input_sweep/20260626-112226/ext3_beats/004/cli/2026-06-26/session_120527/combined.mid` |
| 3 | 005 | 1272 | 719 | 26 | 0 | 344 | `/data/home/bowenzheng/mbzuai-projects/StreamMUSE-v1/output/prompt_extension_old_input_sweep/20260626-112226/ext3_beats/005/cli/2026-06-26/session_120722/combined.mid` |
| 3 | nyan_cat | 512 | 72 | 2 | 0 | 36 | `/data/home/bowenzheng/mbzuai-projects/StreamMUSE-v1/output/prompt_extension_old_input_sweep/20260626-112226/ext3_beats/nyan_cat/cli/2026-06-26/session_121002/combined.mid` |
| 3 | princess_mononoke | 532 | 495 | 61 | 0 | 239 | `/data/home/bowenzheng/mbzuai-projects/StreamMUSE-v1/output/prompt_extension_old_input_sweep/20260626-112226/ext3_beats/princess_mononoke/cli/2026-06-26/session_121106/combined.mid` |
| 3 | river_flows | 484 | 341 | 52 | 0 | 163 | `/data/home/bowenzheng/mbzuai-projects/StreamMUSE-v1/output/prompt_extension_old_input_sweep/20260626-112226/ext3_beats/river_flows/cli/2026-06-26/session_121213/combined.mid` |
| 3 | rush_e | 745 | 369 | 2 | 0 | 182 | `/data/home/bowenzheng/mbzuai-projects/StreamMUSE-v1/output/prompt_extension_old_input_sweep/20260626-112226/ext3_beats/rush_e/cli/2026-06-26/session_121313/combined.mid` |
| 3 | spirited_away | 528 | 0 | 0 | 0 | 0 | `/data/home/bowenzheng/mbzuai-projects/StreamMUSE-v1/output/prompt_extension_old_input_sweep/20260626-112226/ext3_beats/spirited_away/cli/2026-06-26/session_121447/combined.mid` |
| 4 | 001 | 1190 | 372 | 1 | 0 | 181 | `/data/home/bowenzheng/mbzuai-projects/StreamMUSE-v1/output/prompt_extension_old_input_sweep/20260626-112226/ext4_beats/001/cli/2026-06-26/session_121605/combined.mid` |
| 4 | 002 | 998 | 643 | 43 | 0 | 294 | `/data/home/bowenzheng/mbzuai-projects/StreamMUSE-v1/output/prompt_extension_old_input_sweep/20260626-112226/ext4_beats/002/cli/2026-06-26/session_121834/combined.mid` |
| 4 | 003 | 1251 | 876 | 131 | 0 | 428 | `/data/home/bowenzheng/mbzuai-projects/StreamMUSE-v1/output/prompt_extension_old_input_sweep/20260626-112226/ext4_beats/003/cli/2026-06-26/session_122039/combined.mid` |
| 4 | 004 | 920 | 297 | 10 | 0 | 144 | `/data/home/bowenzheng/mbzuai-projects/StreamMUSE-v1/output/prompt_extension_old_input_sweep/20260626-112226/ext4_beats/004/cli/2026-06-26/session_122316/combined.mid` |
| 4 | 005 | 1272 | 140 | 0 | 0 | 65 | `/data/home/bowenzheng/mbzuai-projects/StreamMUSE-v1/output/prompt_extension_old_input_sweep/20260626-112226/ext4_beats/005/cli/2026-06-26/session_122511/combined.mid` |
| 4 | nyan_cat | 512 | 268 | 12 | 0 | 127 | `/data/home/bowenzheng/mbzuai-projects/StreamMUSE-v1/output/prompt_extension_old_input_sweep/20260626-112226/ext4_beats/nyan_cat/cli/2026-06-26/session_122751/combined.mid` |
| 4 | princess_mononoke | 532 | 488 | 69 | 0 | 240 | `/data/home/bowenzheng/mbzuai-projects/StreamMUSE-v1/output/prompt_extension_old_input_sweep/20260626-112226/ext4_beats/princess_mononoke/cli/2026-06-26/session_122855/combined.mid` |
| 4 | river_flows | 484 | 408 | 72 | 0 | 197 | `/data/home/bowenzheng/mbzuai-projects/StreamMUSE-v1/output/prompt_extension_old_input_sweep/20260626-112226/ext4_beats/river_flows/cli/2026-06-26/session_123002/combined.mid` |
| 4 | rush_e | 745 | 291 | 17 | 0 | 141 | `/data/home/bowenzheng/mbzuai-projects/StreamMUSE-v1/output/prompt_extension_old_input_sweep/20260626-112226/ext4_beats/rush_e/cli/2026-06-26/session_123102/combined.mid` |
| 4 | spirited_away | 528 | 142 | 1 | 0 | 70 | `/data/home/bowenzheng/mbzuai-projects/StreamMUSE-v1/output/prompt_extension_old_input_sweep/20260626-112226/ext4_beats/spirited_away/cli/2026-06-26/session_123236/combined.mid` |

## 6. 快速听文件建议

可以优先听这些输出：

- ext1 / 005: `/data/home/bowenzheng/mbzuai-projects/StreamMUSE-v1/output/prompt_extension_old_input_sweep/20260626-112226/ext1_beats/005/cli/2026-06-26/session_113144/combined.mid`
- ext2 / 004: `/data/home/bowenzheng/mbzuai-projects/StreamMUSE-v1/output/prompt_extension_old_input_sweep/20260626-112226/ext2_beats/004/cli/2026-06-26/session_114738/combined.mid`
- ext3 / 005: `/data/home/bowenzheng/mbzuai-projects/StreamMUSE-v1/output/prompt_extension_old_input_sweep/20260626-112226/ext3_beats/005/cli/2026-06-26/session_120722/combined.mid`
- ext4 / 003: `/data/home/bowenzheng/mbzuai-projects/StreamMUSE-v1/output/prompt_extension_old_input_sweep/20260626-112226/ext4_beats/003/cli/2026-06-26/session_122039/combined.mid`

## 7. 备注

- 本次输出目录下每个 case 都有 `stdout.log`、`stderr.log`、`prompt_continuation_trace.jsonl`、`summary.json` 和 `cli/session_*`。
- 因为没有固定 seed，重复运行不会保证生成完全一致。
- 本次验证的是 realtime simulation 完整 pipeline：MIDI file input -> HTTP prompt-continuation server -> local streaming scheduler -> session `combined.mid`。
