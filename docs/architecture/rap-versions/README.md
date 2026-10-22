# RAP 各版本系统图

每个 tag 一张图，风格一致；由 `build_diagrams.py` 生成（`python3 docs/architecture/rap-versions/build_diagrams.py`），改图请改脚本里的 `CONFIGS` 再重新生成。

概念图：[rap-concept.svg](rap-concept.svg)，上面是 server（写词模型 → Lyric Pool → TTS → Audio Bank，两个模型都自回归地一直往前生成），下面是 client（发 theme / flow template，收歌词，按需取音频）。对应 v1.x 的歌词流架构。

修改时间线：[rap-edit-timeline.svg](rap-edit-timeline.svg)，4 个快照展示 bar 4 中间改 theme / flow template（只提前 1 个 bar）时 Lyric Pool、Audio Bank 和播放怎么变，底部对比提前 2 个 bar 的情况。

| 版本 | 图 | 一句话 |
|---|---|---|
| `rap-v0.0` · `0a493cb8` | [rap-v0.0.svg](rap-v0.0.svg) | 两小节 chunk，协议 v1，进程内 HF MOSS，服务器端 R3 |
| `rap-v0.1` · `9b76415e` | [rap-v0.1.svg](rap-v0.1.svg) | MOSS 改为独立的 SGLang-Omni 服务（未打补丁） |
| `rap-v0.2` · `0ea74a7f` | [rap-v0.2.svg](rap-v0.2.svg) | 协议 v2：R3 移到 Mac；并发写词；SGLang-Omni 延迟补丁；Opus level 5 |
| `rap-v1.0` · `19526512` | [rap-v1.0.svg](rap-v1.0.svg) | 协议 v3：歌词流 + 逐 bar MOSS 续写，支持实时修改 |
| `rap-v1.1` · `99565f91` | [rap-v1.1.svg](rap-v1.1.svg) | 只换模型：Qwen3.5-9B + MOSS-TTS-Local v1.5，单 GPU |

数字来源：`developing-logs/reports/2026-10-12-rap-version-demo-page-report.md`（v0.0–v1.0 的 real-time 中位数）、
`2026-10-01-rap-latency-optimization-implementation-report.md`、`2026-10-12-rap-lyric-stream-implementation-report.md`、
`developing-logs/plans/2026-09-03-sglang-omni-moss-serving-plan.md`。v1.1 的实测数字没有入库，图中不标延迟。
