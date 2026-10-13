# 和弦约束线（强拍三和弦 logit boost）交接文档

面向接手 follow-up 的同学。A/B/C 三臂实验已收官（正信号），本文覆盖
项目位置、参数、算法和待做的 follow-up。

## 0. 一句话

offline 生成时，在每小节强拍/次强拍对"当前调主三和弦"的全部八度音高
做 additive logit bias（τ），让伴奏更容易落在和弦音上。已有实验证明
全曲窗口版本 CR/UR 双改善；待做的是 τ 细扫 + 副作用检查 + 进出场平滑。

## 1. 项目位置

| 内容 | 本地（以本地 git 为准） | masdar |
|---|---|---|
| 代码 worktree | `C:\Users\User\.codex\worktrees\bar-quality-history-experiment\StreamMUSE-v1`（branch `codex/bar-quality-history-experiment`） | `/data/home/yuanxin/StreamMUSE-bar-quality-history-experiment`（文件级 scp 同步，远端 git 落后） |
| runner / 打分 | `experiments/chord_law/run_law_ab.py`、`score_law_ab.py`、`run_law_ab.sh` | 同路径（已同步） |
| boost 实现 | `src/streammuse/infrastructure/inference/lekai_http_backend.py`（搜 `LEKAI_TONAL_BOOST_TAU`，约 1186-1205 行）+ `tonal_constraint.py`（调性估计 TwoBarKeyTracker） | 同路径 |
| 结果（原跑） | `F:/repos/StreamMUSE-v1/remote_results/chord_law_20261008_results/` | `/data/home/yuanxin/experiments/chord_law_20261008/run_*` |
| 修正窗口重打分 | — | `/data/home/yuanxin/experiments/rescore_fixed_window_20261012/` |
| 访问 | — | `ssh MBZ-Martina`（10.127.30.168，需 VPN） |

## 2. 参数

后端从环境变量读（每次 generate 时读，运行中可改）：

| 变量 | 含义 | A 臂 | B 臂 | C 臂 |
|---|---|---|---|---|
| `LEKAI_TONAL_BOOST_TAU` | 强拍 logit bias 强度；0=关 | 0 | 2.0 | 2.0 |
| `LEKAI_TONAL_BOOST_START_TICK` | 生效起点（事件 tick，4=1拍） | 32 | 32 | 32 |
| `LEKAI_TONAL_BOOST_END_TICK` | 生效终点；0=不封顶（全曲） | — | 256 | 0 |
| `LEKAI_CONTINUATION_TONAL_CONSTRAINT` | 调外音硬掩码（三臂共有的基线约束） | 1 | 1 | 1 |

其余固定：采样 t=1.05 / top_p 0.98 / top_k 0 / rep 1.0；prompt=Rule
If-Else N=10（三臂共享同一条 prompt，故 A/B/C 严格配对）；cohort=user10
（真人旋律 ×3 seeds）；打分窗口 [84, stop) 事件 tick，指标 CR/UR
（frozen evaluator，见 `score_law_ab.py`）。

## 3. 算法（boost 的完整链路）

1. **调性估计**（`tonal_constraint.py`，原有机制）：每 32 tick 用累计
   旋律音长直方图（Krumhansl 风格相关）估计 (tonic, major/minor,
   confidence)；第 3 小节起 key_state=determined。
2. **硬约束**（原有，三臂都有）：采样时把 7 个调内 pitch class 以外的
   全部 logit 置 −inf。
3. **强拍 boost**（本线新增）：当拍为小节第 0/2 拍（4/4 强拍/次强拍）且
   在 [start, end) 窗口内时，对 {tonic, tonic+3或4(小/大), tonic+7}
   三个 pitch class 的**所有八度**加 additive logit bias +τ
   （`tonal_kwargs['tonal_boost_pcs'/'tonal_boost_tau']` → 采样层
   `TonalBeatMask.apply`）。硬掩码之后加 bias，所以 boost 只在调内
   音之间重新分配概率，不会违规出调。

## 4. 已有结果（修正窗口 [84, stop)，n=90）

| 臂 | CR | UR | 备注 |
|---|---|---|---|
| A（τ=0，纯硬掩码） | 0.7324 | 0.0408 | =offline baseline 逐位一致（确定性校验过） |
| B（τ=2，bar3-16） | 0.7527 | 0.0432 | CR 升但 UR 持平略升；中途窗口切换有惩罚痕迹 |
| **C（τ=2，全曲）** | **0.7856** | **0.0283** | **CR+UR 双改善，当前最优臂** |

注意：2026-10-12 前的旧数字全部作废（打分器 4× 窗口 bug，已修
`d699cded`；引用一律用上表修正值）。BEAT-40 cohort（120bpm 数据集旋律）
与 user10（90bpm 回放）分数不可同表。

## 5. Follow-up（转交部分，均未开跑）

1. **τ sweep**：C 臂窗口（全曲）下 τ ∈ {1, 2, 4} × user10 × 3 seeds，
   ~2h GPU。回答"强度-收益曲线"，找最优点。
2. **音高多样性检查**：对比各 τ 与 A 臂的 onset pitch 直方图熵——防
   boost 把伴奏压塌成三和弦反复。
3. **进出场平滑**：B 臂中途窗口切换有惩罚提示，设计 boost 的渐入渐出
   （如线性 ramp 或只设起点）。
4. （可选）新指标**调性稳定度**：逐 bar 估调漂移量；CR/UR 对预防性
   干预可能不敏感。

## 6. 环境 SOP（masdar 跑法，坑都在这）

```bash
ssh MBZ-Martina   # 需 VPN；断了找负责人重连
cd /data/home/yuanxin/StreamMUSE-bar-quality-history-experiment
export PYTHONPATH=$PWD:$PWD/src:$PWD/transformers/src   # 否则 import 到旧包
export PYTHONDONTWRITEBYTECODE=1
py=/data/home/yuanxin/StreamMUSE-v1/.venv/bin/python     # 共享 venv，严禁 pip install
CUDA_VISIBLE_DEVICES=1 $py -u -m experiments.chord_law.run_law_ab ...
```

- GPU 用 1/2/3（SnapshotProbe 硬校验），**用前确认该卡空且已预约**。
- 代码改动只在本地 worktree commit，然后 scp 覆盖 masdar 对应文件
  （masdar 无 git 历史，本地为准）。
- 长任务用 tmux（断线不死）：`tmux new-session -d -s 名字 '命令 > log 2>&1'`。
- mido 只在上述 venv 里有；系统 python3 没有 MIDI 库。
