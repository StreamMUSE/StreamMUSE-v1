# H200 验收手册：`/inference` 页面优化的第 1 到 3 阶段（2026-11-12）

这份手册写给能连上 H200 的人。照着做一遍大约 1 小时，其中等待约 40 分钟。做完把第 8 节列的文件发回给 Bowen。

## 1. 要验收什么

分支 `feature/rap_optimization` 上有三项改动，代码和测试都做完了，但 H200 一直连不上，没有在真实模型上跑过。

| 阶段 | 改了什么 | 要确认的事 | 通过标准 |
|---|---|---|---|
| 1 | Cyclic 模式回合之间保持连接 | 用户敲完到模型出声的等待（WAIT）变短 | 5 个回合的 WAIT 中位数低于 2000 ms，缺失为 0 |
| 2 | 新回合、新请求的第一个 bar 接着上一句续写 | 续写不在回合边界断开；两段之间声音不再变 | 除整场第一个 bar 外，续写比例 95% 以上；另外要人听 |
| 3 | Steering 模式的长会话适配器（默认关闭，要加参数才启用） | 改主题后新内容更快播出 | 在 bar 前半改的，下一个 bar 播出；后半改的，最多晚一个 bar；缺失为 0；续写比例 95% 以上 |

对照组是改动前的代码（提交 `c90d0d8e`）。链路每天不一样，所以对照组要在同一次验收里重新测，不能用 11-11 的旧数字。

## 2. 准备

### 2.1 代码

Mac 和 H200 上都更新到同一个提交：

```bash
git fetch origin && git checkout feature/rap_optimization && git pull
git log --oneline -1          # 记下这个提交号，发回时一起给
uv sync
```

Mac 上还需要 `espeak-ng`、`rubberband`、`ffmpeg`（`brew install espeak-ng rubberband ffmpeg`）。

### 2.2 可选：不连 H200 先在本机演练一遍

这一步用没有模型的替身服务器检查 Mac 这边的环境和脚本，大约 5 分钟。它不能代替验收。

```bash
# 终端 A：替身服务器，18641 是带延迟的端口
uv run python scripts/experiments/inference_opt/standin_server.py --port 18640 --delay-port 18641
# 终端 B
uv run python scripts/experiments/inference_milestone/launch_mac.py --port 8031 \
  --renderer http://127.0.0.1:18641 --audio-output wav --output output/rehearsal/client
sleep 15
uv run python scripts/experiments/inference_milestone/live_acceptance.py --url http://127.0.0.1:8031 \
  --cycles 3 --output output/rehearsal/cycle
uv run python scripts/analyze_inference_trace.py output/rehearsal/cycle/trace.json
```

正常的话最后一条命令会打印 WAIT 中位数约 1300 ms、`missing 0 ms`、`errors 0`。然后按第 7 节的方法停掉客户端，在终端 A 按 Ctrl-C。

### 2.3 H200 上启动服务

H200 是共享机器。**只动自己的进程，只按记录的 PID 停，不要按名字杀进程。**

```bash
nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv   # 先看 GPU 1、2 的占用
cd ~/mbzuai-projects/StreamMUSE-v1          # Bowen 的账号下的路径
touch /tmp/accept-start-$(id -u)             # 之后用它找这次验收产生的文件
CUDA_DEVICE_ORDER=PCI_BUS_ID bash scripts/model_upgrade_stack.sh start qwen25 mosslocal 2 1
```

脚本打印 `stack qwen25 (GPU 1) + mosslocal (GPU 2) ready` 才算启动成功，通常要几分钟。它的路径写死在 Bowen 的账号下（`/data/home/bowenzheng/...`）。如果你用自己的部署，要保证 render server 跑的是这个分支的代码，并且是在拉取代码之后重启过的。

确认 render server 是新代码：

```bash
curl -s http://127.0.0.1:8620/v1/rap/capabilities
```

返回的 `capabilities` 里必须有 `continuation_handoff_v1`。没有就说明服务器还是旧代码，第 2 阶段不会生效，先解决这个再往下做。

### 2.4 Mac 上建隧道

```bash
ssh -N -L 18620:127.0.0.1:8620 -L 18711:127.0.0.1:8601 <你的 H200 别名>
```

留着这个终端别关。下面所有命令都在 Mac 上、仓库根目录执行。先设两个变量：

```bash
export REN=http://127.0.0.1:18620
export OUT=output/inference_opt_accept
mkdir -p $OUT
curl -s $REN/v1/rap/capabilities      # 应该和 H200 上看到的一样
```

## 3. 每一轮测量前后都记一次链路往返

```bash
uv run python scripts/experiments/inference_opt/measure_round_trips.py --url $REN > $OUT/roundtrip-$(date +%H%M%S).json
```

第 4、5、6 节每节开始前和结束后各跑一次。往返时间决定了 WAIT 的大部分，没有它前后的数字没法比。

## 4. 对照组：改动前的客户端

用 git worktree 取出改动前的代码，只在 Mac 上跑旧客户端；H200 上的服务器不用换。

```bash
git worktree add /tmp/streammuse_before c90d0d8e
mkdir -p $OUT/client-before
( cd /tmp/streammuse_before && PYTHONPATH=/tmp/streammuse_before/src nohup "$OLDPWD/.venv/bin/python" \
    -m streammuse.presentation.rap_demo.cli \
    --rap-audio-renderer moss_aligned_remote --rap-protocol v3 --rap-audio-transport opus \
    --rap-render-url $REN --tempo 60 --audio-output wav --audio-device x --max-bars 0 --port 8034 \
    --log-dir "$OLDPWD/$OUT/client-before/sessions" --terminal-layout stream --terminal-detail summary \
    --interaction-mode continuous --response-bars 2 --rap-warp-policy all_onsets_r3 \
    --drum-gain 0.5 --vocal-gain 0.8 --chord-loop off --chord-gain 0 \
    > "$OLDPWD/$OUT/client-before/client.log" 2>&1 &
  echo $! > "$OLDPWD/$OUT/client-before/client.pid" )
sleep 20
curl -s http://127.0.0.1:8034/api/inference/state | python3 -c \
  "import json,sys; print(json.load(sys.stdin)['state'])"            # 应该打印 stopped
```

然后依次跑三项（每项之间等上一项打印出结果）：

```bash
uv run python scripts/experiments/inference_milestone/live_acceptance.py --url http://127.0.0.1:8034 \
  --cycles 5 --timeout 600 --output $OUT/before-cycle
uv run python scripts/experiments/inference_opt/steering_edits.py --url http://127.0.0.1:8034 \
  --bpm 60 --edits 6 --output $OUT/before-steering-60
uv run python scripts/experiments/inference_opt/steering_edits.py --url http://127.0.0.1:8034 \
  --bpm 90 --edits 6 --output $OUT/before-steering-90
```

停掉旧客户端，删掉 worktree：

```bash
kill -INT $(cat $OUT/client-before/client.pid); sleep 5
git worktree remove --force /tmp/streammuse_before
```

## 5. 第 1、2 阶段，以及 steering 的默认路径

启动新代码的客户端。不加参数时 steering 用的还是“每个请求一个短会话”（`bounded`）。

```bash
uv run python scripts/experiments/inference_milestone/launch_mac.py --port 8018 --renderer $REN \
  --audio-output wav --output $OUT/client-new
sleep 20
uv run python scripts/experiments/inference_milestone/live_acceptance.py --url http://127.0.0.1:8018 \
  --cycles 5 --timeout 600 --output $OUT/new-cycle
uv run python scripts/experiments/inference_opt/steering_edits.py --url http://127.0.0.1:8018 \
  --bpm 60 --edits 6 --output $OUT/new-bounded-steering-60
uv run python scripts/experiments/inference_opt/steering_edits.py --url http://127.0.0.1:8018 \
  --bpm 90 --edits 6 --output $OUT/new-bounded-steering-90
```

看结果：

```bash
uv run python scripts/analyze_inference_trace.py $OUT/before-cycle/trace.json | tee $OUT/before-cycle/analysis.txt
uv run python scripts/analyze_inference_trace.py $OUT/new-cycle/trace.json    | tee $OUT/new-cycle/analysis.txt
```

**第 1 阶段**看 `new-cycle` 的这两行：

- `WAIT  median ... ms`：低于 2000 算通过。同时和 `before-cycle` 的 WAIT 比一下。
- 第二行的 `missing 0 ms`、`errors 0`。

**第 2 阶段**看 `new-cycle` 的 `Continuation:` 这一行，例如 `9/9 bars (100.0%) continued the line before; 4 across a session boundary`：

- 百分比 95% 以上算通过。
- `across a session boundary` 应该等于回合数减 1（5 个回合就是 4）。如果是 0，说明续写没有跨过回合，看 `not continued:` 后面列的原因：
  - `no_previous_line`：服务器或客户端有一边是旧代码，回到 2.3 节检查。
  - `handoff_not_cached`：服务器找不到上一句的音频（服务器重启过、两个回合 BPM 不同，或缓存被挤掉）。
  - `disabled`：render server 启动时没带 `--lyric-continuation`。

按第 7 节的方法停掉这个客户端。

## 6. 第 3 阶段：长会话适配器

用 `--steering-adapter session` 重新启动客户端（换一个端口，避免上一个端口还没释放）：

```bash
uv run python scripts/experiments/inference_milestone/launch_mac.py --port 8019 --renderer $REN \
  --audio-output wav --steering-adapter session --output $OUT/client-session
sleep 20
curl -s http://127.0.0.1:8019/api/inference/state | python3 -c \
  "import json,sys; print(json.load(sys.stdin)['capabilities']['steering_adapter'])"     # 必须打印 session
uv run python scripts/experiments/inference_opt/steering_edits.py --url http://127.0.0.1:8019 \
  --bpm 60 --edits 6 --output $OUT/session-steering-60
uv run python scripts/experiments/inference_opt/steering_edits.py --url http://127.0.0.1:8019 \
  --bpm 90 --edits 6 --output $OUT/session-steering-90
```

对六个 steering 目录都跑一次分析：

```bash
for d in before-steering-60 before-steering-90 new-bounded-steering-60 new-bounded-steering-90 \
         session-steering-60 session-steering-90; do
  uv run python scripts/analyze_inference_trace.py $OUT/$d/trace.json > $OUT/$d/analysis.txt
  echo "== $d"; sed -n '/Steering edits/,/latency:/p' $OUT/$d/analysis.txt; grep -E "^eligible|^Continuation" $OUT/$d/analysis.txt
done
```

每次修改一行，例如：

```text
  version  bar  half    new version after  bars later  old-plan bars started after the edit
       11   15  second            1722 ms           1           0 ms
```

**第 3 阶段**看 `session-steering-60` 和 `session-steering-90`：

- `half` 是 `first` 的三行，`bars later` 必须是 1。
- `half` 是 `second` 的三行，`bars later` 是 1 或 2。
- `eligible ... missing 0 ms ... errors 0`。
- `Continuation:` 的百分比 95% 以上。

`before-*` 和 `new-bounded-*` 是对照，不设通过标准，只用来比较。

按第 7 节的方法停掉这个客户端。

## 7. 停止

Mac 客户端（把目录换成对应的那个）：

```bash
kill -INT $(python3 -c "import json; print(json.load(open('$OUT/client-new/client.pid.json'))['pid'])")
```

H200 上，先存一份现场再停：

```bash
cd ~/mbzuai-projects/StreamMUSE-v1
E=logs/model_upgrade_20261014/stack
nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv > /tmp/accept-gpu-after.csv
tail -n 400 $E/render.log > /tmp/accept-render-tail.log
# 这次验收里服务器合成的每个 bar 的原始人声，之后用来算说话人相似度
find $E/render_artifacts -name source.wav -newer /tmp/accept-start-$(id -u) > /tmp/accept-sources.txt
tar czf /tmp/accept-sources.tgz -T /tmp/accept-sources.txt
bash scripts/model_upgrade_stack.sh stop
```

`stop` 只会停它自己启动并记录下来的三个进程。

## 8. 要发回的东西

- Mac：整个 `output/inference_opt_accept/` 目录。WAV 模式下每个会话目录里的 `inference.wav` 可以不带：
  ```bash
  tar czf inference_opt_accept.tgz --exclude='*.wav' output/inference_opt_accept
  ```
- H200：`/tmp/accept-gpu-after.csv`、`/tmp/accept-render-tail.log`、`/tmp/accept-sources.tgz`，以及 2.3 节开始时那条 `nvidia-smi` 的输出。
- 两边的提交号（`git log --oneline -1`）。
- 验收当天有没有别人在用 GPU 1、2，隧道有没有断过。

## 9. 可选：实际听一次

上面都是 WAV 模式，不出声。要确认“两段之间声音变了”的问题是否消失，需要有人听：

```bash
uv run python scripts/experiments/inference_milestone/launch_mac.py --port 8020 --renderer $REN \
  --audio-output composite --device "<这台 Mac 的输出设备名>" --output $OUT/client-listen
```

打开 `http://127.0.0.1:8020/inference`，Cyclic 模式下按空格敲节奏，跑 5 个回合以上，听回合和回合之间声音是否一致。录音在 `$OUT/client-listen/sessions/<会话>/inference.wav`，这个文件请一起发回。最好请 Andrew 也听一次，上次是他报告的这个问题。

## 10. 出问题时

| 现象 | 先看什么 |
|---|---|
| `launch_mac.py` 报 `Address already in use` | 上一个客户端的端口还没释放，等几秒，或换一个 `--port` |
| `launch_mac.py` 报 `Recorded client still exists` | 这个输出目录对应的客户端还在跑，先按第 7 节停掉 |
| 验收脚本报 `Stop the existing performance` 或 `stop the current performance` | 页面上还在播放，先在页面上点 Stop，或重启客户端 |
| 页面或脚本报错后停住 | `$OUT/client-*/client.log` 的最后几十行，和 H200 上的 `render.log` |
| WAIT 没有变短 | `$OUT/roundtrip-*.json` 里复用连接的往返是多少；`new-cycle/analysis.txt` 里 `first bar's lyric at Mac` 是多少。歌词这一项应该接近一次往返加写词时间；如果仍然是两三秒，说明提前握手没生效，把 `client.log` 发回 |
| 需要退回旧行为 | Steering：不加 `--steering-adapter session` 就是旧适配器。服务器的 keep-alive：启动 render server 时加 `--http-keep-alive-seconds 5`。续写交接：只有服务器声明了 `continuation_handoff_v1` 才启用 |

任何一步结果不符合预期都不用自己排查到底，把现象和第 8 节的文件发回即可。

## 11. 这次验收不包括的

- 回合边界的说话人相似度：要用 7 节打包的 `source.wav` 在 Mac 上算，计算脚本还没写，拿到文件后再做。
- 把 steering 的默认适配器改成 `session`：验收通过后由 Bowen 决定。
- 70、110 这类 BPM 下 steering 的表现：服务层有一个已知的取整问题会让约一半的修改多晚一个 bar，和这次改动无关，60 和 90 BPM 不受影响。详见[合并计划](../plans/2026-11-11-rap-inference-optimization-and-carryover-plan.md)实施记录的第 3 阶段。
