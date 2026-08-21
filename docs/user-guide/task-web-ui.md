# 交互式任务 Web 观察界面

`streammuse-task play` 可以启用一个只读浏览器界面，用于显示当前回合、最近应答、判定结果和 deadline 时间轴。它同时支持 `zip_zap_zop` 与 `animal_naming`，但不接受浏览器作答或游戏命令。

## 启动

```bash
uv run --extra voice --extra speech streammuse-task play \
  --task animal_naming \
  --human-input voice \
  --speech-output audio \
  --deadline-mode soft \
  --deadline-ms 3000 \
  --web-ui \
  --web-port 8002 \
  --model-url http://127.0.0.1:8101/v1 \
  --model Qwen/Qwen3.6-27B
```

CLI 会先输出带随机 session token 的 URL：

```text
Task Web UI: http://127.0.0.1:8002/?token=...
Waiting for viewer...
```

打开该 URL 后，浏览器先建立 WebSocket、接收完整 boot snapshot、完成首次 DOM render，再发送 `viewer_ready`。只有这一步成功，CLI 才会构造和初始化 microphone/STT、TTS 与 LLM client，并输出：

```text
Viewer ready. Initializing game...
```

等待期间可按 `Ctrl-C` 取消；该局写入 `user_interrupt` startup manifest，并返回 130。

## 参数

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `--web-ui` | 关闭 | 启用浏览器观察界面，并将首个 viewer 作为强制开局门槛 |
| `--web-host` | `127.0.0.1` | HTTP/WebSocket 监听地址 |
| `--web-port` | `8002` | 监听端口 |
| `--web-allow-remote` | 关闭 | 允许非 loopback 监听；仍要求随机 token 与合法 Origin |

未启用 `--web-ui` 时传入其他 `--web-*` 参数会被拒绝。端口已占用、静态资源或 uvicorn 启动失败都会在第一回合前终止，不会无界面降级开局。

## 时间轴语义

- 时间轴从运行时真实 deadline 起点开始，不包含人类回合前的 speech guard 或 prompt build。
- `:hint`、`:expected` 等终端命令产生新的 attempt，时间轴会重置。
- 时间轴不显示精确数字，也不参与计分；权威结果仍是 trace 中的 `latency_ms` 与 `deadline_missed`。
- soft deadline 超时后时间轴进入红色 overrun 状态，而不是停在零。
- challenge stage 切换使用服务端事件更新，不从终端文本解析。

## 断线与重连

首个 `viewer_ready` 释放开局门槛后，浏览器断线不会暂停或改变对局。每次连接都会先收到服务端维护的完整 snapshot，因此第二个 viewer 或重连 viewer 可以恢复当前回合、stage、累计统计、最后应答和 ASR/TTS 状态。有界队列溢出时会丢弃旧 backlog，并以最新完整 snapshot 恢复一致状态。

## 安全与隐私

- 动态 ASR/LLM 文本只通过 `textContent` 写入页面；页面配置 CSP，WebSocket 校验随机 token 与 `Origin`。
- 默认只绑定 loopback。远程绑定必须显式使用 `--web-allow-remote`，并通过受信网络或 SSH tunnel 访问。
- 页面会显示 raw ASR transcript，可能包含误收录的私人语音。不要在屏幕共享或远程绑定时暴露不应公开的内容。
- token 只保护当前运行中的 WebSocket session；不要把终端打印的完整 URL 发给不受信任的人。

## 产物与诊断

成功启用时，`manifest.json` 的 `task_web` 记录 host、port、session ID 和 remote policy。`run_summary.json` 与 manifest 的 `task_event_error_count` 记录开局后 observer 事件投递异常；该计数大于零表示页面可能缺失更新，但不会改变游戏判定或 trace。
