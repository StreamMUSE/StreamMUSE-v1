#!/usr/bin/env python3
"""Generate one system diagram (SVG) per tagged RAP version into this directory.

Run: python3 docs/architecture/rap-versions/build_diagrams.py
"""

from __future__ import annotations

from html import escape
from pathlib import Path

OUT = Path(__file__).resolve().parent
SLIDES_PATH = OUT.parents[2] / "developing-logs" / "slides" / "2026-10-22-rap-system-versions-slides.html"
VERSIONS = ("v0.0", "v0.1", "v0.2", "v1.0", "v1.1")

BLUE = "#2F6BDB"
PURPLE = "#7448D1"
LILAC = "#9B7FE0"
RED = "#D9483B"
AMBER = "#D08A00"
GREEN = "#2E9B5F"
GRAY = "#6B7180"
LINE = "#A9B3C6"
NEW = "#E8590C"
REST = "#C9C3D6"

# Timeline scale: one 90 BPM bar (2.667 s) is 270 px wide.
TL_X0 = 240
BAR_W = 270
PX_PER_S = BAR_W / (60 / 90 * 4)
STAGE_PX_PER_S = 60
OFFSET = 70  # vertical space taken by the title and the change strip

STYLE = """
    text { font-family: "PingFang SC", "Hiragino Sans GB", -apple-system, "Helvetica Neue", Arial, sans-serif; fill: #1F2430; }
    .title { font-size: 28px; font-weight: 700; }
    .subtitle { font-size: 15px; fill: #5B6170; }
    .panel-title { font-size: 19px; font-weight: 700; }
    .panel-tag { font-size: 12px; font-weight: 700; letter-spacing: 1px; fill: #FFFFFF; }
    .box-title { font-size: 16px; font-weight: 600; }
    .box-sub { font-size: 13px; fill: #5B6170; }
    .small { font-size: 12px; fill: #6B7180; }
    .label { font-size: 13px; font-weight: 600; }
    .time { font-size: 13px; font-weight: 600; fill: #3A3F4B; }
    .section-title { font-size: 18px; font-weight: 700; }
    .change { font-size: 14px; fill: #7A3A0C; }
    .pill { font-size: 13px; font-weight: 700; }
"""


def _marker(name: str, color: str) -> str:
    return (
        f'<marker id="arrow-{name}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="8" '
        f'markerHeight="8" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="{color}"/></marker>'
    )


DEFS = (
    "<defs><style>"
    + STYLE
    + "</style>"
    + "".join(
        _marker(n, c)
        for n, c in (("gray", GRAY), ("blue", BLUE), ("purple", PURPLE), ("green", GREEN), ("amber", AMBER), ("line", LINE))
    )
    + '<filter id="shadow" x="-5%" y="-5%" width="110%" height="120%">'
    '<feDropShadow dx="0" dy="1.5" stdDeviation="1.5" flood-color="#1F2430" flood-opacity="0.10"/></filter>'
    "</defs>"
)


class Svg:
    def __init__(self, width: int, height: int) -> None:
        self.width = width
        self.height = height
        self.parts: list[str] = []

    def add(self, raw: str) -> None:
        self.parts.append(raw)

    def text(self, x: float, y: float, s: str, cls: str = "", anchor: str | None = None, fill: str | None = None, weight: int | None = None) -> None:
        attrs = f' class="{cls}"' if cls else ""
        if anchor:
            attrs += f' text-anchor="{anchor}"'
        style = []
        if fill:
            style.append(f"fill:{fill}")
        if weight:
            style.append(f"font-weight:{weight}")
        if style:
            attrs += f' style="{";".join(style)}"'
        self.add(f'<text x="{x:.1f}" y="{y:.1f}"{attrs}>{escape(s)}</text>')

    def rect(self, x: float, y: float, w: float, h: float, fill: str = "#FFFFFF", stroke: str | None = None, rx: float = 10, dash: str | None = None, shadow: bool = False, width: float = 1.0) -> None:
        attrs = f'x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="{rx}" fill="{fill}"'
        if stroke:
            attrs += f' stroke="{stroke}" stroke-width="{width}"'
        if dash:
            attrs += f' stroke-dasharray="{dash}"'
        if shadow:
            attrs += ' filter="url(#shadow)"'
        self.add(f"<rect {attrs}/>")

    def path(self, d: str, color: str, marker: str | None, width: float = 1.8, dash: str | None = None, both: bool = False) -> None:
        attrs = f'd="{d}" fill="none" stroke="{color}" stroke-width="{width}"'
        if dash:
            attrs += f' stroke-dasharray="{dash}"'
        if marker:
            attrs += f' marker-end="url(#arrow-{marker})"'
            if both:
                attrs += f' marker-start="url(#arrow-{marker})"'
        self.add(f"<path {attrs}/>")

    def badge(self, right: float, y: float, label: str, color: str) -> float:
        w = 14 + 8 * len(label) if label.isascii() else 16 + 13 * len(label)
        x = right - w
        self.rect(x, y, w, 20, fill=color, rx=10)
        self.text(x + w / 2, y + 14, label, "small", anchor="middle", fill="#FFFFFF", weight=700)
        return x - 6

    def box(self, x: float, y: float, w: float, h: float, title: str, subs: tuple[str, ...] = (), stroke: str = PURPLE, fill: str = "#FFFFFF", badges: tuple[tuple[str, str], ...] = (), dash: str | None = None, title_fill: str | None = None, sub_fill: str | None = None, small: bool = False) -> None:
        self.rect(x, y, w, h, fill=fill, stroke=stroke, dash=dash, shadow=dash is None, width=1.8 if stroke == RED else 1.0)
        pad = 14 if small else 16
        self.text(x + pad, y + (23 if small else 26), title, "label" if small else "box-title", fill=title_fill)
        first = y + (43 if small else 48)
        for i, s in enumerate(subs):
            self.text(x + pad, first + i * (16 if small else 18), s, "small" if small else "box-sub", fill=sub_fill)
        right = x + w - 10
        for label, color in badges:
            right = self.badge(right, y + 8, label, color)

    def render(self) -> str:
        body = "\n  ".join(self.parts)
        return (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{self.width}" height="{self.height}" '
            f'viewBox="0 0 {self.width} {self.height}">\n  {DEFS}\n  '
            f'<rect width="{self.width}" height="{self.height}" fill="#FFFFFF"/>\n  {body}\n</svg>\n'
        )


def header(svg: Svg, cfg: dict) -> None:
    svg.text(40, 52, f"StreamMUSE 实时 Rap · {cfg['version']}", "title")
    svg.text(40, 82, cfg["subtitle"], "subtitle")
    # Version navigation pills, current one highlighted.
    x = 1400 - len(VERSIONS) * 72 + 6
    for v in VERSIONS:
        current = v == cfg["version"]
        svg.rect(x, 30, 66, 28, fill=BLUE if current else "#FFFFFF", stroke=BLUE, rx=14)
        svg.text(x + 33, 49, v, "pill", anchor="middle", fill="#FFFFFF" if current else BLUE)
        x += 72
    svg.rect(40, 96, 1360, 58, fill="#FFF4EC", stroke="#F3C6A5", rx=10)
    svg.rect(56, 108, 74, 22, fill=NEW, rx=11)
    svg.text(93, 124, "本版变化", "small", anchor="middle", fill="#FFFFFF", weight=700)
    lines = cfg["changes"]
    ys = (125,) if len(lines) == 1 else (124, 145)
    if len(lines) == 1:
        ys = (131,)
    for line, y in zip(lines, ys):
        svg.text(144, y, line, "change")


def mac_frame(svg: Svg) -> None:
    svg.rect(40, 110, 620, 625, fill="#F1F6FF", stroke=BLUE, rx=16, width=1.5)
    svg.rect(64, 128, 56, 22, fill=BLUE, rx=11)
    svg.text(92, 144, "MAC", "panel-tag", anchor="middle")
    svg.text(132, 146, "本地：时钟 · 兜底 · 混音播放", "panel-title")
    svg.box(64, 170, 360, 66, "① 节拍时钟 Rap Clock", ("90 BPM · 每小节 16 tick · 唯一的时间权威",), stroke=BLUE)
    svg.path("M244,236 L244,260", GRAY, "gray")


def h200_frame(svg: Svg, title: str, stages: list[dict], timing_note: str, bottom_left: str, bottom_right: str) -> None:
    svg.rect(780, 110, 620, 625, fill="#F6F2FF", stroke=PURPLE, rx=16, width=1.5)
    svg.rect(804, 128, 62, 22, fill=PURPLE, rx=11)
    svg.text(835, 144, "H200", "panel-tag", anchor="middle")
    svg.text(878, 146, title, "panel-title")
    svg.text(1196, 164, timing_note, "small")
    for i, st in enumerate(stages):
        y = 170 + i * 86
        badges = []
        if st.get("new"):
            badges.append(("NEW", NEW))
        if st.get("bottleneck"):
            badges.append(("瓶颈", RED))
        ghost = st.get("ghost", False)
        svg.box(
            804, y, 376, 68, st["title"], (st["sub"],),
            stroke=RED if st.get("bottleneck") else (LINE if ghost else PURPLE),
            fill="#FFEDEA" if st.get("bottleneck") else ("#F7F7F9" if ghost else "#FFFFFF"),
            dash="5 4" if ghost else None,
            title_fill=GRAY if ghost else None,
            badges=tuple(badges),
        )
        if i < len(stages) - 1:
            svg.path(f"M992,{y + 68} L992,{y + 84}", GRAY, "gray")
        if st.get("time"):
            color = RED if st.get("bottleneck") else "#3A3F4B"
            svg.text(1196, y + 28, st["time"], "time", fill=color)
            if st.get("seconds"):
                w = max(3.0, st["seconds"] * STAGE_PX_PER_S)
                svg.rect(1196, y + 36, w, 10, fill=RED if st.get("bottleneck") else PURPLE, rx=3)
    svg.add('<line x1="1196" y1="684" x2="1376" y2="684" stroke="#C9BEEA"/>')
    svg.text(1196, 708, bottom_right, "label")
    svg.text(820, 708, bottom_left, "small")


def tunnel(svg: Svg) -> None:
    svg.rect(672, 176, 96, 490, fill="#F7F7F9", stroke=LINE, dash="6 4", rx=12)
    svg.text(720, 690, "SSH 隧道", "label", anchor="middle")
    svg.text(720, 708, "只转发 :8020", "small", anchor="middle")


def _text_px(s: str) -> float:
    return sum(7.5 if c.isascii() else 13 for c in s)


def chip(svg: Svg, x: float, y: float, w: float, label: str, kind: str) -> None:
    fill, stroke, color = {
        "good": ("#E7F6EE", GREEN, "#1F7A48"),
        "warn": ("#FFF6E5", AMBER, "#9A6500"),
        "bad": ("#FFEDEA", RED, "#B3362B"),
        "info": ("#FFFFFF", LINE, "#3A3F4B"),
    }[kind]
    svg.rect(x, y, w, 34, fill=fill, stroke=stroke, rx=17)
    svg.text(x + w / 2, y + 22, label, "label", anchor="middle", fill=color)


def playback_row(svg: Svg, y: float, solid: int) -> None:
    svg.text(64, y + 25, "Mac 正在播放", "label")
    for i in range(4):
        x = TL_X0 + i * BAR_W
        name = "bar N" if i == 0 else f"bar N+{i}"
        if i < solid:
            svg.rect(x, y, BAR_W - 2, 40, fill=BLUE, rx=6)
            svg.text(x + BAR_W / 2, y + 25, name, "label", anchor="middle", fill="#FFFFFF")
        else:
            svg.rect(x, y, BAR_W - 2, 40, stroke=BLUE, dash="5 3", rx=6)
            svg.text(x + BAR_W / 2, y + 25, f"{name}（正在准备）", "label", anchor="middle", fill=BLUE)


# ---------------------------------------------------------------- v0.x layout


def chunk_diagram(cfg: dict) -> str:
    svg = Svg(1440, 1080 + OFFSET)
    header(svg, cfg)
    svg.add(f'<g transform="translate(0,{OFFSET})">')

    mac_frame(svg)
    svg.box(448, 170, 188, 66, "实时监控", ("Terminal 面板 · Web :8012", "Start / Stop / Reset"), stroke=LINE, dash="4 3", small=True)
    svg.path("M424,203 L446,203", LINE, "line", 1.5, "3 3")
    svg.box(64, 262, 360, 88, "② 滚动调度器 Rolling Controller",
            ("为未来 2 个小节打包一个请求：", "主题 · flow 模板（音节槽位 + 重音）", "前 4 行歌词 · 剩余时间预算"), stroke=BLUE)
    svg.path("M244,350 L244,397", GRAY, "gray")
    svg.add('<polygon points="244,399 394,455 244,511 94,455" fill="#FFFFFF" stroke="#2F6BDB" stroke-width="1.5" filter="url(#shadow)"/>')
    svg.text(244, 451, "③ 截止前到达", "box-title", anchor="middle")
    svg.text(244, 473, "且校验通过？", "box-title", anchor="middle")

    dec = cfg["mac_decode"]
    svg.box(448, 417, 188, 76, dec["title"], dec["subs"], stroke=PURPLE, small=True,
            badges=(("NEW", NEW),) if dec.get("new") else ())
    svg.path("M448,455 L398,455", PURPLE, "purple")

    svg.box(448, 522, 188, 88, "兜底 Fallback", ("预验证歌词库 + 本地 eSpeak", "请求发出前就已备好", "（保证节奏不断）"), stroke=AMBER, fill="#FFF6E5", small=True)
    svg.path("M319,483 L444,548", AMBER, "amber")
    svg.text(400, 508, "否 / 超时", "label", fill="#B37600")
    svg.path("M244,511 L244,543", GREEN, "green")
    svg.text(254, 532, "是 → 用服务器人声", "label", fill=GREEN)
    svg.box(64, 545, 360, 60, "④ 冻结小节 Freeze", ("冻结后不可再改，后到的结果直接丢弃",), stroke=BLUE)
    svg.path("M448,590 L428,590", AMBER, "amber")
    svg.path("M244,605 L244,633", GRAY, "gray")
    svg.box(64, 635, 360, 70, "⑤ 本地鼓 + 混音 → 扬声器", ("人声与鼓按 tick 精确对齐后播放并录制",), stroke=BLUE, fill=BLUE, title_fill="#FFFFFF", sub_fill="#DCE7FF")
    svg.box(448, 635, 188, 70, "会话产物 logs/rap/", ("mixed.wav · events.jsonl", "bars.csv · summary.json"), stroke=LINE, dash="4 3", small=True)
    svg.path("M424,670 L446,670", LINE, "line", 1.5, "3 3")

    tunnel(svg)
    svg.path("M424,290 H704 V204 H800", BLUE, "blue", 2.2)
    svg.text(436, 262, "POST /v1/rap/chunks/render", "small")
    svg.text(436, 282, f"请求：未来 2 小节（{cfg['protocol']}）", "label", fill=BLUE)
    svg.path("M804,634 H736 V455 H640", PURPLE, "purple", 2.2)
    svg.rect(672, 520, 96, 40, stroke=PURPLE, rx=6)
    svg.text(720, 537, "返回 Opus 包", "small", anchor="middle", fill=PURPLE, weight=600)
    svg.text(720, 553, cfg["response_note"], "small", anchor="middle", fill=PURPLE)

    h200_frame(svg, "GPU 服务器 :8020：写词 → 唱 → 对拍", cfg["stages"], "real-time 中位数", cfg["h200_note"], cfg["wait_label"])

    # Timeline
    svg.rect(40, 760, 1360, 296, fill="#FAFAFB", stroke="#E3E5EA", rx=16)
    svg.text(64, 795, "时间线：一个请求必须在 2 个小节内回来", "section-title")
    svg.text(420, 795, "（90 BPM，1 小节 ≈ 2.67 s，图中 1 秒 ≈ 101 px）", "small")
    playback_row(svg, 820, 2)
    svg.text(64, 902, "Mac 等一个请求", "label")
    svg.text(64, 919, "bar N 开始前发出", "small")
    x = TL_X0
    for label, seconds, color in cfg["segments"]:
        w = seconds * PX_PER_S
        svg.rect(x, 886, w, 40, fill=color, rx=6)
        dark = color in (PURPLE, RED, LILAC)
        if w >= _text_px(label) + 16:
            svg.text(x + w / 2, 911, label, "label", anchor="middle", fill="#FFFFFF" if dark else None)
        else:
            svg.text(x, 880, label, "small", fill=GRAY, weight=600)
        x += w
    svg.text(x, 880, cfg["wait_total"], "label", anchor="end")
    deadline = TL_X0 + 5.0 * PX_PER_S
    svg.add(f'<line x1="{deadline:.1f}" y1="810" x2="{deadline:.1f}" y2="940" stroke="{RED}" stroke-width="2" stroke-dasharray="6 4"/>')
    svg.text(deadline, 956, "截止 ≈ 5.0 s", "label", anchor="middle", fill=RED)
    if cfg.get("timeline_note"):
        svg.text(deadline + 64, 956, cfg["timeline_note"], "small", fill=RED, weight=600)

    cx = TL_X0
    for label, w, kind in cfg["results"]:
        chip(svg, cx, 972, w, label, kind)
        cx += w + 16
    for i, line in enumerate(cfg["footnote"]):
        svg.text(64, 1028 + i * 18, line, "small")
    svg.add("</g>")
    return svg.render()


# ---------------------------------------------------------------- v1.x layout


def stream_diagram(cfg: dict) -> str:
    svg = Svg(1440, 1080 + OFFSET)
    header(svg, cfg)
    svg.add(f'<g transform="translate(0,{OFFSET})">')

    mac_frame(svg)
    svg.box(448, 170, 188, 66, "实时监控 + 修改", ("Web :8012 · 改主题 / flow", "POST /api/control/plan"), stroke=LINE, dash="4 3", small=True,
            badges=(("NEW", NEW),) if cfg.get("mac_new") else ())
    svg.path("M424,203 L446,203", LINE, "line", 1.5, "3 3")
    svg.box(64, 262, 360, 88, "② 逐 bar 控制器 BarStream",
            ("收到歌词 → 取接下来 2 个 bar 的音频", "回报播放头（committed）", "把听众的修改（update）发给服务器"), stroke=BLUE,
            badges=(("NEW", NEW),) if cfg.get("mac_new") else ())
    svg.path("M244,350 L244,378", GRAY, "gray")

    # Commit-time selection order.
    svg.rect(64, 380, 360, 150, stroke=BLUE, shadow=True)
    svg.text(80, 406, "③ 提交时按顺序选择", "box-title")
    rows = (
        (GREEN, "1  最新版本的服务器人声", "remote_latest"),
        (BLUE, "2  旧版本人声（修改来不及时）", "remote_older"),
        (AMBER, "3  服务器歌词 + 本地 eSpeak", "espeak_stream"),
        ("#B36B00", "4  本地歌词 + 本地 eSpeak", "espeak_local"),
    )
    for i, (color, label, key) in enumerate(rows):
        y = 432 + i * 24
        svg.add(f'<circle cx="86" cy="{y - 4}" r="5" fill="{color}"/>')
        svg.text(98, y, label, "box-sub", fill="#1F2430")
        svg.text(410, y, key, "small", anchor="end")
    if cfg.get("mac_new"):
        svg.badge(414, 388, "NEW", NEW)

    svg.box(448, 417, 188, 76, "解码 Opus + 本地 R3", ("每个 bar 单独取、单独对拍", "解码 · 对拍 · 混音 ≈ 35 ms"), stroke=PURPLE, small=True)
    svg.path("M448,455 L428,455", PURPLE, "purple")
    svg.box(448, 522, 188, 88, "本地兜底", ("本地 eSpeak + 本地歌词", "只在没有服务器音频时用", "（对应第 3、4 级）"), stroke=AMBER, fill="#FFF6E5", small=True)
    svg.path("M448,566 L428,510", AMBER, "amber")
    svg.path("M244,530 L244,543", GRAY, "gray")
    svg.box(64, 545, 360, 60, "④ 冻结小节 Freeze", ("冻结后不可再改；听众修改只影响未冻结的 bar",), stroke=BLUE)
    svg.path("M244,605 L244,633", GRAY, "gray")
    svg.box(64, 635, 360, 70, "⑤ 本地鼓 + 混音 → 扬声器", ("人声与鼓按 tick 精确对齐后播放并录制",), stroke=BLUE, fill=BLUE, title_fill="#FFFFFF", sub_fill="#DCE7FF")
    svg.box(448, 635, 188, 70, "会话产物 logs/rap/", ("mixed.wav · events.jsonl", "bars.csv · plan_edited 事件"), stroke=LINE, dash="4 3", small=True)
    svg.path("M424,670 L446,670", LINE, "line", 1.5, "3 3")

    tunnel(svg)
    # WebSocket: Mac controller <-> lyric record.
    svg.path("M428,290 H800", BLUE, "blue", 2.2, both=True)
    svg.text(436, 262, "WebSocket /v1/rap/sessions/{id}", "small")
    svg.text(436, 282, "歌词流（双向）", "label", fill=BLUE)
    svg.text(436, 310, "→ start · update · committed · stop", "small")
    svg.text(436, 326, "← lyric · superseded · audio_ready", "small")
    # Long-poll audio fetch.
    svg.path("M636,440 H704 V620 H800", BLUE, "blue", 1.8, dash="6 4")
    svg.path("M804,648 H736 V470 H640", PURPLE, "purple", 2.2)
    svg.rect(676, 360, 88, 56, stroke=BLUE, rx=6)
    svg.text(720, 378, "GET …/bars/", "small", anchor="middle", fill=BLUE, weight=600)
    svg.text(720, 393, "{bar}/audio", "small", anchor="middle", fill=BLUE, weight=600)
    svg.text(720, 408, "长轮询", "small", anchor="middle", fill=BLUE)
    svg.rect(672, 520, 96, 40, stroke=PURPLE, rx=6)
    svg.text(720, 537, "返回 Opus 包", "small", anchor="middle", fill=PURPLE, weight=600)
    svg.text(720, 553, "1 个 bar", "small", anchor="middle", fill=PURPLE)

    h200_frame(svg, "GPU 服务器 :8020：歌词流 + 逐 bar 合成", cfg["stages"], cfg["timing_note"], cfg["h200_note"], cfg["wait_label"])

    # Timeline
    svg.rect(40, 760, 1360, 296, fill="#FAFAFB", stroke="#E3E5EA", rx=16)
    svg.text(64, 795, "时间线：歌词提前 3 个 bar 写好，音频提前 2 个 bar 合成好", "section-title")
    svg.text(600, 795, "（90 BPM，1 小节 ≈ 2.67 s）", "small")
    playback_row(svg, 812, 1)
    playhead = TL_X0 + BAR_W / 3
    svg.add(f'<line x1="{playhead:.1f}" y1="806" x2="{playhead:.1f}" y2="958" stroke="#1F2430" stroke-width="2"/>')
    svg.text(playhead + 6, 866, "播放头", "small", fill="#1F2430", weight=700)

    svg.text(64, 885, "服务器歌词", "label")
    svg.text(64, 902, "写到播放头前 3 bar", "small")
    svg.text(64, 937, "服务器音频", "label")
    svg.text(64, 954, "合成到播放头前 2 bar", "small")
    for i in range(1, 4):
        x = TL_X0 + i * BAR_W + 10
        svg.rect(x, 866, BAR_W - 22, 30, fill="#E3ECFF", stroke=BLUE, rx=6)
        svg.text(x + (BAR_W - 22) / 2, 886, f"N+{i} 歌词已推给 Mac", "small", anchor="middle", fill=BLUE, weight=600)
        if i < 3:
            svg.rect(x, 918, BAR_W - 22, 30, fill=PURPLE, rx=6)
            svg.text(x + (BAR_W - 22) / 2, 938, f"N+{i} 音频已合成（续写 N+{i - 1}）" if i > 1 else "N+1 音频已合成（续写 N）", "small", anchor="middle", fill="#FFFFFF", weight=600)
        else:
            svg.rect(x, 918, BAR_W - 22, 30, stroke=PURPLE, dash="5 3", rx=6)
            svg.text(x + (BAR_W - 22) / 2, 938, "N+3 等播到 N+1 再合成", "small", anchor="middle", fill=PURPLE, weight=600)

    cx = TL_X0
    for label, w, kind in cfg["results"]:
        chip(svg, cx, 972, w, label, kind)
        cx += w + 16
    for i, line in enumerate(cfg["footnote"]):
        svg.text(64, 1028 + i * 18, line, "small")
    svg.add("</g>")
    return svg.render()


# ---------------------------------------------------------------- versions

DEMO_PAGE = "数据：2026-10-12 版本对比 demo 页（同一张 H200 + SSH 隧道，15 段 × 16 bar，90 BPM，real-time 中位数）。"
DEADLINE = "截止 = min(请求发出后 5.0 s, bar N+2 开始前 1 tick)。"

CONFIGS = {
    "v0.0": dict(
        version="v0.0",
        subtitle="tag rap-v0.0 · commit 0a493cb8 · 2026-08-30 · 两小节 chunk · 协议 v1 · 进程内 HF MOSS · 服务器端 R3",
        changes=("第一个远端版本：H200 上用进程内 HF Transformers 跑 MOSS，一次合成两小节人声，在服务器端做 R3 对拍，再用 Opus 回传；",
                 "Mac 负责时钟、eSpeak 兜底和播放。"),
        protocol="协议 v1",
        response_note="2 小节·已对拍",
        mac_decode=dict(title="解码 Opus → PCM", subs=("校验 manifest · 时长 · hash", "直接可播（已在服务器对拍）")),
        stages=[
            dict(title="A  Qwen2.5-7B 写歌词候选", sub="vLLM · 两个 bar 依次写（初始 + 补救）", time="0.35 s", seconds=0.348),
            dict(title="B  过滤 + 打分，选出最佳两行", sub="音节数必须精确匹配 → 重音 · 断句 · 押韵 · 主题"),
            dict(title="C  MOSS-TTS v1.5（进程内 HF）", sub="model.generate() · 两小节连成一句 → source.wav", time="2.77 s", seconds=2.765, bottleneck=True),
            dict(title="D  MMS 强制对齐", sub="找出每个音节在音频里的起点", time="≈ 0.02 s", seconds=0.02),
            dict(title="E  Rubber Band R3 对拍（服务器）", sub="all_onsets_r3：每个音节起点都对齐拍点", time="≈ 0.09 s", seconds=0.09),
            dict(title="F  打包 + Opus 48 kbps", sub="长度恰好 2 小节的 vocal.wav · 产物缓存", time="≈ 0.05 s", seconds=0.05),
        ],
        h200_note="vLLM 与 MOSS 都只绑定 127.0.0.1",
        wait_label="Mac 等一个请求 4.67 s",
        segments=[("写词", 0.348, LILAC), ("MOSS 2.77 s", 2.765, RED), ("其余 + 网络 1.55 s", 1.554, REST)],
        wait_total="共 4.67 s",
        timeline_note="只剩 ≈ 0.3 s 余量；一次超时还会让下一个请求立刻失败（连锁）",
        results=[("real-time 服务器人声 24.2%", 260, "bad"), ("fake real-time 95.8%", 200, "info"),
                 ("兜底 182 bar，其中来不及 178", 260, "warn"), ("开场等待 5.8 s", 150, "info")],
        footnote=(DEMO_PAGE + DEADLINE, "本版默认的 gentle_sparse_r3 会被它自己的 Mac 客户端全部拒收（v0.2 修复），所以 demo 用 all_onsets_r3。D–F 的耗时取自 09-03 计划里的 warm baseline。"),
    ),
    "v0.1": dict(
        version="v0.1",
        subtitle="tag rap-v0.1 · commit 9b76415e · 2026-10-01 · 两小节 chunk · 协议 v1 · SGLang-Omni MOSS（未打补丁）· 服务器端 R3",
        changes=("MOSS 改由独立的 SGLang-Omni 服务提供（未打补丁），render server 通过 HTTP 调用：MOSS 每个请求 2.77 s → 0.83 s；",
                 "Mac 端和其余流程与 v0.0 相同。"),
        protocol="协议 v1",
        response_note="2 小节·已对拍",
        mac_decode=dict(title="解码 Opus → PCM", subs=("校验 manifest · 时长 · hash", "直接可播（已在服务器对拍）")),
        stages=[
            dict(title="A  Qwen2.5-7B 写歌词候选", sub="vLLM（seqs 8）· 两个 bar 依次写", time="0.32 s", seconds=0.323),
            dict(title="B  过滤 + 打分，选出最佳两行", sub="音节数必须精确匹配 → 重音 · 断句 · 押韵 · 主题"),
            dict(title="C  MOSS-TTS v1.5 @ SGLang-Omni", sub="独立服务 :8030（未打补丁）· 两小节一句", time="0.83 s", seconds=0.833, new=True),
            dict(title="D  MMS 强制对齐", sub="找出每个音节在音频里的起点", time="≈ 0.02 s", seconds=0.02),
            dict(title="E  Rubber Band R3 对拍（服务器）", sub="all_onsets_r3：每个音节起点都对齐拍点", time="≈ 0.09 s", seconds=0.09),
            dict(title="F  打包 + Opus（level 10）", sub="长度恰好 2 小节的 vocal.wav · 产物缓存", time="Opus ≈ 0.13 s", seconds=0.129),
        ],
        h200_note="vLLM · SGLang-Omni · render server 都只绑定 127.0.0.1",
        wait_label="Mac 等一个请求 2.67 s",
        segments=[("写词", 0.323, LILAC), ("MOSS", 0.833, PURPLE), ("其余 + 网络 1.51 s", 1.511, REST)],
        wait_total="共 2.67 s",
        timeline_note="大部分能按时到；网络往返 ≈ 1.0 s 成了大头",
        results=[("real-time 服务器人声 92.5%", 260, "good"), ("fake real-time 97.5%", 200, "info"),
                 ("兜底 18：来不及 12 · 渲染失败 4 · 校验 2", 330, "warn"), ("开场等待 3.7 s", 150, "info")],
        footnote=(DEMO_PAGE + DEADLINE, "demo 同样用 all_onsets_r3（gentle_sparse_r3 会被本版 Mac 拒收）。D、E 沿用 09-03 baseline，Opus 编码取自 10-01 报告的“改动前”。"),
    ),
    "v0.2": dict(
        version="v0.2",
        subtitle="tag rap-v0.2 · commit 0ea74a7f · 2026-10-01 · 两小节 chunk · 协议 v2 · SGLang-Omni + 延迟补丁 · Mac 端 R3",
        changes=("R3 对拍移到 Mac（协议 v2，gentle_sparse_r3 终于能用）· 两个 bar 并发写词（vLLM --max-num-seqs 32）·",
                 "SGLang-Omni MOSS 延迟补丁（输出逐字节不变）· Opus 压缩级别 10 → 5。服务器快了，但网络仍是大头。"),
        protocol="协议 v2",
        response_note="人声 + onsets",
        mac_decode=dict(title="解码 → 本地 R3", subs=("gentle_sparse_r3 · ≈ 0.05 s", "共享 phrase_warp 建锚点"), new=True),
        stages=[
            dict(title="A  Qwen2.5-7B 写歌词候选", sub="两个 bar 并发写（vLLM seqs 32）", time="0.12 s", seconds=0.119, new=True),
            dict(title="B  过滤 + 打分，选出最佳两行", sub="音节数必须精确匹配 → 重音 · 断句 · 押韵 · 主题"),
            dict(title="C  MOSS-TTS v1.5 @ SGLang-Omni", sub=":8030 · 打了延迟补丁（输出逐字节不变）", time="0.77 s", seconds=0.765, new=True),
            dict(title="D  MMS 强制对齐", sub="每个音节的 source onset + 置信度", time="≈ 0.02 s", seconds=0.02),
            dict(title="E  R3 对拍 → 已移到 Mac", sub="服务器不再拉伸，只下发原始人声", ghost=True),
            dict(title="F  打包 + Opus（level 5）", sub="原始人声 PCM + 每个音节 onset", time="Opus ≈ 0.06 s", seconds=0.06, new=True),
        ],
        h200_note="vLLM · SGLang-Omni · render server 都只绑定 127.0.0.1",
        wait_label="Mac 等一个请求 2.62 s",
        segments=[("写词", 0.119, LILAC), ("MOSS", 0.765, PURPLE), ("其余 + 网络 + Mac R3", 1.733, REST)],
        wait_total="共 2.62 s",
        timeline_note="服务器省下的时间被网络吃掉：2.67 s → 2.62 s",
        results=[("real-time 服务器人声 92.5%", 260, "good"), ("fake real-time 97.5%", 200, "info"),
                 ("兜底 18：来不及 12 · 写词失败 6", 280, "warn"), ("开场等待 3.6 s", 150, "info")],
        footnote=(DEMO_PAGE + DEADLINE, "Opus 编码与 Mac 端 R3 耗时取自 10-01 延迟优化实施报告。"),
    ),
    "v1.0": dict(
        version="v1.0",
        subtitle="tag rap-v1.0 · commit 19526512 · 2026-10-12 · 协议 v3 · 歌词流 + 逐 bar MOSS 续写 · 每 bar 32 条候选",
        changes=("架构重写（协议 v3）：服务器持续写歌词流（播放头前 3 bar）并逐 bar 续写合成（前 2 bar），Mac 按歌词 ID 取音频；",
                 "支持实时修改主题 / flow；修复歌词结尾念出 “A” 的问题。不再有“一个请求包两小节”的截止竞赛。"),
        mac_new=True,
        timing_note="real-time 中位数",
        stages=[
            dict(title="A  写词线程 · Qwen2.5-7B", sub="每 bar 32 条候选 → 过滤 → 选一句（对句押韵）", time="≈ 0.1 s / 行", seconds=0.1, new=True),
            dict(title="B  歌词记录 BarLyricRecord", sub="lyric_id = sha256 · 版本号 · 修改后版本 +1", new=True),
            dict(title="C  合成线程 · MOSS-TTS v1.5 续写", sub="SGLang-Omni（延迟 + 续写补丁）· 上一 bar 做前缀", time="0.62 s / bar", seconds=0.619, new=True),
            dict(title="D  MMS 强制对齐", sub="每个音节的 source onset", time="≈ 0.015 s", seconds=0.015),
            dict(title="E  打包 + Opus（level 5）", sub="原始人声 + onsets · 1 个 bar", time="≈ 0.056 s", seconds=0.056),
            dict(title="F  音频缓存（按 lyric_id）", sub="ready / older_version / pending / failed …", new=True),
        ],
        h200_note="vLLM · SGLang-Omni · render server 三个服务在同一张 GPU",
        wait_label="音频提前到 Mac：3.8 s",
        results=[("real-time 服务器人声 100%", 250, "good"), ("fake real-time 100%", 190, "info"),
                 ("0 兜底 · 0 underrun", 190, "good"), ("开场等待 3.7 s", 150, "info"), ("改主题后 2.8–5.5 s 听到", 220, "info")],
        footnote=(DEMO_PAGE, "音频 / 歌词提前量（中位数 3.8 s / 7.4 s）与改主题反应时间取自 10-12 歌词流实施报告的 96 bar 会话（中途 8 次修改）。"),
    ),
    "v1.1": dict(
        version="v1.1",
        subtitle="tag rap-v1.1 · commit 99565f91 · 2026-10-14 · 协议 v3（与 v1.0 相同）· Qwen3.5-9B + MOSS-TTS-Local v1.5 · 单 GPU",
        changes=("只换模型，架构和 Mac 客户端与 v1.0 完全相同：写词 Qwen2.5-7B → Qwen3.5-9B（关闭 thinking）；",
                 "声音 MOSS-TTS v1.5 → MOSS-TTS-Local-Transformer v1.5（SGLang-Omni 0.1.8 + 移植的续写补丁，48 kHz 输出重采样）。"),
        timing_note="本版延迟未入库",
        stages=[
            dict(title="A  写词线程 · Qwen3.5-9B", sub="thinking 关闭 · 每 bar 32 条候选 → 选一句", new=True),
            dict(title="B  歌词记录 BarLyricRecord", sub="lyric_id = sha256 · 版本号 · 修改后版本 +1"),
            dict(title="C  合成线程 · MOSS-TTS-Local v1.5", sub="SGLang-Omni 0.1.8 + 续写补丁 · 48 kHz → 重采样", new=True),
            dict(title="D  MMS 强制对齐", sub="每个音节的 source onset"),
            dict(title="E  打包 + Opus（level 5）", sub="原始人声 + onsets · 1 个 bar"),
            dict(title="F  音频缓存（按 lyric_id）", sub="ready / older_version / pending / failed …"),
        ],
        h200_note="vLLM · SGLang-Omni 0.1.8 · render server 同一张 GPU",
        wait_label="客户端同 v1.0",
        results=[("Mac 客户端与 v1.0 相同", 230, "info"), ("server: model_upgrade_stack.sh qwen35 mosslocal", 400, "info")],
        footnote=("本版的听测与实时会话结果在 model-upgrade 听测页面（scripts/build_model_upgrade_page.py 生成）；仓库里没有提交这些数字，所以图中不标延迟。",),
    ),
}


# ---------------------------------------------------------------- concept view


def _self_loop(svg: Svg, left: float, right: float, top: float, color: str, label: str, sub: str) -> None:
    """Arc from a block's tail (right) back to its head (left): autoregressive generation."""
    peak = top - 78
    svg.path(f"M{right - 30},{top} C{right - 30},{peak} {left + 30},{peak} {left + 30},{top - 4}", color, "purple" if color == PURPLE else "blue", 2.2)
    mid = (left + right) / 2
    svg.text(mid, peak - 30, label, "label", anchor="middle", fill=color)
    svg.text(mid, peak - 12, sub, "small", anchor="middle")


def _wave(svg: Svg, x: float, y: float, seed: int, color: str) -> None:
    heights = (4, 9, 14, 7, 16, 11, 5, 13, 18, 8, 12, 6, 15, 10, 4, 9, 13, 7)
    for i in range(18):
        h = heights[(i + seed * 5) % len(heights)]
        svg.rect(x + i * 6, y - h / 2, 3.5, h, fill=color, rx=1.5)


def concept_diagram() -> str:
    svg = Svg(1440, 900)
    svg.text(40, 52, "StreamMUSE 实时 Rap：歌词流架构（概念图）", "title")
    svg.text(40, 82, "服务器一直往前写歌词、往前合成音频，结果分别存进 Lyric Pool 和 Audio Bank；客户端只发主题和 flow，并按需取回音频。对应 v1.0 / v1.1（协议 v3）。", "subtitle")

    # ---- Server
    svg.rect(40, 104, 1360, 436, fill="#F6F2FF", stroke=PURPLE, rx=16, width=1.5)
    svg.rect(64, 122, 76, 22, fill=PURPLE, rx=11)
    svg.text(102, 138, "SERVER", "panel-tag", anchor="middle")
    svg.text(152, 140, "H200：持续写词 → 持续合成", "panel-title")

    # Lyric writer
    svg.rect(80, 280, 270, 120, stroke=BLUE, shadow=True, width=1.8)
    svg.text(215, 316, "写词模型 Lyric Writer", "box-title", anchor="middle")
    svg.text(215, 342, "LLM · 每 bar 生成多条候选", "box-sub", anchor="middle")
    svg.text(215, 362, "过滤 + 打分 → 选出一句", "box-sub", anchor="middle")
    svg.text(215, 384, "一直写到播放头前 3 个 bar", "small", anchor="middle", fill=BLUE, weight=600)
    _self_loop(svg, 80, 350, 280, BLUE, "自回归：一句接一句往下写", "上一句（和押韵）作为下一句的上下文")

    # Lyric pool
    svg.rect(430, 262, 250, 156, fill="#EEF4FF", stroke=BLUE, shadow=True, width=1.8)
    svg.text(555, 290, "Lyric Pool 歌词池", "box-title", anchor="middle")
    cards = (("bar 7 · v1", "rising through the dark", False), ("bar 8 · v1", "stars align and spark", False), ("bar 9 · v2", "（主题修改后已重写）", True))
    for i, (head, body, edited) in enumerate(cards):
        y = 304 + i * 36
        svg.rect(446, y, 218, 30, stroke=NEW if edited else BLUE, rx=6)
        svg.text(456, y + 20, head, "small", fill=NEW if edited else BLUE, weight=700)
        svg.text(530, y + 20, body, "small", fill="#3A3F4B")

    # TTS
    svg.rect(760, 280, 270, 120, stroke=PURPLE, shadow=True, width=1.8)
    svg.text(895, 316, "TTS 模型", "box-title", anchor="middle")
    svg.text(895, 342, "MOSS · 每次合成 1 个 bar", "box-sub", anchor="middle")
    svg.text(895, 362, "对齐每个音节的起点", "box-sub", anchor="middle")
    svg.text(895, 384, "一直合成到播放头前 2 个 bar", "small", anchor="middle", fill=PURPLE, weight=600)
    _self_loop(svg, 760, 1030, 280, PURPLE, "自回归续写：一个 bar 接一个 bar", "上一个 bar 的音频作为前缀，声音连贯")

    # Audio bank
    svg.rect(1110, 262, 270, 156, fill="#F3EEFF", stroke=PURPLE, shadow=True, width=1.8)
    svg.text(1245, 290, "Audio Bank 音频库", "box-title", anchor="middle")
    for i, (label, ready) in enumerate((("bar 7", True), ("bar 8", True), ("bar 9", False))):
        y = 304 + i * 36
        svg.rect(1126, y, 238, 30, stroke=PURPLE, dash=None if ready else "5 3", rx=6)
        svg.text(1136, y + 20, label, "small", fill=PURPLE, weight=700)
        if ready:
            _wave(svg, 1186, y + 15, i, PURPLE)
            svg.text(1354, y + 20, "✓", "small", anchor="end", fill=GREEN, weight=700)
        else:
            svg.text(1255, y + 20, "排队（只提前 2 bar）", "small", anchor="middle", fill=PURPLE)

    # Server-internal flow
    svg.path("M350,340 L426,340", BLUE, "blue", 2.4)
    svg.path("M680,340 L756,340", PURPLE, "purple", 2.4)
    svg.text(718, 330, "歌词", "small", anchor="middle", fill=PURPLE, weight=600)
    svg.path("M1030,340 L1106,340", PURPLE, "purple", 2.4)
    svg.text(1068, 330, "音频", "small", anchor="middle", fill=PURPLE, weight=600)

    # ---- Client
    svg.rect(40, 640, 1360, 190, fill="#F1F6FF", stroke=BLUE, rx=16, width=1.5)
    svg.rect(64, 788, 72, 22, fill=BLUE, rx=11)
    svg.text(100, 804, "CLIENT", "panel-tag", anchor="middle")
    svg.text(148, 806, "Mac：时钟 · 播放 · 兜底", "panel-title")
    svg.box(80, 660, 270, 100, "听众 / Web 控制", ("选择 theme 和 flow template", "播放中随时修改"), stroke=BLUE)
    svg.box(430, 660, 250, 100, "收到的歌词", ("知道下一个 bar 唱什么", "以及它的 lyric_id"), stroke=BLUE)
    svg.box(760, 660, 270, 100, "逐 bar 控制器", ("按节拍时钟决定", "该取哪个 bar 的音频"), stroke=BLUE)
    svg.box(1110, 660, 270, 100, "对拍 + 混音 → 扬声器", ("没拿到音频就用 eSpeak 兜底", "节奏永不中断"), stroke=BLUE, fill=BLUE, title_fill="#FFFFFF", sub_fill="#DCE7FF")
    svg.path("M680,710 L756,710", GRAY, "gray")
    svg.path("M1030,710 L1106,710", GRAY, "gray")

    # ---- Client <-> server
    # 1. theme / flow template -> writer
    svg.path("M215,658 L215,404", BLUE, "blue", 2.6)
    svg.rect(232, 564, 172, 52, stroke=BLUE, rx=8)
    svg.text(318, 586, "theme · flow template", "label", anchor="middle", fill=BLUE)
    svg.text(318, 605, "（以及实时修改）", "small", anchor="middle")
    # 2. lyric pool -> client
    svg.path("M555,420 L555,656", BLUE, "blue", 2.2)
    svg.rect(572, 564, 112, 52, stroke=BLUE, rx=8)
    svg.text(628, 586, "lyric 推送", "label", anchor="middle", fill=BLUE)
    svg.text(628, 605, "写好就发", "small", anchor="middle")
    # 3. client request -> audio bank, 4. audio -> client
    svg.path("M1200,658 L1200,422", BLUE, "blue", 2.0, dash="7 5")
    svg.path("M1300,420 L1300,656", PURPLE, "purple", 2.6)
    svg.rect(1034, 558, 150, 64, stroke=BLUE, rx=8)
    svg.text(1109, 580, "request", "label", anchor="middle", fill=BLUE)
    svg.text(1109, 598, "bar + lyric_id", "small", anchor="middle")
    svg.text(1109, 614, "（client 发请求时）", "small", anchor="middle")
    svg.rect(1316, 564, 72, 52, stroke=PURPLE, rx=8)
    svg.text(1352, 586, "音频", "label", anchor="middle", fill=PURPLE)
    svg.text(1352, 605, "才发回", "small", anchor="middle")

    svg.text(40, 868, "两个模型各自不停地往前生成，彼此只通过 Lyric Pool 衔接；客户端不等任何一次生成，只在需要某个 bar 时取回已经准备好的音频。", "small")
    return svg.render()


# ---------------------------------------------------------------- edit timeline series

TEAL = "#0F9D8A"
TEAL_TEXT = "#0B7A6B"
OLD_BLUE = "#9DB8EC"
SERIES_X0 = 250
SERIES_W = 226
SERIES_FIRST_BAR = 4


def _series_x(bar: int) -> float:
    return SERIES_X0 + (bar - SERIES_FIRST_BAR) * SERIES_W


def _series_chip(svg: Svg, bar: int, y: float, h: float, kind: str, label: str, align: str = "center") -> None:
    x, w = _series_x(bar) + 6, SERIES_W - 12
    fill, stroke, color, dash = {
        "lyricA": ("#E3ECFF", BLUE, BLUE, None),
        "lyricB": ("#DDF5F1", TEAL, TEAL_TEXT, None),
        "dead": ("#F3F4F6", LINE, GRAY, "5 3"),
        "writing": ("#FFFFFF", TEAL, TEAL_TEXT, "5 3"),
        "audioA": (BLUE, None, "#FFFFFF", None),
        "audioB": (TEAL, None, "#FFFFFF", None),
        "oldA": (OLD_BLUE, None, "#1E3F8A", None),
        "synth": ("#FFFFFF", TEAL, TEAL_TEXT, "5 3"),
        "idle": ("#FFFFFF", LINE, GRAY, "3 3"),
        "futureA": ("#FFFFFF", BLUE, BLUE, "5 3"),
        "futureB": ("#FFFFFF", TEAL, TEAL_TEXT, "5 3"),
    }[kind]
    svg.rect(x, y, w, h, fill=fill, stroke=stroke, dash=dash, rx=6)
    if align == "left":
        svg.text(x + 12, y + h / 2 + 4.5, label, "small", fill=color, weight=600)
    else:
        svg.text(x + w / 2, y + h / 2 + 4.5, label, "small", anchor="middle", fill=color, weight=600)


def _series_frame(svg: Svg, fy: float, title: str, desc: str, play, lyrics, audio, playhead: tuple[int, float], edit: bool = False) -> None:
    svg.rect(40, fy, 1360, 196, fill="#FAFAFB", stroke="#E3E5EA", rx=14)
    svg.text(64, fy + 30, title, "section-title")
    svg.text(64, fy + 52, desc, "box-sub")
    svg.text(64, fy + 87, "Mac 播放", "label")
    svg.text(64, fy + 127, "Lyric Pool", "label")
    svg.text(64, fy + 163, "Audio Bank", "label")
    for chip_args in play:
        _series_chip(svg, chip_args[0], fy + 64, 36, *chip_args[1:])
    for bar, kind, label in lyrics:
        _series_chip(svg, bar, fy + 108, 28, kind, label)
    for bar, kind, label in audio:
        _series_chip(svg, bar, fy + 144, 28, kind, label)
    bar, frac = playhead
    x = _series_x(bar) + frac * SERIES_W
    color = NEW if edit else "#1F2430"
    svg.add(f'<line x1="{x:.1f}" y1="{fy + 60}" x2="{x:.1f}" y2="{fy + 178}" stroke="{color}" stroke-width="2.2"/>')
    svg.text(x + 5, fy + 190, "✎ 听众在这里修改" if edit else "播放头", "small", fill=color, weight=700)


def edit_timeline_series() -> str:
    svg = Svg(1440, 1170)
    svg.text(40, 52, "歌词流时间线：中途换 theme / flow template 会发生什么", "title")
    svg.text(40, 82, "v1.0 / v1.1（协议 v3）· 90 BPM · 歌词写到播放头前 3 bar，音频合成到前 2 bar · P = Mac 最后提交的 bar · 主题名只是示例", "subtitle")

    legend = (
        ("lyricA", "旧主题 space 的歌词"), ("audioA", "旧主题的音频"), ("lyricB", "新主题 ocean 的歌词"),
        ("audioB", "新主题的音频"), ("dead", "作废的旧歌词"), ("oldA", "旧版音频（留作兜底）"),
    )
    x = 40
    for kind, label in legend:
        fill, stroke, dash = {
            "lyricA": ("#E3ECFF", BLUE, None), "audioA": (BLUE, None, None), "lyricB": ("#DDF5F1", TEAL, None),
            "audioB": (TEAL, None, None), "dead": ("#F3F4F6", LINE, "5 3"), "oldA": (OLD_BLUE, None, None),
        }[kind]
        svg.rect(x, 104, 30, 18, fill=fill, stroke=stroke, dash=dash, rx=4)
        svg.text(x + 38, 118, label, "small", fill="#3A3F4B")
        x += 38 + _text_px(label) + 30

    frames = (
        dict(
            title="① 正常运行：bar 4 播放中（P = 4）",
            desc="歌词池里有 bar 5–7（写到前 3 bar），音频库里有 bar 5、6（合成到前 2 bar），Mac 已经把这两个 bar 的音频取回。",
            play=((4, "audioA", "bar 4 · space"), (5, "futureA", "bar 5"), (6, "futureA", "bar 6"), (7, "futureA", "bar 7"), (8, "futureA", "bar 8")),
            lyrics=((5, "lyricA", "bar 5 · v1 · space"), (6, "lyricA", "bar 6 · v1 · space"), (7, "lyricA", "bar 7 · v1 · space"), (8, "idle", "bar 8 · 等 P = 5 再写")),
            audio=((5, "audioA", "bar 5 · v1 ✓"), (6, "audioA", "bar 6 · v1 ✓"), (7, "idle", "bar 7 · 等 P = 5 再合成")),
            playhead=(4, 0.25),
        ),
        dict(
            title="② bar 4 中间，听众改主题（只提前 1 个 bar）",
            desc="Mac 发 update {theme: ocean, flow template: T3, from_bar: 5}。服务器把 ≥ 5 的未提交 bar 全部作废、按新主题重写；已合成的旧音频不删。",
            play=((4, "audioA", "bar 4 · space", "left"), (5, "futureB", "bar 5"), (6, "futureB", "bar 6"), (7, "futureB", "bar 7"), (8, "futureB", "bar 8")),
            lyrics=((5, "writing", "bar 5 · v2 写词中…"), (6, "dead", "bar 6 · v1 作废，待重写"), (7, "dead", "bar 7 · v1 作废，待重写")),
            audio=((5, "oldA", "bar 5 · v1 旧版（兜底）"), (6, "oldA", "bar 6 · v1 旧版（兜底）"), (7, "idle", "bar 7 · 等 P = 5 再合成")),
            playhead=(4, 0.5),
            edit=True,
        ),
        dict(
            title="③ bar 4 最后 1 tick：提交 bar 5，新音频来不及",
            desc="新歌词很快写好（≈ 0.1 s / 行），bar 5 v2 被插队合成；但离提交只剩 ≈ 1.3 s，不够合成 + 取回（≈ 2 s）→ bar 5 播 v1 旧版本。",
            play=((4, "audioA", "bar 4 · space"), (5, "oldA", "bar 5 · 播旧版本（remote_older）"), (6, "futureB", "bar 6"), (7, "futureB", "bar 7"), (8, "futureB", "bar 8")),
            lyrics=((5, "lyricB", "bar 5 · v2 · ocean"), (6, "lyricB", "bar 6 · v2 · ocean"), (7, "lyricB", "bar 7 · v2 · ocean")),
            audio=((5, "synth", "bar 5 · v2 合成中（插队）"), (6, "oldA", "bar 6 · v1 兜底 · v2 排队"), (7, "idle", "bar 7 · 等 P = 5 再合成")),
            playhead=(4, 0.985),
        ),
        dict(
            title="④ bar 5 播放中：恢复正常（P = 5）",
            desc="bar 6 的 v2 音频已经就绪，从 bar 6 开始听到的全是新主题；歌词池往前写到 bar 8，音频合成到 bar 7。节奏从头到尾没有中断。",
            play=((4, "audioA", "bar 4 · space"), (5, "oldA", "bar 5 · 旧版本"), (6, "futureB", "bar 6 · ocean"), (7, "futureB", "bar 7 · ocean"), (8, "futureB", "bar 8 · ocean")),
            lyrics=((6, "lyricB", "bar 6 · v2 · ocean"), (7, "lyricB", "bar 7 · v2 · ocean"), (8, "lyricB", "bar 8 · ocean（新写）")),
            audio=((6, "audioB", "bar 6 · v2 ✓"), (7, "synth", "bar 7 · v2 合成中")),
            playhead=(5, 0.25),
        ),
    )
    for i, frame in enumerate(frames):
        _series_frame(svg, 146 + i * 210, **frame)

    # Comparison: the same edit with two bars of notice.
    fy = 146 + 4 * 210
    svg.rect(40, fy, 1360, 124, fill="#F2FBF8", stroke="#BFE6DD", rx=14)
    svg.text(64, fy + 30, "对比：如果提前 2 个 bar 修改（同样在 bar 4 中间，from_bar = 6）", "section-title")
    svg.text(64, fy + 52, "bar 5 照常播 space；bar 6 的新音频有 ≈ 4 s 准备时间，足够 → 修改生效的那个 bar 直接就是新内容，不会播旧版本。", "box-sub")
    svg.text(64, fy + 87, "Mac 播放", "label")
    for bar, kind, label, align in ((4, "audioA", "bar 4 · space", "left"), (5, "audioA", "bar 5 · space（v1，正常）", "center"), (6, "audioB", "bar 6 · ocean ✓ 按时", "center"), (7, "audioB", "bar 7 · ocean", "center"), (8, "audioB", "bar 8 · ocean", "center")):
        _series_chip(svg, bar, fy + 64, 36, kind, label, align)
    x = _series_x(4) + 0.5 * SERIES_W
    svg.add(f'<line x1="{x:.1f}" y1="{fy + 60}" x2="{x:.1f}" y2="{fy + 106}" stroke="{NEW}" stroke-width="2.2"/>')
    svg.text(x + 5, fy + 118, "✎ 修改", "small", fill=NEW, weight=700)

    svg.text(40, 1150, "实测（10-12 歌词流实施报告，96 bar 会话、8 次修改）：提前 2 bar 的 4 次都在生效 bar 听到新内容；提前 1 bar 的 4 次各播了 1 个旧版本 bar；0 次 eSpeak、0 underrun。", "small")
    return svg.render()


# ---------------------------------------------------------------- slides

SLIDES_CSS = """
* { box-sizing: border-box; margin: 0; padding: 0; }
html, body { height: 100%; background: #FFFFFF; overflow: hidden;
  font-family: "PingFang SC", "Hiragino Sans GB", -apple-system, "Helvetica Neue", Arial, sans-serif; color: #1F2430; }
.slide { position: absolute; inset: 0 0 44px 0; display: flex; align-items: center; justify-content: center;
  padding: 16px 24px; opacity: 0; pointer-events: none; transition: opacity .25s ease; }
.slide.active { opacity: 1; pointer-events: auto; }
.slide img { max-width: 100%; max-height: 100%; object-fit: contain; }
.cover { width: min(1240px, 92vw); }
.summary { width: min(1440px, 94vw); }
.summary td:nth-child(2) { white-space: nowrap; }
.cover h1 { font-size: clamp(32px, 4.4vw, 60px); font-weight: 700; letter-spacing: .5px; }
.cover .sub { margin-top: 14px; font-size: clamp(16px, 1.6vw, 22px); color: #5B6170; }
.cover .track { margin-top: 56px; display: flex; gap: 14px; flex-wrap: wrap; }
.cover .step { flex: 1 1 0; min-width: 170px; border: 1.5px solid #2F6BDB; border-radius: 14px; padding: 18px 18px 20px; background: #F1F6FF; }
.cover .step b { display: inline-block; font-size: 15px; color: #FFFFFF; background: #2F6BDB; border-radius: 12px; padding: 3px 12px; }
.cover .step p { margin-top: 12px; font-size: 15px; line-height: 1.55; color: #3A3F4B; }
.cover .step.v1 { border-color: #7448D1; background: #F6F2FF; }
.cover .step.v1 b { background: #7448D1; }
.cover .extra { margin-top: 28px; font-size: 15px; color: #6B7180; }
.summary h2 { font-size: clamp(24px, 2.6vw, 36px); font-weight: 700; }
.summary .sub { margin-top: 8px; font-size: 15px; color: #5B6170; }
.summary table { margin-top: 28px; width: 100%; border-collapse: separate; border-spacing: 0; font-size: clamp(14px, 1.25vw, 18px); }
.summary th { white-space: nowrap; text-align: left; font-weight: 600; color: #5B6170; font-size: .85em; padding: 10px 14px; border-bottom: 1.5px solid #D8DCE4; }
.summary td { padding: 14px; border-bottom: 1px solid #ECEEF2; vertical-align: middle; }
.summary td.num { text-align: right; font-variant-numeric: tabular-nums; white-space: nowrap; }
.summary th.num { text-align: right; }
.summary .tag { display: inline-block; font-weight: 700; color: #FFFFFF; background: #2F6BDB; border-radius: 11px; padding: 2px 10px; font-size: .85em; }
.summary .tag.v1 { background: #7448D1; }
.summary .bad { color: #B3362B; font-weight: 700; }
.summary .good { color: #1F7A48; font-weight: 700; }
.summary .muted { color: #9AA0AC; }
.summary .bar { display: inline-block; height: 10px; border-radius: 3px; background: #7448D1; vertical-align: middle; margin-left: 8px; }
.summary .note { margin-top: 22px; font-size: 13px; color: #6B7180; line-height: 1.6; }
footer { position: fixed; left: 0; right: 0; bottom: 0; height: 44px; display: flex; align-items: center; gap: 16px;
  padding: 0 24px; font-size: 13px; color: #6B7180; border-top: 1px solid #ECEEF2; background: #FFFFFF; }
footer .title { flex: 1; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
footer button { border: 1px solid #D8DCE4; background: #FFFFFF; border-radius: 8px; width: 34px; height: 28px; font-size: 15px;
  color: #3A3F4B; cursor: pointer; }
footer button:hover { background: #F1F6FF; border-color: #2F6BDB; }
.progress { position: fixed; left: 0; bottom: 44px; height: 3px; background: #2F6BDB; transition: width .25s ease; }
"""

SLIDES_JS = """
const slides = [...document.querySelectorAll('.slide')];
const counter = document.getElementById('counter');
const title = document.getElementById('slide-title');
const progress = document.querySelector('.progress');
let current = 0;
function show(i) {
  current = Math.max(0, Math.min(slides.length - 1, i));
  slides.forEach((s, k) => s.classList.toggle('active', k === current));
  counter.textContent = `${current + 1} / ${slides.length}`;
  title.textContent = slides[current].dataset.title;
  progress.style.width = `${(current + 1) / slides.length * 100}%`;
  history.replaceState(null, '', `#${current + 1}`);
}
document.addEventListener('keydown', (e) => {
  if (['ArrowRight', 'ArrowDown', 'PageDown', ' ', 'Enter'].includes(e.key)) { e.preventDefault(); show(current + 1); }
  else if (['ArrowLeft', 'ArrowUp', 'PageUp', 'Backspace'].includes(e.key)) { e.preventDefault(); show(current - 1); }
  else if (e.key === 'Home') show(0);
  else if (e.key === 'End') show(slides.length - 1);
  else if (e.key === 'f') document.fullscreenElement ? document.exitFullscreen() : document.documentElement.requestFullscreen();
});
document.getElementById('prev').onclick = () => show(current - 1);
document.getElementById('next').onclick = () => show(current + 1);
document.getElementById('fs').onclick = () => document.fullscreenElement ? document.exitFullscreen() : document.documentElement.requestFullscreen();
show((parseInt(location.hash.slice(1), 10) || 1) - 1);
"""

COVER_HTML = """
<div class="cover">
  <h1>StreamMUSE 实时 Rap 系统</h1>
  <div class="sub">从“一个请求包两小节”到“歌词流 + 逐 bar 续写”：五个版本的系统结构与实时表现</div>
  <div class="track">
    <div class="step"><b>v0.0</b><p>进程内 HF MOSS<br>两小节 chunk · 服务器 R3</p></div>
    <div class="step"><b>v0.1</b><p>MOSS 改用<br>SGLang-Omni 服务</p></div>
    <div class="step"><b>v0.2</b><p>R3 移到 Mac<br>并发写词 · 延迟补丁</p></div>
    <div class="step v1"><b>v1.0</b><p>歌词流 + 逐 bar 续写<br>支持实时修改</p></div>
    <div class="step v1"><b>v1.1</b><p>换模型：Qwen3.5-9B<br>+ MOSS-TTS-Local</p></div>
  </div>
  <div class="extra">之后两页：v1.x 歌词流架构的概念图，以及中途修改 theme / flow template 时的时间线。← → 或空格翻页，F 全屏。</div>
</div>
"""

SUMMARY_ROWS = (
    # tag, class, architecture, MOSS ms, wait ms (label), real-time %, fake %, samples with fallback
    ("v0.0", "", "进程内 HF MOSS · 两小节 chunk", 2765, "4667 ms", "24.2%", "95.8%", "15 / 15", "bad"),
    ("v0.1", "", "SGLang-Omni MOSS（未打补丁）", 833, "2667 ms", "92.5%", "97.5%", "5 / 15", ""),
    ("v0.2", "", "+ Mac 端 R3 · 并发写词 · 延迟补丁", 765, "2617 ms", "92.5%", "97.5%", "5 / 15", ""),
    ("v1.0", "v1", "歌词流 + 逐 bar 续写（1 个 bar）", 619, "1816 ms（长轮询）", "100%", "100%", "0 / 15", "good"),
)


def _summary_html() -> str:
    rows = []
    for tag, cls, arch, moss, wait, rt, fake, samples, tone in SUMMARY_ROWS:
        bar = f'<span class="bar" style="width:{moss / 2765 * 120:.0f}px"></span>'
        rows.append(
            f'<tr><td><span class="tag {cls}">{tag}</span></td><td>{escape(arch)}</td>'
            f'<td class="num">{moss} ms{bar}</td><td class="num">{escape(wait)}</td>'
            f'<td class="num {tone}">{rt}</td><td class="num">{fake}</td><td class="num">{samples}</td></tr>'
        )
    rows.append(
        '<tr><td><span class="tag v1">v1.1</span></td><td>只换模型：Qwen3.5-9B + MOSS-TTS-Local v1.5</td>'
        '<td class="num muted" colspan="5">实测数字在 model-upgrade 听测页面，未入库</td></tr>'
    )
    return (
        '<div class="summary"><h2>五个版本的实时表现</h2>'
        '<div class="sub">同一张 H200 + SSH 隧道 · 15 段 × 16 bar · 90 BPM · real-time 中位数</div>'
        '<table><thead><tr><th>版本</th><th>结构</th><th class="num">MOSS / 请求</th><th class="num">Mac 等一个请求</th>'
        '<th class="num">real-time 服务器人声</th><th class="num">fake real-time</th><th class="num">有兜底的 sample</th></tr></thead>'
        f'<tbody>{"".join(rows)}</tbody></table>'
        '<div class="note">real-time 与 fake real-time（服务器工作时时钟停住）之差就是“实时”带来的损失：v0.0 损失 72 个百分点，v0.1 / v0.2 各 5 个，v1.0 为 0。'
        '<br>v0.2 服务器更快，但 Mac 端等待只从 2.67 s 降到 2.62 s，大头是网络；v1.0 的等待是长轮询，可能在音频合成前就发出，不能直接和 v0.x 比。'
        '<br>数据：developing-logs/reports/2026-10-12-rap-version-demo-page-report.md。</div></div>'
    )


def build_slides() -> str:
    import base64

    def image(name: str) -> str:
        data = base64.b64encode((OUT / name).read_bytes()).decode("ascii")
        return f'<img alt="{escape(name)}" src="data:image/svg+xml;base64,{data}">'

    slides = (
        ("StreamMUSE 实时 Rap 系统", COVER_HTML),
        ("v0.0 · 进程内 HF MOSS，两小节 chunk", image("rap-v0.0.svg")),
        ("v0.1 · SGLang-Omni MOSS", image("rap-v0.1.svg")),
        ("v0.2 · R3 移到 Mac，延迟优化", image("rap-v0.2.svg")),
        ("v1.0 · 歌词流 + 逐 bar 续写", image("rap-v1.0.svg")),
        ("v1.1 · 换模型", image("rap-v1.1.svg")),
        ("五个版本的实时表现", _summary_html()),
        ("歌词流架构（概念图）", image("rap-concept.svg")),
        ("中途修改 theme / flow template 的时间线", image("rap-edit-timeline.svg")),
    )
    body = "\n".join(
        f'<section class="slide" data-title="{escape(title)}">{content}</section>' for title, content in slides
    )
    return (
        '<!doctype html>\n<html lang="zh-CN"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        f"<title>StreamMUSE 实时 Rap 系统</title><style>{SLIDES_CSS}</style></head><body>\n{body}\n"
        '<div class="progress"></div><footer><span class="title" id="slide-title"></span>'
        '<span id="counter"></span><button id="prev" title="上一页（←）">‹</button>'
        '<button id="next" title="下一页（→）">›</button><button id="fs" title="全屏（F）">⤢</button></footer>'
        f"<script>{SLIDES_JS}</script></body></html>\n"
    )


def main() -> None:
    for version, cfg in CONFIGS.items():
        svg = chunk_diagram(cfg) if version.startswith("v0") else stream_diagram(cfg)
        path = OUT / f"rap-{version}.svg"
        path.write_text(svg, encoding="utf-8")
        print(path)
    path = OUT / "rap-concept.svg"
    path.write_text(concept_diagram(), encoding="utf-8")
    print(path)
    path = OUT / "rap-edit-timeline.svg"
    path.write_text(edit_timeline_series(), encoding="utf-8")
    print(path)
    SLIDES_PATH.write_text(build_slides(), encoding="utf-8")
    print(SLIDES_PATH)


if __name__ == "__main__":
    main()
