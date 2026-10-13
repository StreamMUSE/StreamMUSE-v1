"""Build the three-version demo page from ``<out>/site/data.json``.

Writes ``<out>/site/index.html``: a static page (data inlined, audio under
``audio/``) with the version overview, one card per sample with every
version in both modes, and the method notes.

    uv run python scripts/build_rap_demo_page.py --out output/rap_demo_page_2026-10
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

VERSIONS = [
    {
        "id": "v0.0", "name": "In-process MOSS", "tag": "rap-v0.0", "commit": "0a493cb8", "date": "2026-08-30",
        "unit": "2 bars per request",
        "lyrics": "Qwen writes both bars on the server when the Mac asks for the chunk",
        "speech": "MOSS (Hugging Face, in the render server process) speaks both bars as one phrase",
        "timing": "MMS alignment and Rubber Band R3 on the server; the Mac mixes the drums",
        "fallback": "eSpeak for the whole 2-bar chunk",
    },
    {
        "id": "v0.1", "name": "SGLang-Omni MOSS", "tag": "rap-v0.1", "commit": "9b76415e", "date": "2026-10-01",
        "unit": "2 bars per request",
        "lyrics": "Same as v0.0",
        "speech": "MOSS served by SGLang-Omni (unpatched)",
        "timing": "Same as v0.0",
        "fallback": "eSpeak for the whole 2-bar chunk",
    },
    {
        "id": "v0.2", "name": "Optimized SGLang-Omni", "tag": "rap-v0.2", "commit": "0ea74a7f", "date": "2026-10-01",
        "unit": "2 bars per request",
        "lyrics": "Qwen writes the two bars concurrently on the server (vLLM allows 32 sequences)",
        "speech": "MOSS on SGLang-Omni with a latency patch",
        "timing": "MMS on the server; the Mac warps the raw voice with R3 itself (protocol v2); Opus level 5",
        "fallback": "eSpeak for the whole 2-bar chunk",
    },
    {
        "id": "v1.0", "name": "Lyric stream", "tag": "rap-v1.0", "commit": "19526512", "date": "2026-10-12",
        "unit": "1 bar, rendered ahead",
        "lyrics": "Qwen writes one line per bar from 32 candidates, three bars ahead, and streams it to the Mac",
        "speech": "MOSS (patched SGLang-Omni) continues the previous bar's speech; audio is ready two bars ahead",
        "timing": "MMS on the server; the Mac fetches each bar by lyric ID and warps it with R3",
        "fallback": "eSpeak for one bar, speaking the streamed line",
    },
]


PAGE = r"""<title>StreamMUSE Rap Versions</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Archivo:wdth,wght@62..125,500..900&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>
/* Layout: a listening room. Overview first, then one card per sample with a
   3 x 2 grid (version x mode); every recording carries a 16-pad strip, one pad
   per bar, lit by where that bar's voice came from. */
:root {
  --bg: #f3f4f6; --surface: #ffffff; --ink: #15171c; --muted: #5d6370; --line: #d9dce2;
  --accent: #2f45c4; --accent-soft: #e6e9fb;
  --remote: #1d8a6d; --older: #c4870a; --espeak: #c4432e; --pad-off: #e7e9ee;
  --display: "Archivo", "Arial Narrow", system-ui, sans-serif;
  --body: "IBM Plex Sans", system-ui, -apple-system, "Segoe UI", sans-serif;
  --mono: "IBM Plex Mono", ui-monospace, "SFMono-Regular", Menlo, monospace;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    --bg: #111318; --surface: #1a1d24; --ink: #e8eaef; --muted: #9aa1ae; --line: #2b303a;
    --accent: #8b99ff; --accent-soft: #232a4a;
    --remote: #3cc39a; --older: #e3aa3c; --espeak: #f06a55; --pad-off: #262a33; color-scheme: dark;
  }
}
:root[data-theme="dark"] {
  --bg: #111318; --surface: #1a1d24; --ink: #e8eaef; --muted: #9aa1ae; --line: #2b303a;
  --accent: #8b99ff; --accent-soft: #232a4a;
  --remote: #3cc39a; --older: #e3aa3c; --espeak: #f06a55; --pad-off: #262a33; color-scheme: dark;
}
* { box-sizing: border-box; }
body { background: var(--bg); color: var(--ink); font: 15px/1.55 var(--body); }
.wrap { max-width: 1180px; margin: 0 auto; padding-inline: 20px; padding-block: 40px 72px; display: grid; gap: 48px; }
h1, h2, h3 { font-family: var(--display); text-wrap: balance; margin: 0; }
h1 { font-size: clamp(2.1rem, 5vw, 3.4rem); font-weight: 850; font-stretch: 72%; letter-spacing: -0.01em; line-height: 1.02; }
h2 { font-size: 1.55rem; font-weight: 800; font-stretch: 80%; }
h3 { font-size: 1.12rem; font-weight: 750; font-stretch: 90%; }
p { margin: 0; max-width: 68ch; }
.muted { color: var(--muted); }
.eyebrow { font: 500 0.72rem/1 var(--mono); letter-spacing: 0.08em; text-transform: uppercase; color: var(--muted); }
code, .mono { font-family: var(--mono); font-size: 0.86em; }
header { display: grid; gap: 14px; }
.legend { display: flex; flex-wrap: wrap; gap: 8px 18px; align-items: center; font-size: 0.88rem; }
.legend span { display: inline-flex; gap: 7px; align-items: center; }
.dot { width: 12px; height: 12px; border-radius: 3px; display: inline-block; flex: none; }
.s-remote { background: var(--remote); } .s-remote_older { background: var(--older); } .s-espeak { background: var(--espeak); }
.defs { display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 16px; }
.defs div { display: grid; gap: 4px; }
.defs b { font-family: var(--display); font-stretch: 90%; font-size: 1.02rem; }
section { display: grid; gap: 18px; }
.versions { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 16px; }
@media (max-width: 1060px) { .versions { grid-template-columns: repeat(2, minmax(0, 1fr)); } }
@media (max-width: 600px) { .versions { grid-template-columns: minmax(0, 1fr); } }
.version { background: var(--surface); border: 1px solid var(--line); border-radius: 10px; padding: 18px; display: grid; gap: 12px; align-content: start; }
.version .head { display: flex; justify-content: space-between; align-items: baseline; gap: 12px; }
.vid { font: 800 1.6rem/1 var(--display); font-stretch: 75%; color: var(--accent); }
.version dl { display: grid; grid-template-columns: max-content 1fr; gap: 6px 12px; margin: 0; font-size: 0.9rem; }
.version dt { color: var(--muted); }
.version dd { margin: 0; min-width: 0; }
.table-scroll { overflow-x: auto; border: 1px solid var(--line); border-radius: 10px; background: var(--surface); }
table { border-collapse: collapse; width: 100%; font-size: 0.9rem; font-variant-numeric: tabular-nums; }
th, td { padding: 9px 12px; text-align: right; border-bottom: 1px solid var(--line); white-space: nowrap; }
th:first-child, td:first-child, th:nth-child(2), td:nth-child(2) { text-align: left; }
thead th { font: 500 0.72rem/1.2 var(--mono); letter-spacing: 0.06em; text-transform: uppercase; color: var(--muted); }
tbody tr:last-child td { border-bottom: 0; }
.bar100 { display: inline-flex; width: 120px; height: 9px; border-radius: 2px; overflow: hidden; background: var(--pad-off); vertical-align: middle; }
.bar100 i { display: block; height: 100%; }
.filters { display: flex; flex-wrap: wrap; gap: 8px; align-items: center; }
.filters .label { margin-right: 4px; }
.chip { font: 500 0.85rem/1 var(--body); padding: 8px 12px; border-radius: 999px; border: 1px solid var(--line); background: var(--surface); color: var(--ink); cursor: pointer; }
.chip[aria-pressed="true"] { background: var(--accent); border-color: var(--accent); color: var(--surface); }
.chip:focus-visible, button:focus-visible, summary:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
.samples { display: grid; gap: 22px; }
.sample { background: var(--surface); border: 1px solid var(--line); border-radius: 12px; padding: 18px; display: grid; gap: 14px; }
.sample .head { display: flex; flex-wrap: wrap; gap: 6px 14px; align-items: baseline; }
.sid { font: 500 0.8rem/1 var(--mono); color: var(--muted); }
.grid6 { display: grid; grid-template-columns: 92px repeat(2, minmax(0, 1fr)); gap: 10px 14px; align-items: start; }
.grid6 .colhead { font: 500 0.72rem/1.2 var(--mono); letter-spacing: 0.06em; text-transform: uppercase; color: var(--muted); }
.rowhead { font: 800 1.15rem/1.1 var(--display); font-stretch: 78%; padding-top: 6px; }
.rowhead small { display: block; font: 400 0.75rem/1.3 var(--body); color: var(--muted); }
.take { display: grid; gap: 8px; min-width: 0; padding: 10px; border: 1px solid var(--line); border-radius: 8px; }
.take.missing { color: var(--muted); font-size: 0.85rem; place-content: center; min-height: 92px; border-style: dashed; }
.take audio { width: 100%; height: 34px; }
.pads { display: grid; grid-template-columns: repeat(16, 1fr); gap: 3px; }
.pads span { height: 14px; border-radius: 2px; background: var(--pad-off); opacity: 0.55; transition: transform 0.12s; }
.pads span.s-remote { background: var(--remote); } .pads span.s-remote_older { background: var(--older); } .pads span.s-espeak { background: var(--espeak); }
.pads span:nth-child(4n+1) { margin-left: 0; }
.pads span.on { opacity: 1; transform: scaleY(1.35); }
.pads.idle span { opacity: 0.85; }
.meta { display: flex; flex-wrap: wrap; gap: 4px 12px; font: 400 0.78rem/1.3 var(--mono); color: var(--muted); }
.line-now { font-size: 0.9rem; min-height: 1.4em; }
details summary { cursor: pointer; font-size: 0.84rem; color: var(--accent); }
details ol { margin: 8px 0 0; padding-left: 26px; font-size: 0.86rem; display: grid; gap: 2px; }
details li { padding-left: 2px; }
details li::marker { font-family: var(--mono); color: var(--muted); font-size: 0.8em; }
details li.cur { background: var(--accent-soft); border-radius: 3px; }
details li .dot { width: 8px; height: 8px; margin-right: 6px; vertical-align: middle; }
.notes { display: grid; gap: 14px; }
.notes ul { margin: 0; padding-left: 20px; display: grid; gap: 8px; max-width: 80ch; }
pre { margin: 0; padding: 12px 14px; background: var(--surface); border: 1px solid var(--line); border-radius: 8px; overflow-x: auto; font: 0.8rem/1.5 var(--mono); }
@media (max-width: 720px) {
  .grid6 { grid-template-columns: minmax(0, 1fr); }
  .grid6 .colhead.corner { display: none; }
  .grid6 .colhead { display: none; }
  .take::before { content: attr(data-label); font: 500 0.7rem/1 var(--mono); letter-spacing: 0.06em; text-transform: uppercase; color: var(--muted); }
}
@media (prefers-reduced-motion: reduce) { .pads span { transition: none; } }
</style>

<div class="wrap">
<header>
  <div class="eyebrow">StreamMUSE · real-time rap · version comparison</div>
  <h1>StreamMUSE Rap Versions</h1>
  <p>Four versions of the same system, each heard twice on the same 15 prompts: five topics, three flow templates, 16 bars at 90 BPM over the same drums. Every pad under a player is one bar, colored by where that bar's voice came from.</p>
  <div class="legend">
    <span><i class="dot s-remote"></i>Server voice (MOSS)</span>
    <span><i class="dot s-remote_older"></i>Server voice, earlier version of the line (v1.0 only)</span>
    <span><i class="dot s-espeak"></i>Local fallback (eSpeak)</span>
  </div>
  <div class="defs">
    <div><b>Fake real-time</b><span class="muted">The same system with the clock stopped whenever it waits for the server, so every bar gets the server's voice unless the server itself fails. This is what the system can produce.</span></div>
    <div><b>Real-time</b><span class="muted">The demo as it runs live: the Mac client plays on the beat and talks to the H200 over the network. A bar whose voice is late falls back to eSpeak. This is what a listener hears.</span></div>
  </div>
</header>

<section aria-labelledby="versions-h">
  <h2 id="versions-h">The four versions</h2>
  <div class="versions" id="versions"></div>
  <div class="table-scroll"><table id="summary"></table></div>
  <p class="muted" id="summary-note"></p>
</section>

<section aria-labelledby="samples-h">
  <h2 id="samples-h">Samples</h2>
  <div class="filters" id="topic-filters"><span class="eyebrow label">Topic</span></div>
  <div class="filters" id="template-filters"><span class="eyebrow label">Flow</span></div>
  <div class="samples" id="samples"></div>
</section>

<section class="notes" aria-labelledby="method-h">
  <h2 id="method-h">How these were made</h2>
  <ul id="method"></ul>
  <pre id="repro"></pre>
</section>
</div>

<script type="application/json" id="data">__DATA__</script>
<script>
const DATA = JSON.parse(document.getElementById("data").textContent);
const VERSIONS = DATA.versions, RUNS = DATA.runs;
const MODES = [["fake_realtime", "Fake real-time"], ["realtime", "Real-time"]];
const BAR_SECONDS = 4 * 60 / 90;
const FLOW = { baseline_straight_9: "straight", baseline_syncopated_9: "syncopated", baseline_staggered_9: "staggered" };
const el = (tag, attrs = {}, ...kids) => {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "class") node.className = v; else if (k === "text") node.textContent = v; else node.setAttribute(k, v);
  }
  for (const kid of kids) if (kid != null) node.append(kid);
  return node;
};
const run = (v, m, s) => RUNS.find(r => r.version === v && r.mode === m && r.sample === s);
const fmt = (x, d = 0) => x == null ? "–" : Number(x).toFixed(d);

// Versions.
for (const v of VERSIONS) {
  const dl = el("dl");
  for (const [k, label] of [["unit", "Unit"], ["lyrics", "Lyrics"], ["speech", "Voice"], ["timing", "Timing"], ["fallback", "Fallback"]]) {
    dl.append(el("dt", { text: label }), el("dd", { text: v[k] }));
  }
  document.getElementById("versions").append(el("article", { class: "version" },
    el("div", { class: "head" }, el("span", { class: "vid", text: v.id }), el("span", { class: "mono muted", text: `${v.tag} · ${v.commit} · ${v.date}` })),
    el("h3", { text: v.name }), dl));
}

// Summary table.
const table = document.getElementById("summary");
table.append(el("thead", {}, el("tr", {}, ...["Version", "Mode", "Bars by source", "Server", "Older", "eSpeak", "Underruns", "MOSS p50 ms", "Round trip p50 ms", "Network p50 s"].map(h => el("th", { text: h })))));
const tbody = el("tbody");
for (const v of VERSIONS) for (const [m, label] of MODES) {
  const rs = RUNS.filter(r => r.version === v.id && r.mode === m);
  if (!rs.length) continue;
  const total = rs.reduce((a, r) => a + r.bars.length, 0);
  const c = k => rs.reduce((a, r) => a + (r.counts[k] || 0), 0);
  const pct = k => total ? 100 * c(k) / total : 0;
  const med = xs => { xs = xs.filter(x => x != null).sort((a, b) => a - b); return xs.length ? xs[Math.floor((xs.length - 1) / 2)] : null; };
  const bar = el("span", { class: "bar100", title: "share of bars by source" });
  for (const k of ["remote", "remote_older", "espeak"]) { const i = el("i", { class: `s-${k}` }); i.style.width = pct(k) + "%"; bar.append(i); }
  const live = m === "realtime";
  tbody.append(el("tr", {},
    el("td", { text: v.id }), el("td", { text: label }), el("td", {}, bar),
    el("td", { text: fmt(pct("remote"), 1) + "%" }), el("td", { text: fmt(pct("remote_older"), 1) + "%" }), el("td", { text: fmt(pct("espeak"), 1) + "%" }),
    el("td", { text: String(rs.reduce((a, r) => a + (r.stats.underruns || 0), 0)) }),
    el("td", { text: live ? fmt(med(rs.map(r => r.stats.server_moss_ms_p50))) : "–" }),
    el("td", { text: live ? fmt(med(rs.map(r => r.stats.remote_total_ms_p50))) : "–" }),
    el("td", { text: live ? fmt(med(rs.map(r => r.rtt_seconds_p50)), 2) : "–" })));
}
table.append(tbody);
document.getElementById("summary-note").textContent = DATA.summary_note;

// Filters.
const samples = [...new Map(RUNS.map(r => [r.sample, r])).values()].sort((a, b) => a.sample.localeCompare(b.sample));
const topics = [...new Set(samples.map(s => s.topic))], templates = [...new Set(samples.map(s => s.template_id))];
const state = { topic: null, template: null };
function chips(id, values, key, name) {
  const host = document.getElementById(id);
  const all = el("button", { class: "chip", type: "button", "aria-pressed": "true", text: "All" });
  const buttons = [all, ...values.map(v => el("button", { class: "chip", type: "button", "aria-pressed": "false", "data-v": v, text: name(v) }))];
  for (const b of buttons) {
    b.addEventListener("click", () => {
      state[key] = b.dataset.v || null;
      buttons.forEach(x => x.setAttribute("aria-pressed", String(x === b)));
      applyFilters();
    });
    host.append(b);
  }
}
chips("topic-filters", topics, "topic", t => t);
chips("template-filters", templates, "template", t => FLOW[t] || t);
function applyFilters() {
  for (const card of document.querySelectorAll(".sample")) {
    card.hidden = (state.topic && card.dataset.topic !== state.topic) || (state.template && card.dataset.template !== state.template);
  }
}

// Sample cards.
const players = [];
function take(r, label) {
  if (!r) return el("div", { class: "take missing", "data-label": label, text: "Not recorded" });
  const audio = el("audio", { controls: "", preload: "none", src: r.audio });
  const pads = el("div", { class: "pads idle", "aria-label": "bar sources" });
  r.bars.forEach(b => pads.append(el("span", { class: `s-${b.source}`, title: `bar ${b.bar + 1}: ${b.text || ""}` })));
  const now = el("div", { class: "line-now muted", text: r.bars[0] ? r.bars[0].text : "" });
  const list = el("ol");
  r.bars.forEach(b => list.append(el("li", {}, el("i", { class: `dot s-${b.source}` }), b.text || "")));
  const why = b => ({ timing: "late", server_rejected: `server: ${b.detail}`, client_rejected: "failed client check" }[b.category] || b.reason || "other");
  const reasons = [...new Set(r.bars.filter(b => b.source === "espeak").map(why))];
  const details = [...new Set(r.bars.filter(b => b.source === "espeak").map(b => b.detail).filter(Boolean))].join("; ");
  const meta = el("div", { class: "meta" },
    el("span", { text: `${r.counts.remote}/${r.bars.length} server` }),
    r.counts.remote_older ? el("span", { text: `${r.counts.remote_older} older` }) : null,
    r.counts.espeak ? el("span", { text: `${r.counts.espeak} eSpeak (${reasons.join(", ")})`, title: details }) : null,
    r.mode === "realtime" && r.rtt_seconds_p50 != null ? el("span", { text: `network ${fmt(r.rtt_seconds_p50, 2)} s` }) : null);
  const box = el("div", { class: "take", "data-label": label }, audio, pads, now, meta, el("details", {}, el("summary", { text: "All 16 lines" }), list));
  const cells = [...pads.children], items = [...list.children];
  let last = -1;
  const paint = () => {
    const i = Math.min(r.bars.length - 1, Math.floor(audio.currentTime / BAR_SECONDS));
    if (i === last) return;
    if (last >= 0) { cells[last].classList.remove("on"); items[last].classList.remove("cur"); }
    cells[i].classList.add("on"); items[i].classList.add("cur"); now.textContent = r.bars[i].text || ""; last = i;
  };
  audio.addEventListener("play", () => { players.forEach(p => { if (p !== audio) p.pause(); }); pads.classList.remove("idle"); paint(); });
  audio.addEventListener("timeupdate", paint);
  audio.addEventListener("ended", () => pads.classList.add("idle"));
  players.push(audio);
  return box;
}
for (const s of samples) {
  const grid = el("div", { class: "grid6" }, el("div", { class: "colhead corner" }), ...MODES.map(([, l]) => el("div", { class: "colhead", text: l })));
  for (const v of VERSIONS) {
    grid.append(el("div", { class: "rowhead", text: v.id }, el("small", { text: v.name })));
    for (const [m, l] of MODES) grid.append(take(run(v.id, m, s.sample), `${v.id} · ${l}`));
  }
  document.getElementById("samples").append(el("article", { class: "sample", "data-topic": s.topic, "data-template": s.template_id },
    el("div", { class: "head" }, el("h3", { text: `${s.topic} · ${FLOW[s.template_id] || s.template_id} flow` }), el("span", { class: "sid", text: `${s.sample} · seed ${s.seed}` })),
    grid));
}

// Method notes.
for (const line of DATA.method) document.getElementById("method").append(el("li", { text: line }));
document.getElementById("repro").textContent = DATA.repro;
</script>
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--notes", type=Path, help="JSON with summary_note, method (list) and repro (text)")
    args = parser.parse_args()
    site = args.out / "site"
    data = json.loads((site / "data.json").read_text())
    notes = json.loads(args.notes.read_text()) if args.notes else {"summary_note": "", "method": [], "repro": ""}
    payload = {"versions": VERSIONS, "runs": data["runs"], **notes}
    text = json.dumps(payload, separators=(",", ":")).replace("</", "<\\/")
    (site / "index.html").write_text(PAGE.replace("__DATA__", text))
    print(f"{site / 'index.html'} ({len(data['runs'])} recordings)")


if __name__ == "__main__":
    main()
