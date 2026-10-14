"""Build the model-upgrade listening page from ``<out>/site/data.json`` (plan 2026-10-14).

    uv run python scripts/build_model_upgrade_page.py --notes demo_page/model_upgrade_notes.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

PAGE = r"""<title>StreamMUSE Model Upgrade</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Archivo:wdth,wght@62..125,500..900&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>
/* Layout: a listening notebook in three parts (lyrics, voices, live sessions), each a summary
   table first, then the material to read or hear. */
:root {
  --bg: #f3f4f6; --surface: #ffffff; --ink: #15171c; --muted: #5d6370; --line: #d9dce2;
  --accent: #2f45c4; --accent-soft: #e6e9fb; --good: #1d8a6d; --warn: #c4870a; --bad: #c4432e;
  --display: "Archivo", "Arial Narrow", system-ui, sans-serif;
  --body: "IBM Plex Sans", system-ui, -apple-system, "Segoe UI", sans-serif;
  --mono: "IBM Plex Mono", ui-monospace, "SFMono-Regular", Menlo, monospace;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    --bg: #111318; --surface: #1a1d24; --ink: #e8eaef; --muted: #9aa1ae; --line: #2b303a;
    --accent: #8b99ff; --accent-soft: #232a4a; --good: #3cc39a; --warn: #e3aa3c; --bad: #f06a55; color-scheme: dark;
  }
}
:root[data-theme="dark"] {
  --bg: #111318; --surface: #1a1d24; --ink: #e8eaef; --muted: #9aa1ae; --line: #2b303a;
  --accent: #8b99ff; --accent-soft: #232a4a; --good: #3cc39a; --warn: #e3aa3c; --bad: #f06a55; color-scheme: dark;
}
* { box-sizing: border-box; }
body { background: var(--bg); color: var(--ink); font: 15px/1.55 var(--body); }
.wrap { max-width: 1180px; margin: 0 auto; padding-inline: 20px; padding-block: 40px 72px; display: grid; gap: 52px; }
h1, h2, h3 { font-family: var(--display); text-wrap: balance; margin: 0; }
h1 { font-size: clamp(2.1rem, 5vw, 3.3rem); font-weight: 850; font-stretch: 72%; line-height: 1.02; }
h2 { font-size: 1.6rem; font-weight: 800; font-stretch: 80%; }
h3 { font-size: 1.1rem; font-weight: 750; font-stretch: 90%; }
p { margin: 0; max-width: 72ch; }
.muted { color: var(--muted); }
.eyebrow { font: 500 0.72rem/1 var(--mono); letter-spacing: 0.08em; text-transform: uppercase; color: var(--muted); }
section { display: grid; gap: 16px; }
header { display: grid; gap: 14px; }
.table-scroll { overflow-x: auto; border: 1px solid var(--line); border-radius: 10px; background: var(--surface); }
table { border-collapse: collapse; width: 100%; font-size: 0.88rem; font-variant-numeric: tabular-nums; }
th, td { padding: 8px 11px; text-align: right; border-bottom: 1px solid var(--line); white-space: nowrap; }
th:first-child, td:first-child { text-align: left; }
thead th { font: 500 0.7rem/1.25 var(--mono); letter-spacing: 0.05em; text-transform: uppercase; color: var(--muted); white-space: normal; vertical-align: bottom; }
tbody tr:last-child td { border-bottom: 0; }
td.best { color: var(--good); font-weight: 600; }
td small { color: var(--muted); font-size: 0.78em; }
.sheets { overflow-x: auto; border: 1px solid var(--line); border-radius: 10px; background: var(--surface); }
.sheet { display: grid; grid-auto-flow: column; grid-auto-columns: minmax(250px, 1fr); }
.sheet > div { padding: 12px 14px; border-right: 1px solid var(--line); min-width: 0; }
.sheet > div:last-child { border-right: 0; }
.sheet ol { margin: 8px 0 0; padding-left: 22px; font-size: 0.86rem; display: grid; gap: 2px; }
.sheet li::marker { font-family: var(--mono); color: var(--muted); font-size: 0.78em; }
.sheet .missing { color: var(--bad); font-style: italic; }
.clip-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(240px, 1fr)); gap: 10px; }
.clip { background: var(--surface); border: 1px solid var(--line); border-radius: 8px; padding: 10px; display: grid; gap: 6px; min-width: 0; }
.clip b { font: 600 0.84rem/1.3 var(--body); }
.clip small { color: var(--muted); font-size: 0.76rem; }
audio { width: 100%; height: 34px; }
.item { display: grid; gap: 10px; padding-block: 6px; }
.lyric { font-size: 0.95rem; }
.lyric span { color: var(--muted); }
.session { background: var(--surface); border: 1px solid var(--line); border-radius: 10px; padding: 14px; display: grid; gap: 8px; min-width: 0; }
.sessions { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 12px; }
@media (max-width: 760px) { .sessions { grid-template-columns: minmax(0, 1fr); } }
.meta { display: flex; flex-wrap: wrap; gap: 4px 12px; font: 400 0.78rem/1.3 var(--mono); color: var(--muted); }
details summary { cursor: pointer; font-size: 0.84rem; color: var(--accent); }
details ol { margin: 6px 0 0; padding-left: 26px; font-size: 0.84rem; max-height: 320px; overflow-y: auto; }
.chip { display: inline-block; font: 500 0.7rem/1 var(--mono); padding: 3px 6px; border-radius: 4px; background: var(--accent-soft); color: var(--accent); }
ul.notes { margin: 0; padding-left: 20px; display: grid; gap: 8px; max-width: 82ch; }
summary:focus-visible, audio:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
</style>

<div class="wrap">
<header>
  <div class="eyebrow">StreamMUSE · real-time rap · model upgrade · 2026-10-14</div>
  <h1>StreamMUSE Model Upgrade</h1>
  <p id="intro"></p>
</header>

<section aria-labelledby="llm-h">
  <h2 id="llm-h">1 · Lyric writers</h2>
  <p class="muted" id="llm-note"></p>
  <div class="table-scroll"><table id="llm-table"></table></div>
  <div id="sheets"></div>
</section>

<section aria-labelledby="tts-h">
  <h2 id="tts-h">2 · Voices</h2>
  <p class="muted" id="tts-note"></p>
  <div class="table-scroll"><table id="tts-table"></table></div>
  <div class="clip-grid" id="reference"></div>
  <div id="items"></div>
</section>

<section aria-labelledby="sess-h">
  <h2 id="sess-h">3 · Live sessions</h2>
  <p class="muted" id="sess-note"></p>
  <div class="table-scroll"><table id="sess-table"></table></div>
  <div class="sessions" id="sessions"></div>
</section>

<section aria-labelledby="method-h">
  <h2 id="method-h">How these were made</h2>
  <ul class="notes" id="method"></ul>
</section>
</div>

<script type="application/json" id="data">__DATA__</script>
<script>
const D = JSON.parse(document.getElementById("data").textContent);
const el = (tag, attrs = {}, ...kids) => {
  const n = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) { if (k === "class") n.className = v; else if (k === "text") n.textContent = v; else n.setAttribute(k, v); }
  for (const kid of kids) if (kid != null) n.append(kid);
  return n;
};
const fmt = (x, d = 0) => x == null ? "–" : Number(x).toFixed(d);
const pct = x => x == null ? "–" : (100 * x).toFixed(1) + "%";
const players = [];
const player = src => {
  const a = el("audio", { controls: "", preload: "none", src });
  a.addEventListener("play", () => players.forEach(p => { if (p !== a) p.pause(); }));
  players.push(a);
  return a;
};
function table(id, headers, rows, best = {}) {
  const t = document.getElementById(id);
  t.append(el("thead", {}, el("tr", {}, ...headers.map(h => el("th", { text: h })))));
  const tb = el("tbody");
  for (const r of rows) tb.append(el("tr", {}, ...r.map((c, i) => {
    const td = el("td", {}, ...(Array.isArray(c) ? c : [c == null ? "–" : String(c)]));
    if (best[i] && best[i] === c) td.className = "best";
    return td;
  })));
  t.append(tb);
}
const bestOf = (rows, i, dir) => { const v = rows.map(r => r[i]).filter(x => x != null && x !== "–"); return v.length ? (dir > 0 ? v.reduce((a, b) => (parseFloat(b) > parseFloat(a) ? b : a)) : v.reduce((a, b) => (parseFloat(b) < parseFloat(a) ? b : a))) : null; };

document.getElementById("intro").textContent = D.notes.intro;
document.getElementById("llm-note").textContent = D.notes.llm;
document.getElementById("tts-note").textContent = D.notes.tts;
document.getElementById("sess-note").textContent = D.notes.sessions;
for (const line of D.notes.method) document.getElementById("method").append(el("li", { text: line }));

// 1. Lyric writers.
const lr = D.llm.map(m => { const s = m.summary; return [m.name, pct(s.first_wave_pass_rate), pct(s.rescue_rate), String(s.failed_bars), pct(s.couplet_rhyme_rate), fmt(s.distinct_end_words_per_16_bars, 1), fmt(s.mean_score, 3), fmt(s.first_wave_ms.p50), fmt(s.bar_wall_ms.p95)]; });
table("llm-table", ["Model", "First-wave pass", "Rescue", "Failed bars", "Couplet rhyme", "End words / 16", "Mean score", "Write p50 ms", "Bar p95 ms"], lr,
  { 1: bestOf(lr, 1, 1), 4: bestOf(lr, 4, 1), 5: bestOf(lr, 5, 1), 6: bestOf(lr, 6, 1), 7: bestOf(lr, 7, -1) });
for (const s of D.lyric_samples) {
  const sheet = el("div", { class: "sheet" });
  for (const m of D.llm.filter(m => !m.label.endsWith("hyphens"))) {
    const ol = el("ol");
    (m.lines[s.sample] || []).forEach(t => ol.append(el("li", t ? { text: t } : { class: "missing", text: "no line" })));
    sheet.append(el("div", {}, el("b", { text: m.name }), ol));
  }
  document.getElementById("sheets").append(el("div", { class: "item" },
    el("h3", { text: `${s.topic} · ${s.template_id.replace("baseline_", "").replace("_9", "")} flow` }), el("div", { class: "sheets" }, sheet)));
}

// 2. Voices.
const tr = D.tts.map(t => { const q = t.quality || {}; return [[t.label, el("br"), el("small", { text: t.note })], fmt(100 * q.wer, 1) + "%", fmt(q.speaker_similarity_median, 3), fmt(q.speaker_similarity_min, 3), fmt(q.reference_similarity_median, 3), fmt(q.seam_loudness_jump_db_median, 1), fmt(q.seam_pitch_jump_semitones_median, 2), t.latency_ms ? fmt(t.latency_ms.p50) : "–", t.latency_ms ? fmt(t.latency_ms.p95) : "–"]; });
table("tts-table", ["Voice", "WER", "Bar-to-bar similarity", "Worst pair", "Similarity to reference", "Seam loudness jump dB", "Seam pitch jump st", "Per bar p50 ms", "Per bar p95 ms"], tr,
  { 1: bestOf(tr, 1, -1), 2: bestOf(tr, 2, 1), 3: bestOf(tr, 3, 1), 4: bestOf(tr, 4, 1), 5: bestOf(tr, 5, -1), 6: bestOf(tr, 6, -1), 7: bestOf(tr, 7, -1) });
document.getElementById("reference").append(el("div", { class: "clip" }, el("b", { text: "Reference voice (what every model clones)" }), player("audio/reference.mp3")));
for (const it of D.tts_items) {
  const grid = el("div", { class: "clip-grid" });
  for (const t of D.tts) { const src = t.clips[String(it.item)]; if (src) grid.append(el("div", { class: "clip" }, el("b", { text: t.label }), el("small", { text: t.note }), player(src))); }
  document.getElementById("items").append(el("div", { class: "item" },
    el("div", { class: "lyric" }, el("span", { text: `Item ${it.item} · ` }), `${it.bar_texts[0]} / ${it.bar_texts[1]}`), grid));
}

// 3. Live sessions.
const sr = D.sessions.map(s => { const m = s.summary; const ch = m.choices || {}; const n = m.bars || 1; const react = (m.edits || []).map(e => e.reaction_ms).filter(x => x != null);
  return [`${s.name} @ ${s.tempo} BPM`, String(n), pct((ch.remote_latest || 0) / n), String(ch.remote_older || 0), String((ch.espeak_stream_lyric || 0) + (ch.espeak_local_lyric || 0)), String(m.underruns ?? "–"), fmt((m.startup_wait_ms || 0) / 1000, 1), fmt((m.lyric_lead_ms || {}).p5), fmt((m.remote_ready_lead_ms || {}).min), fmt((m.remote_ready_lead_ms || {}).p5), fmt((m.server_moss_ms || {}).p50), react.length ? fmt(Math.max(...react)) : "–"]; });
table("sess-table", ["Stack", "Bars", "Latest", "Older", "eSpeak", "Underruns", "Startup wait s", "Lyric lead p5 ms", "Audio lead min ms", "Audio lead p5 ms", "MOSS p50 ms", "Edit → new bar max ms"], sr);
for (const s of D.sessions) {
  const ol = el("ol");
  s.lines.forEach(l => ol.append(el("li", { text: `${l.text || ""}${l.choice && l.choice !== "remote_latest" ? "  [" + l.choice + "]" : ""}` })));
  const m = s.summary;
  document.getElementById("sessions").append(el("div", { class: "session" },
    el("h3", { text: `${s.name}` }), el("span", { class: "chip", text: `${s.tempo} BPM · ${m.bars} bars` }), player(s.audio),
    el("div", { class: "meta" }, el("span", { text: `startup ${fmt((m.startup_wait_ms || 0) / 1000, 1)} s` }), el("span", { text: `audio lead min ${fmt((m.remote_ready_lead_ms || {}).min)} ms` }), el("span", { text: `continued ${pct(m.continued_rate)}` })),
    el("details", {}, el("summary", { text: "All lines" }), ol)));
}
</script>
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=ROOT / "output" / "model_upgrade_20261014")
    parser.add_argument("--notes", type=Path, required=True)
    args = parser.parse_args()
    site = args.out / "site"
    data = json.loads((site / "data.json").read_text())
    data["notes"] = json.loads(args.notes.read_text())
    text = json.dumps(data, separators=(",", ":")).replace("</", "<\\/")
    (site / "index.html").write_text(PAGE.replace("__DATA__", text))
    print(site / "index.html")


if __name__ == "__main__":
    main()
