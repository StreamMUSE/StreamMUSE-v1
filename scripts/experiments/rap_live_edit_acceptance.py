"""Measure live topic/flow edits against an already running, local audible GUI.

No models or servers are launched. Event monotonic timestamps share the local
host clock. Playback latency is an observer/device-callback scheduling proxy,
not a DAC timestamp or microphone measurement. The tool owns transport controls
while running and uses stop-after-bar on every exit.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import shutil
import threading
import time
import urllib.error
import urllib.request
from urllib.parse import urlsplit

if __package__:
    from .rap_interaction_acceptance import Acceptance, require_device_output
else:
    from rap_interaction_acceptance import Acceptance, require_device_output


TOPICS = (
    "quiet trains at sunrise", "rain across a neon city", "mountain paths in winter",
    "satellites above the ocean", "late nights in the library", "summer basketball on the block",
)
CURRENT_VOCAL_SOURCES = {"remote_latest": "moss_lyric_stream", "espeak_stream_lyric": "lyric_stream_espeak"}


def quantiles(values):
    values = sorted(values)
    if not values:
        return {"n": 0, "p50": None, "p95": None, "p99": None, "max": None}

    def percentile(p):
        index = (len(values) - 1) * p
        lo = math.floor(index)
        return values[lo] + (values[math.ceil(index)] - values[lo]) * (index - lo)

    return {"n": len(values), "p50": percentile(.5), "p95": percentile(.95),
            "p99": percentile(.99), "max": values[-1]}


def stage_quantiles(events):
    stages = defaultdict(list)
    for event in events:
        if event["event_type"] != "chunk_remote_completed":
            continue
        for name, value in event["payload"].get("stage_timings_ms", {}).items():
            if type(value) in (int, float) and math.isfinite(value) and value >= 0:
                stages[name].append(value)
    return {name: quantiles(values) for name, values in stages.items()}


def marker(event, start_ns):
    return {"bar": event["bar"], "event_sequence": event["sequence"],
            "monotonic_ns": event["monotonic_ns"],
            "latency_ms": (event["monotonic_ns"] - start_ns) / 1e6}


def identity_matches(payload, edit):
    return (payload.get("topic") == edit["topic"]
            and payload.get("template_id") == edit["template_id"]
            and type(payload.get("revision")) is int
            and isinstance(payload.get("lyric_id"), str) and bool(payload["lyric_id"]))


def played_choices(events):
    """Freeze the source classification at each playback notice, never later."""
    frozen, choices = {}, {}
    for event in sorted(events, key=lambda e: e["sequence"]):
        key = (event["session_id"], event["bar"])
        if event["event_type"] == "bar_frozen":
            frozen[key] = event["payload"].get("choice") or "unknown"
        elif event["event_type"] == "bar_playback_started":
            choices[event["sequence"]] = frozen.get(key, "unknown")
    return choices


def summarize_edit(events, edit, *, now_ns):
    """Join exact bar/lyric identities, never edit_seq to per-bar revision.

    The observation window is capped at the request deadline even in offline
    reanalysis. Missing/late playback is not replaced with a timeout-valued
    latency. eSpeak events currently lack revision IDs, so they remain unproven.
    """
    start = edit["request_start_ns"]
    deadline = start + round(edit["timeout_seconds"] * 1e9)
    end = min(now_ns, deadline)
    events = sorted((e for e in events if e["monotonic_ns"] <= end), key=lambda e: e["sequence"])
    resets = [e["sequence"] for e in events
              if e["event_type"] == "session_reset" and e["monotonic_ns"] < start]
    if resets:
        events = [e for e in events if e["sequence"] > max(resets)]
    window = [e for e in events if e["monotonic_ns"] >= start]
    result = {**edit, "status": "pending", "ack_ms": None,
              "lyric_ready": None, "first_matching_commit": None,
              "first_matching_playback": None, "first_scheduled_vocal_onset": None,
              "matching_playback_completed": None,
              "muted_vocals": False,
              "espeak_ready": {"status": "unavailable_revision_identity", "latency_ms": None,
                               "confirmed_ready_by_commit_ms": None,
                               "bar_observations": [],
                               "reason": "audio_render_completed has no lyric_id or revision"},
              "underruns": sum(e["event_type"] == "audio_underrun" for e in window),
              "render_stage_quantiles_ms": stage_quantiles(window)}
    if edit.get("ack_ns") is not None:
        result["ack_ms"] = (edit["ack_ns"] - start) / 1e6
    effective = edit.get("ack", {}).get("effective_from_bar")
    lyrics = [e for e in window if e["event_type"] == "bar_reserved"
              and type(effective) is int and type(e["bar"]) is int and e["bar"] >= effective
              and identity_matches(e["payload"], edit)]
    if lyrics:
        result["lyric_ready"] = {**marker(lyrics[0], start), **{
            key: lyrics[0]["payload"][key] for key in ("lyric_id", "revision")}}

    frozen = {e["bar"]: e for e in events if e["event_type"] == "bar_frozen"}
    completed = {e["bar"]: e for e in events if e["event_type"] == "bar_playback_completed"}
    all_plays = [e for e in events if e["event_type"] == "bar_playback_started"]
    preceding = [e for e in all_plays if e["monotonic_ns"] < start]
    current = preceding[-1:]  # A newer bar start proves older bars are no longer current.
    plays = [e for e in current if completed.get(e["bar"], {}).get("monotonic_ns", end) >= start]
    plays += [e for e in all_plays if e["monotonic_ns"] >= start]
    matching = {}
    for lyric in lyrics:
        bar, record = lyric["bar"], lyric["payload"]
        selected = frozen.get(bar)
        if selected is None or selected["sequence"] < lyric["sequence"]:
            continue
        payload = selected["payload"]
        choice = payload.get("choice")
        if (not identity_matches(payload, edit)
                or any(payload.get(k) != record[k] for k in ("lyric_id", "revision"))
                or payload.get("latest_revision") != record["revision"]
                or selected.get("request_id") != record["lyric_id"]
                or choice not in CURRENT_VOCAL_SOURCES
                or payload.get("source") != CURRENT_VOCAL_SOURCES[choice]
                or not payload.get("text", "").strip() or payload["text"] != record.get("text")):
            continue
        commits = [e for e in window if e["event_type"] == "bar_audio_committed" and e["bar"] == bar
                   and e["sequence"] > selected["sequence"]
                   and e.get("request_id") == record["lyric_id"]
                   and e["payload"].get("choice") == choice
                   and e["payload"].get("source") == CURRENT_VOCAL_SOURCES[choice]]
        if not commits:
            continue
        commit = commits[0]
        identity = {key: record[key] for key in ("topic", "template_id", "revision", "lyric_id")}
        identity["choice"] = payload.get("choice")
        matching[bar] = (commit, identity)
        candidate = {**marker(commit, start), **identity}
        if result["first_matching_commit"] is None or candidate["monotonic_ns"] < result["first_matching_commit"]["monotonic_ns"]:
            result["first_matching_commit"] = candidate

    for play in plays:
        if play["bar"] not in matching:
            continue
        commit, identity = matching[play["bar"]]
        if play["sequence"] <= commit["sequence"]:
            continue
        result["first_matching_playback"] = {**marker(play, start), **identity,
                                             "absolute_frame": play["payload"].get("absolute_frame")}
        if identity["choice"] == "espeak_stream_lyric":
            result["espeak_ready"]["confirmed_ready_by_commit_ms"] = marker(commit, start)["latency_ms"]
        evidence = commit["payload"].get("timing_evidence") or {}
        mix = evidence.get("mix") or {}
        result["muted_vocals"] = any(mix.get(name) == 0 for name in ("vocal_gain", "master_gain"))
        rate, frames = evidence.get("sample_rate_hz"), evidence.get("frame_count")
        onsets = [item["sample"] for item in evidence.get("requested_onsets", [])
                  if type(item.get("sample")) is int and item["sample"] >= 0]
        frame = play["payload"].get("absolute_frame")
        done = completed.get(play["bar"])
        if (done and done["sequence"] > play["sequence"] and type(frame) is int
                and type(frames) is int and frames > 0
                and done["payload"].get("absolute_frame") == frame + frames):
            result["matching_playback_completed"] = marker(done, start)
        if (onsets and type(rate) is int and rate > 0 and type(frame) is int and frame >= 0
                and type(frames) is int and min(onsets) < frames and evidence.get("bar") == play["bar"]
                and result["matching_playback_completed"] and not result["muted_vocals"]):
            offset = min(onsets)
            onset_ns = play["monotonic_ns"] + round(offset / rate * 1e9)
            if onset_ns <= end:
                result["first_scheduled_vocal_onset"] = {
                    "bar": play["bar"], "kind": "device_callback_schedule_proxy",
                    "is_acoustic_observation": False, "monotonic_ns": onset_ns,
                    "latency_ms": (onset_ns - start) / 1e6, "sample_rate_hz": rate,
                    "bar_absolute_frame": frame, "onset_offset_samples": offset,
                    "playback_sample_index": frame + offset,
                    "bar_end_sample_index": frame + frames,
                    "limitation": "observer-emitted callback event plus scheduled offset; not measured acoustic onset",
                }
        break

    lyric_bars = {e["bar"] for e in lyrics}
    result["espeak_ready"]["bar_observations"] = [marker(e, start) for e in window
        if e["event_type"] == "audio_render_completed" and e["bar"] in lyric_bars
        and e["payload"].get("source") == "lyric_stream_espeak"]
    match = result["first_matching_playback"]
    old = [e["bar"] for e in plays if match is None or e["sequence"] < match["event_sequence"]]
    result["old_bars"] = old
    result["old_bar_count"] = len(old)
    result["in_flight_bars_at_request"] = [e["bar"] for e in plays if e["monotonic_ns"] < start]
    choices = played_choices(events)
    counts = Counter(choices[e["sequence"]] for e in plays)
    result["choice_counts"] = {key: counts[key] for key in
                              ("remote_latest", "remote_older", "espeak_stream_lyric", "espeak_local_lyric", "drums_only", "silence", "unknown")}
    result["espeak_count"] = sum(n for name, n in counts.items() if name.startswith("espeak"))
    if result["underruns"] or result["muted_vocals"] or any(e["event_type"] in ("audio_device_failed", "session_reset") for e in window):
        result.update(status="failed", error="underrun, muted vocals, device failure, or reset during edit")
    elif (result["first_scheduled_vocal_onset"] and result["matching_playback_completed"]
          and edit.get("ack_ns") is not None and edit["ack_ns"] <= deadline):
        result["status"] = "passed"
    elif now_ns >= deadline:
        result.update(status="timed_out", error="matching revision playback/onset/completion not observed before deadline")
    return result


def select_templates(templates):
    curated = [t for t in templates if t.get("template_id", "").startswith("mcflow_") and t.get("slots")]
    delayed = [t for t in curated if min(s["tick_in_bar"] for s in t["slots"]) > 0]
    if not delayed:
        raise ValueError("curated templates must include a leading rest")
    return [delayed[0], *[t for t in curated if t != delayed[0]]]


class EventEvidence:
    """Retain canonical events across bounded snapshots and flag missing spans."""

    def __init__(self, output):
        self.path = Path(output) / "events.jsonl"
        self._events = {}
        self._last = {}
        self.gaps = []
        self.state = {}
        self._lock = threading.RLock()

    @property
    def events(self):
        with self._lock:
            return list(self._events.values())

    def observe(self, state, received_ns):
        with self._lock, self.path.open("a") as output:
            if self.state and any(state.get(key) != self.state.get(key)
                                  for key in ("session_id", "coordinator_epoch")):
                raise ValueError("session or epoch changed during continuous measurement")
            self.state = state
            for event in sorted(state.get("recent_events", []), key=lambda e: e["sequence"]):
                session, seq = event["session_id"], event["sequence"]
                if (session, seq) in self._events:
                    continue
                if event["monotonic_ns"] > received_ns:
                    raise ValueError("event clock is ahead of local receipt; same-host clock required")
                last = self._last.get(session)
                if last is not None and seq > last + 1:
                    self.gaps.append({"session_id": session, "after": last, "before": seq})
                self._last[session] = max(seq, last or seq)
                self._events[session, seq] = event
                output.write(json.dumps({"observed_ns": received_ns, "event": event}) + "\n")


class LiveAcceptance(Acceptance):
    """Existing control helpers with precise, thread-safe HTTP evidence."""

    def __init__(self, url, output):
        super().__init__(url, output)
        self._http_lock = threading.Lock()

    def request_timed(self, path, payload=None, *, timeout=15):
        data = None if payload is None else json.dumps(payload).encode()
        request = urllib.request.Request(self.url + path, data=data,
                                         headers={"Content-Type": "application/json"})
        row = {"path": path, "input": payload, "status": None, "body": None,
               "request_start_ns": time.monotonic_ns()}
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                row["status"] = response.status
                row["body"] = json.load(response)
        except urllib.error.HTTPError as error:
            row["status"] = error.code
            raw = error.read().decode("utf-8", errors="replace")
            try:
                row["body"] = json.loads(raw)
            except ValueError:
                row["body"] = raw
            row["error"] = str(error)
        except Exception as error:
            row["error"] = repr(error)
        row["response_ns"] = time.monotonic_ns()
        row["elapsed_ms"] = (row["response_ns"] - row["request_start_ns"]) / 1e6
        with self._http_lock:
            self.evidence.write(json.dumps(row) + "\n")
            self.evidence.flush()
        return row

    def request(self, path, payload=None):
        return self.checked(self.request_timed(path, payload))

    @staticmethod
    def checked(row):
        if row.get("error") or row["status"] is None or row["status"] >= 400:
            raise RuntimeError(f"{row['path']}: {row.get('error') or row['body']}")
        return row["body"]

    def stop(self):
        deadline = time.monotonic() + 65
        while time.monotonic() < deadline:
            state = self.request("/api/control/settings")
            if state["state"] == "stopped":
                return state
            if state["state"] != "stop_requested":
                # Start can temporarily hold the control lock while priming.
                row = self.request_timed("/api/control/stop", {})
                if row["status"] == 409:
                    time.sleep(.15)
                    continue
                self.checked(row)
            # This endpoint requests a musical bar boundary, not abrupt shutdown.
            return self.wait("/api/control/settings", lambda s: s["state"] == "stopped",
                             timeout=max(.01, deadline - time.monotonic()))
        raise TimeoutError("stop-after-bar did not leave the GUI ready")


class StatePoller:
    """Poll independently so a blocking edit ACK cannot hide the first playback."""

    def __init__(self, test, evidence, interval):
        self.test, self.evidence, self.interval = test, evidence, interval
        self.error = None
        self.settings = {}
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="live-edit-evidence", daemon=True)

    def sample(self):
        row = self.test.request_timed("/api/state", timeout=5)
        self.evidence.observe(self.test.checked(row), row["response_ns"])
        self.settings = self.test.checked(self.test.request_timed("/api/control/settings", timeout=5))

    def start(self):
        self.sample()
        self._thread.start()

    def _run(self):
        try:
            while not self._stop.wait(self.interval):
                self.sample()
                self.check()
        except Exception as error:
            self.error = repr(error)

    def check(self):
        if self.error:
            raise RuntimeError(f"event collection failed: {self.error}")
        if self.evidence.gaps:
            raise RuntimeError(f"event evidence has gaps: {self.evidence.gaps}")
        if self.settings.get("state") == "running":
            output = self.settings.get("audio_output", {})
            require_device_output(output)
            mix = output.get("live_mix") or {}
            assert all((mix.get(key) or {}).get("vocal_gain") != 0 for key in ("applied", "requested")), "live vocal gain is muted"

    def close(self):
        self._stop.set()
        if self._thread.ident is not None:
            self._thread.join()


def warmed_bars(events):
    frozen = {e["bar"]: e["payload"] for e in events if e["event_type"] == "bar_frozen"}
    return sorted({e["bar"] for e in events if e["event_type"] == "bar_playback_completed"
                   and frozen.get(e["bar"], {}).get("choice") == "remote_latest"})


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def measure_edit(test, poller, *, index, topic, template, args):
    row = test.request_timed("/api/control/plan", {"topic": topic, "template_id": template["template_id"]},
                             timeout=args.timeout)
    accepted = not row.get("error") and row["status"] is not None and row["status"] < 400
    edit = {"index": index, "topic": topic, "template_id": template["template_id"],
            "first_flow_tick": min(s["tick_in_bar"] for s in template["slots"]),
            "request_start_ns": row["request_start_ns"], "ack_ns": row["response_ns"] if accepted else None,
            "http_response_ns": row["response_ns"], "http_status": row["status"],
            "ack": row["body"] if isinstance(row["body"], dict) else {},
            "timeout_seconds": args.timeout}
    try:
        test.checked(row)
        if type(edit["ack"].get("effective_from_bar")) is not int:
            raise ValueError("plan ACK did not identify effective_from_bar")
        while True:
            poller.check()
            result = summarize_edit(poller.evidence.events, edit, now_ns=time.monotonic_ns())
            if result["status"] != "pending":
                return result
            if poller.settings.get("state") == "stopped":
                raise RuntimeError("session stopped during edit")
            time.sleep(args.poll_interval)
    except Exception as error:
        result = summarize_edit(poller.evidence.events, edit, now_ns=time.monotonic_ns())
        result.update(status="failed", error=repr(error), first_scheduled_vocal_onset=None)
        return result


def run_bpm(args, bpm, output):
    test = LiveAcceptance(args.url, output)
    evidence = EventEvidence(output)
    poller = StatePoller(test, evidence, args.poll_interval)
    summary = {"bpm": bpm, "status": "running", "edits": [], "ready": False,
               "measurement_kind": "device_callback_schedule_proxy", "microphone_capture": None,
               "clock_basis": "same-host time.monotonic_ns; canonical event emission, not hardware DAC time"}
    try:
        test.stop()
        test.request("/api/control/settings", {"mode": "continuous", "tempo_bpm": bpm, "vocal_gain": .8})
        test.request("/api/control/reset", {})
        options = test.request("/api/control/plan")
        if not options.get("supported"):
            raise ValueError("GUI does not support live plan edits")
        templates = select_templates(options["templates"])
        summary["templates"] = templates
        summary["session"] = test.request("/api/session")
        poller.start()
        test.request("/api/control/start", {})
        deadline = time.monotonic() + args.warmup_timeout
        while time.monotonic() < deadline:
            poller.check()
            warm = warmed_bars(evidence.events)
            if poller.settings.get("state") == "running" and len(warm) >= args.warmup_bars:
                break
            time.sleep(args.poll_interval)
        else:
            raise TimeoutError("warmed remote_latest device playback did not complete before deadline")
        summary["warmup"] = {"remote_completed_bars": warm, "completed_ns": time.monotonic_ns()}
        require_device_output(poller.settings["audio_output"])
        recording = poller.settings["audio_output"].get("recording", {})
        if recording.get("source") != "completed_callback_pcm" or not recording.get("exact_completed_bars"):
            raise ValueError("completed device-callback WAV recording required (use GUI composite audio output)")
        summary["audio_output"] = poller.settings["audio_output"]
        if not evidence.state.get("audio", {}).get("recording_path"):
            raise ValueError("GUI did not expose a session WAV recording path")
        for index in range(args.edits):
            poller.check()
            topic = TOPICS[index % len(TOPICS)]
            if index >= len(TOPICS):
                topic += f" verse {index + 1}"
            result = measure_edit(test, poller, index=index, topic=topic,
                                  template=templates[index % len(templates)], args=args)
            summary["edits"].append(result)
            with (output / "edits.jsonl").open("a") as handle:
                handle.write(json.dumps(result) + "\n")
            write_json(output / "summary.json", summary)
            print(json.dumps({"bpm": bpm, "edit": index, "status": result["status"],
                              "onset": result["first_scheduled_vocal_onset"]}), flush=True)
            if index + 1 < args.edits:
                time.sleep(args.edit_spacing)
        summary["status"] = "passed" if all(e["status"] == "passed" for e in summary["edits"]) else "failed"
    except BaseException as error:
        summary.update(status="failed", error=repr(error))
    finally:
        # Persist measurements even when stopping, recording, or polling fails.
        try:
            summary["final_settings"] = test.stop()
            summary["ready"] = summary["final_settings"]["state"] == "stopped"
        except BaseException as error:
            summary.update(status="failed", cleanup_error=repr(error))
        poller.close()
        try:
            poller.sample()
            poller.check()
        except Exception as error:
            summary.update(status="failed", evidence_error=repr(error))
        summary["event_gaps"] = evidence.gaps
        summary["underruns"] = evidence.state.get("audio", {}).get("underruns")
        if summary["underruns"] != 0:
            summary.update(status="failed", underrun_error="session underruns missing or nonzero")
        summary["render_stage_quantiles_ms"] = stage_quantiles(evidence.events)
        summary["latency_quantiles_ms"] = {
            key: quantiles([e[key]["latency_ms"] for e in summary["edits"] if e.get(key) is not None])
            for key in ("lyric_ready", "first_matching_commit", "first_matching_playback", "first_scheduled_vocal_onset")
        }
        summary["latency_quantiles_note"] = "observed milestones only; timed-out edits retained separately, never timeout-valued samples"
        frozen = {e["bar"]: e["payload"] for e in evidence.events if e["event_type"] == "bar_frozen"}
        summary["committed_choice_counts"] = dict(Counter(e.get("choice", "unknown") for e in frozen.values()))
        summary["choice_counts"] = dict(Counter(played_choices(evidence.events).values()))
        source = evidence.state.get("audio", {}).get("recording_path")
        if source:
            summary["recording"] = {"source_path": source, "preserved_path": None,
                                     "sample_index_basis": "play_started absolute_frame + requested onset; reset-origin run",
                                     "is_acoustic_capture": False}
            try:
                if not summary["ready"]:
                    raise RuntimeError("cannot preserve a final recording until stop-after-bar completes")
                destination = output / "session.wav"
                shutil.copyfile(source, destination)
                summary["recording"]["preserved_path"] = str(destination.resolve())
                for edit in summary["edits"]:
                    edit["session_wav_path"] = str(destination.resolve())
                    onset = edit["first_scheduled_vocal_onset"]
                    if onset:
                        rate = onset["sample_rate_hz"]
                        start = max(0, onset["playback_sample_index"] - round(rate * .5))
                        edit["audition"] = {
                            "wav_path": str(destination.resolve()), "sample_rate_hz": rate,
                            "start_sample": start, "onset_sample": onset["playback_sample_index"],
                            "end_sample": onset["bar_end_sample_index"],
                            "start_seconds": start / rate,
                            "duration_seconds": (onset["bar_end_sample_index"] - start) / rate,
                        }
            except Exception as error:
                summary.update(status="failed", recording_error=repr(error))
        (output / "edits.jsonl").write_text("".join(json.dumps(e) + "\n" for e in summary["edits"]))
        write_json(output / "summary.json", summary)
        test.evidence.close()
    return summary


def run(args):
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "summary.json").exists():
        raise FileExistsError("choose a fresh output directory to preserve prior evidence")
    summary = {"status": "running", "url": args.url, "runs": []}
    for index, bpm in enumerate(args.bpms, 1):
        output = args.output / f"{index:02d}_{bpm:g}bpm"
        output.mkdir(exist_ok=False)
        summary["runs"].append(run_bpm(args, bpm, output))
        write_json(args.output / "summary.json", summary)
        if not summary["runs"][-1]["ready"] or "error" in summary["runs"][-1]:
            break
    summary["status"] = "passed" if (len(summary["runs"]) == len(args.bpms)
                                      and all(r["status"] == "passed" for r in summary["runs"])) else "failed"
    write_json(args.output / "summary.json", summary)
    return summary


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8018")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bpm", "--bpms", dest="bpms", type=float, nargs="+", default=[60])
    parser.add_argument("--edits", type=int, default=6, help="Edits per BPM")
    parser.add_argument("--edit-spacing", type=float, default=2, help="Seconds after each edit outcome")
    parser.add_argument("--timeout", type=float, default=60, help="Per-edit deadline including ACK and bar completion")
    parser.add_argument("--warmup-timeout", type=float, default=120)
    parser.add_argument("--warmup-bars", type=int, default=2, help="Required remote_latest completed bars before editing")
    parser.add_argument("--poll-interval", type=float, default=.1)
    args = parser.parse_args(argv)
    url = urlsplit(args.url)
    if url.scheme != "http" or url.hostname not in ("127.0.0.1", "localhost", "::1") or url.path not in ("", "/"):
        parser.error("GUI must be a local HTTP URL: request/event monotonic clocks must share a host")
    if args.edits < 1 or args.warmup_bars < 1:
        parser.error("edits and warmup-bars must be positive")
    for value in [*args.bpms, args.timeout, args.warmup_timeout, args.poll_interval]:
        if not math.isfinite(value) or value <= 0:
            parser.error("BPMs and timeout/poll values must be finite and positive")
    if any(not 30 <= bpm <= 180 for bpm in args.bpms):
        parser.error("GUI supports BPMs from 30 to 180")
    if not math.isfinite(args.edit_spacing) or args.edit_spacing < 0:
        parser.error("edit-spacing must be finite and non-negative")
    return args


if __name__ == "__main__":
    args = parse_args()
    result = run(args)
    print(json.dumps({"status": result["status"], "output": str(args.output)}))
    raise SystemExit(0 if result["status"] == "passed" else 1)
