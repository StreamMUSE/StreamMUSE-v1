"""Observe three directed-story edits through an already running local v3 demo.

Owns demo transport controls for the run: stop, configure, reset, start, and
stop-after-bar in finally. Launches no models or services. Playback callbacks
are software observations, not acoustic measurements or semantic fulfillment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time
from urllib.parse import urlsplit
from uuid import uuid4

if __package__:
    from ..rap_live_edit_acceptance import LiveAcceptance, EventEvidence, StatePoller, write_json
else:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from rap_live_edit_acceptance import LiveAcceptance, EventEvidence, StatePoller, write_json


DEFAULT_EVENTS = (
    "A courier loses the map.",
    "A stranger offers help.",
    "The stranger wants the package.",
)


def playback_observations(events):
    """Join the chosen lyric to a subsequent callback, never to a newer draft."""
    frozen, seen, observations = {}, set(), []
    for event in sorted(events, key=lambda item: item["sequence"]):
        key = (event["session_id"], event["bar"])
        if event["event_type"] == "bar_frozen":
            frozen[key] = event
        elif event["event_type"] == "bar_playback_started" and key not in seen:
            seen.add(key)
            chosen = frozen.get(key)
            payload = chosen["payload"] if chosen else {}
            provenance = payload.get("provenance") or {}
            version = payload.get("story_version", provenance.get("story_version"))
            ids = payload.get("story_event_ids", provenance.get("story_event_ids", []))
            identity = payload.get("lyric_id")
            callback_id = event["payload"].get("lyric_id")
            verified = (bool(identity) and type(version) is int and version > 0
                        and isinstance(ids, (list, tuple)) and bool(ids)
                        and (callback_id is None or callback_id == identity))
            observations.append({
                "session_id": event["session_id"], "bar": event["bar"],
                "event_sequence": event["sequence"], "monotonic_ns": event["monotonic_ns"],
                "absolute_frame": event["payload"].get("absolute_frame"),
                "frozen_event_sequence": chosen["sequence"] if chosen else None,
                "lyric_id": identity, "revision": payload.get("revision"),
                "text": payload.get("text"), "choice": payload.get("choice"),
                "source": payload.get("source"), "story_version": version if verified else None,
                "story_event_ids": list(ids) if verified else [],
                "identity_evidence": "same_session_bar_frozen_before_callback" if verified else "unverified",
                "is_acoustic_observation": False, "semantic_fulfillment": "not_evaluated",
            })
    return observations


def first_playback_for(story_event, observations):
    return next((row for row in observations
                 if story_event["event_id"] in row["story_event_ids"]
                 and row["story_version"] >= story_event["sequence"]), None)


def preserve_mix_artifact(session, output, *, stopped):
    """Best-effort per-run copy; recording availability never changes acceptance."""
    result = {"status": "unavailable", "source_path": None, "path": None,
              "sha256": None, "warning": None, "is_acoustic_capture": False}
    try:
        source = session.get("metadata", {}).get("audio", {}).get("artifact_paths", {}).get("wav")
        result["source_path"] = source
        if not stopped:
            raise RuntimeError("WAV copy skipped because stopped state was not confirmed")
        if not isinstance(source, str) or not source:
            raise ValueError("session metadata has no audio.artifact_paths.wav")
        destination = output.resolve() / "story_mix.wav"
        digest = hashlib.sha256()
        with Path(source).open("rb") as reader, destination.open("xb") as writer:
            while chunk := reader.read(1024 * 1024):
                writer.write(chunk)
                digest.update(chunk)
        result.update(status="copied", path=str(destination), sha256=digest.hexdigest())
    except Exception as error:
        result["warning"] = f"WAV preservation failed: {type(error).__name__}: {error}"
    return result


class StoryPoller(StatePoller):
    def sample(self):
        super().sample()
        with (self.test.output / "states.jsonl").open("a") as handle:
            handle.write(json.dumps({"observed_ns": time.monotonic_ns(), "state": self.evidence.state,
                                     "settings": self.settings}) + "\n")


def submission_timing(observations, trigger_bar, *, bpm, phase, now_ns):
    trigger = next((row for row in reversed(observations) if row["bar"] == trigger_bar), None)
    latest = max((row["bar"] for row in observations if type(row["bar"]) is int), default=-1)
    duration = 240 / bpm * 1e9
    start = trigger["monotonic_ns"] if trigger else None
    due = start + round(phase * duration) if start is not None else None
    end = start + round(duration) if start is not None else None
    if latest > trigger_bar or (end is not None and now_ns >= end):
        state = "late"
    elif due is None or now_ns < due:
        state = "waiting"
    else:
        state = "eligible"
    return {"edit_phase": phase, "trigger_playback_ns": start, "bar_duration_ns": duration,
            "scheduled_submit_ns": due, "next_bar_ns": end, "eligibility": state}


def _submit(test, index, text, trigger_bar, observed_bar, run_id, *, timing=None):
    now = time.monotonic_ns()
    if timing is not None and (timing["eligibility"] == "late"
                               or (timing["next_bar_ns"] is not None and now >= timing["next_bar_ns"])):
        return {"index": index + 1, "text": text, "trigger_bar": trigger_bar, **timing,
                "observed_bar_at_submission": observed_bar, "accepted": False,
                "request_start_ns": None, "checked_ns": now, "first_playback_callback": None,
                "phase_timing_status": "late_not_submitted", "error": "trigger bar window was missed"}
    row = test.request_timed("/api/story/events", {"text": text, "event_id": f"{run_id}-{index + 1}"})
    phase_evidence = {}
    if timing is not None:
        request_ns = row["request_start_ns"]
        offset = request_ns - timing["trigger_playback_ns"]
        phase_evidence = {**timing, "request_offset_ms": offset / 1e6,
                          "actual_request_phase": offset / timing["bar_duration_ns"],
                          "request_phase_error_ms": (request_ns - timing["scheduled_submit_ns"]) / 1e6,
                          "available_until_next_bar_ms": (timing["next_bar_ns"] - request_ns) / 1e6,
                          "phase_timing_status": "within_trigger_bar" if request_ns < timing["next_bar_ns"] else "late_submitted"}
    accepted = not row.get("error") and row["status"] is not None and row["status"] < 400
    body = row["body"] if isinstance(row["body"], dict) else {}
    event = body.get("event")
    valid_ack = (body.get("status") == "accepted" and isinstance(event, dict)
                 and isinstance(event.get("event_id"), str) and bool(event["event_id"])
                 and type(event.get("sequence")) is int and event["sequence"] > 0
                 and type(body.get("effective_from_bar")) is int)
    error = row.get("error")
    if accepted and not valid_ack:
        accepted, error = False, "invalid story acknowledgement"
    return {"index": index + 1, "text": text, "trigger_bar": trigger_bar, **phase_evidence,
            "observed_bar_at_submission": observed_bar, "accepted": accepted,
            "http_status": row["status"], "ack": body,
            "request_start_ns": row["request_start_ns"], "response_ns": row["response_ns"],
            "ack_ms": row["elapsed_ms"], "effective_from_bar": body.get("effective_from_bar"),
            "event": event, "error": error,
            "first_playback_callback": None}


def run(args):
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("choose an empty output directory to preserve existing evidence")
    output.mkdir(parents=True, exist_ok=True)
    test = LiveAcceptance(args.url, output)
    evidence = EventEvidence(output)
    poller = StoryPoller(test, evidence, args.poll_interval)
    summary = {"status": "running", "url": args.url, "bpm": args.bpm, "target_bars": args.bars,
               "edit_phase": args.edit_phase,
               "events": [], "ready": False, "target_callback_observed": False,
               "measurement_kind": "software_playback_callback", "is_acoustic_observation": False,
               "semantic_fulfillment": "not_evaluated", "output_dir": str(output),
               "clock_basis": "same-host monotonic event emission and HTTP receipt",
               "limitation": "accepted and prompt-version playback do not prove plot fulfillment or audibility"}
    run_id = uuid4().hex[:16]
    try:
        test.stop()
        test.request("/api/control/settings", {"mode": "continuous", "tempo_bpm": args.bpm})
        test.request("/api/control/reset", {})
        summary["initial_story"] = test.request("/api/story")
        if summary["initial_story"].get("version") != 0 or summary["initial_story"].get("events") != []:
            raise RuntimeError("reset did not clear the story session; refusing to mix runs")
        summary["session"] = test.request("/api/session")
        poller.start()
        deadline = time.monotonic() + args.timeout
        test.request("/api/control/start", {})
        while time.monotonic() < deadline:
            poller.check()
            observations = playback_observations(evidence.events)
            bars = [row["bar"] for row in observations if type(row["bar"]) is int]
            last_bar = max(bars, default=-1)
            index = len(summary["events"])
            timing = (submission_timing(observations, args.event_bars[index], bpm=args.bpm,
                                        phase=args.edit_phase, now_ns=time.monotonic_ns()) if index < 3 else None)
            if timing is not None and timing["eligibility"] != "waiting":
                result = _submit(test, index, args.events[index], args.event_bars[index], last_bar, run_id,
                                 timing=timing)
                summary["events"].append(result)
                write_json(output / "summary.json", summary)
                if not result["accepted"]:
                    raise RuntimeError(f"story event {index + 1} was not accepted: {result}")
                if result["phase_timing_status"] == "late_submitted":
                    raise RuntimeError("request crossed the trigger bar boundary before HTTP submission")
                if (not isinstance(result["event"], dict)
                        or type(result["effective_from_bar"]) is not int):
                    raise RuntimeError("story acknowledgement lacks event identity or effective bar")
            if last_bar >= args.bars - 1 and len(summary["events"]) == 3:
                summary["target_callback_observed"] = True
                break
            time.sleep(args.poll_interval)
        else:
            raise TimeoutError("target playback bar was not observed before the run deadline")
        summary["status"] = "collected"
    except BaseException as error:
        summary.update(status="failed", error=repr(error))
    finally:
        try:
            summary["final_settings"] = test.stop()
            summary["ready"] = summary["final_settings"].get("state") == "stopped"
        except BaseException as error:
            summary.update(status="failed", cleanup_error=repr(error))
        summary["audio_artifact"] = preserve_mix_artifact(summary.get("session", {}), output,
                                                          stopped=summary["ready"])
        poller.close()
        try:
            poller.sample()
            summary["final_story"] = test.request("/api/story")
        except Exception as error:
            summary["final_capture_error"] = repr(error)
            summary["status"] = "failed"
        summary["event_gaps"] = evidence.gaps
        observations = playback_observations(evidence.events)
        for result in summary["events"]:
            if result["accepted"] and result.get("event"):
                eligible = [row for row in observations
                            if row["monotonic_ns"] >= result["request_start_ns"]
                            and type(row["bar"]) is int and type(result["effective_from_bar"]) is int
                            and row["bar"] >= result["effective_from_bar"]]
                first = first_playback_for(result["event"], eligible)
                result["first_playback_callback"] = (None if first is None else {
                    **first, "request_to_callback_ms": (first["monotonic_ns"] - result["request_start_ns"]) / 1e6})
        summary["played_bar_count"] = len(observations)
        summary["chosen_lyrics"] = observations
        if summary["status"] != "failed":
            complete = (summary["ready"] and summary["target_callback_observed"] and not evidence.gaps
                        and len(summary["events"]) == 3
                        and all(row["first_playback_callback"] is not None for row in summary["events"]))
            summary["status"] = "observed" if complete else "incomplete"
        write_json(output / "summary.json", summary)
        write_json(output / "story_events.json", summary["events"])
        write_json(output / "chosen_lyrics.json", observations)
        test.evidence.close()
    return summary


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8018")
    parser.add_argument("--output-dir", "--output", type=Path, required=True)
    parser.add_argument("--bpm", type=float, default=60)
    parser.add_argument("--bars", type=int, default=12)
    parser.add_argument("--event-bars", type=int, nargs=3, default=[0, 4, 8])
    parser.add_argument("--edit-phase", type=float, default=0,
                        help="Submission fraction within each trigger bar, in [0,1); .75 is tick 12")
    parser.add_argument("--events", nargs=3, default=list(DEFAULT_EVENTS))
    parser.add_argument("--timeout", type=float, default=240, help="Run deadline including startup, in seconds")
    parser.add_argument("--poll-interval", type=float, default=.1)
    args = parser.parse_args(argv)
    if not math.isfinite(args.edit_phase) or not 0 <= args.edit_phase < 1:
        parser.error("edit-phase must be finite and in [0,1)")
    url = urlsplit(args.url)
    if (url.scheme != "http" or url.hostname not in ("127.0.0.1", "localhost", "::1")
            or url.path not in ("", "/") or url.query or url.fragment or url.username or url.password):
        parser.error("use a local HTTP demo URL; event and request clocks must share a host")
    if (args.bars < 4 or sorted(set(args.event_bars)) != args.event_bars
            or args.event_bars[0] < 0 or args.event_bars[-1] >= args.bars - 1):
        parser.error("three strictly increasing event bars must leave at least one later playback bar")
    if (not all(math.isfinite(v) and v > 0 for v in (args.bpm, args.timeout, args.poll_interval))
            or not 30 <= args.bpm <= 180):
        parser.error("BPM must be 30..180; timeout and poll interval must be finite and positive")
    if any(not text.strip() for text in args.events) or sum(len(t.encode("utf-8")) for t in args.events) > 512:
        parser.error("directions must be nonblank and fit the 512-byte cumulative story budget")
    return args


if __name__ == "__main__":
    result = run(parse_args())
    print(json.dumps({"status": result["status"], "output_dir": result["output_dir"], "ready": result["ready"]}))
    raise SystemExit(0 if result["status"] == "observed" else 1)
