#!/usr/bin/env python3
"""Build the PI brief from recorded evidence; never contact production services."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
RESEARCH = ROOT / "docs/research"
PREFIX = "2026-10-22"
TMP = ROOT / "tmp/pdfs/scheduling-pi-brief"
OUTPUT = ROOT / "output/pdf/2026-10-22-scheduling-pi-brief.pdf"
INK = "#202735"
MUTED = "#566171"
LINE = "#DCE2EA"
BLUE = "#2F6BDB"
PURPLE = "#7448D1"
TEAL = "#168574"
AMBER = "#B47513"
RED = "#B34C42"
COLORS = [BLUE, PURPLE, TEAL, RED, "#52778D", "#897132", "#647F4B"]
GUIDE_TITLES = [
    "Start here: the units and the work",
    "G1: Time available before playback",
    "G2: Work, waiting and execution capacity",
    "G3: Searching for a usable lyric",
    "G4: Changes and work that must be redone",
    "G5: From server speech to local audio",
    "G6: Chunk sizes and allowed start times",
    "G7: Ready audio we are allowed to play",
]
MODEL_GUIDE_TITLES = [
    "M1: Beats become clock times",
    "M2: Playback becomes a finish deadline",
    "M3: Readiness, waiting and finishing",
    "M4: Shared execution capacity",
    "M5: Required work and remaining time",
    "M6: On-time and fresh success",
    "M7: Valid, selectable and best candidates",
    "M8: Coverage without gaps",
    "M9: Allowed boundaries and turn-taking",
    "M10: Moving a deadline on a saved trace",
]
PAGE_TITLES = GUIDE_TITLES + ["Parameterization", "System mapping"] + MODEL_GUIDE_TITLES + [
    "Timing model reference",
    "Eligibility and granularity", "Observed evidence", "Deadline sensitivity",
]


def load_evidence():
    summary = json.loads((RESEARCH / f"{PREFIX}-scheduling-evidence.json").read_text())
    for item in summary["primary_inputs"].values():
        path = ROOT / item["path"]
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != item["sha256"]:
            raise ValueError(f"Evidence hash mismatch: {path}")
    raw = ROOT / summary["primary_inputs"]["budget_h200"]["path"]
    budget = [json.loads(line) for line in raw.read_text().splitlines()]
    assert len(budget) == 36
    assert len(summary["deadline_counterfactuals"]) == 90
    for n in (16, 32, 64):
        rows = [r for r in budget if r["condition"] == f"n{n}"]
        recorded = summary["budget_h200"][f"n{n}"]
        assert len(rows) == recorded["rounds"] == 12
        assert abs(statistics.mean(r["valid_count"] for r in rows) - recorded["valid_count_mean"]) < 1e-8
    for cell in summary["deadline_counterfactuals"]:
        for row in cell["outcomes"]:
            expected = ((cell["lookahead_bars"] + row["bar"]) * 4 - 0.25) * 60000 / cell["bpm"]
            assert abs(row["deadline_ms"] - expected) < 1e-7
        assert cell["misses"] == sum(row["miss"] for row in cell["outcomes"])
    return summary, budget


def worked_examples():
    """Keep teaching arithmetic separate from the recorded experimental data."""
    e = {}
    e["clock_s"] = 4*60/90 + 4*60/120
    e["play_ms"] = 1000*e["clock_s"]
    e["sample_frame"] = round(48000*e["clock_s"])
    e["deadline_ms"] = 8000-200
    e["budget_ms"] = e["deadline_ms"]-4000
    e["release_ms"] = max(4000,4300,4500)
    e["wait_ms"] = 5000-e["release_ms"]
    e["finish_ms"] = 5000+1800
    e["second_finish_ms"] = e["finish_ms"]+1800
    e["critical_path_ms"] = max(200+1800+100+50,100+50)
    e["slack_ms"] = e["deadline_ms"]-6000-1500
    e["delayed_slack_ms"] = e["deadline_ms"]-6000-(1500+600)
    e["success_cases"] = [(f-e["deadline_ms"],int(f<=e["deadline_ms"] and fresh))
                          for f,fresh in [(7500,True),(7500,False),(7900,True)]]
    e["candidates"] = [("A",9,True,82),("B",9,True,65),("C",10,False,None),("D",9,True,78)]
    e["valid_count"] = sum(valid for _,_,valid,_ in e["candidates"])
    selected = [score for _,_,valid,score in e["candidates"] if valid and score>=75]
    e["selectable_count"],e["best_score"] = len(selected),max(selected)
    e["bar_ms"] = 4*60000/90
    e["capture_ms"] = e["bar_ms"]
    e["ready_intervals"] = (6200+200-e["capture_ms"])/e["bar_ms"]
    e["wait_intervals"] = max(0,math.ceil(e["ready_intervals"]))
    e["response_ms"] = e["capture_ms"]+e["wait_intervals"]*e["bar_ms"]
    e["cutoffs_ms"] = [(4*lead-.25)*60000/90 for lead in (1,2,3)]
    e["trace_misses"] = [int(5000>d) for d in e["cutoffs_ms"]]
    assert math.isclose(e["clock_s"],14/3) and e["sample_frame"]==224000
    assert (e["deadline_ms"],e["budget_ms"])==(7800,3800)
    assert (e["release_ms"],e["wait_ms"],e["finish_ms"],e["second_finish_ms"])==(4500,500,6800,8600)
    assert (e["critical_path_ms"],e["slack_ms"],e["delayed_slack_ms"])==(2150,300,-300)
    assert e["success_cases"]==[(-300,1),(-300,0),(100,0)]
    assert (e["valid_count"],e["selectable_count"],e["best_score"])==(3,2,82)
    assert math.isclose(e["ready_intervals"],1.4) and e["wait_intervals"]==2
    assert math.isclose(e["response_ms"],8000) and e["trace_misses"]==[1,0,0]
    return e


def setup_plotting():
    global plt, Rectangle, FancyArrowPatch, np
    os.environ.setdefault("MPLCONFIGDIR", str(TMP / "matplotlib-cache"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle, FancyArrowPatch
    import numpy as np
    matplotlib.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 12,
        "svg.fonttype": "none", "pdf.fonttype": 42,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.labelcolor": INK, "text.color": INK,
        "xtick.color": MUTED, "ytick.color": MUTED,
        "axes.edgecolor": LINE, "savefig.facecolor": "white",
    })


def text(ax, x, y, value, size=12, color=INK, weight="normal", ha="left", **kw):
    return ax.text(x, y, value, fontsize=size, color=color, weight=weight,
                   ha=ha, va="top", linespacing=1.4, **kw)


def rect(ax, x, y, w, h, fill="white", edge=LINE, lw=1, dashed=False):
    patch = Rectangle((x, y), w, h, facecolor=fill, edgecolor=edge,
                      linewidth=lw, linestyle="--" if dashed else "-")
    ax.add_patch(patch)
    return patch


def arrow(ax, start, end, color=MUTED, dashed=False, lw=1.5, style="-|>"):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle=style, mutation_scale=12,
                                linewidth=lw, color=color, linestyle="--" if dashed else "-"))


def path(ax, points, color=MUTED, dashed=False):
    xs, ys = zip(*points[:-1])
    ax.plot(xs, ys, color=color, lw=1.5, ls="--" if dashed else "-")
    arrow(ax, points[-2], points[-1], color, dashed)


def page(number, phase, title, subtitle):
    fig = plt.figure(figsize=(16, 10), dpi=120)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set(xlim=(0, 1600), ylim=(1000, 0))
    ax.axis("off")
    text(ax, 48, 26, f"STREAMMUSE  /  RESEARCH BRIEF  /  {phase}", 10, MUTED, "bold")
    text(ax, 48, 60, title, 25, INK, "bold")
    text(ax, 48, 112, subtitle, 12, MUTED)
    ax.plot([48, 1552], [151, 151], color=LINE, lw=1)
    return fig, ax


def block(ax, x, y, w, h, title, body, color=BLUE, fill="white", size=11.2):
    rect(ax, x, y, w, h, fill, color)
    rect(ax, x, y, 5, h, color, color)
    text(ax, x+17, y+13, title, 12, color, "bold")
    text(ax, x+17, y+43, body, size, INK)


def save_page(fig, name, number):
    fig.savefig(RESEARCH / f"{PREFIX}-{name}.svg")
    fig.savefig(RESEARCH / f"{PREFIX}-{name}.png", dpi=180)
    fig.savefig(TMP / f"page-{number}.pdf")
    plt.close(fig)


def foundations():
    fig, ax = page(1, "START HERE", "What are we scheduling?",
                   "A lyric must become playable audio before its musical start time. The following pages define the seven groups before combining them.")
    text(ax, 48, 184, "ONE BAR OF OUTPUT", 12, BLUE, "bold")
    labels = ["Choose topic + flow", "Write / select lyric", "Make source speech", "Prepare timed audio", "Freeze -> queue -> play"]
    bodies = ["Flow = target syllable\ntimes and stresses.", "Candidate = one\nproposed lyric line.", "Source = speech before\nfitting it to the flow.", "Prepare = fit to flow,\nadd drums, mix.", "Commit = freeze choice.\nEnqueue = hand to playback."]
    for i, (label, body) in enumerate(zip(labels, bodies)):
        x = 48 + i*306
        block(ax, x, 224, 280, 112, label, body, [BLUE, TEAL, PURPLE, BLUE, AMBER][i], size=12)
        if i < 4:
            arrow(ax, (x+282, 279), (x+302, 279))
    text(ax, 48, 365, "A job is the work for one output bar/version. A stage is one step in that job. Some stages can overlap; others must wait for earlier steps.", 12)
    text(ax, 48, 399, "Two modes: steer the model while it keeps playing, or take turns entering a flow and waiting for a model response.", 11, MUTED)

    text(ax, 48, 424, "MUSICAL UNITS", 13, BLUE, "bold")
    text(ax, 48, 463, "Beat: the regular pulse. Bar: a group of beats.\nTick: a subdivision used for flow positions.\nCurrent setup: 4 beats/bar and 4 ticks/beat.", 13)
    for i in range(16):
        x = 48+i*42
        rect(ax, x, 578, 39, 52, "#DCE8FF" if i%4==0 else "#F3F6FB", BLUE if i%4==0 else LINE)
        if i%4==0:
            text(ax, x+19, 592, str(i//4+1), 13, BLUE, "bold", ha="center")
    text(ax, 48, 648, "One bar: 16 ticks. At 90 BPM: beat = 0.667 s,\nbar = 2.667 s, tick = 0.167 s. BPM = beats per minute.", 12, MUTED)

    text(ax, 850, 424, "HOW TO READ THE SYMBOLS", 13, PURPLE, "bold")
    text(ax, 850, 463, r"$j$ = output bar/job; $v$ = processing stage; $t$ = clock time.", 13)
    text(ax, 850, 509, r"$C_{jv}$ means: the work duration for bar $j$, stage $v$.", 13)
    text(ax, 850, 555, "Times are milliseconds (ms) unless marked seconds (s).\nA timestamp says when; a duration says how long.\nSubscripts identify which job, stage or version is meant.", 13)
    text(ax, 850, 650, r"$\mathbf{1}[\mathrm{condition}]$ = 1 if true, otherwise 0.", 13)
    text(ax, 850, 689, "A sum of these 1s counts how many tests pass.", 12, MUTED)

    ax.plot([48,1552],[753,753],color=LINE,lw=1)
    text(ax, 48, 784, "Separate the setting from the result", 14, INK, "bold")
    text(ax, 48, 825, "CONTROL / INPUT\nTempo, lead, candidate budget, edits", 12, BLUE)
    text(ax, 570, 825, "CURRENT STATE\nWaiting work, versions, ready audio", 12, TEAL)
    text(ax, 1100, 825, "MEASURED / DERIVED RESULT\nWait, usable count, deadline misses", 12, RED)
    text(ax, 48, 913, "All guide examples are illustrative unless explicitly marked measured. A prepared or queued sound is not proof that the listener has heard it.", 11, MUTED)
    save_page(fig, "guide-foundations", 1)


def guide_page(group, title, subtitle, rows, results, why):
    color = COLORS[group-1]
    fig, ax = page(group+1, f"GROUP GUIDE / G{group}", title, subtitle)
    text(ax, 48, 182, "VARIABLE / UNIT", 11, MUTED, "bold")
    text(ax, 405, 182, "PLAIN-LANGUAGE MEANING", 11, MUTED, "bold")
    text(ax, 1080, 182, "WHAT WE DERIVE OR CHECK", 11, color, "bold")
    for i, (name, unit, meaning) in enumerate(rows):
        y = 219 + i*78
        rect(ax, 48, y, 973, 72, "#F8FAFC" if i%2==0 else "white", "white")
        rect(ax, 48, y, 4, 72, color, color)
        text(ax, 63, y+8, name, 12.4, color, "bold")
        text(ax, 63, y+40, unit, 10.8, MUTED)
        text(ax, 405, y+10, meaning, 12.1)
    y = 221
    for name, meaning in results:
        text(ax, 1080, y, name, 12.8, color, "bold")
        text(ax, 1080, y+32, meaning, 11.7)
        y += 96
    ax.plot([1048,1048],[181,839],color=LINE,lw=1)
    ax.plot([48,1552],[863,863],color=LINE,lw=1)
    text(ax, 48, 886, "WHY ONE GROUP?", 11, color, "bold")
    text(ax, 248, 882, why, 12)
    return fig, ax


def guide_time():
    rows = [
        (r"Tempo $B(\beta)$; position $\beta$", "beats/min; beats", "Tempo tells how fast beats pass. Beat position says\nwhere we are in the song; tempo may change there."),
        (r"Meter $m$; tick resolution $r$", "beats/bar; ticks/beat", "How many beats make a bar, and how many ticks\ndivide a beat. This system uses m = r = 4."),
        (r"Stage lead $h_v$ or $L_v$", "beats or bars; h = mL", "How far ahead stage v is asked to work. Writing and\naudio can have different leads; this is not ready audio."),
        (r"Early-freeze guard $\gamma$", "ticks", "How early before playback the bar choice is frozen.\nThe usual whole-bar cutoff is one tick early."),
        (r"Remaining-time reserve $M_{jv}$", "ms", "Time set aside after stage v for later steps and safety.\nInclude the guard here once; do not add it twice."),
        (r"Musical clock $T(\beta)$", "seconds", "Maps a beat position to a clock time using the tempo\nat each beat. T0 is the session start in seconds."),
        (r"Clock origins $T_0,\ t_{ref}$", "seconds", "T0 starts the musical timeline; t_ref is the reference\nused to compare timestamps across jobs."),
        (r"Earliest allowed start $R_{jv}$", "ms timestamp", "When required inputs and earlier steps are ready\nand policy permits this work. Waiting may still follow."),
    ]
    results = [
        (r"Play time $P_j$", "beta_j is the starting beat of bar j.\nP = 1000 [T(beta_j) - t_ref]."),
        (r"Finish deadline $D_{jv}=P_j-M_{jv}$", "Latest finish for this stage that leaves\nthe reserved time before playback."),
        (r"Available budget $b_{jv}=D_{jv}-R_{jv}$", "How long this stage has after release.\nA negative value means already too late."),
    ]
    fig, ax = guide_page(1, "G1  How much time do we have?",
                        "Formal name: time budget. Tempo, advance planning and safety margins jointly set the deadline.", rows, results,
                        "They turn musical distance into clock time. Two bars of lead mean different seconds at different tempos;\na deadline also needs a named stage, its release time and the time reserved afterward.")
    text(ax, 1080, 533, "EXAMPLE / CONSTANT 90 BPM", 11, BLUE, "bold")
    text(ax, 1080, 574, "Two-bar lead = 5.333 s.\nOne-tick guard = 0.167 s.\nFinish cutoff = 5.167 s after release.", 13)
    ax.plot([1090,1520],[713,713],color=INK,lw=1.4)
    for x, label, color in [(1090,"release\n0 s",INK),(1435,"cutoff\n5.167 s",RED),(1520,"play\n5.333 s",BLUE)]:
        ax.plot([x,x],[692,726],color=color,lw=2)
        text(ax,x,742,label,10.8,color,ha="center" if x>1090 else "left")
    text(ax, 1080, 813, "Illustration: no reserve beyond the guard.", 10.5, MUTED)
    save_page(fig, "guide-g1-time", 2)


def guide_work():
    rows = [
        (r"Stage work $C_{jv}$", "ms duration", "Time spent executing one stage, excluding the wait\nwe model separately. It need not be constant."),
        (r"Placement $p(v)$", "resource / process identifier", "Which computer, device or protected worker runs\nstage v; for example, the shared H200 renderer."),
        (r"Execution capacity $k_q$", "simultaneous execution slots", "How many jobs resource q can actually run at once.\nq names a resource; it is not tick resolution r."),
        (r"Admitted sessions $N_{sessions}$", "session count", "How many sessions are allowed into the system.\nMore admitted work does not create execution slots."),
        (r"Queue rule $\pi_q$", "ordering policy", "Which waiting job resource q serves next: for example,\narrival order or earliest deadline first."),
        (r"Start $S_{jv}$; finish $F_{jv}$", "ms timestamps", "When execution actually begins and ends. Start can\nbe later than the earliest allowed start R."),
        (r"Waiting time $Q_{jv}=S_{jv}-R_{jv}$", "ms duration", "Time between being ready to run and actually running.\nA busy resource or scheduling choice can cause it."),
        (r"Remaining-time estimate $\widehat W^{rem}_j$", "ms duration; not fitted here", "Prediction of all work and waits still needed before\nthe chosen finish boundary, including later steps."),
    ]
    results = [
        (r"Completion $F=S+C$", "Start plus execution duration.\nReady does not mean started."),
        (r"Slack $\sigma_j(t)=D_j-t-\widehat W^{rem}_j(t)$", "Estimated time left after remaining work.\nNegative slack predicts a late finish."),
        ("Critical path", "Longest chain of required earlier work.\nWork time alone excludes resource waits."),
    ]
    fig, ax = guide_page(2, "G2  What must run, and what must wait?",
                        "Formal name: work and capacity. Resource means something jobs compete to use, such as a GPU worker or a locked renderer.", rows, results,
                        "Completion depends on both execution cost and competition for the same resource. Model, input length,\nhardware and cache state can change C; the queue rule and available slots determine additional waiting.")
    text(ax, 1080, 533, "EXAMPLE / ONE EXECUTION SLOT", 11, PURPLE, "bold")
    text(ax, 1080, 572, "Two sessions arrive together.\nEach needs 1.8 s of rendering.", 13)
    for y,label in [(661,"A"),(732,"B")]:
        text(ax,1080,y+13,label,12,PURPLE,"bold")
    rect(ax,1120,661,190,46,"#E8DEFA",PURPLE)
    text(ax,1215,673,"render 1.8 s",11,PURPLE,ha="center")
    rect(ax,1120,732,190,46,"#EEF0F3",LINE)
    text(ax,1215,744,"wait 1.8 s",11,MUTED,ha="center")
    rect(ax,1310,732,190,46,"#E8DEFA",PURPLE)
    text(ax,1405,744,"render 1.8 s",11,PURPLE,ha="center")
    text(ax,1080,811,"Illustration: B finishes at 3.6 s, not 1.8 s.",11,MUTED)
    save_page(fig, "guide-g2-work", 3)


def guide_search():
    rows = [
        (r"Candidates per wave $n_w$", "requested completions; w = wave index", "A wave asks the language model for n proposed lines.\nw = 0 is the initial request."),
        (r"Rescue limit $W_{max}$", "additional wave count", "Maximum extra requests after the first when there\nare too few valid lines. A rescue wave can be smaller."),
        (r"Valid-count target $k_{min}$", "candidate count", "Desired number of valid lines before stopping search.\nThis is a target, not a guarantee or final score check."),
        (r"Validity test $V(c)$", "1 = pass; 0 = fail", "Does candidate c satisfy the required checks, such as\nsyllable count and the applicable repetition rules?"),
        (r"Ranking score $s(c)$", "internal score units", "A rule-based measure used to rank valid lines,\nincluding fit to the flow. It is not a listening score."),
        (r"Selectable threshold $\tau_s$", "same units as the ranking score", "Minimum score a valid line must also meet before\nit is eligible for selection."),
        (r"Evaluated set $\mathcal{X}_{jw}$", "distinct candidates after waves 0..w", "The proposed lines that were parsed and evaluated,\nwith duplicates removed. Not necessarily n in total."),
        (r"Search policy $\theta_s$", "the settings above, kept together", "The candidate counts, retry limit, stopping target,\nvalidity tests, scoring rule and threshold."),
    ]
    results = [
        (r"Valid count $Y_{jw}$", "Number of distinct evaluated candidates\nthat pass V, whether or not selectable."),
        (r"Selectable count $U_{jw}$", "Number passing V and the score threshold.\nAt least one is needed to choose a line."),
        (r"Best selectable score $s^*_{jw}$", "Highest score among selectable lines.\nUndefined when there are none."),
    ]
    fig, ax = guide_page(3, "G3  How do we find a usable lyric?",
                        "Formal name: useful search. Requested, parseable, valid and selectable are different counts; parseable means text we can read from the response.", rows, results,
                        "A candidate budget only has meaning together with the rules for accepting and ranking its results.\nThe counts and best score are outcomes, not knobs; more samples are not independent guarantees.")
    text(ax,1080,533,"EXAMPLE / ONE WAVE",11,TEAL,"bold")
    stages=[("32 requested",410),("24 distinct, evaluated",325),("6 valid",220),("4 selectable",155)]
    for i,(label,width) in enumerate(stages):
        y=574+i*56
        rect(ax,1080,y,width,39,"#E7F3EE",TEAL)
        text(ax,1091,y+10,label,11.5,TEAL)
    text(ax,1080,812,"Illustration. Current stopping target: Y >= 3;\nselectability is checked afterward.",10.7,MUTED)
    save_page(fig, "guide-g3-search", 4)


def guide_revisions():
    rows = [
        (r"Edit record $E$", "ordered list of accepted edits", "The history of changes, preserving when each arrived,\nwhat changed and which output bars it affects."),
        (r"Edit time $e_l$; scope $scope_l$", "ms timestamp; affected bar range", "l labels an edit. Time records its acceptance; scope\nsays which unfrozen bars should respond to it."),
        (r"Changed input $\delta_l$", "topic / flow / tempo / settings", "Which input changed and its new value. Different\ninputs can require different parts of the job to rerun."),
        (r"Dependency map $D_{dep}$", "directed graph: what depends on what", "If a lyric changes, its speech usually changes too.\nArrows record which later results must be reconsidered."),
        (r"Cancellation rule $\chi_v$", "stop / finish / discard policy", "Whether running stage v can be stopped, or must\nfinish even when its result is no longer wanted."),
        (r"Revision IDs", "content / timeline / settings versions", "Content tracks lyric/flow changes; timeline tracks\nbeat timing; configuration tracks rendering settings."),
        (r"Session epoch", "new session-generation identifier", "Distinguishes a restarted session from an older one\nso old asynchronous results cannot overwrite it."),
        (r"Input cutoff $\kappa_j$", "declared input/version requirement", "The input version output j is judged against when\nfrozen. Later edits do not invalidate that played bar."),
    ]
    results = [
        ("Latest-version response delay", "Time until a result for the requested\nversion is ready, not until it is heard."),
        ("Obsolete work", "Execution spent on a result no longer\nneeded. Not measured in this study."),
        ("Fresh vs older", "Fresh matches the required input cutoff.\nAn older version may be a permitted fallback."),
    ]
    fig,ax=guide_page(4,"G4  What happens when the user changes their mind?",
                     "Formal name: revision work. Revision means a new version; obsolete means the current request no longer needs that work.",rows,results,
                     "An edit costs work only through the results it changes and the work that can or cannot be stopped.\nAn average edit rate hides bursts, affected bars and whether edits arrive before or after freezing.")
    text(ax,1080,533,"EXAMPLE / TOPIC EDIT",11,RED,"bold")
    text(ax,1080,574,"Topic: space -> gardens\nApply from unfrozen bar 5 onward.",13)
    block(ax,1080,648,450,68,"Bar 4: already frozen","Keep the chosen audio.",BLUE,size=11.5)
    block(ax,1080,735,450,77,"Bar 5: old work in progress","Rewrite; old speech may still finish.",RED,size=11.5)
    save_page(fig,"guide-g4-revisions",5)


def guide_delivery():
    rows = [
        ("Route", "network path / host placement", "Where audio bytes travel. Here: server on H200 to\nMac through an SSH-forwarded connection."),
        ("Codec / format", "PCM or Opus", "PCM stores sample values directly; Opus compresses\naudio. Format can affect size, decoding and quality."),
        (r"Payload size $z_j$", "bytes", "Size of the audio package and metadata transferred\nfor output j. Metadata describes the audio and timing."),
        (r"Transfer time $C_x$", "ms duration", "Time for the named transfer step, not the whole wait\nfrom asking for a lyric until its audio arrives."),
        (r"Preparation time $C_p$", "ms duration", "Mac time to decode, fit speech to the flow, add drums\nand mix. Prepared means locally playable."),
        (r"Playback reserve $M_{sink}$", "ms duration", "Time left after preparation for freezing, enqueueing\nand device delivery. Sink means audio output device."),
        (r"Sample rate $f_s$", "sample frames per second", "Converts time to a position in digital audio.\nAt 48,000 frames/s, one frame is about 0.021 ms."),
        ("Source onsets / target slots", "source seconds / musical positions", "Onset: estimated syllable start in ordinary speech.\nTarget slot: where that syllable should start in the flow."),
    ]
    results = [
        ("Server-ready vs received", "Audio may exist on H200 before its bytes\nreach the Mac. Measure both boundaries."),
        ("Received vs prepared", "Received audio still needs local work.\nPreparation adds a separate duration."),
        ("Prepared vs heard", "Queued frames still pass through a device.\nAcoustic timing requires measurement."),
    ]
    fig,ax=guide_page(5,"G5  How does server speech become playable audio?",
                     "Formal name: delivery. Downstream means the steps after the stage currently being measured.",rows,results,
                     "These variables describe the remaining path after server speech generation. They are part of total\nwork in G2, not extra costs to count twice; G1 reserves time for whichever of these steps remain.")
    text(ax,1080,533,"NAMES USED IN THE SYSTEM DIAGRAM",11,COLORS[4],"bold")
    text(ax,1080,574,"MOSS: neural speech generator.\nMMS: estimates speech-to-text alignment.\nR3: Rubber Band pitch-preserving stretch.\neSpeak: local robotic speech fallback.\nSSH: encrypted network connection.",11.8)
    text(ax,1080,753,"EXAMPLE",11,COLORS[4],"bold")
    text(ax,1080,787,"Receive at 2.000 s + prepare for 50 ms\n= prepared at 2.050 s, not yet heard.",11.8)
    save_page(fig,"guide-g5-delivery",6)


def guide_granularity():
    rows = [
        (r"Stage unit $g_v$", "beats or a named chunk type", "Size of the piece stage v handles: a bar, beat, phrase\nor audio frame. This is its granularity."),
        (r"Boundary offset $\phi_v$", "same unit as g for periodic boundaries", "Where boundaries fall in each cycle. For 16-tick bars,\nphi = -1 gives freeze boundaries at ticks 15, 31, ..."),
        (r"Input-ready time $A_{jv}$", "ms timestamp", "When every input version required by this stage\nis available. Future user taps cannot be used early."),
        (r"Earlier required jobs $pred(j,v)$", "set of job/stage pairs", "Which previous steps must finish first. Example:\nselected text of bar j informs writing bar j+1."),
        (r"Policy start time $r^{pol}_{jv}$", "ms timestamp", "Earliest time the scheduling rule permits work,\nsuch as the next allowed bar boundary."),
        (r"User/model spans $L_{user},L_{model}$", "bars per turn", "How long the user enters flow and how many model\nresponse bars follow. Current input capture: one bar."),
        (r"Capture end $t_{cap}$; bar time $\tau$", "ms timestamp; ms duration", "When the input bar ends, and how long each bar\nlasts at constant tempo."),
        (r"Finish $F_{elig}$; wait $N_{wait}$", "ms timestamp; whole-bar interval count", "When an allowed response is ready, and how many\nbar intervals after capture the response can start."),
    ]
    results=[
        ("Earliest allowed start R", "Wait for inputs, previous work and policy.\nHaving a free GPU is not enough."),
        ("First response time", "Choose a boundary after readiness and\nreserve: P = t_cap + N_wait times tau."),
        ("Turn waiting rules", "Grace: accept late taps for 100 ms.\nTransition: one bar before response.\nTimeout: limit on waiting for remote audio."),
    ]
    fig,ax=guide_page(6,"G6  How small a piece can we respond to?",
                     "Temporal contract means the rules for chunk size, allowed boundaries and required inputs. Fine audio timing is not fine lyric generation.",rows,results,
                     "A smaller unit helps only if inputs, generation, delivery and commitment support that unit.\nThe current writer produces whole bars even though syllables are placed on ticks and audio on samples.")
    text(ax,1080,533,"EXAMPLE / ALTERNATING MODE",11,AMBER,"bold")
    for i,(label,color) in enumerate([("Capture 1 bar",BLUE),("Replay while waiting",AMBER),("Model response",PURPLE)]):
        y=575+i*71
        rect(ax,1080,y,450,50,"#FAF9F5",color)
        text(ax,1305,y+14,label,12,color,ha="center")
        if i<2:
            arrow(ax,(1305,y+51),(1305,y+68),color)
    text(ax,1080,804,"Replay may repeat. N_wait counts all intervals\nafter capture, not only replay bars.",10.9,MUTED)
    save_page(fig,"guide-g6-granularity",7)


def guide_coverage():
    rows = [
        (r"Result $a$; dependency key $K_a$", "artifact identifier; input/version record", "Artifact means a saved lyric or audio result. Its key\nrecords the inputs/settings used to produce it."),
        (r"Coverage interval $I_a$", "start/end beat positions", "The part of the song this result would cover if played.\nAn interval describes where, not whether it is usable."),
        (r"Eligibility checks $\eta_a$", "ready / valid / ownership / policy tests", "Is it finished and intact, from the correct session,\nand allowed under the selected playback policy?"),
        (r"Playback position $\beta_{play}$", "beats", "The current point reached in the musical timeline.\nReady coverage is measured forward from here."),
        (r"Fallback order $\pi_f$", "ordered preference list", "Which alternative to try when fresh model audio is\nunavailable: older audio, robotic speech or replay."),
        ("Eligibility class", "fresh-only or fallback-permitted, etc.", "A named set of rules for what counts as usable.\nFresh-only and continuity can accept different results."),
        (r"Input cutoff $\kappa_j$ [G4]", "required input/version for output j", "Fresh must match this requirement. A later edit\ndoes not retroactively make committed audio wrong."),
        (r"Fresh success $Z_j^{fresh}$", "1 = success; 0 = failure", "Required-version prepared audio finishes before its\ndeadline and passes eligibility. Not acoustic proof."),
    ]
    results=[
        (r"Ready coverage $\mathcal{C}_{class}(t)$", "All beat intervals covered by usable\nready results under the chosen class."),
        (r"Unbroken ready lead $H_{class}(t)$", "Beats playable from the current position\nwithout reaching a missing interval."),
        ("Continuity vs freshness", "Continuity: no unintended gap.\nFreshness: output matches required inputs."),
    ]
    fig,ax=guide_page(7,"G7  What can we actually play next?",
                     "Formal name: eligible reserve. Eligible simply means allowed to use; reserve here means ready output ahead, not G1's time margin.",rows,results,
                     "A buffer count is misleading without checking which versions are usable and where gaps occur.\nA replay can keep time without providing new lyrics; report fresh output and uninterrupted output separately.")
    text(ax,1080,533,"EXAMPLE / A HOLE IN READY AUDIO",11,COLORS[6],"bold")
    for i,(label,fill,color) in enumerate([("ready","#E7F3EE",TEAL),("missing","#FFF2EB",RED),("ready","#E7F3EE",TEAL)]):
        x=1080+i*151
        rect(ax,x,584,140,81,fill,color)
        text(ax,x+70,597,f"bar {i+1}",11,color,ha="center")
        text(ax,x+70,630,label,11,color,"bold",ha="center")
    arrow(ax,(1080,698),(1218,698),TEAL,style="<->")
    text(ax,1080,729,"Unbroken ready lead = 1 bar, not 3.\nThe third bar cannot fill the second.",12)
    text(ax,1080,805,"Check fresh-only coverage separately from\ncoverage that permits fallback audio.",11,MUTED)
    save_page(fig,"guide-g7-coverage",8)


def parameter_map():
    fig, ax = page(1, "01 PARAMETERIZATION", "What exactly can change in this system?",
                   "Now combine the seven groups. Keep each variable's meaning and unit; the arrows show what we calculate or measure from them.")
    text(ax, 58, 176, "CONTROL", 10, BLUE, "bold")
    text(ax, 160, 176, "chosen setting", 10, MUTED)
    text(ax, 345, 176, "WORKLOAD / ENVIRONMENT", 10, RED, "bold")
    text(ax, 610, 176, "arriving inputs + conditions", 10, MUTED)
    text(ax, 960, 176, "STATE", 10, TEAL, "bold")
    text(ax, 1030, 176, "waiting work, versions, ready audio", 10, MUTED)
    text(ax, 58, 221, "VARIABLES AND UNITS", 11, MUTED, "bold")
    text(ax, 705, 221, "RETAINED GROUP", 11, MUTED, "bold")
    text(ax, 1120, 221, "DERIVED / OBSERVED", 11, MUTED, "bold")
    rows = [
        ("Tempo B [beats/min]; meter m; resolution r [ticks/beat]\nStage lead h [beats]; guard gamma [ticks]; reserve M [ms]",
         "G1  Time budget", "Timeline + stage leads + margins",
         r"$D_{jv}=P_j-M_{jv}$"+"\nBudget = deadline - earliest start"),
        ("Stage work C [ms]; placement p; capacity k_q [slots]\nAdmitted sessions N_sessions; queue rule pi_q",
         "G2  Work and capacity", "Execution time + shared resources",
         "Queue waits Q; completion times F\nRequests in flight are not render slots"),
        ("Candidates per wave n; rescue limit W_max; valid target k_min\nValidity V; score s; score threshold tau_s [score units]",
         "G3  Useful search", "How many to try + which to accept",
         "Valid yield Y; selectable count U\nBest score only when U > 0"),
        ("Edit times e [ms]; changed inputs + affected spans\nInvalidation dependencies; cancellation policy",
         "G4  Revision work", "Edits + what must be redone",
         "Latest-eligible response delay\nObsolete work, if instrumented"),
        ("Route + codec; payload z [bytes]; transfer Cx [ms]\nPreparation Cp [ms]; sink reserve [ms]",
         "G5  Delivery", "Server audio -> local ready audio",
         "Source receipt; prepared readiness\nDownstream reserve used by G1"),
        ("Stage chunk sizes g + boundary offsets phi\nInput-ready times A [ms]; prior steps; turn spans",
         "G6  Temporal contract", "Chunk sizes + allowed start times",
         "Allowed start and freeze boundaries\nBar / beat / tick are not equivalent"),
        ("Artifact keys K; intervals I [beats]; eligibility eta\nPlayback frontier [beats]; fallback order",
         "G7  Eligible reserve", "Ready coverage for each allowed class",
         "Contiguous fresh / fallback coverage\nA future ready bar cannot fill a hole"),
    ]
    for i, (primitives, title, representation, derived) in enumerate(rows):
        y = 258 + i*85
        rect(ax, 48, y, 1504, 76, "#F8FAFC" if i % 2 == 0 else "white", "white")
        rect(ax, 48, y, 5, 76, COLORS[i], COLORS[i])
        text(ax, 65, y+14, primitives, 11.4)
        arrow(ax, (636, y+37), (687, y+37), COLORS[i])
        text(ax, 705, y+10, title, 13, COLORS[i], "bold")
        text(ax, 705, y+40, representation, 10.5, MUTED)
        arrow(ax, (1055, y+37), (1101, y+37), COLORS[i])
        text(ax, 1120, y+14, derived, 10.9)
    text(ax, 48, 879, "Merge rule", 12, INK, "bold")
    text(ax, 175, 879, "Group variables that jointly answer one scheduling question. Keep the individual values; do not replace the group with one number.", 11.5)
    text(ax, 48, 916, "Groups are connected: G3 changes work in G2; edits in G4 change usable audio in G7; G5 and G6 change the time available in G1.", 10.8, MUTED)
    save_page(fig, "parameter-map", 9)


def architecture():
    fig, ax = page(2, "02 SYSTEM MAPPING", "One pipeline, two hosts, explicit timing obligations",
                   "Current v3 implementation + parameter overlays. Layout follows the historical v1.1 diagram; benchmark versions are separate.")
    rect(ax, 48, 181, 530, 586, "#F1F6FF", BLUE)
    rect(ax, 1022, 181, 530, 586, "#F6F2FF", PURPLE)
    text(ax, 68, 198, "MAC  /  TIME, INPUT, PREPARATION, PLAYBACK", 12, BLUE, "bold")
    text(ax, 1042, 198, "H200  /  GENERATION AND SOURCE SPEECH", 12, PURPLE, "bold")
    block(ax, 70, 239, 486, 83, "Input + musical timeline   [G1, G4, G6]",
          "Continuous: topic / flow / tempo edits\nAlternating: one bar of keyboard flow capture", BLUE)
    block(ax, 70, 345, 486, 83, "Mode controller + stage leads   [G1, G6]",
          "Apply edits as soon as unfrozen work allows\nWriting and audio can have different advance leads", BLUE)
    block(ax, 70, 453, 486, 90, "Decode, local R3, drums and mix   [G5]",
          "Source onsets -> target flow -> prepared bar\nRetain timing / flow / format / mix identity", PURPLE)
    block(ax, 70, 558, 486, 88, "Choose eligible -> freeze   [G4, G7]",
          "Continuous: latest -> older -> eSpeak -> emergency\nWhole-bar cutoff: start minus one tick [G1, G6]", AMBER)
    rect(ax, 306, 652, 250, 27, "#FFF7E7", AMBER)
    text(ax, 431, 656, "Local eSpeak / echo [G7]", 9.8, AMBER, ha="center")
    arrow(ax, (431, 652), (431, 646), AMBER)
    path(ax, [(556, 387), (566, 387), (566, 665), (557, 665)], AMBER, True)
    block(ax, 70, 683, 486, 61, "Audio sink + progress events   [G1, G7]",
          "Frozen choice != queued frames != sound heard", BLUE, size=10.5)
    for y0, y1 in [(322, 345), (428, 453), (543, 558), (646, 683)]:
        arrow(ax, (220, y0), (220, y1), BLUE)
    path(ax, [(70, 710), (58, 710), (58, 386), (69, 386)], BLUE, True)
    text(ax, 70, 770, "Dashed feedback: sink progress updates the controller.", 9.5, BLUE)

    block(ax, 1044, 239, 486, 70, "Session coordinator   [G1, G4, G6]",
          "Edits / commits / separate lyric and audio leads", PURPLE, size=10.6)
    block(ax, 1044, 329, 486, 70, "LLM lyric writer + candidate waves   [G2, G3]",
          "LLM = language model; topic + flow + prior text", PURPLE, size=10.6)
    block(ax, 1044, 419, 486, 70, "Validate / score / select -> lyric record   [G3]",
          "Selected text(j) -> writer(j+1); identity + revisions", TEAL, size=10.6)
    block(ax, 1044, 509, 486, 80, "MOSS + MMS alignment   [G2]",
          "Source speech + source-onset estimates\nShared lock: one executing source render", PURPLE, size=10.6)
    block(ax, 1044, 609, 486, 70, "Package [G5] + source store [G4, G7]",
          "PCM / Opus + metadata; ready / older source versions", PURPLE, size=10.6)
    text(ax, 1044, 700, "Dashed: optional prior source -> MOSS [G6, G7].\nEnabled continuation requires a compatible predecessor.\nRender(j) can overlap writer(j+1); obsolete work may finish.", 10.3, MUTED)
    for y0, y1 in [(309, 329), (399, 419), (489, 509), (589, 609)]:
        arrow(ax, (1300, y0), (1300, y1), PURPLE)
    path(ax, [(1530, 467), (1542, 467), (1542, 353), (1532, 353)], TEAL, True)
    path(ax, [(1530, 652), (1542, 652), (1542, 551), (1532, 551)], PURPLE, True)

    text(ax, 800, 201, "SSH-FORWARDED CONNECTION", 10, MUTED, "bold", ha="center")
    text(ax, 800, 250, "CONTROL / EVENTS", 11, BLUE, "bold", ha="center")
    text(ax, 800, 279, "WebSocket: ongoing message connection\nstart, update, committed, stop\ntempo_update / acknowledgment [new]", 10.2, ha="center")
    path(ax, [(578, 370), (1003, 370), (1003, 275), (1042, 275)], BLUE)
    text(ax, 800, 385, "lyric / superseded / audio_ready", 10, BLUE, ha="center")
    path(ax, [(1042, 454), (1010, 454), (1010, 419), (578, 419)], BLUE)
    text(ax, 800, 456, "AUDIO DATA", 11, PURPLE, "bold", ha="center")
    text(ax, 800, 485, "HTTP: fetch or wait for one bar's audio\nidentity + version checks", 10.2, ha="center")
    path(ax, [(578, 527), (620, 527), (620, 610), (1020, 610)], MUTED, True)
    path(ax, [(1020, 637), (970, 637), (970, 555), (595, 555), (595, 500), (558, 500)], PURPLE)
    text(ax, 800, 654, "Source audio is transferred.\nSpeaker timing stays on the Mac.", 11, PURPLE, ha="center")
    text(ax, 800, 712, "G5: route / bytes / codec / reserves", 10, MUTED, ha="center")

    text(ax, 48, 800, "Model job identity", 12, INK, "bold")
    text(ax, 270, 800, "session + epoch + span + content revision + timeline revision + configuration + stage + attempt", 11.5)
    text(ax, 48, 839, "Observation targets", 12, INK, "bold")
    text(ax, 270, 839, "release -> resource start -> source ready -> receipt -> prepared -> committed -> sink -> acoustic  (partially instrumented)", 10.6)
    text(ax, 48, 885, "[new] Tempo / alternating features are post-baseline. Job identity, full observation chain and dependency rules are model overlays, not one implemented wire key.", 10, MUTED)
    text(ax, 48, 918, "Measured baseline: bde2f62c; Qwen2.5-7B + MOSS-TTS-v1.5 / MMS, PCM source transfer. This is not the historical v1.1 model/deployment benchmark.", 10, MUTED)
    save_page(fig, "comprehensive-system", 10)


def model_guide(index, group, title, equations, reading, symbols, worked, result, note, takeaway):
    color = COLORS[group-1]
    fig, ax = page(10+index, f"MODEL GUIDE / M{index} / G{group}", title,
                   "Read the equation, substitute concrete values, then interpret the result. Numbers on this page are illustrative, not benchmark measurements.")
    text(ax,48,180,"EQUATION",11,color,"bold")
    equation_y = 208
    renderer = fig.canvas.get_renderer()
    for expression,size in equations:
        artist = text(ax,48,equation_y,expression,size)
        extent = ax.transData.inverted().transform(artist.get_window_extent(renderer).get_points())
        equation_y = max(extent[:,1])+12
    assert equation_y-12 < 428, f"Equation block exceeds its teaching layout: M{index}"
    text(ax,48,440,"IN PLAIN ENGLISH",11,color,"bold")
    text(ax,48,475,reading,13)
    text(ax,48,560,symbols,11,MUTED)
    ax.plot([923,923],[180,587],color=LINE,lw=1)
    text(ax,965,180,"WORKED RAP EXAMPLE",11,color,"bold")
    text(ax,965,218,worked,12.5)
    text(ax,965,468,result,15,color,"bold")
    text(ax,965,530,note,11.2,MUTED)
    ax.plot([48,1552],[604,604],color=LINE,lw=1)
    ax.plot([48,1552],[867,867],color=LINE,lw=1)
    text(ax,48,891,"WHAT THIS TELLS US",11,color,"bold")
    text(ax,286,884,takeaway,12)
    return fig,ax


def model_clock(e):
    fig,ax=model_guide(1,1,"Musical beats determine when audio should start",
        [(r"$T(\beta)=T_0+\int_0^\beta 60/B(u)\,du$",21),
         (r"$P_j=1000\,[T(\beta_j)-t_{ref}]$",21),
         (r"$frame_j=\mathrm{round}\{f_s[T(\beta_j)-T_0]\}$",20)],
        "Add the duration of each beat to locate the output in clock time.\nSubtract the common reference and convert seconds to milliseconds.\nMultiply elapsed seconds by the sample rate to locate its audio frame.",
        "T: clock seconds; beta_j: bar j's start beat; B: BPM along the song.\nT0: song start; t_ref: reference origin; f_s: frames/second; round: nearest integer.",
        "First 4 beats at 90 BPM: 4 x 60/90 = 2.667 s.\nNext 4 beats at 120 BPM: 4 x 60/120 = 2.000 s.\nT0 = t_ref = 0. Next output starts at beat 8.\n\nT(8) = 2.667 + 2.000 = 4.667 s.\nP = 4,666.667 ms. Sample rate = 48,000.",
        f"Schedule audio frame {e['sample_frame']:,}",
        "Use exact fractions before rounding.\nThe integral adds durations even when tempo changes.",
        "A tempo edit changes future playback times. The lyric can stay the same while its target audio positions change.\nThis equation specifies intended timing; measuring the sound actually emitted is a separate task.")
    text(ax,48,628,"EIGHT BEATS WITH A TEMPO CHANGE",11,BLUE,"bold")
    x0=180
    scale=210
    split=x0+(4*60/90)*scale
    end=x0+e["clock_s"]*scale
    rect(ax,x0,692,split-x0,64,"#E9F0FE",BLUE)
    rect(ax,split,692,end-split,64,"#E7F3EE",TEAL)
    text(ax,(x0+split)/2,711,"4 beats at 90 BPM",14,BLUE,ha="center")
    text(ax,(split+end)/2,711,"4 beats at 120 BPM",14,TEAL,ha="center")
    for x,label in [(x0,"0 s / beat 0"),(split,"2.667 s / beat 4"),(end,"4.667 s / beat 8")]:
        ax.plot([x,x],[763,783],color=INK,lw=1)
        text(ax,x,800,label,11.7,ha="center")
    text(ax,1280,693,"Next bar starts here",12,BLUE,"bold")
    arrow(ax,(1260,729),(end+3,729),BLUE)
    text(ax,1280,746,"frame 224,000",12,BLUE)
    save_page(fig,"model-guide-01-clock",11)


def model_deadline(e):
    fig,ax=model_guide(2,1,"The finish deadline comes before playback",
        [(r"$D_{jv}=P_j-M_{jv}$",23),
         (r"$b_{jv}=D_{jv}-R_{jv}$",23)],
        "Work backward from playback, reserving time for later steps.\nThen subtract the earliest allowed start to find the available budget.\nThe same bar has different deadlines for speech, preparation and playback.",
        "P: playback timestamp; M: reserve after this stage; D: latest finish.\nR: earliest allowed start; b: available duration. All values here use milliseconds.",
        "A bar must start playing at P = 8,000 ms.\nFor prepared audio, reserve M = 200 ms.\nThat total already includes the early-freeze guard.\n\nD = 8,000 - 200 = 7,800 ms.\nRequired work can begin at R = 4,000 ms.",
        f"Available budget: {e['budget_ms']:,} ms",
        "b = 7,800 - 4,000 = 3,800 ms.\nThe 200-ms margin is illustrative, not a tuned default.",
        "Lookahead becomes useful only after the required inputs are ready. Reserve each later step once.\nIf the stage becomes ready after its deadline, its available budget is already negative.")
    text(ax,48,628,"PREPARED-AUDIO OBLIGATION",11,BLUE,"bold")
    start,cutoff,play=180,1282,1340
    rect(ax,start,717,cutoff-start,52,"#E9F0FE",BLUE)
    rect(ax,cutoff,717,play-cutoff,52,"#FAE8E2",RED)
    arrow(ax,(start,685),(cutoff,685),BLUE,style="<->")
    text(ax,(start+cutoff)/2,646,"3,800 ms to complete the required work",14,BLUE,ha="center")
    ax.plot([start,play],[790,790],color=INK,lw=1)
    for x in [start,cutoff,play]:
        ax.plot([x,x],[705,800],color=RED if x==cutoff else INK,lw=1.4,ls="--")
    text(ax,start,817,"R = 4,000 ms",11.5)
    text(ax,cutoff-8,817,"D = 7,800 ms",11.5,RED,ha="right")
    text(ax,play+8,817,"P = 8,000 ms",11.5)
    text(ax,1380,696,"200 ms\nreserved",12,RED)
    save_page(fig,"model-guide-02-deadline",12)


def model_readiness(e):
    fig,ax=model_guide(3,2,"Ready work can still wait for a busy renderer",
        [(r"$R_{jv}=\max\{r^{pol}_{jv},A_{jv},\max_{(i,u)\in pred(j,v)}F_{iu}\}$",19),
         (r"$S_{jv}\geq R_{jv},\qquad Q_{jv}=S_{jv}-R_{jv}$",21),
         (r"$F_{jv}=S_{jv}+C_{jv}$",22)],
        "Wait until policy, required inputs and earlier jobs all permit starting.\nThe latest of these times is R. Actual execution starts at S, possibly later.\nThe wait is S minus R; the finish is S plus the work duration.",
        "r_pol: policy start; A: input-ready time; pred: required earlier job/stage pairs.\nS/F: actual start/finish; C: execution duration; Q: wait. i and u index prior jobs/stages.",
        "Policy allows rendering at 4,000 ms.\nThe required topic/flow is available at 4,300 ms.\nThe earlier lyric-writing step finishes at 4,500 ms.\nR = max(4,000, 4,300, 4,500) = 4,500 ms.\n\nBusy renderer starts this bar at S = 5,000 ms.\nC = 1,800 ms. Q = 5,000 - 4,500 = 500 ms.",
        f"Speech finishes at {e['finish_ms']:,} ms",
        "F = 5,000 + 1,800.\nThese equations separate execution time from waiting.",
        "An edit may delay input readiness before a request can run. Shared resources may add waiting afterward.\nA faster speech model reduces C, but does not by itself remove either kind of delay.")
    block(ax,48,649,370,152,"Conditions for source speech",
          "Policy: 4,000 ms\nTopic/flow: 4,300 ms\nSelected lyric: 4,500 ms",PURPLE,size=12)
    arrow(ax,(420,725),(460,725),PURPLE)
    block(ax,463,682,260,86,"Ready: 4,500 ms","Take the latest condition.",PURPLE,size=11)
    arrow(ax,(726,725),(765,725),PURPLE)
    block(ax,768,682,250,86,"Wait: 500 ms","Start at 5,000 ms.",AMBER,size=11.5)
    arrow(ax,(1021,725),(1060,725),PURPLE)
    block(ax,1063,682,455,86,"Execute: 1,800 ms","Finish at 6,800 ms.",PURPLE,size=12)
    text(ax,463,813,"Readiness is a condition; execution start is a scheduling outcome.",12,MUTED)
    save_page(fig,"model-guide-03-readiness",13)


def model_capacity(e):
    fig,ax=model_guide(4,2,"Two admitted sessions may share one execution slot",
        [(r"$\sum_{j,v:p(v)=q}\mathbf{1}[S_{jv}\leq t<F_{jv}]\leq k_q$",22)],
        "At time t, count jobs currently executing on the same resource.\nThe count cannot exceed that resource's actual execution capacity.\nAccepting another request does not make another worker available.",
        "p(v)=q: stage v runs on resource q; k_q: simultaneous slots.\nEach indicator is 1 between that job's start S and finish F, and 0 otherwise.",
        "The shared source-render worker has k_q = 1.\nSession A runs from 5,000 to 6,800 ms.\nSession B is ready at 5,000 ms but must wait.\nB then runs from 6,800 to 8,600 ms.\n\nAt t = 6,000: A counts as 1 and B as 0.\nTotal executing = 1, which satisfies capacity 1.",
        "Two executing jobs would exceed capacity 1",
        "This models the protected render worker.\nIt does not claim the entire GPU can run only one task.",
        "Concurrent requests can increase waiting without speeding up this shared stage. A different worker or lock\ndesign could change capacity, but a larger admission limit alone does not establish that improvement.")
    scale=.30
    x=lambda ms:220+(ms-5000)*scale
    text(ax,48,630,"A",13,PURPLE,"bold")
    text(ax,48,724,"B",13,PURPLE,"bold")
    rect(ax,x(5000),638,x(6800)-x(5000),58,"#E8DEFA",PURPLE)
    text(ax,(x(5000)+x(6800))/2,655,"executing",13,PURPLE,ha="center")
    rect(ax,x(5000),732,x(6800)-x(5000),58,"#EEF0F3",LINE)
    rect(ax,x(6800),732,x(8600)-x(6800),58,"#E8DEFA",PURPLE)
    text(ax,(x(5000)+x(6800))/2,749,"waiting for the same slot",13,MUTED,ha="center")
    text(ax,(x(6800)+x(8600))/2,749,"executing",13,PURPLE,ha="center")
    for ms in (5000,6800,8600):
        ax.plot([x(ms),x(ms)],[797,813],color=INK,lw=1)
        text(ax,x(ms),824,f"{ms:,} ms",11.5,ha="center")
    ax.plot([x(6000),x(6000)],[625,795],color=RED,lw=1.2,ls="--")
    text(ax,1380,647,"At 6,000 ms:\n1 executing\n1 waiting",12,RED)
    save_page(fig,"model-guide-04-capacity",14)


def model_remaining(e):
    fig,ax=model_guide(5,2,"Required work and remaining-time estimates answer different questions",
        [(r"$CP_j=\max_{\rho\in paths\_to(j)}\quad\sum_{(i,v)\in\rho}C_{iv}$",21),
         (r"$\sigma_j(t)=D_j-t-\widehat W^{rem}_j(t)$",22)],
        "For the critical path, add execution times along each required chain\nand take the longest. For slack, subtract all estimated remaining work\nand waits from the time left before the deadline.",
        "rho: one dependency path; CP: longest required work chain, excluding queue waits.\nsigma: estimated spare time; W_hat_rem: remaining-time estimate, including waits.",
        "Speech path: 200 + 1,800 + 100 + 50 = 2,150 ms.\nDrum path: 100 + 50 = 150 ms.\nCP = max(2,150, 150) = 2,150 ms.\n\nLater: now t = 6,000 ms; deadline D = 7,800 ms.\nEstimate remaining work and waits as 1,500 ms.\nSlack = 7,800 - 6,000 - 1,500 = 300 ms.",
        f"Estimated spare time: {e['slack_ms']} ms",
        "An extra 600-ms wait changes slack to -300 ms.\nThis is a hypothetical estimate, not a fitted predictor.",
        "Parallel work should not be counted as if every stage ran in sequence. The critical path is a lower bound\non required execution time; using it alone as a completion forecast ignores waiting and input availability.")
    blocks=[(48,245,"Write: 200 ms",TEAL),(350,285,"Speech: 1,800 ms",PURPLE),
            (695,275,"Transfer: 100 ms",COLORS[4]),(1220,295,"Mix: 50 ms",BLUE)]
    for x,w,label,color in blocks:
        rect(ax,x,657,w,63,"#F8FAFC",color)
        text(ax,x+w/2,677,label,12.5,color,"bold",ha="center")
    for a,b in [(293,350),(635,695),(970,1220)]:
        arrow(ax,(a+2,689),(b-3,689),PURPLE)
    rect(ax,695,767,275,57,"#EFF7F3",TEAL)
    text(ax,832,784,"Drums: 100 ms",12.5,TEAL,"bold",ha="center")
    path(ax,[(972,795),(1110,795),(1110,701),(1217,701)],TEAL)
    text(ax,48,775,"Drum preparation can run alongside speech.\nMix waits for both branches.",12,MUTED)
    save_page(fig,"model-guide-05-remaining-time",15)


def model_success(e):
    fig,ax=model_guide(6,7,"A timely result must also match the required version",
        [(r"$\ell_j=F_{j,prep}-D_{j,prep}$",23),
         (r"$Z_j^{fresh}=\mathbf{1}[\ell_j\leq0]\;\mathbf{1}[eligible(a_j;\kappa_j)]$",20)],
        "Lateness is the preparation finish time minus its deadline.\nFresh success requires both an on-time finish and an allowed result\nthat matches the input versions required for this output.",
        "ell: lateness, negative means early; prep: local preparation; Z: success flag.\na_j: result for job j; kappa_j: required input/version cutoff. Indicators multiply as AND.",
        "Prepared-audio deadline: D = 7,800 ms.\nMatching-topic audio is ready at F = 7,500 ms.\nLateness = 7,500 - 7,800 = -300 ms.\n\nOn-time test gives 1. Eligibility test gives 1.\nFresh success Z = 1 x 1 = 1.",
        "300 ms early, with the right inputs",
        "An early result for the old topic instead gives 1 x 0 = 0.\nOther session, format and integrity checks also apply.",
        "Continuity and fresh success need separate reporting. An older bar may avoid a gap while failing this test.\nA later edit does not retroactively invalidate a frozen bar. Prepared success still needs playback verification.")
    cols=[65,440,710,960,1350]
    for x,label in zip(cols,["RESULT","FINISH","LATENESS","REQUIRED VERSION?","FRESH SUCCESS"]):
        text(ax,x,630,label,10.5,MUTED,"bold")
    data=[("Matching topic",7500,"-300 ms","Yes","1"),
          ("Old topic",7500,"-300 ms","No","0"),
          ("Matching topic, late",7900,"+100 ms","Yes","0")]
    for i,row in enumerate(data):
        y=670+i*59
        rect(ax,48,y-7,1504,51,"#F8FAFC" if i%2==0 else "white","white")
        for x,value in zip(cols,[row[0],f"{row[1]:,} ms",*row[2:]]):
            text(ax,x,y+4,value,12.6,TEAL if i==0 else RED)
    save_page(fig,"model-guide-06-success",16)


def model_search(e):
    fig,ax=model_guide(7,3,"Counting accepted lines explains the selection step",
        [(r"$Y_j=\sum_{c\in\mathcal{X}_j}\mathbf{1}[V(c)=1]$",21),
         (r"$U_j=\sum_{c\in\mathcal{X}_j:V(c)=1}\mathbf{1}[s(c)\geq\tau_s]$",20),
         (r"$s_j^*=\max_{c\in\mathcal{X}_j:V(c)=1,\ s(c)\geq\tau_s}s(c)$",20)],
        "Count each distinct evaluated line that passes the validity checks.\nCount again, now requiring the minimum score too. Finally choose\nthe highest score among those selectable lines.",
        "X: distinct evaluated candidates; V: pass/fail checks; s: ranking score.\nY: valid count; U: selectable count; tau_s: score threshold; s*: best selectable score.",
        "The flow requires 9 syllables. Four distinct lines\nhave been evaluated. Assume other validity tests pass.\nUse an illustrative score threshold of 75.\n\nY = 1 + 1 + 0 + 1 = 3 valid lines.\nU = 1 + 0 + 1 = 2 selectable valid lines.\ns* = max(82, 78) = 82. Choose candidate A.",
        f"{e['valid_count']} valid; {e['selectable_count']} selectable; best {e['best_score']}",
        "Toy 0-100 scores, not the production score calibration.\nThe recorded benchmark used threshold 0.",
        "Valid-count targets and score thresholds are separate rules. Raising the toy threshold to 90 leaves Y = 3\nbut U = 0; then no best selectable score exists. None of these scores directly measures intelligibility.")
    cols=[65,420,690,960,1265]
    for x,label in zip(cols,["CANDIDATE","SYLLABLES","VALID?","TOY SCORE","SELECTABLE?"]):
        text(ax,x,625,label,11,MUTED,"bold")
    for i,(label,syllables,valid,score) in enumerate(e["candidates"]):
        y=664+i*44
        rect(ax,48,y-5,1504,39,"#F8FAFC" if i%2==0 else "white","white")
        values=[label,str(syllables),"Yes" if valid else "No",str(score) if score is not None else "Not ranked",
                "Yes" if valid and score>=75 else "No"]
        for x,value in zip(cols,values):
            text(ax,x,y+2,value,12.4,TEAL if valid and score>=75 else MUTED)
    save_page(fig,"model-guide-07-search",17)


def model_coverage(e):
    fig,ax=model_guide(8,7,"The next missing interval limits ready playback",
        [(r"$\mathcal{C}_{class}(t)=\bigcup_{a:eligible_{class}(a,t)} I_a$",20),
         (r"$H_{class}(t)=\sup\{h\geq0:[\beta_{play},\beta_{play}+h)\subseteq\mathcal{C}_{class}(t)\}$",17)],
        "Combine the intervals covered by results allowed under one rule set.\nStarting at the current beat, measure how far that coverage continues\nwithout a hole. Stop at the first missing interval.",
        "Union combines intervals; subset means every beat is covered; sup gives longest lead.\nI_a: covered interval; class: allowed-output rules; beta_play: current beat; H: ready beats.",
        "Take one snapshot with playback at beat 0.\nFresh audio covers [0,4) and [8,12).\nBeats 4 through 8 are missing. H_fresh = 4 beats.\n\nAllow an eSpeak fallback covering [4,8).\nCombined allowed coverage becomes [0,12).\nH_allowed = 12 beats, although only 8 are fresh.",
        "1 fresh bar ahead; 3 with fallback",
        "At 90 BPM: 2.667 s fresh lead, 8.000 s allowed lead.\n[a,b) includes a but excludes b, so adjacent bars meet.",
        "The amount of finished audio alone can overstate safety. Its location and allowed version matter.\nReady coverage describes available material; the audio device must still schedule and play it correctly.")
    start,w=430,340
    for row,y in enumerate([650,748]):
        text(ax,65,y+12,"Fresh only" if row==0 else "Fallback allowed",13,COLORS[6],"bold")
        for i in range(3):
            fill="#E7F3EE" if i!=1 else ("#FFF2EB" if row==0 else "#FFF4DC")
            color=TEAL if i!=1 else (RED if row==0 else AMBER)
            rect(ax,start+i*w,y,w-8,49,fill,color)
            label="fresh bar" if i!=1 else ("missing" if row==0 else "eSpeak fallback")
            text(ax,start+i*w+(w-8)/2,y+13,label,12,color,ha="center")
        arrow(ax,(start,y+67),(start+(w if row==0 else 3*w)-8,y+67),COLORS[6],style="<->")
    for i in range(4):
        text(ax,start+i*w,836,str(i*4),10.8,MUTED,ha="center")
    text(ax,65,831,"Beat position",11,MUTED)
    save_page(fig,"model-guide-08-coverage",18)


def model_turns(e):
    fig,ax=model_guide(9,6,"A ready response waits for an allowed bar boundary",
        [(r"$b_{next}(x;g,\phi)=\phi+g\lceil(x-\phi)/g\rceil$",20),
         (r"$N_{wait}=\max(0,\lceil(F_{elig}+M_{sink}-t_{cap})/\tau\rceil)$",18.5),
         (r"$P_{response}=t_{cap}+N_{wait}\tau$",21)],
        "Round a required time upward to the next permitted boundary.\nFor alternating turns, first add the playback reserve to ready time,\nthen count whole bar intervals from capture end to that boundary.",
        "g: boundary spacing; phi: offset; x: required time. tau: bar duration; t_cap: capture end.\nF_elig: allowed prepared response ready; M_sink: reserve; N_wait: whole intervals. All times: ms.",
        "At 90 BPM, tau = 2,666.667 ms per bar.\nCapture ends at t_cap = 2,666.667 ms.\nResponse ready at F_elig = 6,200 ms; reserve = 200.\nRequired boundary time x = 6,400 ms.\n\n(x - t_cap)/tau = 1.4; round up to 2 intervals.\nP = 2,666.667 + 2 x 2,666.667 = 8,000 ms.",
        f"First response begins at {e['response_ms']/1000:.3f} s",
        "Here g = tau and phi = t_cap. Use exact bar duration.\nThe 5.333-s boundary is too early; the next is 8.000 s.",
        "Even a small readiness delay can move the response by a whole bar. Replaying rhythm fills the wait.\nN_wait includes transition time, not just echoes. Without a finite policy-eligible result, record no response.")
    x=lambda seconds:100+seconds*130
    stages=[(0,8/3,"Capture",BLUE),(8/3,16/3,"Transition",MUTED),
            (16/3,8,"Replay while waiting",AMBER),(8,32/3,"Model response",PURPLE)]
    for a,b,label,color in stages:
        rect(ax,x(a),691,x(b)-x(a)-5,61,"#F8FAFC",color)
        text(ax,(x(a)+x(b))/2,710,label,12,color,ha="center")
    ax.plot([x(6.4),x(6.4)],[654,766],color=RED,lw=1.5,ls="--")
    text(ax,x(6.4),623,"Ready + reserve = 6.400 s",11.5,RED,ha="center")
    for second,label in [(0,"0"),(8/3,"2.667 / capture end"),(16/3,"5.333"),(8,"8.000 / response")]:
        ax.plot([x(second),x(second)],[764,782],color=INK,lw=1)
        text(ax,x(second),800,label,11.5,ha="center" if second else "left")
    text(ax,1450,805,"seconds",10.8,MUTED)
    save_page(fig,"model-guide-09-turns",19)


def model_trace(e):
    fig,ax=model_guide(10,1,"A saved completion can be early for one deadline and late for another",
        [(r"$D_j^{cf}=[(L+j)m-1/r]\,60000/B$",21),
         (r"$miss_j^{cf}=1-\mathbf{1}[f_j\leq D_j^{cf}]\mathbf{1}[success_j]\mathbf{1}[latest_j]$",17.5)],
        "Compute playback from tempo and bar lead, then subtract one tick.\nKeep the saved completion time fixed and compare it with that cutoff.\nCount a miss if it is late, unsuccessful or for the wrong version.",
        "cf: hypothetical comparison; L: lead bars; j: bar index starting at 0; m: beats/bar.\nr: ticks/beat; B: BPM; f: saved completion timestamp. Missing results count as misses.",
        "Assume a successful, latest-version source arrives\nat f = 5,000 ms after the initial request.\nB = 90; m = r = 4; first bar j = 0.\n\nLead 1: D = (4 - 0.25) x 60000/90 = 2,500 ms.\nLead 2: D = 5,166.667 ms.\nLead 3: D = 7,833.333 ms.",
        "Lead 1: late. Leads 2 and 3: on time.",
        "A wrong-version result at 2,000 ms still counts as a miss.\nThis example is hypothetical; saved experiments follow.",
        "This isolates the arithmetic of changing deadlines. It does not predict how a live tempo or lead edit\nchanges queues, generation or delivery. These source cutoffs also omit local preparation and acoustic timing.")
    x=lambda ms:160+ms*.16
    for i,(lead,cutoff) in enumerate(zip((1,2,3),e["cutoffs_ms"])):
        y=650+i*62
        text(ax,48,y+12,f"Lead {lead}",12,BLUE,"bold")
        rect(ax,x(0),y,x(cutoff)-x(0),40,"#E9F0FE",BLUE)
        text(ax,x(cutoff)+10,y+9,f"{cutoff/1000:.3f} s",11.5,BLUE)
    ax.plot([x(5000),x(5000)],[635,824],color=RED,lw=1.6,ls="--")
    text(ax,x(5000),618,"Saved readiness: 5.000 s",11.5,RED,ha="center")
    text(ax,x(0),829,"Time since request: 0 s",10.8,MUTED)
    save_page(fig,"model-guide-10-trace",20)


def timing_model():
    fig, ax = page(3, "03 MODEL / TIME AND RESOURCES", "A deadline is not a latency measurement",
                   "All stage times share one reference clock in milliseconds. T maps beat position to seconds; the conversion to P is explicit.")
    text(ax, 48, 179, "1  Map beats to playback times and finish deadlines", 14, BLUE, "bold")
    text(ax, 48, 222, r"$T(\beta)=T_0+\int_0^\beta 60/B(u)\,du$", 18)
    text(ax, 735, 222, r"$P_j=1000\,[T(\beta_j)-t_{ref}],\quad D_{jv}=P_j-M_{jv}$", 18)
    text(ax, 48, 290, "The integral adds time across beats even when tempo changes. P is playback time; D is the stage deadline; M reserves later work once.", 11.2, MUTED)
    text(ax, 48, 334, "2  Only ready work may start; only available capacity may execute it", 14, PURPLE, "bold")
    text(ax, 48, 378, r"$R_{jv}=\max\{r^{pol}_{jv}, A_{jv}, \max_{(i,u)\in pred(j,v)} F_{iu}\}$", 18)
    text(ax, 48, 442, r"$S_{jv}\geq R_{jv},\quad Q_{jv}=S_{jv}-R_{jv},\quad F_{jv}=S_{jv}+C_{jv}$", 18)
    text(ax, 945, 432, r"$\sum_{j,v:p(v)=q}\mathbf{1}_{[S_{jv},F_{jv})}(t)\leq k_q$", 17)
    text(ax, 48, 510, "max takes the latest required time. The sum counts jobs executing on resource q; it cannot exceed k_q slots. Measured render capacity: 1.", 11, MUTED)
    text(ax, 48, 553, "3  Follow the required steps, then check whether the result is allowed", 14, TEAL, "bold")
    text(ax, 48, 595, "Illustrative one-job path; widths are schematic, not measured", 10.5, MUTED)
    spans = [(80, 190, "Write / select", TEAL), (270, 160, "Resource wait", "#D6DAE0"),
             (430, 295, "Synthesize / align", PURPLE), (725, 165, "Transfer", "#52778D"),
             (890, 160, "Prepare", BLUE)]
    for x, w, label, color in spans:
        rect(ax, x, 632, w-4, 57, color, "white")
        text(ax, x+(w-4)/2, 648, label, 11, "white" if color!="#D6DAE0" else INK, ha="center")
    ax.plot([80, 1480], [712, 712], color=INK, lw=1)
    for x, label, color in [(80, "R: earliest start", INK), (1050, "F: prepared", BLUE),
                            (1230, "D: cutoff", RED), (1480, "P: playback", INK)]:
        ax.plot([x, x], [620, 720], color=color, lw=1.2, ls="--")
        text(ax, x, 732, label, 10.8, color, ha="center" if x>100 else "left")
    arrow(ax, (1233, 609), (1478, 609), RED, style="<->")
    text(ax, 1355, 578, "sink reserve M", 10.5, RED, ha="center")
    text(ax, 48, 783, r"$\sigma_j(t)=D_j-t-\widehat W_j^{rem}(t)$", 19)
    text(ax, 790, 783, r"$Z_j^{fresh}=\mathbf{1}[F_{j,prep}\leq D_{j,prep}]\;\mathbf{1}[eligible(a_j;\kappa_j)]$", 17)
    text(ax, 48, 838, "Remaining-work estimate must include queues and downstream work.\nA service-only critical path is a lower bound; no predictor is fitted here.", 11.1, MUTED)
    text(ax, 790, 838, "kappa: declared input/revision cutoff for this output obligation.\nPrepared-stage success is not proof of acoustic success.", 11.1, MUTED)
    text(ax, 48, 919, "Do not count waiting twice inside render_total. Adding separate stage percentiles does not give a whole-pipeline percentile. Work times need not be independent.", 10.3, RED)
    save_page(fig, "timing-model", 21)


def contract_model():
    fig, ax = page(4, "04 MODEL / SEARCH, REVISION, GRANULARITY", "Useful output must pass checks, match inputs and arrive in time",
                   "Search settings change the work. Edits can invalidate results. Chunk sizes and required inputs determine when a response is allowed.")
    text(ax, 48, 180, "G3  Requested candidates != usable candidates", 14, TEAL, "bold")
    text(ax, 48, 215, r"$Y_j=\sum_{c\in\mathcal{X}_j}\mathbf{1}[V(c)=1]$", 18)
    text(ax, 680, 215, r"$U_j=\sum_{c\in\mathcal{X}_j:V(c)=1}\mathbf{1}[s(c)\geq\tau_s]$", 18)
    text(ax, 48, 291, "Sum over distinct evaluated lines X: count valid lines as Y, and valid lines above the score threshold as U. Selection needs U >= 1.\nThe desired valid count of 3 stops search; it is not a quality guarantee. These scores do not measure what listeners understand.", 11.3, MUTED)
    text(ax, 48, 351, "G4 + G7  Keep fresh coverage separate from fallback-permitted coverage", 14, RED, "bold")
    text(ax, 48, 390, r"$\mathcal{C}_{class}(t)=\bigcup_{a:eligible_{class}(a,t)} I_a$", 17)
    text(ax, 690, 390, r"$H_{class}(t)=\sup\{h\geq0:[\beta_{play},\beta_{play}+h)\subseteq\mathcal{C}_{class}(t)\}$", 16)
    text(ax, 48, 462, "Union combines covered intervals; sup takes the longest forward span with no hole. Compute this separately for each allowed class of output.\nAn older voice or replay can avoid a gap without providing fresh lyrics. Judge frozen output against its declared input cutoff.", 11, MUTED)
    text(ax, 48, 526, "G6  Finer playback timing does not mean smaller lyric chunks", 14, AMBER, "bold")
    xs = [48, 365, 682, 999, 1316]
    widths = [292, 292, 292, 292, 236]
    for x, w, title, body in zip(xs, widths,
                               ["Input", "Write / select", "Speech / prepare", "Commit", "Placement"],
                               ["One captured bar", "One lyric bar", "One bar / phrase", "Bar; freeze 1 tick early", "Ticks -> samples"]):
        rect(ax, x, 567, w, 65, "#F7F8FA", LINE)
        text(ax, x+13, 577, title, 12, INK, "bold")
        text(ax, x+13, 607, body, 10.7, MUTED)
    for i in range(4):
        arrow(ax, (xs[i]+widths[i]+1, 600), (xs[i+1]-2, 600), MUTED)
    text(ax, 48, 657, "TWO INTERACTION CONTRACTS", 11, MUTED, "bold")
    text(ax, 48, 697, "Continuous\nsteering", 12, BLUE, "bold")
    text(ax, 48, 782, "Alternating\nturns", 12, PURPLE, "bold")
    starts = [265, 480, 695, 910, 1125, 1340]
    for i, x in enumerate(starts):
        rect(ax, x, 691, 200, 58, "#EFF5FF", BLUE)
        text(ax, x+100, 710, "model output", 11, BLUE, ha="center")
        label = ["capture", "transition", "echo if needed", "response 1", "echo if needed", "response 2 / ..."][i]
        color = [BLUE, MUTED, AMBER, PURPLE, AMBER, PURPLE][i]
        rect(ax, x, 776, 200, 58, "#FAF9FC", color)
        text(ax, x+100, 795, label, 10.8, color, ha="center")
    text(ax, 270, 849, "Schematic: continuous edits target unfrozen bars; alternating topics affect next capture. Replay may separate logical response bars.", 10.3, MUTED)
    text(ax, 48, 886, r"$P_{response}=t_{cap}+N_{wait}\tau$", 16)
    text(ax, 48, 928, r"$N_{wait}=\max(0,\lceil(F_{elig}+M_{sink}-t_{cap})/\tau\rceil)$", 16)
    text(ax, 915, 886, "Ceiling brackets round up to a whole bar interval.\nConstant tempo; M_sink includes the freeze guard.\nNo finite eligible readiness means no response time.", 10.8, MUTED)
    save_page(fig, "eligibility-granularity", 22)


def chart_style(ax, title, ylabel):
    ax.set_title(title, loc="left", fontsize=13, weight="bold", pad=16)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.tick_params(labelsize=10)
    ax.grid(axis="y", color="#E9EDF2", zorder=0)
    ax.set_axisbelow(True)


def observations(summary, budget):
    fig, ax = page(5, "05 EVIDENCE / OBSERVED", "Small experiments identify mechanisms, not universal laws",
                   "Baseline Qwen2.5-7B + MOSS/MMS. Real model executions; all candidate attempts retained. Separate groups are not pooled.")
    axes = [fig.add_axes([.07, .58, .39, .20]), fig.add_axes([.57, .58, .39, .20]),
            fig.add_axes([.07, .225, .39, .20]), fig.add_axes([.57, .225, .39, .20])]
    pairs = sorted(set(r["pair_id"] for r in budget))
    for plot, key in zip(axes[:2], ("valid_count", "elapsed_ms")):
        for pair in pairs:
            rows = {int(r["condition"][1:]): r for r in budget if r["pair_id"] == pair}
            ys = [rows[n][key] for n in (16, 32, 64)]
            plot.plot([0, 1, 2], ys, "o-", ms=3, lw=.8, alpha=.4, color=TEAL if key=="valid_count" else BLUE)
        values = [statistics.median(r[key] for r in budget if r["condition"]==f"n{n}") for n in (16, 32, 64)]
        plot.plot([0, 1, 2], values, "D", ms=6, color=INK, label="Median", zorder=5)
        plot.set_xticks([0, 1, 2], ["16", "32", "64"])
        plot.set_xlabel("Requested candidates per one-wave attempt", fontsize=10)
        plot.set_xlim(-.2, 2.2)
    chart_style(axes[0], "G3  More search produced more valid options", "Valid candidates / attempt")
    chart_style(axes[1], "G2 + G3  Host-local cost also changed", "Writer elapsed time (ms)")
    axes[0].set_ylim(bottom=0)
    axes[1].set_ylim(bottom=0)
    text(ax, 112, 475, "Mean valid: 4.17 / 6.50 / 11.17. Attempts with a selectable line:\n10/12 / 10/12 / 12/12. Lines pair workloads; diamonds are medians.", 10.3, MUTED)
    text(ax, 912, 475, "Median: 91 / 95 / 156 ms. Cold n16 observation is retained.\nNo fitted cost curve; n32 always occupied the middle order position.", 10.3, MUTED)

    categories = ["MOSS", "Alignment", "Render total", "Residual"]
    keys = ["moss", "aligner", "render_total", "render_residual"]
    for offset, condition, label, color in [(-.18, "sequential", "One session at a time", BLUE),
                                          (.18, "parallel", "Two sessions admitted", PURPLE)]:
        vals = [summary["stages"][condition]["stages"][key]["median_ms"]/1000 for key in keys]
        axes[2].bar(np.arange(4)+offset, vals, width=.34, label=label, color=color)
    axes[2].set_xticks(np.arange(4), categories)
    chart_style(axes[2], "G2  More admission did not create render slots", "Median stage span (s)")
    axes[2].legend(fontsize=8.5, frameon=False, loc="upper left")
    axes[2].set_ylim(0, 4.4)
    for x, condition in enumerate(("no_edits", "one_edit", "burst_edits")):
        vals = [v/1000 for v in summary["edit_first_bar_ms"][condition]]
        axes[3].scatter([x-.045,x,x+.045], vals, color=RED, s=25, alpha=.8)
        axes[3].plot([x-.17,x+.17], [statistics.median(vals)]*2, color=INK, lw=2)
    axes[3].set_xticks([0,1,2], ["No edit", "One edit", "Burst edits"])
    axes[3].set_ylim(0, 4.4)
    chart_style(axes[3], "G4  Latest-source readiness changed after edits", "Request to latest bar-0 source (s)")
    text(ax, 112, 815, "Three paired groups; 12 bars/condition, from shared sessions.\nMOSS ~1.78 s in both; total 1.81 -> 3.58 s. Residual = total minus\nmeasured synthesis/alignment. Separate medians do not add up.", 10.2, MUTED)
    text(ax, 912, 815, "Three matched seeds/condition; medians 1.917 / 3.799 / 3.826 s.\nFinal topics differ: this is not a causal cost-per-edit estimate.\nSource receipt is not an audible response measurement.", 10.2, MUTED)
    text(ax, 48, 895, "G5: 12 Mac preparation observations, median 49.99 ms / max 59.17 ms. Two Mac receipts: 5.786 / 7.794 s, including prior work and transport.", 10.8)
    text(ax, 48, 926, "No listening-quality study, acoustic onset measurement, tail guarantee or statistical significance claim. Route-median subtraction cannot isolate SSH latency.", 10.5, RED)
    save_page(fig, "observed-patterns", 23)


def deadlines(summary):
    fig, ax = page(6, "06 EVIDENCE / DEADLINE SENSITIVITY", "Available time changes which recorded results would be late",
                   "Shift the deadline while holding every observed source-ready time fixed. This is sensitivity analysis, not a new live tempo benchmark.")
    text(ax, 48, 181, r"$D_j^{cf}=[(L+j)m-1/r]\,60000/B\quad (m=r=4)$", 21)
    text(ax, 920, 181, "L: lead in bars; first bar is j = 0; guard: 1 tick.\ncf means hypothetical deadline, not a new run.\n90 BPM cutoffs: 2.500 / 5.167 / 7.833 s.", 11, MUTED)
    heat = fig.add_axes([.17, .385, .31, .32])
    conditions = ["sequential", "parallel", "no_edits", "one_edit", "burst_edits", "mac_two_bars"]
    labels = ["Sequential", "Concurrent", "No edits", "One edit", "Burst edits", "Mac receipt"]
    cells = {(c["condition"],c["bpm"],c["lookahead_bars"]):c for c in summary["deadline_counterfactuals"]}
    matrix = [[cells[(c,90,l)]["miss_rate"] for l in (1,2,3)] for c in conditions]
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("misses", ["#EFF7F3", "#E9C088", "#B34C42"])
    heat.imshow(matrix, cmap=cmap, vmin=0, vmax=1, aspect="auto")
    heat.set_xticks([0,1,2], ["1 bar", "2 bars", "3 bars"])
    heat.set_yticks(range(6), labels)
    heat.tick_params(length=0, labelsize=11, pad=10)
    heat.set_title("90-BPM source-ready misses / observations", fontsize=13, loc="left", weight="bold", pad=18)
    for y,c in enumerate(conditions):
        for x,l in enumerate((1,2,3)):
            v = cells[(c,90,l)]
            heat.text(x,y,f"{v['misses']}/{v['rounds']}",ha="center",va="center",fontsize=15,
                      weight="bold",color="white" if v["miss_rate"]>.6 else INK)
    for spine in heat.spines.values():
        spine.set_visible(False)
    plot = fig.add_axes([.61,.385,.35,.32])
    bpms = np.linspace(60,180,121)
    for lead,color in [(1,BLUE),(2,TEAL),(3,PURPLE)]:
        plot.plot(bpms,(4*lead-.25)*60/bpms,label=f"Lead {lead}",color=color,lw=2)
    plot.axvline(90,color=LINE,ls="--")
    plot.set_xticks([60,90,120,150,180])
    plot.set_xlabel("Tempo (BPM)", fontsize=11)
    chart_style(plot,"G1  Deadline budget follows the musical clock", "First-bar source cutoff (s)")
    plot.legend(frameon=False,fontsize=10)
    plot.set_ylim(0,12)
    text(ax, 48, 658, "What this establishes", 13, TEAL, "bold")
    text(ax, 48, 697, "Tempo, lead and guard jointly determine physical budget.\nA complete source can miss an earlier obligation.\nOriginal obligations remain after edits; stale results are not fresh success.", 11.7)
    text(ax, 835, 658, "Interpretation limits", 13, RED, "bold")
    text(ax, 835, 697, "Live lookahead changes may alter queues and obsolete work.\nSource-ready excludes preparation, enqueue and acoustics.\nZero observed misses does not bound future tail risk.", 11.7)
    ax.plot([48,1552],[790,790],color=LINE,lw=1)
    text(ax, 48, 817, "Compact study design", 13, INK, "bold")
    text(ax, 305, 817, "Vary time budget, admitted load, search policy and edit trace. Fix and record delivery, temporal contract and fallback rules.\nMeasure stage waits + revisions + contiguous coverage; report continuity, fresh success, validity, response gap and quality separately.", 11.2)
    text(ax, 48, 898, "90 cells: 5 tempos x 3 leads x 6 trace conditions. No resimulation of changed admission or queues; local preparation / enqueue / acoustics omitted.", 10.5, MUTED)
    text(ax, 48, 928, "Sources: saved evidence JSON + raw SHA-256-verified traces; technical companion contains full definitions, registry crosswalk, code anchors and prior-paper reference.", 10.1, MUTED)
    save_page(fig, "deadline-sensitivity", 24)


def assemble_pdf():
    from reportlab.pdfgen import canvas
    from pypdf import PdfReader, PdfWriter
    from io import BytesIO
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    writer = PdfWriter()
    titles = PAGE_TITLES
    for i, title in enumerate(titles, 1):
        page = PdfReader(TMP / f"page-{i}.pdf").pages[0]
        overlay = BytesIO()
        c = canvas.Canvas(overlay, pagesize=(1152,720))
        c.setFillColorRGB(.34,.38,.44)
        c.setFont("Helvetica",8)
        c.drawString(35,13,"StreamMUSE  |  PI research brief  |  Revised 23 October 2026")
        c.drawRightString(1117,13,f"{i:02d} / {len(titles):02d}  |  {title}")
        c.linkURL("https://arxiv.org/html/2606.11886v1",(35,7,295,22),relative=0,thickness=0)
        c.showPage()
        c.save()
        overlay.seek(0)
        page.merge_page(PdfReader(overlay).pages[0])
        writer.add_page(page)
        writer.add_outline_item(title, i-1)
    writer.add_metadata({"/Title":"Parameterizing and Modeling a Real-Time Generative System",
                         "/Author":"StreamMUSE research working notes",
                         "/Subject":"PI brief: parameterization, structural models, measured evidence"})
    with OUTPUT.open("wb") as stream:
        writer.write(stream)
    result = PdfReader(OUTPUT)
    assert len(result.pages)==len(PAGE_TITLES)
    assert all(len(page.extract_text())>500 for page in result.pages)
    print(json.dumps({"pdf":str(OUTPUT),"pages":len(result.pages),"text_extractable":True}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdf-python",default=sys.executable,help="Python with ReportLab and pypdf for vector PDF assembly")
    parser.add_argument("--assemble-only",action="store_true",help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.assemble_only:
        assemble_pdf()
        return
    summary,budget = load_evidence()
    RESEARCH.mkdir(parents=True,exist_ok=True)
    TMP.mkdir(parents=True,exist_ok=True)
    setup_plotting()
    foundations()
    guide_time()
    guide_work()
    guide_search()
    guide_revisions()
    guide_delivery()
    guide_granularity()
    guide_coverage()
    parameter_map()
    architecture()
    examples = worked_examples()
    for build in [model_clock,model_deadline,model_readiness,model_capacity,model_remaining,
                  model_success,model_search,model_coverage,model_turns,model_trace]:
        build(examples)
    timing_model()
    contract_model()
    observations(summary,budget)
    deadlines(summary)
    subprocess.run([args.pdf_python,str(Path(__file__).resolve()),"--assemble-only"],check=True)
    print(json.dumps({"verified_primary_inputs":len(summary["primary_inputs"]),"verified_deadline_cells":90,
                      "worked_example_pages":len(MODEL_GUIDE_TITLES),"figures":len(PAGE_TITLES)}))


if __name__=="__main__":
    main()
