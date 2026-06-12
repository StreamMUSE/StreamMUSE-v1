#!/usr/bin/env bash
set -euo pipefail

# One-command strict realtime demo/validation entrypoint for Lekai prompt-continuation.
# Defaults are the current 0509 handoff settings. Override by prefixing env vars,
# e.g. DEVICE=5 OUT_ROOT=realtime_runs/my_demo scripts/run_lekai_prompt_continuation_realtime_demo.sh

export DEVICE=${DEVICE:-0}
export PORT=${PORT:-8001}
export IDS=${IDS:-"6217163 5472152 5472153 6144911 332891 3329166 6303422"}
export OUT_ROOT=${OUT_ROOT:-realtime_runs/lekai_prompt_continuation_demo}

# Use metadata tempo, but clamp fast pieces so realtime playback does not outrun catch-up.
export CLI_TEMPO=${CLI_TEMPO:-metadata}
export CLI_TEMPO_MAX=${CLI_TEMPO_MAX:-120}

# `auto` keeps 8 beats for 4/4 and 2/4, and uses 6 beats for 3/4.
export PROMPT_BEATS=${PROMPT_BEATS:-auto}
export PROMPT_CONTINUATION_ENGINE=${PROMPT_CONTINUATION_ENGINE:-standard}

# Long enough for demo listening; reduce for smoke tests if needed.
export MAX_TICKS=${MAX_TICKS:-432}

# Prompt-stage settings that matched the RT offline prompt reference in validation.
export PROMPT_TEMPERATURE=${PROMPT_TEMPERATURE:-1.1}
export PROMPT_TOP_K=${PROMPT_TOP_K:-0}
export PROMPT_TOP_P=${PROMPT_TOP_P:-0.95}

# Current author-recommended continuation sampling settings.
export RT_TEMPERATURE=${RT_TEMPERATURE:-0.8}
export RT_TOP_K=${RT_TOP_K:-50}
export RT_TOP_P=${RT_TOP_P:-0.98}
export RT_REPETITION_PENALTY=${RT_REPETITION_PENALTY:-1.2}

# Realtime audible scheduling policy used in the current demo bundle.
export RECOVER_LATE_EVENTS=${RECOVER_LATE_EVENTS:-1}
export BOUND_LATE_RECOVERY=${BOUND_LATE_RECOVERY:-1}
export RECOVER_LATE_MAX_TICKS=${RECOVER_LATE_MAX_TICKS:-4}

exec scripts/run_cli_prompt_alignment_batch.sh
