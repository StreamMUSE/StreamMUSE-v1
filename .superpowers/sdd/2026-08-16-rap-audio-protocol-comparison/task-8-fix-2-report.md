# Task 8 Fix Round 2 Report

## Review Finding

The package stage built its listening matrix from whichever `mix.wav` files
were present. Missing selected files were silently skipped, so the unfiltered
default could succeed with fewer than the required three songs by four
protocols.

## Root Cause And Fix

`package_campaign()` used `continue` when a selected mix was missing. It now
constructs the Cartesian product of the selected songs and protocols and
validates every expected `mix.wav` with `is_file()` before emitting progress or
writing package artifacts. A failure lists the missing selected paths and is
recorded through the existing immediate `campaign_errors.jsonl` path.

The preserved selection contracts are:

| CLI filters | Required package matrix |
| --- | ---: |
| none | 3 songs x 4 protocols = 12 mixes |
| `--song` | 1 song x 4 protocols = 4 mixes |
| `--protocol` | 3 songs x 1 protocol = 3 mixes |
| `--song` and `--protocol` | 1 song x 1 protocol = 1 mix |

## TDD Evidence

The initial regression run reproduced the review finding:

```text
1 failed, 3 passed, 6 deselected
Failed: DID NOT RAISE <class 'ValueError'>
```

After the preflight validation, package-specific coverage passed:

```text
7 passed, 5 deselected
```

Regression cases cover rejection of incomplete default, `--song`, and
`--protocol` matrices. Separate cases prove complete `--song`, `--protocol`,
and combined-filter matrices still package with the expected audit counts.

## Verification

Focused Task 8 tests:

```text
34 passed in 8.04s
```

Backend-inclusive Task 8 tests:

```text
101 passed in 8.10s
```

Python compilation and `git diff --check` also completed successfully.

## Changed Files

- `scripts/run_rap_audio_protocol_comparison.py`
- `tests/unit/scripts/test_run_rap_audio_protocol_comparison.py`
- `.superpowers/sdd/2026-08-16-rap-audio-protocol-comparison/task-8-fix-2-report.md`

Unrelated pre-existing worktree files were not modified or staged for this
fix.
