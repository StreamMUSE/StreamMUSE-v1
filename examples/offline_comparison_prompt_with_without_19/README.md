# Offline comparison set: prompt_with_without_19_scored

This directory contains a minimal, paired subset extracted from:

`prompt_with_without_19_scored.zip`

The source archive includes multiple conditions for each case. For the first
offline comparison pass, this folder keeps only:

- `gt/`: ground-truth MIDI files, selected from `*_GT.mid`.
- `base_noprompt_zero/`: no-prompt baseline MIDI files, selected from
  `*_base_noprompt.mid`.

`base_noprompt_zero` is the closest set in the archive to the "generated from
zero / no prompt" baseline. Other archive variants such as `base_prompt`,
`base_melprompt`, and `initdrop*` are intentionally left out for now so the
first comparison stays paired and easy to interpret.

## Layout

```text
examples/offline_comparison_prompt_with_without_19/
  gt/
    case01_6383897_GT.mid
    ...
    case19_6175749_GT.mid
  base_noprompt_zero/
    case01_6383897_base_noprompt.mid
    ...
    case19_6175749_base_noprompt.mid
  manifest.csv
  score_metadata.csv
```

## Pairing

Use `manifest.csv` as the authoritative mapping between source archive paths
and normalized local filenames.

Each case has two rows:

- `role=gt`
- `role=base_noprompt_zero`

For offline testing, compare new framework outputs against `gt/`, then compare
the same case-level scores against `base_noprompt_zero/` as the previous
no-prompt baseline.

## Suitability Scores

The source zip stores the task-suitability score in each source folder name.
For example, `03-5` means folder `03` has score `5`, and `15-7` means folder
`15` has score `7`.

`score_metadata.csv` makes this explicit with one row per case. Scores ending
with `?` come from source folder names that included a question mark-like suffix
in the archive, so treat them as uncertain labels rather than clean integers.
