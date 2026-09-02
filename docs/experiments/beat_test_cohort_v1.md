# BEAT checkpoint-aligned test cohort v1

Use `scripts/build_beat_test_cohort.py` to build the formal music-metric cohort.

```bash
python scripts/build_beat_test_cohort.py \
  --data-dir /data/home/yuanxin/data/allxml_npz_dual_track_optimized_no_underscore \
  --output-dir output/metrics/beat_test_cohort_v1
```

Defaults are `--count 40` and `--selection-seed 20260901`.

This builder deliberately uses the dataset contract of the current Prompt and
Continuation checkpoints:

- 192,788 legacy four-channel NPZ files;
- `.lengths_cache.pkl["data_files"]` as the authoritative ordering;
- NumPy legacy shuffle with seed 42;
- the final 10 percent (19,278 files) as held-out test data;
- frozen ordered test-list SHA256
  `f0366d936d815f775d398e4658ab393a55762a6cc176357693a9a1b4e389733e`.

It must not be replaced with the newer BEAT v3 169,283-piece 80/10/10 split:
that split is not aligned with these checkpoints.

Eligibility is fixed to the requested inference window: every measure must be
`(4, 88, 16)`, all `time_signature_idx` values must be the confirmed 4/4 index
0, total length must be at least 128 steps, and melody/accompaniment onset
channels must both be non-empty in `[0,32)` and `[32,128)`. No genre,
complexity, title, composer, or human selection is performed.

The generated directory contains the complete ordered test list and hash,
`candidate_audit.csv`, CSV/JSON cohort manifests, and one directory per piece.
Each piece includes the untouched `source.npz`, `melody_120bpm.mid`, and
`gt_120bpm.mid`. The MIDI files are derived directly from the piano roll at
fixed 120 BPM with explicit `Melody` and `Accompaniment` track names.
