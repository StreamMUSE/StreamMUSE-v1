# Dataset V2 leading-trimmed NPZ

This preparation removes only the prefix before the first nonzero Melody onset
in channel 1. After concatenating `measure_0..N`, the output roll is exactly:

```python
output_full_roll = source_full_roll[:, :, first_melody_onset_step:]
```

Channel order is `[mel_sustain, mel_onset, acc_sustain, acc_onset]`. Internal
rests, trailing music, and trailing rests are retained exactly. Measures use 16
steps (4/4 at 4 steps per beat); the final measure may be partial and is never
padded. The tool does not restore measures or change `valid_measures`.
Source NPZ files may also begin with a partial pickup measure; after shifting,
the timeline is re-split every 16 steps and the final measure remains unpadded.

Run on the remote dataset with:

```bash
bash experiments/lekai_failcase_dataset_v2/run_prepare_leading_trimmed_npz.sh
```

The default output is
`/data/home/yuanxin/data/lekai_failcase_dataset_v2_leading_trimmed`, containing
`input_npz/{id}.npz`, `manifest.json`, and `manifest.csv`. Set `NPZ_ROOT`,
`OUTPUT_ROOT`, `STREAMMUSE_ROOT`, or `PYTHON_BIN` to override the defaults.
