# Lekai Melody Timing Robustness Task 1 v2

## Question

When Melody onset timing is perturbed on the model-visible 4-steps-per-beat grid,
how much does the generated continuation change, and is the response different
between greedy decoding and nucleus sampling?

## Frozen setting

| Item | Setting |
|---|---|
| Pieces | 5889045, 5817083, 332891, 5472153 |
| Melody conditions | clean, 10% onset-perturbed, 50% onset-perturbed |
| Perturbation | selected Melody notes shifted by exactly -1 or +1 model tick |
| Perturbation seed | 0 |
| Prompt | 8-beat GT accompaniment prompt |
| Model | `/data/home/yuanxin/RT-accompanimentV2/checkpoints-resume/epoch_15_0307_1858/model.safetensors` |
| Greedy | temperature 1.0, top-k 1, top-p 1.0, seed 0 |
| Sampling | temperature 1.1, top-k disabled, top-p 0.95, seeds 0/1/2 |
| Outputs | 48 MIDI files |

Clean and perturbed inputs are inferred in separate singleton runs. For each
decoder seed, this resets the run-level RNG before every condition and prevents
unrelated RNG advancement from being mistaken for a perturbation effect.

## Input audit

All 12 prepared inputs passed these checks before inference:

- The perturbation remains visible after 4tpb quantization.
- Pitch, quantized duration, note count, accompaniment, and NPZ metadata are preserved.
- Only Melody onset/sustain cells differ from the clean input.

| Piece | Melody notes | Changed at 10% | Changed at 50% |
|---|---:|---:|---:|
| 5889045 | 620 | 62 | 310 |
| 5817083 | 171 | 17 | 86 |
| 332891 | 334 | 33 | 167 |
| 5472153 | 291 | 29 | 146 |

## Objective results

All 48 runs completed. No generated continuation was empty.

Remote run root:
`/data/home/yuanxin/StreamMUSE-v1/output/lekai_melody_timing_robustness/task1_v2_greedy_nucleus_new4_20260806`

| Decoder | Perturbed notes | Pairs | Exact matches | Note-record Jaccard | Pianoroll Jaccard | Exact-beat rate |
|---|---:|---:|---:|---:|---:|---:|
| Greedy | 10% | 4 | 1 | 0.554 | 0.660 | 0.649 |
| Greedy | 50% | 4 | 1 | 0.506 | 0.649 | 0.590 |
| Nucleus sampling | 10% | 12 | 0 | 0.162 | 0.235 | 0.229 |
| Nucleus sampling | 50% | 12 | 0 | 0.088 | 0.167 | 0.137 |

The response is strongly piece-dependent:

| Piece | Greedy roll Jaccard, 10% | Greedy roll Jaccard, 50% | Sampling mean, 10% | Sampling mean, 50% |
|---|---:|---:|---:|---:|
| 332891 | 0.170 | 0.203 | 0.101 | 0.050 |
| 5472153 | 1.000 | 1.000 | 0.544 | 0.366 |
| 5817083 | 0.534 | 0.438 | 0.215 | 0.208 |
| 5889045 | 0.936 | 0.954 | 0.082 | 0.042 |

For greedy decoding, piece 5472153 is exactly unchanged at both perturbation rates,
while 332891 changes substantially. The aggregate 10%-to-50% trend is weak and not
strictly monotonic by piece, which is expected from discrete autoregressive argmax
trajectories.

For nucleus sampling, the same random seed does not guarantee the same output once the
perturbed context changes the token distribution. A first differing sampled token can
send the remaining autoregressive trajectory down a different path. The lower
similarity therefore shows greater output sensitivity under sampling, not by itself
lower musical quality.

## Listening results

Ratings remain pending in `analysis/ratings.csv`. The listening directory hides
decoder, seed, and perturbation labels; the mapping is retained only in
`analysis/_technical/listening_key.csv`.

## Interpretation boundary

Objective output difference is a sensitivity measure, not a direct music-quality
score. A conclusion about robustness must combine paired output-change statistics
with blind ratings of musical quality, Melody-accompaniment fit, stability, and
variety.
