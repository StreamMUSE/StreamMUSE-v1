# Lekai Cross-Style Evaluation

Task 3 uses metadata-derived style subsets. These labels are candidate-selection labels,
not ground-truth genre annotations.

Classification uses only semantic metadata: title, subtitle, movement title, composer,
creator, artist, arranger, and explicit franchise/source names in `credits` and
`creators`. Note events, pianorolls, NPZ density, BPM, duration, and listening impressions
must not determine the label. The keyword rules retrieve high-precision candidates;
the frozen set is finalized by a piece-by-piece semantic audit of title and creator.

NPZ information is used only after classification to reject unusable files and to
record evaluation covariates. It does not move a piece from one category to another.

## Initial categories

- `classical`: explicit canonical classical composer match.
- `pop`: explicit pop artist/songwriter match.
- `film_tv_soundtrack`: explicit film/TV composer or franchise match.
- `game_anime_soundtrack`: explicit game/anime composer or franchise match.
- `jazz_blues_ragtime`: explicit jazz composer or style-title match.

Generic words such as `theme`, `love`, `song`, and `prayer` are intentionally excluded.
Candidates matching more than one category are excluded from the first round.

## Candidate filters

- Single piano/keyboard part in the MusicXML metadata.
- Corresponding MusicXML and no-underscore NPZ exist.
- NPZ measure shape is `(4, 88, T)` with both Melody and accompaniment non-empty.
- 4/4 time signature.
- 32 to 160 measures.
- At most three shortlist entries per primary composer/franchise/style-term evidence.

The resulting CSV must be manually reviewed before any subset is frozen. BPM, length,
Melody density, and accompaniment density are retained for later matching across
categories.

`frozen_selection.json` records the approved four-piece metadata selection for each of
the five subsets.

## Frozen inference round

Task 3 primarily evaluates one sampling setting across metadata-derived style subsets.
One greedy output per piece is included as a paired control. Greedy and sampling results
must be summarized separately; the single greedy output is not treated as another
sampling seed.

```text
Checkpoint: base epoch15 continuation model
Condition: 8-beat GT accompaniment prompt
Primary decoder: nucleus sampling
  temperature=1.1, top_k=0, top_p=0.95; seeds=0,1,2
Paired control: greedy
  temperature=1.0, top_k=1, top_p=1.0; seed=0
Scale: 5 styles x 4 pieces x 4 outputs = 80 outputs
```

The 20 metadata assignments and their piece-specific reasoning are recorded in
`remote_results/lekai_cross_style_task3_frozen20_preflight_20260802/metadata_classification_audit.md`.

Before GPU inference, review the 20 reconstructed source arrangements with
`remote_results/lekai_cross_style_task3_frozen20_preflight_20260802/source_review.csv`.
This audio review checks only arrangement usability and Melody/accompaniment track
assignment. It must not relabel style from musical content.

After the source review is accepted, launch the formal runner with four GPUs:

```bash
cd /data/home/yuanxin/StreamMUSE-v1
GPUS=3,4,5,6 \
  bash experiments/lekai_cross_style/run_task3_offline_sampling.sh
```

It uses the existing `RT-accompanimentV2/inference_new.py` without modifying model
inference. The result analyzer produces objective diagnostics by style and decoder plus
a listening bundle organized as
`style/piece/source_gt + greedy + sampling_seed0..2`.

## Build candidates on MBZ

```bash
cd /data/home/yuanxin/StreamMUSE-v1
.venv/bin/python experiments/lekai_cross_style/build_style_candidates.py \
  --metadata /data/home/yuanxin/data/allxml_xml/xml_metadata.jsonl \
  --xml-root /data/home/yuanxin/data/allxml_xml/allxml_npz_dual_track_optimized_no_underscore_xml \
  --npz-root /data/home/yuanxin/data/allxml_npz_dual_track_optimized_no_underscore \
  --rules experiments/lekai_cross_style/style_rules.json \
  --output-dir output/lekai_cross_style/task3_candidate_audit_20260802
```
