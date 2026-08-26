# Dataset V2 candidate scout

## Corrected retention rule

The earlier one-retained-piece-per-style interpretation was incorrect. Dataset V2 keeps every eligible piece from the corrected constant-4/4 V1 main cohort, then adds new non-overlapping pieces until each style contains five pieces.

- Retained from V1: 13 pieces.
- New V2 candidates: 12 pieces.
- Provisional total: 25 unique pieces, five per style.
- No model inference was run and no remote file was created or modified.
- Remote source paths remain MBZ-Martina metadata, MusicXML, and NPZ paths. The eight restored retained audits use the read-only remote artifacts cached during the full 35-piece meter audit.

## Composition

| Style | Retained | New | Total |
|---|---:|---:|---:|
| classical | 2 | 3 | 5 |
| pop_contemporary | 4 | 1 | 5 |
| film | 1 | 4 | 5 |
| anime | 3 | 2 | 5 |
| game | 3 | 2 | 5 |

Required split: classical 2+3, pop_contemporary 4+1, film 1+4, anime 3+2, game 3+2.

## Provisional 25-piece selection

| Style | Role | ID | Title | XML signatures | First-8 distinct onsets | Risk flags |
|---|---|---:|---|---|---:|---|
| classical | retained_from_v1 | 6154808 | "Fur Elise" | 4/4 | 12 | - |
| classical | retained_from_v1 | 5610221 | Rondo Alla Turca | 4/4 | 10 | - |
| classical | new_v2_candidate | 4240641 | Prelude in C Major BWV 846 | 4/4 | 24 | high_large_leap_rate |
| classical | new_v2_candidate | 110272 | Piano Sonata No. 16 in C Major (K 545) | 4/4 | 7 | - |
| classical | new_v2_candidate | 4892542 | Nocturne | 4/4 | 9 | wide_pitch_range |
| pop_contemporary | retained_from_v1 | 149761 | Someone like you | 4/4 | 16 | - |
| pop_contemporary | retained_from_v1 | 2757411 | River Flows in You | 4/4 | 10 | - |
| pop_contemporary | retained_from_v1 | 5120186 | Let It Be | 4/4 | 9 | - |
| pop_contemporary | retained_from_v1 | 5578678 | Clocks | 4/4 | 16 | - |
| pop_contemporary | new_v2_candidate | 1151366 | Yesterday | 4/4 | 16 | - |
| film | retained_from_v1 | 3943646 | Time | 4/4 | 1 | first8_sparse, high_polyphony |
| film | new_v2_candidate | 121149 | My Heart Will Go On | 4/4 | 12 | - |
| film | new_v2_candidate | 1662546 | The Pink Panther | 4/4 | 4 | - |
| film | new_v2_candidate | 5331846 | Interstellar Medley for Piano | 4/4 | 8 | wide_pitch_range |
| film | new_v2_candidate | 5588159 | Speechless | 4/4 | 4 | high_chord_onset |
| anime | retained_from_v1 | 6386770 | One Summer's Day | 4/4 | 2 | first8_sparse, high_chord_onset, high_polyphony |
| anime | retained_from_v1 | 4827295 | Tonari no Totoro | 4/4 | 10 | - |
| anime | retained_from_v1 | 1213481 | aLIEz | 4/4 | 14 | wide_pitch_range |
| anime | new_v2_candidate | 172486 | Kimi wo Nosete (Carrying You) | 4/4 | 1 | first8_sparse |
| anime | new_v2_candidate | 1705281 | Gurenge | 4/4 | 6 | - |
| game | retained_from_v1 | 188997 | Dearly Beloved | 4/4 | 16 | - |
| game | retained_from_v1 | 5990547 | Megalovania | 4/4 | 8 | - |
| game | retained_from_v1 | 4792790 | Minecraft - Calm | 4/4 | 2 | first8_sparse, high_chord_onset, high_polyphony, high_large_leap_rate |
| game | new_v2_candidate | 1315621 | Undertale | 4/4 | 13 | - |
| game | new_v2_candidate | 1531901 | Fallen Down | 4/4 | 15 | - |

## Eligibility and risk policy

All selected records have unique MusicXML signatures [4/4], a loadable NPZ, and nonempty Melody in the first eight beats. Risk flags are listening prompts only and do not disqualify an already validated retained piece.

- first8_sparse: first-8-beat distinct Melody onset count < 4.
- high_chord_onset: chord-onset ratio > 0.4.
- high_polyphony: polyphonic-active ratio > 0.5.
- high_large_leap_rate: large-leap (>7 semitones) ratio > 0.2.
- wide_pitch_range: robust p95-p05 pitch range > 30 semitones.

## Preserved long-list inventory

candidate_inventory.jsonl retains all previously audited long-list and ineligible records. Records not in the provisional 25 remain available as reserves.

### classical

| Status | ID | Title | Eligible | XML signatures |
|---|---:|---|---|---|
| retained_from_v1 | 6154808 | "Fur Elise" | yes | 4/4 |
| retained_from_v1 | 5610221 | Rondo Alla Turca | yes | 4/4 |
| new_v2_candidate | 4240641 | Prelude in C Major BWV 846 | yes | 4/4 |
| new_v2_candidate | 110272 | Piano Sonata No. 16 in C Major (K 545) | yes | 4/4 |
| reserve_or_exclusion | 1156331 | Prélude in E Minor | no | 2/2 |
| reserve_or_exclusion | 1015261 | Invention No. 8 in F Major | no | 3/4 |
| reserve_or_exclusion | 1674336 | Pathetique Sonata op 13 mvt 2 | no | 3/4 |
| reserve_or_exclusion | 1198646 | Arabesque No. 1 | no | 4/4/2/4 |
| reserve_or_exclusion | 5002382 | Sonate | yes | 4/4 |
| new_v2_candidate | 4892542 | Nocturne | yes | 4/4 |
| reserve_or_exclusion | 6285642 | Summer | yes | 4/4 |
| reserve_or_exclusion | 5854325 | Piano Concerto No. 5 | yes | 4/4 |

### pop_contemporary

| Status | ID | Title | Eligible | XML signatures |
|---|---:|---|---|---|
| retained_from_v1 | 149761 | Someone like you | yes | 4/4 |
| retained_from_v1 | 2757411 | River Flows in You | yes | 4/4 |
| retained_from_v1 | 5120186 | Let It Be | yes | 4/4 |
| retained_from_v1 | 5578678 | Clocks | yes | 4/4 |
| new_v2_candidate | 1151366 | Yesterday | yes | 4/4 |
| reserve_or_exclusion | 25689 | Hey Jude | no | 4/4 |
| reserve_or_exclusion | 1445611 | Imagine | no | 4/4 |
| reserve_or_exclusion | 1356471 | Hello | yes | 4/4 |
| reserve_or_exclusion | 3123876 | Viva la Vida | yes | 4/4 |
| reserve_or_exclusion | 2089646 | The Scientist | yes | 4/4 |
| reserve_or_exclusion | 2978961 | Fix You | yes | 4/4 |
| reserve_or_exclusion | 293156 | All of me | yes | 4/4 |
| reserve_or_exclusion | 142159 | Bohemian Rhapsody | no | 4/4/5/4/2/4 |

### film

| Status | ID | Title | Eligible | XML signatures |
|---|---:|---|---|---|
| retained_from_v1 | 3943646 | Time | yes | 4/4 |
| new_v2_candidate | 121149 | My Heart Will Go On | yes | 4/4 |
| reserve_or_exclusion | 5314555 | Schindlers List Theme | no | 3/4 |
| reserve_or_exclusion | 1177676 | Jurassic Park | no | 2/2 |
| reserve_or_exclusion | 1277821 | Forrest Gump - Main Title | yes | 4/4 |
| reserve_or_exclusion | 2667586 | Interstellar | no | 3/4/6/8/12/8/10/8/4/4 |
| reserve_or_exclusion | 3493366 | Cinema Paradiso | no | 2/2 |
| new_v2_candidate | 1662546 | The Pink Panther | yes | 4/4 |
| new_v2_candidate | 5331846 | Interstellar Medley for Piano | yes | 4/4 |
| reserve_or_exclusion | 5277609 | Lily's Theme | yes | 4/4 |
| new_v2_candidate | 5588159 | Speechless | yes | 4/4 |

### anime

| Status | ID | Title | Eligible | XML signatures |
|---|---:|---|---|---|
| retained_from_v1 | 6386770 | One Summer's Day | yes | 4/4 |
| retained_from_v1 | 4827295 | Tonari no Totoro | yes | 4/4 |
| retained_from_v1 | 1213481 | aLIEz | yes | 4/4 |
| reserve_or_exclusion | 4844224 | The Path of the Wind - 風のとおり道 | yes | 4/4 |
| reserve_or_exclusion | 5353503 | A Town With an Ocean View (Umi No Mieru Machi) | no | 4/4/3/4 |
| new_v2_candidate | 172486 | Kimi wo Nosete (Carrying You) | yes | 4/4 |
| reserve_or_exclusion | 5193802 | The Legend of Ashitaka | no | 4/4 |
| reserve_or_exclusion | 1031936 | Guren no Yumiya | yes | 4/4 |
| reserve_or_exclusion | 1064766 | Tokyo Ghoul Opening: Unravel | yes | 4/4 |
| new_v2_candidate | 1705281 | Gurenge | yes | 4/4 |
| reserve_or_exclusion | 4136376 | Secret Base | yes | 4/4 |

### game

| Status | ID | Title | Eligible | XML signatures |
|---|---:|---|---|---|
| retained_from_v1 | 188997 | Dearly Beloved | yes | 4/4 |
| retained_from_v1 | 5990547 | Megalovania | yes | 4/4 |
| retained_from_v1 | 4792790 | Minecraft - Calm | yes | 4/4 |
| reserve_or_exclusion | 228826 | LOST WOODS (SARIA'S SONG) | no | 2/4 |
| reserve_or_exclusion | 2816421 | Great Fairy Fountain | no | 4/4 |
| new_v2_candidate | 1315621 | Undertale | yes | 4/4 |
| new_v2_candidate | 1531901 | Fallen Down | yes | 4/4 |
| reserve_or_exclusion | 1724121 | "Hopes and Dreams" | no | 2/2 |
| reserve_or_exclusion | 150778 | Piano 2, "Wet Hands" | no | 4/4/2/4 |
| reserve_or_exclusion | 125533 | Mice on Venus | yes | 4/4 |
| reserve_or_exclusion | 5504322 | Final Fantasy Main Theme | no | 4/4 |
| reserve_or_exclusion | 5185045 | Aerith's Theme | yes | 4/4 |
| reserve_or_exclusion | 2477361 | Gusty Garden Galaxy | yes | 4/4 |
| reserve_or_exclusion | 1518486 | Snowy | yes | 4/4 |

## Bohemian Rhapsody exclusion

ID 142159 remains exclusion evidence. Its NPZ metadata reports 4/4, but the full MusicXML contains 2/4, 4/4, and 5/4, so it is not part of the constant-4/4 retained cohort.
