# Provisional Selection Review

## Correction

Dataset V2 retains all 13 eligible pieces from the corrected V1 main cohort; it does not retain only one old piece per style. Twelve new pieces fill the remaining slots to five per style.

- Composition: classical 2 retained + 3 new; pop_contemporary 4+1; film 1+4; anime 3+2; game 3+2.
- Status: all 25 records are provisional_review and eligible=true.
- New-candidate overlap: no old-35 ID or normalized-title overlap.
- Excluded evidence: Bohemian Rhapsody (142159) mixes 2/4, 4/4, and 5/4.
- keep and comments are intentionally blank for manual audition.

## Selection Table

| order | style | id | title | role | first8 distinct onsets | robust pitch range | risk flags |
|---:|---|---:|---|---|---:|---:|---|
| 1 | classical | 6154808 | "Fur Elise" | retained_from_v1 | 12 | 12 | - |
| 2 | classical | 5610221 | Rondo Alla Turca | retained_from_v1 | 10 | 14 | - |
| 3 | classical | 4240641 | Prelude in C Major BWV 846 | new_v2_candidate | 24 | 21 | high_large_leap_rate |
| 4 | classical | 110272 | Piano Sonata No. 16 in C Major (K 545) | new_v2_candidate | 7 | 20 | - |
| 5 | classical | 4892542 | Nocturne | new_v2_candidate | 9 | 31 | wide_pitch_range |
| 6 | pop_contemporary | 149761 | Someone like you | retained_from_v1 | 16 | 13 | - |
| 7 | pop_contemporary | 2757411 | River Flows in You | retained_from_v1 | 10 | 17 | - |
| 8 | pop_contemporary | 5120186 | Let It Be | retained_from_v1 | 9 | 22 | - |
| 9 | pop_contemporary | 5578678 | Clocks | retained_from_v1 | 16 | 10 | - |
| 10 | pop_contemporary | 1151366 | Yesterday | new_v2_candidate | 16 | 19 | - |
| 11 | film | 3943646 | Time | retained_from_v1 | 1 | 23 | first8_sparse, high_polyphony |
| 12 | film | 121149 | My Heart Will Go On | new_v2_candidate | 12 | 16.6 | - |
| 13 | film | 1662546 | The Pink Panther | new_v2_candidate | 4 | 15.8 | - |
| 14 | film | 5331846 | Interstellar Medley for Piano | new_v2_candidate | 8 | 31 | wide_pitch_range |
| 15 | film | 5588159 | Speechless | new_v2_candidate | 4 | 17 | high_chord_onset |
| 16 | anime | 6386770 | One Summer's Day | retained_from_v1 | 2 | 21.55 | first8_sparse, high_chord_onset, high_polyphony |
| 17 | anime | 4827295 | Tonari no Totoro | retained_from_v1 | 10 | 10 | - |
| 18 | anime | 1213481 | aLIEz | retained_from_v1 | 14 | 32 | wide_pitch_range |
| 19 | anime | 172486 | Kimi wo Nosete (Carrying You) | new_v2_candidate | 1 | 26.25 | first8_sparse |
| 20 | anime | 1705281 | Gurenge | new_v2_candidate | 6 | 17 | - |
| 21 | game | 188997 | Dearly Beloved | retained_from_v1 | 16 | 14 | - |
| 22 | game | 5990547 | Megalovania | retained_from_v1 | 8 | 29 | - |
| 23 | game | 4792790 | Minecraft - Calm | retained_from_v1 | 2 | 19 | first8_sparse, high_chord_onset, high_polyphony, high_large_leap_rate |
| 24 | game | 1315621 | Undertale | new_v2_candidate | 13 | 25.7 | - |
| 25 | game | 1531901 | Fallen Down | new_v2_candidate | 15 | 8 | - |

## Risk Flag Definitions

These flags are review prompts, not exclusion thresholds.

- first8_sparse: first-8-beat Melody distinct onset count < 4.
- high_chord_onset: chord onset ratio > 0.4.
- high_polyphony: polyphonic active-step ratio > 0.5.
- high_large_leap_rate: large-leap (>7 semitones) ratio > 0.2.
- wide_pitch_range: robust pitch range (p95-p05) > 30 semitones.
