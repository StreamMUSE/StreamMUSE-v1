# Lekai Cross-Style Task 3: Candidate Design

Date: 2026-08-02

## Label scope

The labels are metadata-derived evaluation subsets, not ground-truth genre labels.
Selection uses explicit title/composer evidence and excludes generic title words and
multi-label matches.

The initial four-way audit combined film, game, and anime music. This was revised
because soundtrack source is not one coherent musical style. The proposed design uses
five subsets:

1. Classical solo-piano repertoire.
2. Pop-song piano arrangements.
3. Jazz-oriented piano arrangements.
4. Film/TV soundtrack piano arrangements.
5. Game/anime soundtrack piano arrangements.

## Dataset audit

The metadata file contains 192,788 records; 148,389 have one piano/keyboard part and an
associated NPZ. High-precision rules produced:

| Subset | Metadata matches | Valid review shortlist |
| --- | ---: | ---: |
| Classical | 4,317 | 30 |
| Pop | 1,617 | 30 |
| Film/TV soundtrack | 1,539 | 30 |
| Game/anime soundtrack | 2,690 | 30 |
| Jazz/Blues/Ragtime | 1,431 | 30 |

Shortlist entries pass the following checks: corresponding MusicXML and NPZ exist,
NPZ shape is `(4, 88, T)`, both tracks are non-empty, time signature is 4/4, length is
32–160 measures, and no primary evidence term contributes more than three entries.

## Provisional 20-piece set

These pieces have clear title/creator evidence and avoid repeated composers within a
subset. They are provisional until the source arrangements are checked.

### Classical

| ID | Piece | Creator |
| --- | --- | --- |
| 4991039 | Prelude and Fugue No. 2 in C minor | J. S. Bach |
| 3129786 | Elegie, Op. 3 No. 1 | Sergei Rachmaninoff |
| 3438341 | Piano Sonata No. 16 in C major, K. 545 | W. A. Mozart |
| 781491 | Reverie | Claude Debussy |

### Pop

| ID | Piece | Creator/artist |
| --- | --- | --- |
| 2887096 | Dancing Queen | ABBA |
| 5842109 | Someone Like You | Adele |
| 6041157 | Your Song | Elton John |
| 5756111 | Bad Guy | Billie Eilish |

Film-associated pop songs such as `Skyfall`, `No Time To Die`, `Beauty and the Beast`,
and songs from `The Lion King` are excluded to avoid overlap with soundtrack subsets.

### Jazz

| ID | Piece | Creator/association |
| --- | --- | --- |
| 4993699 | I'm Confessin' (That I Love You) | Thelonious Monk |
| 5971066 | My Foolish Heart | Bill Evans |
| 5000747 | Freddie Freeloader | Miles Davis |
| 6290728 | Spain | Chick Corea |

### Film/TV soundtrack

| ID | Piece | Creator |
| --- | --- | --- |
| 5621276 | Star Wars Theme | John Williams |
| 5331846 | Interstellar Medley | Hans Zimmer |
| 1832046 | Lily's Theme | Alexandre Desplat |
| 5556853 | Remember / Cinema Paradiso | Ennio Morricone |

### Game/anime soundtrack

| ID | Piece | Creator/source |
| --- | --- | --- |
| 5185045 | Aerith's Theme | Nobuo Uematsu / Final Fantasy VII |
| 6268873 | Ardyn | Yoko Shimomura / Final Fantasy XV |
| 5332096 | Totoro Theme Song | Joe Hisaishi |
| 6391539 | Aria Math | Minecraft |

## Proposed inference design

Use four pieces per subset and three fixed seeds per piece, yielding 60 outputs. Piece
is the experimental unit; seeds measure within-piece decoding variability.

The decoder must be frozen after Task 2 listening. Running Task 3 before that would
confound style with the sampling candidate's known piece- and seed-dependent silence.
Report BPM, measure count, Melody density, and accompaniment density for every selected
piece rather than claiming perfectly matched subsets.

## Input preflight

After approval, the 20-piece metadata selection was frozen in
`experiments/lekai_cross_style/frozen_selection.json`. All selected NPZ files were
checksummed and passed the official `inference_new.py --prepare-only` dataset,
tokenizer, and schedule path. All 20 GT MIDI files exported successfully.

The official input selection fingerprint is
`05533801a0f7bbce7f640d623bf456fc6caf630c`. This preflight did not load a checkpoint or
generate continuation.
