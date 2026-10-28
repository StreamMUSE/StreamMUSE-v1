# Curated MCFlow Catalog

## Selection and Provenance

The six built-in choices are whole extracted measures from the existing local
MCFlow song manifests, not hand-authored timing or stress variants. Only timing,
stress, boundary strength, rhyme labels, anonymous IDs, and provenance are copied
into `src/streammuse/infrastructure/rap/data/curated_mcflow_templates.json`.
No source lyrics, phonetics, audio, or render outputs are included.

All three manifests identify the source repository as
[Computational-Cognitive-Musicology-Lab/MCFlow](https://github.com/Computational-Cognitive-Musicology-Lab/MCFlow)
at commit `2f743913424f256924d9e70b8201cbb0e39b8d89`.
Source identity is recorded from those manifests, not independently re-extracted
or verified against a fresh upstream checkout.

| Choice / source measure ordinal | Template ID | Slots | Onsets (ticks) | Selection rationale |
| --- | --- | ---: | --- | --- |
| Basketball m01 - Syncopated 9 | `mcflow_4b52a40dac67813b8d53` | 9 | 0, 2, 3, 5, 7, 8, 10, 11, 15 | Uneven 1/2-tick spacing and a late isolated slot |
| Basketball m03 - Delayed 10 | `mcflow_0a782d792d272ce35c37` | 10 | 1, 3, 4, 5, 7, 8, 10, 11, 14, 15 | One-tick entrance rest, internal boundary, sustained rhyme |
| Basketball m20 - Delayed 8 | `mcflow_4fa0466611a111afb52f` | 8 | 2, 4, 5, 7, 8, 9, 11, 14 | Two-tick entrance rest and a three-tick rhyme slot |
| Boom Boom Pow m17 - Long-short 8 | `mcflow_5db8ad408d09359b078d` | 8 | 0, 3, 6, 10, 11, 12, 13, 14 | Long opening durations followed by one-tick slots; no annotated rhymes |
| Boom Boom Pow m35 - Dense 12 | `mcflow_778c4db0b4f291db7620` | 12 | 0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14 | Repeated 1/1/2-tick groups and two matching rhyme labels |
| U Can't Touch This m07 - Clustered 11 | `mcflow_522b72072b7907f3288f` | 11 | 0, 1, 2, 4, 6, 8, 9, 10, 12, 13, 14 | Alternating spacious and clustered sections |

The ordinal is the extractor's `anonymous_measure_N`, not an independently
verified recording bar number. All six use 4 beats/bar and 4 ticks/beat, with
reported quantization error `0.0` ticks. Original durations, stress values,
boundaries, and rhyme labels are preserved exactly, including rests between
slots and rhymes before the last slot. Descriptive display names are applied
only after loading; the JSON retains the extractor's original names and IDs.
Selection is structural, not a claim of listening validation or acoustic timing
accuracy. Source tempo is provenance only; the session controls playback tempo.

### Source Manifests

- Basketball (105 BPM): `output/mcflow_source_tempo_two_songs_20260821/basketball_105bpm/mcflow_song_manifest.json`
  Source file SHA-256: `782d61eac81af7524a7ecccb42132f93c3fd1afc937af87fd6add1a6ce190d08`.
- Boom Boom Pow (130 BPM): `output/mcflow_source_tempo_two_songs_20260821/boom_boom_pow_130bpm/mcflow_song_manifest.json`
  Source file SHA-256: `95d1fcb862d6c55822ef70aa71be2bfddf03f8d6549a83b07289018d9f3c7e5b`.
- U Can't Touch This (133 BPM): `output/mcflow_sfw_ucanttouchthis_133bpm_20260820/mcflow_song_manifest.json`
  Source file SHA-256: `f1f6e8691f40d5da3262644e849f7b8d6929f290f94d8f3131a4ba4eca0e4442`.

The standard anonymous extraction schema validates the packaged JSON. Its
aggregate counts describe this six-template subset from three source files,
not a new full-corpus extraction or the original rejection totals. Tests pin
canonical JSON SHA-256 fingerprints calculated from the six source manifest
objects, so every copied structural and provenance field is checked without
requiring the untracked `output/` directory at runtime or in CI.

## Catalog and Defaults

`BUILTIN_TEMPLATES.values()` returns an immutable tuple of exactly these six
selectable templates in the table's order. `get(template_id)` still accepts the
three `baseline_*_9` IDs and six `untested_*_9` stress variants for saved scenarios;
they are lookup-only, not picker choices. Custom `TemplateCatalog` instances
enumerate their own supplied templates, including legacy IDs when explicitly
configured by a caller.

The default scenario retains 90 BPM and the 12-bar space / deep sea / code loop.
Each topic occupies four bars, now split into two two-bar patterns, following
catalog order. The six original StreamMUSE fallback lines cover 8, 9, 10, 11,
and 12 syllables. Tests validate every line with both CMU and heuristic prosody
analyzers, and check fallback lookup after a live topic edit. These fallback
lines are not copied from MCFlow lyrics.

Custom scenarios also receive startup-validated default lines for missing
syllable counts among their active catalog's curated choices. Existing configured
lines are never replaced or supplemented for a count they already cover, so a
legacy nine-slot scenario keeps its own text for nine-slot edits. Other catalogs
do not gain fallback counts for curated templates they do not expose. This is a
direct validation pass, not a recursive fallback-catalog build; malformed or
empty user fallback configurations still fail startup validation.

## Integration

`_MonitorLifecycle.plan_control_options` enumerates the catalog's public
`values()` method instead of reading its private `_by_id` lookup table:

```python
catalog = getattr(controller, "_templates", BUILTIN_TEMPLATES)
result["templates"] = [flow_template_payload(template) for template in catalog.values()]
```

This also supports the existing mapping-based catalog case. `get()` remains for
lookup and saved-plan compatibility. Presentation tests verify the public six
choices and custom-catalog behavior.

For wheel/sdist distribution, the new data file is included under
`[tool.setuptools.package-data]` in `pyproject.toml`:

```toml
"streammuse.infrastructure.rap" = ["data/*.json"]
```

The catalog is self-contained in a source/editable checkout; it never loads
the experiment manifests.
