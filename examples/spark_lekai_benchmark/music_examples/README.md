# Music Examples

Small MIDI examples copied from `prompts/` for Spark/H200 benchmark runs.

| Melody file | Reference accompaniment | Suggested use |
|---|---|---|
| `pop909_291_mel.mid` | `pop909_291_acc.mid` | Default smoke test; this is the file used in local 3050 measurements. |
| `pop909_013_mel.mid` | `pop909_013_acc.mid` | Additional A minor case. |
| `pop909_007_mel.mid` | `pop909_007_acc.mid` | Additional B minor case. |

Run with one of these files:

```bash
MIDI_FILE=examples/spark_lekai_benchmark/music_examples/pop909_291_mel.mid \
bash scripts/run_spark_lekai_benchmark.sh
```

For direct model/scheduler timing only:

```bash
MIDI_FILE=examples/spark_lekai_benchmark/music_examples/pop909_291_mel.mid \
RUN_PUBLIC_CLIENT=0 \
bash scripts/run_spark_lekai_benchmark.sh
```

The benchmark uses the melody file. The accompaniment files are included only as reference material for listening or offline comparison.

