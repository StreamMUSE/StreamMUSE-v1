# Rap Candidate Batch Structure Experiment - 2026-08-16

## Question

For a fixed budget of 36 requested lyric lines, does Qwen2.5-7B produce a
better selectable candidate pool when the request is split into smaller model
calls? Can later calls continue the same chat, for example:

1. user supplies the bar context and asks for `X` lines;
2. assistant returns those lines;
3. user asks for another `X` non-repeating lines; and
4. the exchange repeats until the pool is large enough?

The multi-turn exchange is supported by the OpenAI-compatible vLLM API. It is
not statistically independent: each later assistant response is conditioned
on the earlier assistant output. Fresh requests share the same bar context but
start new decoding trajectories and are therefore less tightly coupled.

## Compared Policies

All policies used the committed `count_verify` prompt, the same saved bar
contexts, Qwen2.5-7B on an otherwise idle H200, temperature `0.8`, actual vLLM
API seeds, the production CMU prosody analyzer and ranker, and minimum score
`0.30`.

- `one_36`: one assistant response containing up to 36 lines.
- `sequential_18x2`: two fresh 18-line requests, made sequentially.
- `parallel_18x2`: two fresh 18-line requests, issued concurrently.
- `conversation_18x2`: one 18-line response followed by "give 18 additional
  lines" in the same conversation.
- `conversation_12x3`: three 12-line turns in one conversation.

Each policy requested 36 total lines per context. Returned lines were parsed,
normalized, deduplicated, analyzed, and reranked as one pool. A selectable line
passed the exact nine-syllable gate and scored at least `0.30`.

## Results

The SSH transport became unresponsive late in a planned 60-context run. The
vLLM log showed HTTP 200 for every request that reached the server, no engine
error, and then no new requests while the client timed out. The comparison
below uses the 36 matched contexts for which every policy completed without an
API error. Those contexts contained 13 space/syncopated, 14 deep sea/straight,
and 9 code/staggered cases. This is an engineering probe, not a statistically
powered model evaluation.

| Policy | Returned | Unique | Hard-valid | Selectable | >=1 | >=3 | Mean / P95 latency |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| One 36 | 24.06 | 24.03 | 5.06 | 3.50 | 77.8% | 44.4% | **2.37 / 4.35 s** |
| Two fresh 18, sequential | 29.72 | 29.58 | 6.97 | 5.17 | 94.4% | **72.2%** | 3.40 / 6.99 s |
| Two fresh 18, parallel | 31.56 | 31.47 | 8.08 | 5.81 | **97.2%** | 66.7% | 4.11 / 8.76 s |
| Two conversational 18 | 26.86 | 26.56 | 6.42 | 4.86 | 88.9% | 47.2% | 3.81 / 9.95 s |
| Three conversational 12 | **31.78** | **31.42** | **8.31** | **6.31** | 88.9% | 61.1% | 4.80 / 12.82 s |

Counts are per context. Duplicate rates were low for every split policy, so
the main difference was not literal repetition. Smaller requests more often
completed the requested list and produced a larger analyzable pool. The
single 36-line response returned only 66.8% of requested lines in this clean
subset, while fresh `18x2` returned 82.6% sequentially and 87.7% in the
parallel sample.

The conversational `12x3` policy had the highest average yield, but its
coverage and tail latency were worse than the fresh split policies. When one
turn adopted an incorrect syllable pattern or style, later turns often
continued it. Conversation memory helps the model avoid literal duplicates,
but it also propagates systematic mistakes.

Parallel `18x2` did not reduce latency on this single H200. Two long decodes
contended for the same model worker, producing a higher mean and P95 than the
sequential calls. It did have the best at-least-one coverage in this sample.

## Recommendation

Use fresh bounded requests rather than a growing conversation for the current
prototype. The strongest balanced starting point is:

1. request 18 lines;
2. validate, score, and merge them immediately;
3. if the pool is below the target, make a fresh 18-line request with a new
   actual sampling seed; and
4. stop as soon as the pool contains enough selectable candidates.

This preserves the reliability gain from smaller requests without forcing the
second call on every round. The next probe should measure that adaptive early
stop policy and compare it with true API-level multi-choice generation
(`n=N`, one independently decoded one-line choice per candidate). The current
production client reads only `choices[0]`, so API-level multi-choice requires a
small client/parser extension before it can be evaluated in the integrated
runtime.

No production policy was changed by this experiment. Temporary evidence is
under `/tmp/streammuse-rap-batch-structure-20260816-60/`; the comparison
harness is `/tmp/rap_batch_structure_compare.py`.
