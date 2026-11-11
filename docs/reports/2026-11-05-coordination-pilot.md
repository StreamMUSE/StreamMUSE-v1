# Two Characters And Concurrent Publication: Pilot

## Question

Can multiple conversational actors share one model while preserving distinct
contexts, and what changes when their requests overlap?

This implementation demonstrates an execution substrate, not a new coordination
algorithm. One Qwen2.5-7B-Instruct endpoint serves two actor-specific clients.
Each actor has a persona, its own proposal history, and a shared accepted-message
record. An actor does not observe another request's partially decoded tokens.
Concurrent conditions operate in synchronized request rounds, not a continuously
listening full-duplex conversation.

## Policies

| Policy | Generation | Publication |
| --- | --- | --- |
| Fixed turns | Request only the assigned next actor | Audit the answer, then append it |
| Naive | Both actors see the same accepted-state version | Publish each completion; fail on a rule violation |
| Coordinated | Both actors see the same accepted-state version | Check version, next number, and turn rules before appending |

Counting requires 1 through 10, both actors participating, and no consecutive
contributions from the same actor. The host knows these rules. It does not rewrite
a model proposal into a correct number. Invalid proposals remain in the trace.

```text
Fixed:       A sees v0 -> "1" accepted -> B sees v1 -> "2" accepted

Concurrent:  A sees v0 -> "1" --+-> first valid proposal accepted -> v1
             B sees v0 -> "1" --+-> naive: publishes duplicate and fails
                                   checked: rejects stale/illegal proposal
```

After each concurrent round, both requests have finished before the next pair
starts. Consequently, the current checked policy deliberately wastes calls on an
actor who cannot legally speak next. A better scheduling policy could avoid that
work; this pilot does not show that speculative concurrency is necessary.

## Measured Results

Source: `output/research_day_20261105/counting/comparison.json` and each run's
proposal/accepted/publication-attempt logs. Three runs per policy, seeds 0-2,
temperature 0, fixed policy order, shared H200 GPU service. No audio in these runs.

| Policy | Completed | Wall time, seconds | Mean seconds | Calls/run |
| --- | ---: | --- | ---: | ---: |
| Fixed turns | 3/3 | 6.993, 7.211, 8.020 | 7.408 | 10 |
| Naive | 0/3 | Failed after 0.954, 1.348, 0.897 | Not a completion metric | 2 |
| Coordinated | 3/3 | 5.844, 6.670, 5.580 | 6.031 | 20 |

All naive runs published a duplicate one after one valid step. This rapid failure
is not a latency advantage. Coordinated runs accepted ten contributions and
rejected ten proposals. Stale-proposal counts were 6, 7, and 4; these overlap with
the invalid/rejected category and must not be added as disjoint failures.

For a run with start time `s` and terminal time `f`:

`T_wall = f - s`

For request `i`, `d_i` is its client-observed response duration:

`T_request_sum = sum_i d_i`

Overlapping requests mean `T_request_sum` can exceed `T_wall`. Neither value is
GPU execution time: both include host, network, queue, and response processing.
The metrics do not isolate prefill, decode, or cache effects.

The measured checked runs were shorter but used twice as many calls. Fixed order,
prefix caching, shared GPU load, and different actor selection confound a causal
speedup claim. Three temperature-zero repetitions are not independent samples of
general language-model coordination ability. Report 3/3 observed completions, not
a 100% reliability estimate.

## Character Dialogue

The eight-turn CLI dialogue completed in 7.240 seconds using eight calls. Its
history hashes and parent-message IDs establish separate prompt construction.
They do not establish strong personalities: the two roles often give similar
practical suggestions, and the final two lines were identical. This is a working
shared-model/multiple-context implementation, not a validated persona-quality
result. The browser tests additionally verified cancellation and restart.

## What To Test Next

1. Randomize policy order and repeat with identical endpoint conditions; log GPU
   utilization, queue delay if exposed, prompt/cache tokens, and request overlap.
2. Match useful work and model-call budgets. Compare assigned-speaker scheduling,
   concurrent proposals, and a policy that avoids generating ineligible actors.
3. Add delayed observations and external interruptions independently. Measure
   stale proposals, accepted-state correctness, wasted tokens, and response delay.
4. Use a task requiring meaningful private information, not just incrementing a
   host-known counter. Keep the publication mechanism separate from task scoring.

The useful research direction is the relationship between observation freshness,
speculative work, and safe publication. Existing multi-agent and real-time serving
work already addresses parts of that problem; novelty has not been established.
