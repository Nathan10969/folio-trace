# Historical Map-Reduce latency comparison

Source record date: **2026-07-16**. These are results reported in the supplied
snapshot, not a new benchmark of this public release. Raw replay artifacts and the
historical deployment were not independently available for recomputation.

The comparison used the same five documents and question about the best ratio,
with a fresh chat ID for each replay.

| Metric | Staged + expanded baseline | Accepted one-pass + compact | Rejected prompt experiment |
| --- | ---: | ---: | ---: |
| Subagent calls | 10 | 8 | 6 |
| Subagent input tokens | 48,102 | 34,097 | 23,985 |
| One-pass repairs | — | 2 | 0 |
| Pre-finalizer time | 73.771 s | 79.353 s | 30.021 s |
| Finalizer input tokens | 4,626 | 2,311 | 1,967 |
| Finalizer time | 38.147 s | 10.343 s | 25.559 s |
| Time to first token | 102.319 s | 83.753 s | 47.302 s |
| Total server time | 111.918 s | 89.696 s | 55.580 s |
| Displayed evidence | 0 in older pre-annotation run | 7 across 4 files | 2 from 1 file |

The accepted variant reported 19.9% lower total server time and 18.1% lower time to
first token. Compact Reduce was the primary observed improvement. One-pass Map
required two deep repairs and made pre-finalizer time slower in that accepted run.

The faster follow-up retained four document summaries in the answer but exposed
annotations from only one of the four relevant documents. It was rejected and
rolled back for a citation-binding regression. Its speed is not an accepted result.

Run IDs, retained for traceability:

- Baseline: `run_20260716_39fc4ac6d196`
- Accepted: `run_20260716_f6a0a0226cc0`
- Rejected: `run_20260716_f713f03867d3`

The historical record identifies deployed HEAD `2a41b9e`, with accepted code content
matching `1c36da3`. The public configuration defaults remain `staged` and `expanded`,
as in the supplied source configuration. Select `one_pass` and `compact` explicitly
to explore the historical variant. This single replay comparison does not establish
statistical significance, universal speedup, or complete citation coverage.
