# SignLingo AI Mini — Retrieval Evaluation

- Samples: 450 (val n=90)
- Index: `models/retrieval_index.pkl`
- Ranker: `models/ranker.pt`

| Protocol | Mode | Model | R@1 | R@5 | R@10 | MRR | p50 ms | p95 ms |
|---|---|---|---|---|---|---|---|---|
| full | text_only | baseline | 1.000 | 1.000 | 1.000 | 1.000 | 29.8 | 42.0 |
| full | text_only | trained | 1.000 | 1.000 | 1.000 | 1.000 | 32.0 | 42.1 |
| full | context_aware | baseline | 1.000 | 1.000 | 1.000 | 1.000 | 66.2 | 88.1 |
| full | context_aware | trained | 0.989 | 1.000 | 1.000 | 0.994 | 68.2 | 91.0 |
| ctx-only | context_aware | baseline | 0.044 | 0.256 | 0.422 | 0.159 | 31.2 | 39.5 |
| ctx-only | context_aware | trained | 0.044 | 0.256 | 0.300 | 0.147 | 32.6 | 44.4 |

`full` includes the query caption in the index (exact-caption retrieval is near-trivial). `ctx-only` hides the current sentence, so it measures whether context + previous pose carry signal. These are retrieval metrics, not ASL correctness; human signer evaluation is pending.
