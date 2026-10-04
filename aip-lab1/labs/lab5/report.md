# Lab 5 Report

## A. Failure diagnosis (automated, preliminary)

Ran `python labs/lab5/diagnose.py --input reports/lab4.json --save reports/lab5_diagnosis.json` from `aip-lab1`. The input contains 45 answers; 10 are below the correctness threshold or have invalid citations. The corpus-wide content heuristic produced:

| Signal | Count | Questions |
|---|---:|---|
| Candidate missing content (mode 1 heuristic) | 6 | Q11, Q23, Q34, Q35, Q37, Q44 |
| Retrieval evidence unavailable; human check required | 4 | Q04, Q20, Q29, Q32 |

```text
candidate mode 1   6  60%  ##################
needs evidence     4  40%  ############
```

This is not a verified seven-mode tally. The Lab 4 JSON contains retrieved IDs (top five), but no top-30 ranking, gold-context outcome, or reranker trace. The four unresolved cases cannot yet be called chunk-boundary failures. The six mode-1 labels also require checking against their gold facts; text matching is not proof that content is absent. Q37 is unanswerable and must be evaluated as a refusal, not automatically counted as missing content.

## B. Prioritisation and prediction

| Candidate cluster | n | Candidate fix | Expected recovery | Cost / latency | Effort |
|---|---:|---|---:|---|---|
| Corpus coverage | 6 | Verify gold facts in source corpus; add source data only if confirmed missing | Not estimated before validation | No query cost; ingest effort unknown | Medium |
| Retrieval/chunking | 4 unresolved | Inspect source chunks and top-30 ranks first | Not estimated | Unknown | Medium |

No fix was selected because the diagnosis evidence is incomplete. No intervention or prediction was recorded, so there is no post-fix result to compare.

## D1. Available Lab 4 baseline

The saved Lab 4 run reports normalized correctness 0.775, faithfulness 0.956, citation validity 1.000, refusal recall 1.000 (5/5), refusal precision 0.714, cost $0.2233/query, and p95 latency 3,460 ms. These are baseline figures only.

| Metric | Lab 4 baseline | Lab 5 after | Change |
|---|---:|---:|---:|
| Correctness (normalized) | 0.775 | Not measured | — |
| Faithfulness | 0.956 | Not measured | — |
| Citation validity | 1.000 | Not measured | — |
| Refusal recall / precision | 1.000 / 0.714 | Not measured | — |
| nDCG@10 / recall@5 | Not in Lab 4 output | Not measured | — |
| Cost / query | $0.2233 | Not measured | — |
| p95 latency | 3,460 ms | Not measured | — |

## D2–D4. Regression, reclassification, next action

No fix was measured, so regression status and post-fix reclassification are unknown. Next, run the gold-context and top-30 checks in the lab procedure, manually inspect Q04/Q20/Q29/Q32, then select and measure one fix. No prediction or failed-fix number is claimed because no intervention was run.

## Fix that did not work

None measured in the available Lab 5 materials. A failed-fix number cannot be reported without running a fix.
