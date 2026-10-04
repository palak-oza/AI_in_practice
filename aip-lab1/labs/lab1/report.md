# Lab 1 - The Reliable Extractor

## Run and scope

Refreshed with `python labs/lab1/run_eval.py --split test --variant c --save reports/lab1_test.json` on 2026-10-05. The run covers all 120 test tickets. All 120 responses were served from cache, so the reported zero cost and latency describe replay, not a fresh production run. The first uncached attempt populated the cache before a Windows console encoding error; the UTF-8 rerun wrote the report successfully.

## Results

| Metric | Result |
|---|---:|
| Schema valid | 120/120 (1.000) |
| Field accuracy | 0.907 |
| Whole-record accuracy | 0.508 (61/120) |
| Needs review | 0 |
| Error rate | 0 |
| Cost / p95 latency in this cached replay | $0 / 0 ms |

| Field | Accuracy |
|---|---:|
| Policy number | 1.000 |
| Contains PII | 1.000 |
| Product | 1.000 |
| Language | 1.000 |
| Category | 0.917 |
| Escalate | 0.892 |
| Sentiment | 0.833 |
| Urgency | 0.617 |

## Analysis

The schema contract held on every example, and the four deterministic fields were exact. Urgency was the weakest judgment field, followed by sentiment. Perfect schema validity does not mean the extracted record is correct: only 50.8% of records had every field correct. The result is below the 0.55 whole-record target and points to urgency calibration as the clearest next improvement.

The saved JSON records 120 cached calls. Its timing and cost must not be compared with uncached Lab 2 timings as if they were measured under the same conditions. Part A's naive-extractor demonstration was not rerun in this refresh; the pre-existing `lab1_report.md` contains that earlier experiment.

## Artifact

Raw per-ticket results: `reports/lab1_test.json`.
