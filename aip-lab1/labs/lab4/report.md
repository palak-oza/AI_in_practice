# Lab 4 - Grounded Answers with Citations

## Run and scope

Refreshed the full evaluator with `python labs/lab4/evaluate.py --full --save reports/lab4.json` on 2026-10-05. The JSON contains 45 questions. All answers in this refresh were served from cache; reported cost and latency describe replay, not an uncached production run.

## Results

| Metric | Result |
|---|---:|
| Questions | 45 (40 answerable, 5 unanswerable) |
| Normalized correctness | 0.838 |
| Faithfulness | 0.956 |
| Citation validity | 45/45 (1.000) |
| Full-refusal recall | 4/5 (0.800) |
| Full-refusal precision | 4/6 (0.667) |
| Repair rate | 1/45 (0.022) |
| Cached replay p95 latency | about 15 ms |
| Cached replay cost | $0.0000 |

There were six full refusals, four on unanswerable questions. The run also included partial declines; retain the distinction between a full refusal and an answer that gives supported facts while declining an unsupported part.

## Analysis

Citation validation remained perfect. Correctness and faithfulness were below perfect, so valid citations alone do not guarantee a correct answer. The refusal set is small: one question changes recall by 0.20. The saved JSON is the authoritative source for each question's answer, citations, refusal flags, and scores.

Earlier report prose cites uncached cost and latency values from a different run. Those figures are not measurements from this refreshed cached JSON and are intentionally not mixed into this report. Judge calibration and strict-refusal comparisons were not rerun in this command.

## Artifact

Per-question evaluation: `reports/lab4.json`.
