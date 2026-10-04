# AI in Practice Labs - Final Results

**Updated:** 2026-10-05  
**Scope:** Labs 1-6 available in this checkout. Results summarize the refreshed JSON artifacts in `aip-lab1/reports/`. Lab 7 is not present in the repository.

## Executive summary

The extraction and citation pipelines maintained schema/citation validity, but quality is uneven. Lab 1's schema validity was 100%, while exact-record accuracy was 50.8%. In Lab 2, a main-model few-shot configuration reached 66.7% record accuracy on the development set at substantially higher cost and latency than small-model options. Lab 3 produced a serious internal discrepancy: dense, BM25, and hybrid metrics were all zero, while Chroma was nonzero. Lab 4 achieved 0.838 normalized correctness and 0.956 faithfulness with valid citations on all 45 answers. Lab 5 classified nine failures but did not apply or measure a fix. Lab 6 blocked all 17 attacks in the layered run, with zero false positives and zero privileged calls, while useful-answer rate was 19/21.

## Results by lab

| Lab | Refreshed result | Main limitation |
|---|---|---|
| 1 - Extraction | 120 test cases; schema validity 1.000; field accuracy 0.907; record accuracy 0.508 | Urgency accuracy was 0.617; all calls were cached in the saved replay, so cost/latency are not live measurements. |
| 2 - Prompt grid | On 60 dev cases, best record accuracy was 0.667 for few_shot_main and few_shot_reasoned_main; schema validity 1.000 | Best-accuracy options cost about $0.255-$0.258 per 60 and had 5.36-5.78 s p95. Cascade was 60/60 cached with zero escalations. |
| 3 - Retrieval | Chroma: hit@5 0.810, recall@5 0.748, MRR 0.741, nDCG@10 0.714 | Dense, BM25, and hybrid metrics were zero across sweeps. Investigate before drawing retriever conclusions. |
| 4 - Grounded RAG | 45 cases; normalized correctness 0.838; faithfulness 0.956; citations valid 45/45; cached p95 about 15 ms | Full-refusal recall 4/5 and precision 4/6; cached metrics do not represent fresh-run cost or latency. |
| 5 - Diagnosis | 9 cases classified: 5 missing-content candidates and 4 chunk-boundary candidates | Heuristic diagnosis only; no fix or before/after evaluation was run. |
| 6 - Red team | Layered run blocked 17/17 attacks; 0/4 control false positives; 0 privileged calls | 19/21 useful answers; no-guards baseline was not refreshed alongside this run. |

## Cross-lab conclusions

1. **Separate validity from correctness.** Labs 1 and 4 show that valid records and citations are necessary safeguards, but they do not ensure correct answers.
2. **Tune extraction around urgency.** Urgency was Lab 1's weakest field (0.617); Lab 2 also shows that gains in exact-record accuracy require a meaningful cost and latency trade-off.
3. **Investigate Lab 3 before recommending a retrieval stack.** The all-zero dense, BM25, and hybrid results conflict with Chroma's nonzero scores on the same corpus. Check model/cache consistency, vector values, document IDs, and evaluation relevance matching.
4. **Treat Lab 5 as a backlog, not a fix result.** Verify all nine classifications, then run one targeted intervention with a comparable Lab 4 baseline.
5. **Keep measuring utility alongside security.** Lab 6 achieved perfect blocking on this small suite, but the layered system produced useful answers for 19/21 cases. Expand indirect-attack coverage and preserve legitimate answer paths.

## Measurement notes

- Lab 1, Lab 4, and the Lab 2 cascade were cache-heavy. Their zero cost/latency fields are cache behavior, not comparable live inference measurements.
- Lab 2's other six variants ran uncached. Paired tests against zero_shot found significant record-accuracy gains for few_shot_main (p=0.0266) and few_shot_reasoned_main (p=0.0347); the other listed comparisons were not significant.
- Lab 3 embeddings used the configured Gemini embedding model. Chunk segmentation and local score calculations run in Python; this does not mean a local embedding model was selected.
- Lab 6's refreshed JSON contains the layered run only. Historical no-guards prose is not treated as a same-run comparison.
- Human-authored report files now point to the refreshed JSONs. The original `labs/lab1/lab1_report.md` remains as the earlier report artifact; the current Lab 1 report is `labs/lab1/report.md`.

## Artifacts

- `aip-lab1/labs/lab1/report.md` and `aip-lab1/reports/lab1_test.json`
- `aip-lab1/labs/lab2/report.md` and `aip-lab1/reports/lab2_grid.json`
- `aip-lab1/labs/lab3/report.md` and `aip-lab1/reports/lab3_sweeps.json`
- `aip-lab1/labs/lab4/report.md` and `aip-lab1/reports/lab4.json`
- `aip-lab1/labs/lab5/report.md` and `aip-lab1/reports/lab5_diagnosis.json`
- `aip-lab1/labs/lab6/report.md` and `aip-lab1/reports/lab6_redteam.json`
