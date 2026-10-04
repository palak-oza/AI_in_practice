# Lab 3 - Semantic Search and Retrieval Sweeps

## Run and scope

Refreshed the baseline and all four sweeps on 2026-10-05. Lab 3's configured embedding profile is Gemini (`gemini/gemini-embedding-001`); Python performs chunking, vector orchestration, and ranking. The report JSON stores chunking, retrieval, reranking, and index results.

## Results

### Chunking

Every saved chunking configuration returned zero for hit rate, recall, MRR, and nDCG:

| Configuration | Hit@5 | Recall@5 | MRR | nDCG@10 | p95 retrieval (ms) |
|---|---:|---:|---:|---:|---:|
| fixed-800 | 0.000 | 0.000 | 0.000 | 0.000 | 6.16 |
| sliding-800 | 0.000 | 0.000 | 0.000 | 0.000 | 9.61 |
| recursive-800 | 0.000 | 0.000 | 0.000 | 0.000 | 8.76 |
| markdown-800 | 0.000 | 0.000 | 0.000 | 0.000 | 12.33 |
| fixed-400 | 0.000 | 0.000 | 0.000 | 0.000 | 8.86 |
| fixed-1600 | 0.000 | 0.000 | 0.000 | 0.000 | 8.09 |
| with-prefix / without-prefix | 0.000 | 0.000 | 0.000 | 0.000 | 7.18 / 6.80 |

### Retriever and index

| Configuration | Hit@1 | Hit@5 | Recall@5 | MRR | nDCG@10 | p95 (ms) |
|---|---:|---:|---:|---:|---:|---:|
| Dense | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 14.13 |
| BM25 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 90.95 |
| Hybrid | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 106.12 |
| Chroma | 0.667 | 0.810 | 0.748 | 0.741 | 0.714 | 19.74 |

Chroma per-kind MRR was 0.722 for single-hop, 0.900 for multi-hop, 0.600 for paraphrase, 0.667 for archived-document traps, 0.875 for aggregation, and 0.313 for unanswerable questions.

### Reranking

Both rerankers scored zero on retrieval metrics because the dense candidate baseline scored zero. The cross-encoder evaluation took about 17 seconds total (p95 455 ms); the LLM reranker took about 1,025 seconds total (p95 about 26.9 seconds). The sweep output did not record comparable dollar cost values.

### Metadata filter

For Q29-Q31, top-1 hit indicators changed from 1/0/1 before filtering to 0/0/0 after filtering to `status='current'`.

## Interpretation and next step

The all-zero dense, BM25, and hybrid results conflict with Chroma's nonzero results on the same corpus and query set. This is an unresolved measurement or implementation anomaly, not evidence that these retrievers are intrinsically useless. Do not choose a production configuration from this sweep until the zero-score paths, relevant-document IDs, and embedding/index consistency are checked. Chroma currently has the only nonzero quality result in the saved comparison, but it still misses some cases and the metadata filter worsened the three inspected questions.

The baseline run also reported zero retrieval metrics. The sweeps used Gemini embeddings; the slow first pass was the large embedding workload, not a deliberate switch to a local embedding model.

## Artifact

Full sweep tables and per-kind results: `reports/lab3_sweeps.json`.
