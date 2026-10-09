# Lab 7 evaluation report

## 1. What it does

Aurora Policy Assistant answers questions about the supplied insurance-policy documents. It finds relevant passages, gives a short answer with numbered sources, and declines when the documents cannot support an answer. The service returns the time, cost, cache status, and trace identifier for each request so a user can inspect the result.

## 2. How well it works

The golden set is `data/eval/rag_golden.jsonl`; Lab 4 results are retained in `reports/lab4.json`. The service uses the Lab 4 citation validator: a non-refusal without an in-range citation is replaced with the exact refusal. The regression gate measures retrieval hit@5 plus citation, refusal, cost, and latency gates offline.

| Measure | Gate | Measurement source |
|---|---:|---|
| Correctness | >= 0.75 | golden set / Lab 4 |
| Faithfulness | >= 0.90 | Lab 4 citation-grounded answer evaluation |
| Citation validity | >= 0.98 | deterministic citation validator |
| Refusal recall / precision | >= 0.80 / >= 0.75 | unanswerable cases |
| Retrieval hit@5 | >= 0.85 | `gate.py` |
| Cost/query | <= $0.010 | `/metrics` |
| p95 latency | <= 6,000 ms | `/metrics` |

Run `AIP_OFFLINE=1 python labs/lab7/gate.py` to reproduce the CI measurement without an API key.

## 3. Where it fails

The remaining failures are mostly partial-support questions: a retrieved source can support one clause while omitting a qualifier such as a plan, location, or monetary limit. A second failure is semantic similarity across plan names; for example, Gold and Silver questions can be close in wording but have different terms. Semantic cache hits are therefore accepted only at cosine >= 0.98; below 0.98 plan-specific questions began to be unsafe in manual adversarial checks. The exact cache has no equivalent correctness risk.

## 4. What it costs

The service exposes actual process cost in every `/ask` response and aggregate cost in `/metrics`. Exact cache hits cost $0.00 because they skip retrieval and generation. The release gate caps uncached average cost at $0.010/query, which corresponds to $10 per 1,000 requests and an upper-bound annual operating cost of $36,500 at 10,000 requests/day. This is a ceiling, not a claim about a particular provider price.

## 5. How fast it is

The dashboard reads JSONL traces and reports p50/p95 per span. The expected cold-path order is query embedding, retrieval, optional reranking, generation, then citation validation. Generation is the stage to optimise first because it dominates wall-clock time; reduce answer length before changing retrieval quality. Cached requests bypass the pipeline and target p95 <= 800 ms; uncached requests target p95 <= 6,000 ms. Streaming sends prose first, then citations and a final validation event, so users get early output while the UI can still react to validation.

## 6. What it is not safe for

This system is not safe to use for coverage, claim-settlement, pricing, medical, or legal decisions without human review. The corpus can be incomplete or stale, retrieval can select an incomplete passage, and citations establish where text came from but do not prove that it is current or applicable to an individual policy. Tool mode applies the Lab 6 allowlist, schema validation, input fencing, output filtering, and call budgets; these controls reduce risk but do not make prompt-injection or data-poisoning impossible.

## 7. What to do next

1. Add human review for every coverage and claims decision. This has the highest safety value because it catches incorrect applicability.
2. Add plan and policy-version metadata filters to retrieval and semantic-cache keys. This prevents cross-plan and stale-document collisions.
3. Expand the golden set with the observed partial-support and cache-collision cases, then recalibrate thresholds from repeated runs.

## Alert and regression demonstration

The dashboard alerts when the refusal rate of the newest ten requests is at least double the earlier baseline; the response is to pause deployments and inspect corpus freshness and retrieval traces. To demonstrate a red gate, temporarily set `hit_rate_at_5: {min: 1.01}` in `thresholds.yml`, run the gate, capture the failure, and restore the tested threshold.
