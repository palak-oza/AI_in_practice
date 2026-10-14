#!/usr/bin/env python3
"""Lab 7 regression gate: evaluate the configured RAG service against golden data.

Run with AIP_OFFLINE=1 to replay committed model responses without provider calls.
"""
from __future__ import annotations

import argparse
import math
import statistics
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def measure() -> dict[str, float]:
    """Measure golden-set answer quality and operations on the Lab 7 RAG path.

    Model-judge calls are evaluated from the LLM response cache in offline CI.
    Cost and latency are measured around answer generation only; judge overhead
    is excluded from the per-query service cost and latency values.
    """
    from aip.cost import global_budget
    from aip.evals import retrieval_metrics
    from aip.retrieval import format_context
    from labs.lab3.search import load_questions
    from labs.lab4.evaluate import judge_correctness, judge_faithfulness, refusal_stats
    from labs.lab4.rag import answer_question
    from labs.lab7 import service

    questions = load_questions(include_unanswerable=True)
    if not questions:
        raise RuntimeError("golden set is empty")

    rag = service.pipeline().retriever
    rows: list[dict] = []
    answer_costs: list[float] = []
    latencies: list[float] = []
    hit5: list[float] = []

    for q in questions:
        before_cost = global_budget().spent_usd
        started = time.perf_counter()
        result = answer_question(q["question"], rag, k=12, final_k=5)
        latencies.append((time.perf_counter() - started) * 1000)
        answer_costs.append(global_budget().spent_usd - before_cost)

        retrieved = list(dict.fromkeys(hit.doc_id for hit in result.hits))
        if q["relevant_docs"]:
            hit5.append(retrieval_metrics(
                retrieved, q["relevant_docs"], ks=(5,)
            )["hit_rate@5"])
        context = format_context(result.hits)

        faithfulness = judge_faithfulness(result.text, context)
        if faithfulness is None:
            raise RuntimeError(f"faithfulness judge returned no score for {q['id']}")
        correctness = None
        if q["relevant_docs"] and q["kind"] != "unanswerable":
            correctness = judge_correctness(q["question"], result.text, q["gold_answer"])
            if correctness is None:
                raise RuntimeError(f"correctness judge returned no score for {q['id']}")

        rows.append({
            "unanswerable": not q["relevant_docs"] or q["kind"] == "unanswerable",
            "refused": result.refused,
            "partial_decline": result.partial_decline,
            "citations_valid": result.citations_valid,
            "faithfulness": float(faithfulness),
            "correctness": correctness,
        })

    answerable_scores = [r["correctness"] for r in rows if r["correctness"] is not None]
    if not answerable_scores:
        raise RuntimeError("golden set contains no answerable cases")
    if not hit5:
        raise RuntimeError("golden set contains no retrieval-labeled cases")
    refusals = refusal_stats(rows)
    ordered = sorted(latencies)
    p95 = ordered[max(0, math.ceil(0.95 * len(ordered)) - 1)]
    return {
        "correctness": statistics.fmean(answerable_scores) / 2.0,
        "faithfulness": statistics.fmean(r["faithfulness"] for r in rows),
        "citation_validity": statistics.fmean(float(r["citations_valid"]) for r in rows),
        "refusal_recall": refusals["recall"],
        "refusal_precision": refusals["precision"],
        "hit_rate_at_5": statistics.fmean(hit5),
        "cost_per_query_usd": statistics.fmean(answer_costs),
        "p95_latency_ms": p95,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="labs/lab7/thresholds.yml")
    args = ap.parse_args()

    thresholds = yaml.safe_load((ROOT / args.config).read_text(encoding="utf-8"))
    metrics = measure()

    failures = []
    width = max(len(k) for k in thresholds)
    print(f"{'metric':<{width}}  {'value':>10}  {'gate':>14}  status")
    print("-" * (width + 40))
    for name, rule in thresholds.items():
        value = metrics.get(name)
        if value is None:
            failures.append(f"{name}: not measured")
            print(f"{name:<{width}}  {'?':>10}  {'':>14}  MISSING")
            continue
        ok, gate = True, ""
        if "min" in rule:
            gate, ok = f">= {rule['min']}", value >= rule["min"]
        if "max" in rule and ok:
            gate, ok = f"<= {rule['max']}", value <= rule["max"]
        if not ok:
            failures.append(f"{name}: {value} violates {gate}")
        print(f"{name:<{width}}  {value:>10.4f}  {gate:>14}  {'ok' if ok else 'FAIL'}")

    if failures:
        print("\nGATE FAILED:")
        for failure in failures:
            print("  " + failure)
        return 1
    print("\nGATE PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
