#!/usr/bin/env python3
"""Lab 3 — retrieval sweeps.

The scaffolding (corpus loading, metric computation, table printing) is
written for you. The sweeps are yours.

    python labs/lab3/search.py --baseline
    python labs/lab3/search.py --sweep chunking
    python labs/lab3/search.py --sweep retrieval
    python labs/lab3/search.py --sweep rerank
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from aip.chunking import STRATEGIES, Chunk  # noqa: E402
from aip.evals import retrieval_metrics  # noqa: E402
from aip.retrieval import Bm25Retriever, DenseRetriever, HybridRetriever, Retriever, CrossEncoderReranker, LLMReranker,ChromaRetriever  # noqa: E402

CORPUS_DIR = ROOT / "data/corpus_scaled"
GOLDEN = ROOT / "data/eval/rag_golden.jsonl"
REPORT_PATH = ROOT / "reports" / "lab3_sweeps.json"

# ---------------------------------------------------------------------------
# scaffolding (provided)
# ---------------------------------------------------------------------------
def load_corpus() -> dict[str, str]:
    return {p.stem: p.read_text(encoding="utf-8") for p in sorted(CORPUS_DIR.glob("*.md"))}


def load_questions(include_unanswerable: bool = False) -> list[dict]:
    rows = [json.loads(l) for l in GOLDEN.open(encoding="utf-8")]
    if include_unanswerable:
        return rows
    # THREE questions (Q36, Q38, Q39) have no relevant document, so recall and
    # nDCG are undefined for them -- you cannot rank correctly against an empty
    # relevant set. Dropping them leaves n = 42.
    #
    # Do not confuse that with the FIVE questions of kind 'unanswerable'
    # (Q36-Q40): two of those do keep relevant documents, because part of what
    # they ask is supported. All five are measured properly in Lab 4, as
    # refusal precision and recall.
    #
    # Excluding the three is correct -- but say so in your report rather than
    # letting an unexplained n = 42 pass for a stated 45.
    return [r for r in rows if r["relevant_docs"]]


def build_chunks(corpus: dict[str, str], strategy: str = "sliding",
                 size: int = 800, **kw) -> list[Chunk]:
    fn = STRATEGIES[strategy]
    out: list[Chunk] = []
    for doc_id, text in corpus.items():
        try:
            out.extend(fn(text, doc_id, size=size, **kw))
        except TypeError:                       # chunker without that kwarg
            out.extend(fn(text, doc_id, size=size))
    return out


def evaluate(retriever: Retriever, questions: list[dict], k: int = 10,
             reranker=None, final_k: int = 5) -> dict:
    """Run every question, return aggregate metrics + per-kind breakdown."""
    agg: dict[str, list[float]] = defaultdict(list)
    by_kind: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    latencies: list[float] = []
    per_q: dict[str, float] = {}
    per_q_mrr: dict[str, float] = {}

    for q in questions:
        t0 = time.perf_counter()
        hits = retriever.search(q["question"], k=k)
        if reranker is not None:
            hits = reranker.rerank(q["question"], hits, k=final_k)
        latencies.append((time.perf_counter() - t0) * 1000)

        # A document counts as retrieved at rank r if any of its chunks does.
        seen, ranked = set(), []
        for h in hits:
            if h.doc_id not in seen:
                seen.add(h.doc_id)
                ranked.append(h.doc_id)

        m = retrieval_metrics(ranked, q["relevant_docs"], ks=(1, 3, 5, 10))
        per_q[q["id"]] = m["hit_rate@5"]
        per_q_mrr[q["id"]] = m["mrr"]
        for key, val in m.items():
            agg[key].append(val)
            by_kind[q["kind"]][key].append(val)

    out = {k2: statistics.fmean(v) for k2, v in agg.items()}
    out["latency_p50_ms"] = statistics.median(latencies)
    out["latency_p95_ms"] = sorted(latencies)[int(0.95 * (len(latencies) - 1))]
    out["_by_kind"] = {kind: {k2: statistics.fmean(v) for k2, v in d.items()}
                       for kind, d in by_kind.items()}
    out["_per_question"] = per_q            # hit_rate@5 -- saturated, see kind_table
    out["_per_question_mrr"] = per_q_mrr    # use this one for Part B
    out["_kind_n"] = {kind: len(d["mrr"]) for kind, d in by_kind.items()}
    return out


def table(rows: dict[str, dict], cols: tuple[str, ...] =
          ("hit_rate@1", "hit_rate@5", "recall@5", "mrr", "ndcg@10",
           "latency_p95_ms")) -> str:
    name_w = max(len(n) for n in rows) + 2
    head = f"{'config':<{name_w}}" + "".join(f"{c:>15}" for c in cols)
    lines = [head, "-" * len(head)]
    for name, m in rows.items():
        lines.append(f"{name:<{name_w}}" + "".join(f"{m.get(c, 0):>15.4f}" for c in cols))
    return "\n".join(lines)


def kind_table(metrics: dict, col: str = "hit_rate@5") -> str:
    """Break a result down by question kind.

    NOTE the default column. `hit_rate@5` is saturated on this corpus -- every
    retriever scores 0.93-0.98 -- so this table will look flat and tell you
    nothing. Pass col='mrr' or col='ndcg@10' for Part B. The default is left
    saturated on purpose.
    """
    bk, counts = metrics["_by_kind"], metrics.get("_kind_n", {})
    w = max(len(k) for k in bk) + 2
    lines = [f"{'kind':<{w}}{col:>12}{'n':>6}", "-" * (w + 18)]
    for kind, m in sorted(bk.items()):
        lines.append(f"{kind:<{w}}{m.get(col, 0):>12.4f}{counts.get(kind, 0):>6}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# sweeps (yours)
# ---------------------------------------------------------------------------
def sweep_baseline() -> None:
    corpus, questions = load_corpus(), load_questions()
    chunks = build_chunks(corpus, "sliding", 800, overlap=150)
    print(f"corpus: {len(corpus)} docs -> {len(chunks)} chunks "
          f"(mean {statistics.fmean(len(c) for c in chunks):.0f} chars)")
    r = DenseRetriever(chunks)
    m = evaluate(r, questions)
    print(table({"baseline sliding-800 dense": m}))
    print()
    print(kind_table(m))
    print("\nWrite these numbers down before you change anything.")


def save_sweep_results(sweep_name, results):
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    if REPORT_PATH.exists():
        try:
            with REPORT_PATH.open("r", encoding="utf-8") as f:
                all_results = json.load(f)
        except json.JSONDecodeError:
            all_results = {}
    else:
        all_results = {}

    all_results[sweep_name] = results

    with REPORT_PATH.open("w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)

    print(f"\nSaved results to: {REPORT_PATH}")



def sweep_chunking() -> None:
    """TODO A1-A3.

    A1: all four strategies at size=800.
    A2: the winner at sizes 400 / 800 / 1600. Plot or tabulate the curve.
    A3: markdown WITH and WITHOUT the '[heading > path]' prefix.
        (Strip it with a list comprehension over the chunks -- do not modify
         aip/chunking.py; other labs depend on it.)

    Report chunk count and index build time alongside quality.
    A configuration that is 1 point better and takes 4x as long to build
    is a real trade-off.
    """

    corpus, questions = load_corpus(), load_questions()

    # ---------------------------------------------------------
    # A1: Compare all four chunking strategies at size = 800
    # ---------------------------------------------------------
    print("\n" + "=" * 80)
    print("A1: CHUNKING STRATEGIES (size=800)")
    print("=" * 80)

    results = {}

    for strategy in STRATEGIES:
        t0 = time.perf_counter()

        chunks = build_chunks(
            corpus,
            strategy=strategy,
            size=800,
            overlap=150
        )

        build_time = (time.perf_counter() - t0) * 1000

        retriever = DenseRetriever(chunks)

        m = evaluate(retriever, questions)

        # Add chunking metadata to the result
        m["chunks"] = len(chunks)
        m["build_ms"] = build_time
        m["mean_chars"] = statistics.fmean(len(c) for c in chunks)

        results[f"{strategy}-800"] = m

        print(
            f"\n{strategy}: "
            f"{len(chunks)} chunks | "
            f"build={build_time:.2f} ms"
        )

        print(
            f"  hit@1={m['hit_rate@1']:.4f}, "
            f"hit@5={m['hit_rate@5']:.4f}, "
            f"recall@5={m['recall@5']:.4f}, "
            f"MRR={m['mrr']:.4f}, "
            f"nDCG@10={m['ndcg@10']:.4f}"
        )

    print("\nA1 SUMMARY")
    print(table(results))

    # ---------------------------------------------------------
    # Find the best strategy using MRR
    # ---------------------------------------------------------
    best_strategy = max(
        results,
        key=lambda s: results[s]["mrr"]
    )

    # Remove "-800" because the result keys are now strategy-800
    best_strategy = best_strategy.rsplit("-", 1)[0]

    print(
        f"\nBest chunking strategy based on MRR: "
        f"{best_strategy}"
    )

    # ---------------------------------------------------------
    # A2: Test the best strategy at 400 / 800 / 1600
    # ---------------------------------------------------------
    print("\n" + "=" * 80)
    print(f"A2: CHUNK SIZE SWEEP ({best_strategy})")
    print("=" * 80)

    size_results = {}

    for size in [400, 800, 1600]:
        t0 = time.perf_counter()

        chunks = build_chunks(
            corpus,
            strategy=best_strategy,
            size=size,
            overlap=150
        )

        build_time = (time.perf_counter() - t0) * 1000

        retriever = DenseRetriever(chunks)

        m = evaluate(retriever, questions)

        config_name = f"{best_strategy}-{size}"

        # Add chunking metadata
        m["chunks"] = len(chunks)
        m["build_ms"] = build_time
        m["mean_chars"] = statistics.fmean(len(c) for c in chunks)

        size_results[config_name] = m

        print(
            f"\n{config_name}: "
            f"{len(chunks)} chunks | "
            f"build={build_time:.2f} ms"
        )

        print(
            f"  hit@1={m['hit_rate@1']:.4f}, "
            f"hit@5={m['hit_rate@5']:.4f}, "
            f"recall@5={m['recall@5']:.4f}, "
            f"MRR={m['mrr']:.4f}, "
            f"nDCG@10={m['ndcg@10']:.4f}"
        )

    print("\nA2 SUMMARY")
    print(table(size_results))

    # ---------------------------------------------------------
    # A3: Markdown WITH vs WITHOUT heading/path prefix
    # ---------------------------------------------------------
    print("\n" + "=" * 80)
    print("A3: MARKDOWN PREFIX EFFECT")
    print("=" * 80)

    prefix_results = {}

    # ---------------------------------------------------------
    # WITH prefix
    # ---------------------------------------------------------
    t0 = time.perf_counter()

    chunks_with = build_chunks(
        corpus,
        strategy=best_strategy,
        size=800,
        overlap=150
    )

    build_time_with = (time.perf_counter() - t0) * 1000

    retriever_with = DenseRetriever(chunks_with)

    m_with = evaluate(
        retriever_with,
        questions
    )

    # Add metadata
    m_with["chunks"] = len(chunks_with)
    m_with["build_ms"] = build_time_with
    m_with["mean_chars"] = statistics.fmean(
        len(c) for c in chunks_with
    )

    prefix_results["with-prefix"] = m_with

    print(
        f"\nwith-prefix: "
        f"{len(chunks_with)} chunks | "
        f"build={build_time_with:.2f} ms"
    )

    print(
        f"  hit@1={m_with['hit_rate@1']:.4f}, "
        f"hit@5={m_with['hit_rate@5']:.4f}, "
        f"recall@5={m_with['recall@5']:.4f}, "
        f"MRR={m_with['mrr']:.4f}, "
        f"nDCG@10={m_with['ndcg@10']:.4f}"
    )

    # ---------------------------------------------------------
    # WITHOUT prefix
    # ---------------------------------------------------------
    # The chunk text contains "[heading > path]" prefix.
    # Remove it without modifying aip/chunking.py.

    t0 = time.perf_counter()

    chunks_without = build_chunks(
        corpus,
        strategy=best_strategy,
        size=800,
        overlap=150
    )

    for chunk in chunks_without:
        if "]" in chunk.text:
            chunk.text = chunk.text.split("]", 1)[1].lstrip()

    build_time_without = (time.perf_counter() - t0) * 1000

    retriever_without = DenseRetriever(chunks_without)

    m_without = evaluate(
        retriever_without,
        questions
    )

    # Add metadata
    m_without["chunks"] = len(chunks_without)
    m_without["build_ms"] = build_time_without
    m_without["mean_chars"] = statistics.fmean(
        len(c) for c in chunks_without
    )

    prefix_results["without-prefix"] = m_without

    print(
        f"\nwithout-prefix: "
        f"{len(chunks_without)} chunks | "
        f"build={build_time_without:.2f} ms"
    )

    print(
        f"  hit@1={m_without['hit_rate@1']:.4f}, "
        f"hit@5={m_without['hit_rate@5']:.4f}, "
        f"recall@5={m_without['recall@5']:.4f}, "
        f"MRR={m_without['mrr']:.4f}, "
        f"nDCG@10={m_without['ndcg@10']:.4f}"
    )

    print("\nA3 SUMMARY")
    print(table(prefix_results))

    # ---------------------------------------------------------
    # Save A1 + A2 + A3 results
    # ---------------------------------------------------------

    all_chunking_results = {}

    # A1
    all_chunking_results.update(results)

    # A2
    all_chunking_results.update(size_results)

    # A3
    all_chunking_results.update(prefix_results)

    save_sweep_results(
        "chunking",
        all_chunking_results
    )

def sweep_retrieval() -> None:
    """B1-B4 retrieval sweep using the best chunking configuration from Part A.

    B1: Dense / BM25 / Hybrid
    B2: Per-kind MRR + Q44 and Q41
    B3: RRF k in {10, 30, 60, 100}
    B4: Unequal fusion weights
    """

    corpus, questions = load_corpus(), load_questions()

    # ---------------------------------------------------------
    # Use the best chunking configuration from Part A
    # ---------------------------------------------------------
    chunks = build_chunks(
        corpus,
        strategy="markdown",
        size=400,
        overlap=150,
    )

    print("\n" + "=" * 80)
    print("PART B: RETRIEVAL SWEEP")
    print("=" * 80)

    print(
        f"\nUsing markdown chunking: "
        f"{len(chunks)} chunks"
    )

    # ---------------------------------------------------------
    # B1: Dense vs BM25 vs Hybrid
    # ---------------------------------------------------------
    print("\n" + "=" * 80)
    print("B1: DENSE vs BM25 vs HYBRID")
    print("=" * 80)

    dense = DenseRetriever(chunks)
    bm25 = Bm25Retriever(chunks)

    hybrid = HybridRetriever(
        [dense, bm25]
    )

    retrievers = {
        "dense": dense,
        "bm25": bm25,
        "hybrid": hybrid,
    }

    results = {}

    for name, retriever in retrievers.items():
        m = evaluate(retriever, questions)

        # Add configuration information
        m["chunks"] = len(chunks)
        m["chunk_strategy"] = "markdown"
        m["chunk_size"] = 400
        m["overlap"] = 150

        results[name] = m

        print(f"\n{name.upper()}")
        print(
            f"  hit@1={m['hit_rate@1']:.4f}, "
            f"hit@5={m['hit_rate@5']:.4f}, "
            f"recall@5={m['recall@5']:.4f}, "
            f"MRR={m['mrr']:.4f}, "
            f"nDCG@10={m['ndcg@10']:.4f}"
        )

    print("\nB1 SUMMARY")
    print(table(results))

    # ---------------------------------------------------------
    # B2: Per-kind MRR
    # ---------------------------------------------------------
    print("\n" + "=" * 80)
    print("B2: PER-QUESTION-KIND MRR")
    print("=" * 80)

    for name, m in results.items():
        print(f"\n{name.upper()}")

        # IMPORTANT:
        # Use MRR because hit_rate@5 is saturated on this corpus.
        print(kind_table(m, col="mrr"))

        print(
            f"\nQ44 MRR: "
            f"{m['_per_question_mrr'].get('Q44', 'N/A')}"
        )

        print(
            f"Q41 MRR: "
            f"{m['_per_question_mrr'].get('Q41', 'N/A')}"
        )

    # ---------------------------------------------------------
    # B3: RRF k sweep
    # ---------------------------------------------------------
    print("\n" + "=" * 80)
    print("B3: RRF k SWEEP")
    print("=" * 80)

    rrf_results = {}

    for rrf_k in [10, 30, 60, 100]:

        hybrid_rrf = HybridRetriever(
            [dense, bm25],
            rrf_k=rrf_k,
        )

        m = evaluate(
            hybrid_rrf,
            questions,
        )

        # Add configuration information
        m["chunks"] = len(chunks)
        m["chunk_strategy"] = "markdown"
        m["chunk_size"] = 400
        m["overlap"] = 150
        m["rrf_k"] = rrf_k

        rrf_results[f"hybrid-rrf-{rrf_k}"] = m

        print(
            f"\nRRF k={rrf_k}: "
            f"nDCG@10={m['ndcg@10']:.4f}, "
            f"MRR={m['mrr']:.4f}, "
            f"Hit@1={m['hit_rate@1']:.4f}, "
            f"Hit@5={m['hit_rate@5']:.4f}"
        )

    print("\nB3 SUMMARY")
    print(table(rrf_results))

    # ---------------------------------------------------------
    # B4: Unequal fusion weights
    # ---------------------------------------------------------
    print("\n" + "=" * 80)
    print("B4: UNEQUAL FUSION WEIGHTS")
    print("=" * 80)

    weighted_results = {}

    # The lab specifically asks for [2.0, 1.0].
    weights_to_test = [
        [2.0, 1.0],
    ]

    for weights in weights_to_test:

        hybrid_weighted = HybridRetriever(
            [dense, bm25],
            weights=weights,
        )

        m = evaluate(
            hybrid_weighted,
            questions,
        )

        name = (
            f"hybrid-weights-"
            f"{weights[0]}-{weights[1]}"
        )

        # Add configuration information
        m["chunks"] = len(chunks)
        m["chunk_strategy"] = "markdown"
        m["chunk_size"] = 400
        m["overlap"] = 150
        m["weights"] = weights

        weighted_results[name] = m

        print(
            f"\nweights={weights}: "
            f"nDCG@10={m['ndcg@10']:.4f}, "
            f"MRR={m['mrr']:.4f}, "
            f"Hit@1={m['hit_rate@1']:.4f}, "
            f"Hit@5={m['hit_rate@5']:.4f}"
        )

    print("\nB4 SUMMARY")
    print(table(weighted_results))

    # ---------------------------------------------------------
    # Save all B1-B4 results
    # ---------------------------------------------------------

    all_retrieval_results = {}

    # B1
    all_retrieval_results.update(results)

    # B3
    all_retrieval_results.update(rrf_results)

    # B4
    all_retrieval_results.update(weighted_results)

    save_sweep_results(
        "retrieval",
        all_retrieval_results
    )
    

def sweep_rerank() -> None:
    """TODO C1-C4.

    Retrieve k=30, rerank to 5: evaluate(r, questions, k=30, reranker=rr,
    final_k=5).

    C1: CrossEncoderReranker. First run downloads ~90 MB.
    C2: LLMReranker -- report cost as well as latency.
    C3: the decision table, and TWO different deployment answers
        (interactive search box vs overnight batch). They should differ.
    C4: find a query reranking made worse, using
        metrics['_per_question_mrr'] before and after.
    """

    corpus, questions = load_corpus(), load_questions()

    # ---------------------------------------------------------
    # Use the best retrieval/chunking configuration from A + B
    # ---------------------------------------------------------
    chunks = build_chunks(
        corpus,
        strategy="markdown",
        size=400,
        overlap=150,
    )

    print("\n" + "=" * 80)
    print("PART C: RERANKING SWEEP")
    print("=" * 80)

    print(
        f"\nUsing markdown chunking: "
        f"{len(chunks)} chunks"
    )

    # ---------------------------------------------------------
    # Base retriever
    # ---------------------------------------------------------
    dense = DenseRetriever(chunks)

    # Retrieve 30 candidates, then rerank to 5.
    RETRIEVAL_K = 30
    FINAL_K = 5

    # ---------------------------------------------------------
    # C1: CrossEncoder reranker
    # ---------------------------------------------------------
    print("\n" + "=" * 80)
    print("C1: CROSS-ENCODER RERANKER")
    print("=" * 80)

    cross_encoder = CrossEncoderReranker()

    t0 = time.perf_counter()

    cross_metrics = evaluate(
        dense,
        questions,
        k=RETRIEVAL_K,
        reranker=cross_encoder,
        final_k=FINAL_K,
    )

    cross_time = (time.perf_counter() - t0) * 1000

    # Add experiment metadata
    cross_metrics["reranker"] = "cross-encoder"
    cross_metrics["retrieval_k"] = RETRIEVAL_K
    cross_metrics["final_k"] = FINAL_K
    cross_metrics["chunks"] = len(chunks)
    cross_metrics["chunk_strategy"] = "markdown"
    cross_metrics["chunk_size"] = 400
    cross_metrics["overlap"] = 150
    cross_metrics["evaluation_time_ms"] = cross_time

    print(
        f"\nCrossEncoder: "
        f"nDCG@10={cross_metrics['ndcg@10']:.4f}, "
        f"MRR={cross_metrics['mrr']:.4f}, "
        f"Hit@1={cross_metrics['hit_rate@1']:.4f}, "
        f"Hit@5={cross_metrics['hit_rate@5']:.4f}"
    )

    print(
        f"Total evaluation time: "
        f"{cross_time:.2f} ms"
    )

    print("\nPer-kind MRR:")
    print(kind_table(cross_metrics, col="mrr"))

    # ---------------------------------------------------------
    # C2: LLM reranker
    # ---------------------------------------------------------
    print("\n" + "=" * 80)
    print("C2: LLM RERANKER")
    print("=" * 80)

    llm_reranker = LLMReranker()

    t0 = time.perf_counter()

    llm_metrics = evaluate(
        dense,
        questions,
        k=RETRIEVAL_K,
        reranker=llm_reranker,
        final_k=FINAL_K,
    )

    llm_time = (time.perf_counter() - t0) * 1000

    # Add experiment metadata
    llm_metrics["reranker"] = "llm"
    llm_metrics["retrieval_k"] = RETRIEVAL_K
    llm_metrics["final_k"] = FINAL_K
    llm_metrics["chunks"] = len(chunks)
    llm_metrics["chunk_strategy"] = "markdown"
    llm_metrics["chunk_size"] = 400
    llm_metrics["overlap"] = 150
    llm_metrics["evaluation_time_ms"] = llm_time

    print(
        f"\nLLM Reranker: "
        f"nDCG@10={llm_metrics['ndcg@10']:.4f}, "
        f"MRR={llm_metrics['mrr']:.4f}, "
        f"Hit@1={llm_metrics['hit_rate@1']:.4f}, "
        f"Hit@5={llm_metrics['hit_rate@5']:.4f}"
    )

    print(
        f"Total evaluation time: "
        f"{llm_time:.2f} ms"
    )

    print("\nPer-kind MRR:")
    print(kind_table(llm_metrics, col="mrr"))

    # ---------------------------------------------------------
    # C3: Decision table
    # ---------------------------------------------------------
    print("\n" + "=" * 80)
    print("C3: RERANKER DECISION TABLE")
    print("=" * 80)

    print(
        f"\n{'configuration':<25}"
        f"{'MRR':>12}"
        f"{'nDCG@10':>12}"
        f"{'Hit@1':>12}"
        f"{'Latency(ms)':>15}"
    )

    print("-" * 76)

    print(
        f"{'CrossEncoder':<25}"
        f"{cross_metrics['mrr']:>12.4f}"
        f"{cross_metrics['ndcg@10']:>12.4f}"
        f"{cross_metrics['hit_rate@1']:>12.4f}"
        f"{cross_metrics['latency_p95_ms']:>15.2f}"
    )

    print(
        f"{'LLM':<25}"
        f"{llm_metrics['mrr']:>12.4f}"
        f"{llm_metrics['ndcg@10']:>12.4f}"
        f"{llm_metrics['hit_rate@1']:>12.4f}"
        f"{llm_metrics['latency_p95_ms']:>15.2f}"
    )

    print("\nDeployment considerations:")

    print(
        "Interactive search box:"
    )

    print(
        "  Consider reranking latency and per-query cost, "
        "because the user is waiting for the result."
    )

    print(
        "\nOvernight batch:"
    )

    print(
        "  Higher latency/cost may be acceptable when results "
        "can be computed ahead of time."
    )

    # ---------------------------------------------------------
    # C4: Find queries where reranking makes MRR worse
    # ---------------------------------------------------------
    print("\n" + "=" * 80)
    print("C4: QUERIES HURT BY RERANKING")
    print("=" * 80)

    baseline_metrics = evaluate(
        dense,
        questions,
        k=RETRIEVAL_K,
    )

    baseline_q = baseline_metrics["_per_question_mrr"]
    cross_q = cross_metrics["_per_question_mrr"]
    llm_q = llm_metrics["_per_question_mrr"]

    found = False

    for q in questions:
        qid = q["id"]

        base = baseline_q.get(qid, 0.0)
        cross = cross_q.get(qid, 0.0)
        llm = llm_q.get(qid, 0.0)

        if cross < base or llm < base:
            found = True

            print(f"\nQuestion: {qid}")
            print(f"Query: {q['question']}")
            print(f"Baseline MRR:       {base:.4f}")
            print(f"CrossEncoder MRR:   {cross:.4f}")
            print(f"LLM reranker MRR:   {llm:.4f}")

            if cross < base:
                print(
                    f"CrossEncoder change: "
                    f"{cross - base:+.4f}"
                )

            if llm < base:
                print(
                    f"LLM change: "
                    f"{llm - base:+.4f}"
                )

            break

    if not found:
        print(
            "\nNo query was found where either reranker "
            "reduced MRR relative to the baseline."
        )

    # ---------------------------------------------------------
    # Save C1 + C2 results
    # ---------------------------------------------------------

    all_rerank_results = {
        "cross-encoder": cross_metrics,
        "llm": llm_metrics,
    }

    save_sweep_results(
        "rerank",
        all_rerank_results
    )
    
    
def sweep_index() -> None:
    """TODO D1-D3.

    D1/D2: ChromaRetriever vs DenseRetriever -- recall gap and latency.
    D3: pass status metadata into the chunks and filter at query time.

        Set chunk.meta['status'] = 'archived' if 'ARCHIVED' in doc_id else 'current'
        then ChromaRetriever.search(..., where={"status": "current"}).

        Report hit_rate@1 on Q29/Q30/Q31 before and after (hit_rate@1, not
        @5 -- @5 is saturated here and will hide the whole effect).
    """

    corpus, questions = load_corpus(), load_questions()

    # ---------------------------------------------------------
    # Use the best chunking configuration from Part A
    # ---------------------------------------------------------
    chunks = build_chunks(
        corpus,
        strategy="markdown",
        size=400,
        overlap=150
    )

    print("\n" + "=" * 80)
    print("D1: CHROMA vs DENSE")
    print("=" * 80)

    # ---------------------------------------------------------
    # D1: DenseRetriever vs ChromaRetriever
    # ---------------------------------------------------------

    dense = DenseRetriever(chunks)
    chroma = ChromaRetriever(chunks)

    dense_results = evaluate(dense, questions)
    chroma_results = evaluate(chroma, questions)

    # Add experiment metadata
    dense_results["chunks"] = len(chunks)
    dense_results["chunk_strategy"] = "markdown"
    dense_results["chunk_size"] = 400
    dense_results["overlap"] = 150
    dense_results["index"] = "dense"

    chroma_results["chunks"] = len(chunks)
    chroma_results["chunk_strategy"] = "markdown"
    chroma_results["chunk_size"] = 400
    chroma_results["overlap"] = 150
    chroma_results["index"] = "chroma"

    results = {
        "dense": dense_results,
        "chroma": chroma_results,
    }

    print("\nD1 SUMMARY")
    print(table(results))

    print("\nQuality gap:")
    print(
        f"Recall@5 gap (Dense - Chroma): "
        f"{dense_results['recall@5'] - chroma_results['recall@5']:+.4f}"
    )

    # ---------------------------------------------------------
    # D2: Latency at different corpus sizes
    # ---------------------------------------------------------

    print("\n" + "=" * 80)
    print("D2: INDEX LATENCY")
    print("=" * 80)

    print(
        "\nCurrent corpus size: "
        f"{len(chunks)} chunks"
    )

    # Time query latency manually so we can compare both indexes.
    def measure_latency(retriever):
        latencies = []

        for q in questions:
            t0 = time.perf_counter()

            retriever.search(
                q["question"],
                k=10
            )

            latencies.append(
                (time.perf_counter() - t0) * 1000
            )

        latencies.sort()

        p50 = latencies[len(latencies) // 2]
        p95 = latencies[int(0.95 * (len(latencies) - 1))]

        return p50, p95

    dense_p50, dense_p95 = measure_latency(dense)
    chroma_p50, chroma_p95 = measure_latency(chroma)

    print("\nLatency at current corpus size:")
    print(
        f"Dense  : p50={dense_p50:.2f} ms, "
        f"p95={dense_p95:.2f} ms"
    )

    print(
        f"Chroma : p50={chroma_p50:.2f} ms, "
        f"p95={chroma_p95:.2f} ms"
    )

    print(
        "\nRun the corpus expansion command separately and repeat this "
        "comparison at ~4k and ~40k chunks."
    )

    # ---------------------------------------------------------
    # D3: Metadata filtering
    # ---------------------------------------------------------

    print("\n" + "=" * 80)
    print("D3: METADATA FILTERING")
    print("=" * 80)

    # Add status metadata exactly as required by the lab.
    for chunk in chunks:
        chunk.meta["status"] = (
            "archived"
            if "ARCHIVED" in chunk.doc_id
            else "current"
        )

    # Rebuild Chroma so the metadata is included in the index.
    chroma_filtered = ChromaRetriever(chunks)

    qids = ["Q29", "Q30", "Q31"]

    # ---------------------------------------------------------
    # Before filtering
    # ---------------------------------------------------------

    print("\nBefore metadata filtering:")

    before_scores = {}

    for qid in qids:
        question = next(
            q for q in questions
            if q["id"] == qid
        )

        retrieved = chroma_filtered.search(
            question["question"],
            k=5
        )

        relevant_docs = set(
            question["relevant_docs"]
        )

        hit = 0

        if retrieved:
            if retrieved[0].doc_id in relevant_docs:
                hit = 1

        before_scores[qid] = hit

        print(
            f"{qid}: hit@1 = {hit}"
        )

    # ---------------------------------------------------------
    # After filtering: current documents only
    # ---------------------------------------------------------

    print("\nAfter filtering status='current':")

    after_scores = {}

    for qid in qids:
        question = next(
            q for q in questions
            if q["id"] == qid
        )

        retrieved = chroma_filtered.search(
            question["question"],
            k=5,
            where={"status": "current"}
        )

        relevant_docs = set(
            question["relevant_docs"]
        )

        hit = 0

        if retrieved:
            if retrieved[0].doc_id in relevant_docs:
                hit = 1

        after_scores[qid] = hit

        print(
            f"{qid}: hit@1 = {hit}"
        )

    # ---------------------------------------------------------
    # D3 summary
    # ---------------------------------------------------------

    print("\nD3 SUMMARY")
    print("-" * 80)

    print(
        f"{'Question':10s}"
        f"{'Before':>10s}"
        f"{'After':>10s}"
    )

    for qid in qids:
        print(
            f"{qid:10s}"
            f"{before_scores[qid]:>10d}"
            f"{after_scores[qid]:>10d}"
        )

    # ---------------------------------------------------------
    # Save D1 + D2 + D3 results
    # ---------------------------------------------------------

    all_index_results = {
        "dense": dense_results,
        "chroma": chroma_results,

        "latency": {
            "chunks": len(chunks),
            "dense_p50_ms": dense_p50,
            "dense_p95_ms": dense_p95,
            "chroma_p50_ms": chroma_p50,
            "chroma_p95_ms": chroma_p95,
        },

        "metadata_filtering": {
            "qids": qids,
            "before": before_scores,
            "after": after_scores,
        },
    }

    save_sweep_results(
        "index",
        all_index_results
    )
    
    
SWEEPS = {
    "chunking": sweep_chunking,
    "retrieval": sweep_retrieval,
    "rerank": sweep_rerank,
    "index": sweep_index,
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", action="store_true")
    ap.add_argument("--sweep", choices=list(SWEEPS))
    args = ap.parse_args()
    if args.baseline or not args.sweep:
        sweep_baseline()
    if args.sweep:
        SWEEPS[args.sweep]()


if __name__ == "__main__":
    main()
