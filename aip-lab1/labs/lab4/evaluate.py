#!/usr/bin/env python3
"""Lab 4 evaluation. Scaffolding provided; the judges are yours.

    python labs/lab4/evaluate.py --full --save reports/lab4.json
    python labs/lab4/evaluate.py --gold-context
    python labs/lab4/evaluate.py --strict          # C4, both settings
    python labs/lab4/evaluate.py --calibrate       # writes the hand-label sheet
    python labs/lab4/evaluate.py --kappa
    python labs/lab4/evaluate.py --failures        # E3 backlog for Lab 5
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from aip.chunking import markdown_chunks  # noqa: E402
from aip.cost import Budget  # noqa: E402
from aip.evals import judge_agreement, llm_judge  # noqa: E402
from aip.retrieval import DenseRetriever, format_context  # noqa: E402
from labs.lab3.search import load_corpus, load_questions  # noqa: E402
from labs.lab4.rag import REFUSAL, answer_question, answer_with_gold_context  # noqa: E402

GOLDEN = ROOT / "data/eval/rag_golden.jsonl"
LABEL_SHEET = ROOT / "labs/lab4/calibration_labels.jsonl"


def build_retriever():
    """My Lab 3 winning configuration: markdown-aware chunking @ 300 chars,
    exact dense retrieval, archived documents excluded.

    Lab 3 D3 applied the `status` filter at query time via Chroma. Here the
    archived chunks are dropped at index build instead. For a filter whose
    value never changes the retrieved set is identical, and it removes a
    persistent index from the hot path -- D1 measured exact and HNSW as
    identical on quality anyway. The audit argument for keeping the archived
    file on disk is unaffected: it is still in data/corpus/, just not indexed.
    """
    corpus = load_corpus()
    chunks = [c for doc_id, text in corpus.items() if "ARCHIVED" not in doc_id
              for c in markdown_chunks(text, doc_id, size=300)]
    return DenseRetriever(chunks, show_progress=False)


# ---------------------------------------------------------------------------
# D1 -- the judges
# ---------------------------------------------------------------------------
# Both rubrics are rewritten from the aip/evals.py starting templates. The
# changes are listed in report.md; the two that matter are partial answers
# (faithfulness) and correct refusal (correctness), neither of which the
# shipped rubrics handle.

RUBRIC_FAITHFULNESS = """\
You are grading whether an ANSWER is fully supported by the provided CONTEXT.

Rules:
- Judge support ONLY. Do not judge helpfulness, style, or whether the answer
  matches your own knowledge of the world.
- An answer is unsupported if it states anything the context does not contain,
  even if that statement is true in the real world.
- Refusing to answer is SUPPORTED whenever the context genuinely lacks the
  answer, and UNSUPPORTED if the context does contain it.
- A PARTIAL answer -- one that answers the supported part and explicitly says
  the sources do not cover the rest -- is SUPPORTED, provided the part it does
  answer is in the context. Do not penalise it for the part it declined.
- Paraphrase is fine. Strengthening is not: if the context says "usually" and
  the answer says "always", or the context gives a range and the answer gives
  a single figure, that claim is UNSUPPORTED.
- A citation index being right or wrong is not your concern; code checks that.
  Judge only whether the content is in the context.

CONTEXT:
{context}

ANSWER:
{answer}

Reply as JSON: {{"score": 0 or 1, "unsupported_claims": [..], "reason": "one sentence"}}
"""

RUBRIC_CORRECTNESS = """\
Compare a CANDIDATE answer to a REFERENCE answer for the same question.

Score 2 = answers the QUESTION AS ASKED with the same substance as the
          reference. Wording may differ.
Score 1 = correct as far as it goes, but omits part of what the QUESTION
          asked for, or states something the reference contradicts.
Score 0 = wrong, or refuses when the reference answers.

SCOPE -- read this before scoring, it decides most borderline cases:
- The reference often carries context beyond what the question asked. Omitting
  that context is NOT a deduction. Ask "did the candidate answer the question
  the user asked?", not "did it reproduce the reference?".
  Example: asked "how long is the grace period for an annual policy", a
  candidate answering "30 days" scores 2 even if the reference also mentions
  instalment policies, because the question asked about annual policies.
- Extra correct information is not penalised. Deduct only if it is wrong or
  contradicts the reference.
- Deduct when the omitted part was actually requested. Asked "which plans
  allow it and what changes", a candidate listing the plans but not the
  changes scores 1.

REFUSAL CASES -- apply these first, they override the scale above:
- If the REFERENCE begins with "REFUSE", the correct behaviour is to decline.
  A candidate that declines scores 2. A candidate that answers confidently
  scores 0, however plausible it sounds.
- If the REFERENCE begins with "PARTIAL REFUSE", the correct behaviour is to
  answer the supported part AND state that the rest is not covered. A
  candidate doing both scores 2; doing only one of them scores 1; inventing
  the missing part scores 0.
- Judge substance, not phrasing. Any clear decline counts as a decline.

QUESTION: {question}
REFERENCE: {reference}
CANDIDATE: {candidate}

Reply as JSON: {{"score": 0|1|2, "reason": "one sentence"}}
"""


def _score(verdict: dict) -> int | None:
    """Pull a score out of a judge verdict, or None if the judge failed.

    THE trap of this lab. `llm_judge` returns `parse_error` when the verdict
    will not parse -- most often because a reasoning judge spent its output
    budget on invisible thinking and the JSON was truncated. That is MISSING
    DATA. Scoring it 0 puts a silent, systematic downward bias on the headline
    metric: it cost this course's own reference solution a faithfulness
    reading of 0.667 when the truth was 0.933.
    """
    if verdict.get("parse_error"):
        return None
    return int(verdict.get("score", 0))


def judge_faithfulness(answer_text: str, context: str) -> int | None:
    return _score(llm_judge(RUBRIC_FAITHFULNESS.format(
        context=context[:8000], answer=answer_text), tier="LARGE"))


def judge_correctness(question: str, candidate: str, reference: str) -> int | None:
    return _score(llm_judge(RUBRIC_CORRECTNESS.format(
        question=question, reference=reference, candidate=candidate), tier="LARGE"))


def _mean(values) -> float:
    """Mean over judged cases only -- None means the judge failed, not zero."""
    kept = [v for v in values if v is not None]
    return statistics.fmean(kept) if kept else 0.0


# ---------------------------------------------------------------------------
def evaluate_all(questions, retriever, *, strict: bool = False,
                 judge: bool = True) -> list[dict]:
    """Answer every question, judge it, and time it end to end."""
    rows = []
    for q in questions:
        t0 = time.perf_counter()
        a = answer_question(q["question"], retriever, strict=strict)
        latency_ms = (time.perf_counter() - t0) * 1000
        ctx = format_context(a.hits)
        rows.append({
            "id": q["id"], "kind": q["kind"],
            "unanswerable": not q["relevant_docs"] or q["kind"] == "unanswerable",
            "question": q["question"], "answer": a.text, "refused": a.refused,
            "partial_decline": a.partial_decline,
            "citations_valid": a.citations_valid,
            "invalid_citations": a.invalid_citations,
            "n_citations": a.n_citations, "repaired": a.repaired,
            "truncated": a.truncated, "latency_ms": latency_ms,
            "faithfulness": judge_faithfulness(a.text, ctx) if judge else None,
            "correctness": (judge_correctness(q["question"], a.text, q["gold_answer"])
                            if judge else None),
            "retrieved": [h.doc_id for h in a.hits],
            "relevant": q["relevant_docs"], "gold_answer": q["gold_answer"],
        })
    return rows


def refusal_stats(rows: list[dict]) -> dict:
    """Both directions, with raw counts, on two definitions of "declined".

    Counts matter more than ratios at n=5: one case moves precision by ~0.12
    and recall by 0.20, so `3/5` is honest where `0.60` is not.

    Two definitions, because the golden set contains partial refusals (Q37,
    Q40) where the right behaviour is to answer part and decline the rest.
    Exact-string matching alone scores those as non-refusals and understates
    recall; counting them as full refusals overstates precision, since a
    partial answer is not a decline of the whole question. Report both.
    """
    una = [r for r in rows if r["unanswerable"]]
    out = {"n_unanswerable": len(una), "n_answerable": len(rows) - len(una)}
    for name, pred in (("", lambda r: r["refused"]),
                       ("_incl_partial",
                        lambda r: r["refused"] or r.get("partial_decline"))):
        dec = [r for r in rows if pred(r)]
        tp = sum(1 for r in una if pred(r))
        out[f"recall{name}"] = tp / len(una) if una else 0.0
        out[f"precision{name}"] = tp / len(dec) if dec else 1.0
        out[f"tp{name}"] = tp
        out[f"n_declined{name}"] = len(dec)
    return out


def print_report(rows: list[dict], budget: Budget | None = None) -> None:
    ans = [r for r in rows if not r["unanswerable"]]
    rs = refusal_stats(rows)
    lat = sorted(r["latency_ms"] for r in rows)
    n_parse_fail = sum(1 for r in rows if r["faithfulness"] is None)

    print(f"\nn = {len(rows)}  ({len(ans)} answerable, {rs['n_unanswerable']} unanswerable)")
    print(f"citation validity   {_mean(r['citations_valid'] for r in rows):.3f}   (target 1.000)")
    print(f"faithfulness        {_mean(r['faithfulness'] for r in rows):.3f}"
          f"   (judged {len(rows) - n_parse_fail}/{len(rows)}, {n_parse_fail} parse failures excluded)")
    corr = _mean(r["correctness"] for r in ans)
    print(f"correctness (0-2)   {corr:.3f}  normalised {corr / 2:.3f}")
    print(f"refusal recall      {rs['recall']:.3f}   ({rs['tp']}/{rs['n_unanswerable']}"
          f" full refusals)")
    print(f"refusal precision   {rs['precision']:.3f}   ({rs['tp']}/{rs['n_declined']} declines)")
    print(f"  incl. partials    recall {rs['recall_incl_partial']:.3f}"
          f" ({rs['tp_incl_partial']}/{rs['n_unanswerable']})"
          f"   precision {rs['precision_incl_partial']:.3f}"
          f" ({rs['tp_incl_partial']}/{rs['n_declined_incl_partial']})")
    print(f"repair rate         {_mean(r['repaired'] for r in rows):.3f}")
    print(f"p95 latency         {lat[int(0.95 * (len(lat) - 1))]:.0f} ms")
    if budget is not None:
        print(f"cost per query      ${budget.spent_usd / max(len(rows), 1):.4f}"
              f"   (total ${budget.spent_usd:.4f})")

    print("\nby question kind (mean correctness / 2):")
    for kind in sorted({r["kind"] for r in ans}):
        sub = [r for r in ans if r["kind"] == kind]
        print(f"  {kind:<16} {_mean(r['correctness'] for r in sub) / 2:.3f}  n={len(sub)}")


def run_full(save: str = "", strict: bool = False) -> None:
    questions = load_questions(include_unanswerable=True)
    retriever = build_retriever()
    with Budget(limit_usd=1.00, label="lab4-full") as b:
        rows = evaluate_all(questions, retriever, strict=strict)
    print_report(rows, b)

    if save:
        p = ROOT / save
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nsaved -> {p}   (Lab 5 reads this file)")


def run_strictness() -> None:
    """C4: both refusal settings, so the trade-off is visible rather than asserted.

    Judging is skipped -- this run is about the refusal dial only, and 45 extra
    judge calls per setting would double the cost to measure nothing new.
    """
    questions = load_questions(include_unanswerable=True)
    retriever = build_retriever()
    out = {}
    with Budget(limit_usd=2.00, label="lab4-strictness"):
        for label, strict in (("default", False), ("strict", True)):
            rows = evaluate_all(questions, retriever, strict=strict, judge=False)
            s = refusal_stats(rows)
            s["wrongly_refused"] = sum(1 for r in rows
                                       if r["refused"] and not r["unanswerable"])
            out[label] = s

    print(f"\n{'setting':<10}{'recall':>9}{'precision':>11}{'caught':>9}"
          f"{'refusals':>10}{'wrongly refused':>17}")
    print("-" * 66)
    for label, s in out.items():
        caught = f"{s['tp']}/{s['n_unanswerable']}"
        print(f"{label:<10}{s['recall']:>9.3f}{s['precision']:>11.3f}{caught:>9}"
              f"{s['n_declined']:>10}{s['wrongly_refused']:>17}")
    n_una = out["default"]["n_unanswerable"]
    print(f"\none unanswerable question = {1 / n_una:.2f} recall, so do not claim "
          f"a difference below ~0.15 on either number (n = {n_una}).")


def run_gold_context() -> None:
    """E2: the decomposition. The highest-value ten minutes in the lab."""
    questions = [q for q in load_questions() if q["relevant_docs"]]
    retriever = build_retriever()
    corpus = load_corpus()

    retrieved, gold = [], []
    with Budget(limit_usd=1.00, label="lab4-decomposition"):
        for q in questions:
            a = answer_question(q["question"], retriever)
            retrieved.append(judge_correctness(q["question"], a.text, q["gold_answer"]))
            g = answer_with_gold_context(
                q["question"], [corpus[d] for d in q["relevant_docs"] if d in corpus])
            gold.append(judge_correctness(q["question"], g.text, q["gold_answer"]))

    A, B = _mean(gold) / 2, _mean(retrieved) / 2
    print(f"\nn = {len(questions)} answerable questions")
    print(f"correctness with GOLD context       A = {A:.3f}   <- generation ceiling")
    print(f"correctness with RETRIEVED context  B = {B:.3f}   <- your system")
    print(f"retrieval-attributable loss   A - B = {A - B:.3f}")
    print(f"generation-attributable loss  1 - A = {1 - A:.3f}")
    print("\nWhichever is larger is where Lab 5 goes.")


# E3 -- the seven failure modes (T4 5). Mode 5 cannot occur: no reranker.
FAILURE_MODES = {
    1: "missing content -- not in the corpus at all",
    2: "chunk boundary -- answer straddles two chunks",
    3: "embedding mismatch -- right chunk exists but ranks below 30",
    4: "ranking -- right chunk in top 30, not in top final_k",
    6: "generation -- right chunk WAS in context, answer still wrong",
    7: "presentation -- answer right, citation wrong or missing",
}


_FACT = re.compile(r"\d[\d,]*")


def evidence_in_context(gold_answer: str, context: str) -> bool | None:
    """Deterministic check: do the gold answer's numbers appear in the context?

    Document-level matching is not enough to tell mode 2 from mode 6. Q04
    retrieves the right DOCUMENT and still cannot answer, because the chunker
    sliced the waiting-period table two rows above "36 months" -- right doc,
    wrong chunk. Comparing the gold answer's figures against the context text
    separates "the evidence was there and generation fluffed it" from "the
    evidence never arrived". Free, and it beats asking a model.

    Returns None when the gold answer contains no numbers to check.
    """
    facts = {f.replace(",", "") for f in _FACT.findall(gold_answer)}
    if not facts:
        return None
    ctx = context.replace(",", "")
    return all(f in ctx for f in facts)


def classify_failure(row: dict, retriever) -> tuple[int, str]:
    """Assign one failure mode to a wrong answer (T4 5).

    Mechanical wherever possible. The order matters: check presentation first
    (it is provable), then whether the evidence reached the context at all,
    and only call it generation when the evidence demonstrably did arrive.
    """
    if not row["citations_valid"]:
        return 7, FAILURE_MODES[7]

    hits = retriever.search(row["question"], k=30)
    context = format_context(hits[:5])
    present = evidence_in_context(row["gold_answer"], context)
    doc_hit = bool(set(row["retrieved"]) & set(row["relevant"]))

    if present:                       # evidence was in context and still wrong
        return 6, FAILURE_MODES[6]

    rank = next((i + 1 for i, h in enumerate(hits)
                 if h.doc_id in row["relevant"]), None)
    if present is None:               # nothing numeric to check
        return (6, FAILURE_MODES[6]) if doc_hit else (4, FAILURE_MODES[4])
    if doc_hit:
        return 2, FAILURE_MODES[2]    # right document, wrong chunk
    if rank:
        return 4, FAILURE_MODES[4]    # in the top 30, not in the top 5
    return 3, FAILURE_MODES[3]        # never ranked


def run_failures() -> None:
    """E3: tag every wrong answer with a failure mode. Lab 5's backlog."""
    rows = json.loads((ROOT / "reports/lab4.json").read_text(encoding="utf-8"))
    wrong = sorted((r for r in rows if r["correctness"] is not None
                    and r["correctness"] < 2), key=lambda r: r["correctness"])
    retriever = build_retriever()

    tally: dict[int, int] = {}
    print(f"{len(wrong)} of {len(rows)} answers scored below 2\n")
    for r in wrong:
        mode, why = classify_failure(r, retriever)
        tally[mode] = tally.get(mode, 0) + 1
        r["failure_mode"] = mode
        print(f"{r['id']:<5} ({r['kind']:<13}) correctness={r['correctness']} "
              f"faith={r['faithfulness']} -> MODE {mode}: {why}")
        print(f"  Q:    {r['question'][:94]}")
        print(f"  gold: {' '.join(r['gold_answer'].split())[:94]}")
        print(f"  got:  {' '.join(r['answer'].split())[:94]}\n")

    print("tally (this is the Lab 5 backlog, largest first):")
    for mode, n in sorted(tally.items(), key=lambda kv: -kv[1]):
        print(f"  mode {mode}  n={n:<3} {FAILURE_MODES[mode]}")

    out = ROOT / "reports/lab4_failures.json"
    out.write_text(json.dumps(wrong, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nsaved -> {out}")


def make_calibration_sheet() -> None:
    """D2: writes 20 answers for you to hand-label BEFORE seeing the judge."""
    rows = json.loads((ROOT / "reports/lab4.json").read_text(encoding="utf-8"))
    sample = rows[:20]
    LABEL_SHEET.write_text("\n".join(json.dumps({
        "id": r["id"], "question": r["question"], "answer": r["answer"],
        "gold_answer": r["gold_answer"],
        "human_faithfulness": None, "human_correctness": None,
    }, ensure_ascii=False) for r in sample) + "\n", encoding="utf-8")
    print(f"wrote {LABEL_SHEET}")
    print("Fill in human_faithfulness (0/1) and human_correctness (0/1/2), then:")
    print("  python labs/lab4/evaluate.py --kappa")


def report_kappa() -> None:
    human = [json.loads(l) for l in LABEL_SHEET.open(encoding="utf-8")]
    machine = {r["id"]: r for r in
               json.loads((ROOT / "reports/lab4.json").read_text(encoding="utf-8"))}
    for field in ("faithfulness", "correctness"):
        # Drop any case the judge failed to parse: it has no machine label to
        # agree or disagree with, and pairing it with 0 would deflate kappa.
        pairs = [(machine[r["id"]][field], r[f"human_{field}"]) for r in human
                 if r[f"human_{field}"] is not None
                 and machine[r["id"]][field] is not None]
        if not pairs:
            print(f"{field}: no usable labels yet")
            continue
        m, h = [p[0] for p in pairs], [p[1] for p in pairs]
        print(f"{field}: {judge_agreement(m, h)}")
        disagree = [r["id"] for (a, b), r in zip(pairs, human) if a != b]
        if disagree:
            print(f"  disagreements: {', '.join(disagree)}")
    print("\nkappa < 0.4 -> fix the rubric, not the model. Read your disagreements.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--gold-context", action="store_true")
    ap.add_argument("--strict", action="store_true", help="C4: both refusal settings")
    ap.add_argument("--calibrate", action="store_true")
    ap.add_argument("--kappa", action="store_true")
    ap.add_argument("--failures", action="store_true", help="E3: the Lab 5 backlog")
    ap.add_argument("--save", default="")
    a = ap.parse_args()
    if a.full:
        run_full(a.save)
    if a.gold_context:
        run_gold_context()
    if a.strict:
        run_strictness()
    if a.calibrate:
        make_calibration_sheet()
    if a.kappa:
        report_kappa()
    if a.failures:
        run_failures()
    if not any([a.full, a.gold_context, a.strict, a.calibrate, a.kappa, a.failures]):
        ap.print_help()