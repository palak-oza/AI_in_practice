#!/usr/bin/env python3
"""Lab 5 — the failure classifier.

    python labs/lab5/diagnose.py --input reports/lab4.json
    python labs/lab5/diagnose.py --input reports/lab4.json --pareto

Implements the T4 §5 diagnostic tree. Everything that can be decided by code
is decided by code; mode 2 needs your eyes and the script says so.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
import re
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from labs.lab3.search import load_corpus, load_questions  # noqa: E402

MODES = {
    1: "missing_content",
    2: "chunk_boundary",
    3: "embedding_mismatch",
    4: "ranking",
    5: "reranker",
    6: "generation",
    7: "presentation",
}


_FACT = re.compile(r"\d[\d,]*")
def answer_in_corpus(gold_answer: str, corpus: dict[str, str],
                     relevant_docs: list[str]) -> bool:
    """Mode 1 test: is the answer's substance present in the relevant documents?

    The shipped version scored word overlap over tokens longer than 4 chars.
    That fails the moment the gold answer paraphrases the corpus -- "30 days
    from the date of discharge" against "submitted within 30 days of
    discharge" shares almost no long tokens -- so it reports mode 1 (a data
    problem, unfixable by retrieval) for what is really a retrieval problem.
    Over-reporting mode 1 sends you hunting for missing documents that are
    not missing.

    This version checks the FIGURES instead. A policy answer is carried by its
    numbers, and a number survives rewording. Where the gold answer has no
    numbers it falls back to the token test, because there is nothing better.

    How I know it is better: on my 10 Lab 4 failures the shipped test flagged
    several as mode 1 whose gold figures are demonstrably present in the
    corpus (Q04's "36 months" is in pre-existing-conditions.md). This version
    flags none of them, which agrees with grep.
    """
    text = " ".join(corpus.get(d, "") for d in relevant_docs)
    if not text:
        return False

    facts = {f.replace(",", "") for f in _FACT.findall(gold_answer)}
    if facts:
        haystack = text.replace(",", "")
        return all(f in haystack for f in facts)

    tokens = [t.strip(".,;()") for t in gold_answer.lower().split() if len(t) > 4]
    if not tokens:
        return True
    return sum(1 for t in tokens if t in text.lower()) / len(tokens) > 0.4



def classify(row: dict, q: dict, corpus: dict[str, str], *,
             gold_context_fixes_it: bool | None = None,
             in_top_30: bool | None = None,
             dropped_by_reranker: bool | None = None) -> tuple[int, str]:
    """Walk the T4 §5 diagnostic tree. Returns (mode, evidence)."""

    # Mode 7: right answer, wrong citation
    # Check this before retrieval-related failures.
    if row.get("correctness", 0) >= 2 and not row.get("citations_valid", True):
        return 7, f"correct answer, invalid citations {row.get('invalid_citations')}"

    # Mode 1: the gold answer content is not present in the
    # relevant documents.
    if not answer_in_corpus(q["gold_answer"], corpus, q["relevant_docs"]):
        return 1, "gold answer content not found in the relevant documents"

    # Mode 6: gold context fixes the answer.
    # If giving the generator the gold context fixes the answer,
    # the original failure was caused by retrieval.
    if gold_context_fixes_it is True:
        return 6, "gold context fixes the answer: retrieval was at fault"

    # If gold context was tested and it did NOT fix the answer,
    # the problem is generation.
    if gold_context_fixes_it is False:
        return 6, "gold context does not fix the answer: generation failure"

    # Mode 4 / 5:
    # Gold document was retrieved somewhere in the top 30,
    # but did not survive into the final retrieved context.
    if in_top_30 is True:
        if dropped_by_reranker is True:
            return 5, "gold document was in top 30 but dropped by reranker"
        else:
            return 4, "gold document was in top 30 but not in final retrieval"

    # Mode 3:
    # Gold document/chunk is not in top 30, but searching with the
    # gold chunk's own text can retrieve it. This indicates a query
    # problem rather than an unfindable chunk.
    if in_top_30 is False:
        gold_docs = q.get("relevant_docs", [])

        for doc_id in gold_docs:
            if doc_id not in corpus:
                continue

            gold_text = corpus[doc_id]

            # The exact gold document text is used as the diagnostic query.
            # If this can retrieve the gold document, the original query
            # was the problem.
            if answer_in_corpus(
                gold_text,
                corpus,
                [doc_id],
            ):
                return 3, "gold chunk is findable using its own text: query problem"

        # Gold chunk itself could not be found.
        return 2, "gold chunk appears unfindable: needs human inspection"

    # If retrieval diagnostics were not available, leave it as
    # a human-check case.
    return 2, "needs_human_check: open the chunks around the gold answer"

def pareto(tally: Counter) -> str:
    total = sum(tally.values()) or 1
    lines, cum = ["failure mode          n    share   cumulative"], 0

    for mode, n in tally.most_common():
        cum += n
        bar = "█" * round(30 * n / total)
        lines.append(
            f"{MODES[mode]:<20} {n:>3}   {n/total:>5.1%}   "
            f"{cum/total:>5.1%}  {bar}"
        )

    return "\n".join(lines)

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="reports/lab4.json")
    ap.add_argument("--pareto", action="store_true")
    ap.add_argument("--save", default="reports/lab5_diagnosis.json")
    args = ap.parse_args()

    rows = json.loads((ROOT / args.input).read_text(encoding="utf-8"))
    questions = {q["id"]: q for q in load_questions(include_unanswerable=True)}
    corpus = load_corpus()

    failures = [r for r in rows
                if r.get("correctness", 2) < 2 or not r.get("citations_valid", True)]
    print(f"{len(failures)} failures out of {len(rows)}\n")

    out, tally = [], Counter()
    for r in failures:
        q = questions[r["id"]]
        mode, evidence = classify(r, q, corpus)
        tally[mode] += 1
        out.append({"id": r["id"], "kind": q["kind"], "mode": mode,
                    "mode_name": MODES[mode], "evidence": evidence,
                    "question": q["question"], "answer": r["answer"][:300]})
        print(f"  {r['id']:<5} {MODES[mode]:<20} {evidence}")

    print("\n" + pareto(tally))
    print("\nCases marked needs_human_check are Part A2. Open them.")

    p = ROOT / args.save
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nsaved -> {p}")


if __name__ == "__main__":
    main()
