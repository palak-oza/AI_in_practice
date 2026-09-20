#!/usr/bin/env python3
"""Lab 2 — the configurations under test.

Each variant is a callable `str -> dict`. `grid.py` runs them all through the
same harness, so the only thing that differs between rows of your table is the
thing you intended to differ.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from labs.lab1.extract import (  # noqa: E402
    SYSTEM_PROMPT, TicketRecord, apply_business_rules, extract_deterministic,structured,
)

# ---------------------------------------------------------------------------
# A1 — your six chosen examples.
# ---------------------------------------------------------------------------
# TODO A1: choose 6 dev-set tickets. For EACH, write one line saying what it
#          teaches that prose cannot. Pick edges, not averages (T2 §2.2):
#            - the billing/complaint boundary
#            - a ticket with no policy number (teaches null)
#            - a Hinglish ticket
#            - a satisfied-but-urgent ticket (the sentiment/urgency trap)
#            - a ticket whose policy number is only in a quoted reply
#            - one you got wrong in Lab 1
FEW_SHOT_IDS: list[str] = [
    # "T0123",   # teaches: ...
"T0020",    #Teaches the LLM to distinguish the actual customer request from forwarded email metadata and correctly extract the policy number while detecting PII.
"T0059",    #Teaches that a customer can be neutral but still urgent because they specify a deadline (“before tomorrow morning”).
"T0086",    #Teaches the LLM to extract product = gold even when policy_number = null, rather than assuming both fields must appear together.
"T0054",    #Teaches the LLM not to infer or hallucinate a policy number or product when the customer only says “my policy.
"T0095",    #Teaches the LLM to independently identify the policy number, product, and PII when all appear together in formatted text.
"T0196"     #Teaches the LLM to recognize Hinglish (hi-en) even when most of the ticket is English, while separately classifying billing, frustration, urgency, and escalation.
]


def load_examples(ids: list[str]) -> list[dict]:
    rows = [json.loads(l) for l in
            (ROOT / "data/eval/extraction_dev.jsonl").open(encoding="utf-8")]
    by_id = {r["id"]: r for r in rows}
    missing = [i for i in ids if i not in by_id]
    if missing:
        raise KeyError(f"unknown example ids: {missing}")
    return [by_id[i] for i in ids]


import dataclasses

def few_shot_block(ids: list[str]) -> str:
    """Render the examples into the prompt."""
    parts = [
        "Here are worked examples. Produce output in exactly this JSON "
        "format — same keys, same order, nothing extra.\n"
    ]

    field_order = list(TicketRecord.model_fields.keys())

    for row in load_examples(ids):
        g = row["expected"]
        record = {key: g[key] for key in field_order if key in g}

        parts.append(
            f"Ticket:\n{row['input']}\n\n"
            f"JSON:\n{json.dumps(record, indent=2, ensure_ascii=False)}"
        )

    return "\n\n---\n\n".join(parts)
def zero_shot(ticket: str, tier: str = "SMALL") -> dict:
    """Lab 1 Part C: no examples. This is the baseline."""
    rec = structured(
        ticket,
        schema=TicketRecord,
        system=SYSTEM_PROMPT,
        tier=tier,
    )

    result = rec.model_dump()
    result.update(extract_deterministic(ticket))
    result = apply_business_rules(result, ticket)

    return result


def few_shot(ticket: str, tier: str = "SMALL") -> dict:
    """Lab 1 Part C: zero_shot + the few-shot block."""
    system = f"""{SYSTEM_PROMPT}

{few_shot_block(FEW_SHOT_IDS)}

Now classify the new ticket.
Return only the JSON object in exactly the same format as the examples.
"""

    rec = structured(
        ticket,
        schema=TicketRecord,
        system=system,
        tier=tier,
    )

    result = rec.model_dump()
    result.update(extract_deterministic(ticket))
    result = apply_business_rules(result, ticket)

    return result

# class TicketRecordReasoned(TicketRecord):
#     """TODO B: add a `reasoning: str` field FIRST (T2 §3.3).

#     Pydantic keeps declaration order, and field order in the JSON Schema
#     influences generation order. Putting reasoning first makes it condition the
#     answer; putting it last makes it a post-hoc rationalisation. You want the
#     first. Measure the difference in output tokens.
#     """

#     reasoning: str

# def few_shot_reasoned(ticket: str, tier: str = "SMALL") -> dict:
#     """TODO B: few_shot with TicketRecordReasoned."""
#     raise NotImplementedError


# def cascade(ticket: str) -> dict:
#     """TODO C: SMALL first; escalate to MAIN on a trigger you choose.

#     Triggers, roughly in ascending order of how well they work:
#       - validation failed                      (free, weak: misses confident errors)
#       - evidence field empty or very short     (free, surprisingly decent)
#       - urgency >= 4                           (free, but it is not a confidence signal)
#       - two SMALL samples at T=0.7 disagree    (2x small cost, much the best)

#     Record which path each ticket took -- set rec['_path'] = 'small' | 'large'
#     so grid.py can report the escalation rate.
#     """
#     raise NotImplementedError


# VARIANTS = {
#     "zero_shot": lambda t: zero_shot(t, "SMALL"),
#     "zero_shot_main": lambda t: zero_shot(t, "MAIN"),
#     "few_shot": lambda t: few_shot(t, "SMALL"),
#     "few_shot_main": lambda t: few_shot(t, "MAIN"),
#     "few_shot_reasoned": lambda t: few_shot_reasoned(t, "SMALL"),
#     "few_shot_reasoned_main": lambda t: few_shot_reasoned(t, "MAIN"),
#     "cascade": cascade,
# }


# ---------------------------------------------------------------------------
# B — Few-shot + reasoning
# ---------------------------------------------------------------------------

class TicketRecordReasoned(TicketRecord):
    """TicketRecord with reasoning generated before the final fields."""

    reasoning: str


def few_shot_reasoned(ticket: str, tier: str = "SMALL") -> dict:
    """Few-shot prompting with a concise reasoning field generated first."""

    parts = [
        f"""{SYSTEM_PROMPT}

Here are worked examples. Learn the classification and extraction behavior
from these examples.

For the new ticket:
- First provide a concise reasoning field.
- Reason only from evidence in the CURRENT message.
- The reasoning must be directly relevant to the classification/extraction.
- Then provide all required structured fields.
- Do not infer or hallucinate information that is not present.
- If a policy number is absent from the current message, return null.
- If no product is explicitly mentioned, return "unknown".
- Sentiment and urgency must be judged independently.
- Hindi mixed with English should be classified as "hi-en".
- Lines beginning with ">" are quoted history, not the current request.
- Return ONLY one valid JSON object.
- Do not use markdown or code fences.
- "reasoning" MUST be the first field.
"""
    ]

    for row in load_examples(FEW_SHOT_IDS):
        expected = row["expected"]

        evidence = expected.get("evidence", "")

        if evidence:
            reasoning = (
                f'The relevant evidence is "{evidence}", which supports '
                "the classification decision."
            )
        else:
            reasoning = (
                "Use the explicit information in the current ticket to "
                "determine the fields without inferring missing information."
            )

        example = {"reasoning": reasoning}

        # Keep the remaining fields in exactly the same order as TicketRecord.
        for key in TicketRecord.model_fields:
            if key in expected:
                example[key] = expected[key]

        parts.append(
            f"Ticket:\n{row['input']}\n\n"
            f"JSON:\n{json.dumps(example, indent=2, ensure_ascii=False)}"
        )

    parts.append(
        """---

Now classify the new ticket.

Return ONLY one valid JSON object.
The first field MUST be "reasoning".
Keep reasoning concise and directly relevant to the ticket.
Use only evidence from the CURRENT message.
Do not guess missing values.
"""
    )

    system = "\n\n".join(parts)

    rec = structured(
        f"Ticket:\n{ticket}",
        schema=TicketRecordReasoned,
        system=system,
        tier=tier,
    )

    result = rec.model_dump()

    # Same post-processing used elsewhere in Lab 2.
    result.update(extract_deterministic(ticket))
    result = apply_business_rules(result, ticket)

    return result


# ---------------------------------------------------------------------------
# C — Cascade
# ---------------------------------------------------------------------------

def cascade(ticket: str) -> dict:
    """Run few-shot SMALL first and escalate to MAIN when evidence is weak."""

    try:
        # First attempt: cheaper SMALL model.
        rec = few_shot(ticket, tier="SMALL")

        # Evidence is the model's explicit supporting span for category.
        evidence = str(rec.get("evidence", "")).strip()

        # Escalate when SMALL provides no useful evidence.
        if len(evidence) < 10:
            rec = few_shot(ticket, tier="MAIN")
            rec["_path"] = "large"
        else:
            rec["_path"] = "small"

        return rec

    except Exception:
        # If the SMALL call itself fails, retry with MAIN.
        rec = few_shot(ticket, tier="MAIN")
        rec["_path"] = "large"
        return rec


# ---------------------------------------------------------------------------
# Variants
# ---------------------------------------------------------------------------

VARIANTS = {
    "zero_shot": lambda t: zero_shot(t, "SMALL"),
    "zero_shot_main": lambda t: zero_shot(t, "MAIN"),
    "few_shot": lambda t: few_shot(t, "SMALL"),
    "few_shot_main": lambda t: few_shot(t, "MAIN"),
    "few_shot_reasoned": lambda t: few_shot_reasoned(t, "SMALL"),
    "few_shot_reasoned_main": lambda t: few_shot_reasoned(t, "MAIN"),
    "cascade": cascade,
}