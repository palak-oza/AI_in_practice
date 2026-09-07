#!/usr/bin/env python3
"""Lab 1, Parts B and C — the extractor you actually ship.

Complete the TODOs. `run_eval.py` imports `extract_b` and `extract_c` from
here, so keep those two function names.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, field_validator

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from aip.guards import _PII_PATTERNS  # noqa: E402
from aip.llm import structured  # noqa: E402

CATEGORIES = Literal["billing", "claims", "policy_change",
                     "technical", "complaint", "information"]


# ===========================================================================
# PART B — the schema
# ===========================================================================
class TicketRecord(BaseModel):
    """The contract. Everything the model is allowed to say, and nothing else.

    Remember from T2 §3.2: field `description`s are shipped to the model as
    part of the JSON Schema. They are the highest-leverage place to put an
    instruction, because they sit next to the thing they govern. Write them as
    instructions to the model, not as documentation for a human.
    """

    # B1a: `evidence` goes BEFORE `category` -- T2 §3.3's WithReasoning idiom.
    # It forces the model to name the deciding span before it commits to a
    # label, so it acts as reasoning rather than a post-hoc citation. Chosen
    # because the gold labels themselves call out the claims/complaint
    # boundary as where most errors land -- exactly what a forced "point at
    # your evidence first" step should help with.
    evidence: str = Field(
        max_length=200,
        description="The exact span of the message that determined `category`, "
                    "quoted verbatim. One sentence at most."
    )

    category: CATEGORIES = Field(
        description="billing = money in: premium, debits, refunds, invoices, "
                    "the 80D tax certificate, instalment options. claims = an "
                    "actual or intended claim: cashless, reimbursement, "
                    "settlement amount, deduction, rejection. policy_change = "
                    "altering the contract: add/remove a member, upgrade, "
                    "port, change contact details. technical = the app, "
                    "portal, OTP, login, locator, or document upload is "
                    "broken. complaint = Aurora's own CONDUCT is the subject "
                    "-- mis-selling, being kept on hold, an ignored grievance "
                    "-- not just anger about a claim or a bill. information = "
                    "a question with no pending transaction behind it. An "
                    "angry message about a claim is 'claims' if the customer "
                    "still wants it processed; it is 'complaint' only when "
                    "Aurora's conduct itself is what the message is about."
    )

    urgency: int = Field(
        ge=1, le=5,
        description="1 = answerable from general product knowledge or a "
                    "self-service how-to; Aurora need not look anything up. "
                    "2 = requires Aurora to look up THIS customer's account, "
                    "act on it, or fix a defect, or a transaction is already "
                    "in flight. 3 = something has already gone wrong or is "
                    "stuck and the customer is waiting. 4 = repeated failure "
                    "to resolve, money or access at risk right now, or an "
                    "explicit threat to escalate. 5 = an emergency in "
                    "progress, a formal denial demanding immediate reversal, "
                    "or the customer states they ARE escalating to the "
                    "Ombudsman (not merely threatening to). Add 1 (capped at "
                    "5) if the message states a same-day or next-morning "
                    "deadline. Judge the situation, not how loudly it is "
                    "written."
    )

    sentiment: Literal["angry", "frustrated", "neutral", "satisfied"] = Field(
        description="Tone only, independent of urgency. angry = hostile, "
                    "shouting, threatening. frustrated = unhappy and tired of "
                    "trying but still civil -- requires the message to "
                    "reference a PRIOR failure (a repeat attempt, an "
                    "unanswered request, or a delay). neutral = matter-of-"
                    "fact, including a terse first-time request. satisfied = "
                    "thanks or praise."
    )

    product: Literal["bronze", "silver", "gold", "platinum", "unknown"] = Field(
        description="The plan name exactly as stated in the message: bronze, "
                    "silver, gold, or platinum. Use 'unknown' if no plan name "
                    "is mentioned -- never infer it from the sum insured or "
                    "any other context."
    )

    language: Literal["en", "hi-en"] = Field(
        description="'hi-en' if Hindi words are mixed into the English, "
                    "including transliterated Hindi in Latin script (e.g. "
                    "kripya, jaldi, bahut, turant). Otherwise 'en'."
    )

    # Part B only: the model decides these. In Part C you will delete them
    # from this schema and compute them in code instead.
    policy_number: str | None = Field(
        default=None,
        description="Format: 'AUR-' followed by exactly 7 digits, copied "
                    "verbatim, character for character, from the CURRENT "
                    "message only -- never from a quoted reply below a '>' "
                    "line. Return null if no such string appears in the "
                    "current message. Never invent one and never reformat a "
                    "number that is close but not exact."
    )
    contains_pii: bool = Field(
        default=False,
        description="True if the message contains a phone number, or an "
                    "email address that is not one of Aurora's own published "
                    "support addresses. A personal name alone does not count."
    )

    # Set by our code, never by the model.
    needs_human_review: bool = False
    review_reason: str = ""

    @field_validator("policy_number")
    @classmethod
    def _policy_format(cls, v: str | None) -> str | None:
        if v is None:
            return None
        # B1j: treat sloppy ways of saying "there isn't one" as absence
        # rather than a validation failure -- the model was told to emit
        # null but some providers return "" or the literal word "null"
        # instead. Anything else that isn't the exact format is a real
        # defect (truncated, reformatted, or invented) and should fail
        # validation so the repair loop sends it back.
        if v.strip().lower() in {"", "null"}:
            return None
        if not re.fullmatch(r"AUR-\d{7}", v):
            raise ValueError(
                "policy_number must be exactly 'AUR-' followed by 7 digits, "
                "copied verbatim from the ticket, or null if none is present"
            )
        return v


SYSTEM_PROMPT = """\
You are the extraction step of Aurora Health Insurance's support-ticket \
triage system. Turn one raw ticket into the structured record the routing \
queue reads next -- your output is consumed by code, not a human, so its \
shape matters as much as its content.

Tickets arrive as forwarded email chains, WhatsApp messages, and web-form \
submissions: quoted history, typos, Hinglish, shouting, and signature blocks \
included. Base every field only on the CURRENT message. Lines starting with \
'>' are a quoted reply from an earlier thread -- context, not the live \
request, and never the source of a policy number.

If a field cannot be determined from the message, use the schema's null or \
"unknown" value. Never guess or infer a value that is not actually present; \
an invented value is worse than an honest gap.
"""


def extract_b(ticket: str) -> TicketRecord:
    """Part B: the model decides everything."""
    try:
        return structured(
            f"Ticket:\n{ticket}",
            schema=TicketRecord,
            system=SYSTEM_PROMPT,
            tier="SMALL",
        )
    except Exception as exc:  # noqa: BLE001 - never raise: degrade any failure (StructuredOutputError, and T1 3 modes 1-3: transport, rate limit, timeout/budget) to a review record
        return TicketRecord(
            evidence="",
            category="information",
            urgency=1,
            sentiment="neutral",
            product="unknown",
            language="en",
            needs_human_review=True,
            review_reason=f"{type(exc).__name__}: {exc}"[:300],
        )


# ===========================================================================
# PART C — move the deterministic work out of the model
# ===========================================================================
POLICY_RE = re.compile(r"\bAUR-\d{7}\b")

# The quoted-reply marker. Everything after this is history, not the current
# message. Part C3 asks you to decide what that means for policy extraction.
QUOTE_MARKER = re.compile(r"^\s*>", re.MULTILINE)

# Aurora's own published addresses never count as PII (data/README.md).
AURORA_EMAILS = {"support@aurorahealth.example", "grievance@aurorahealth.example"}


def _live_text(ticket: str) -> str:
    """Every line that is NOT a quoted reply (see QUOTE_MARKER)."""
    return "\n".join(
        line for line in ticket.splitlines() if not QUOTE_MARKER.match(line)
    )


def extract_deterministic(ticket: str) -> dict:
    """Return {'policy_number', 'contains_pii'} without a model call.

    policy_number:
        Find AUR-<7 digits>.

        C3 -- the trap, resolved: a ticket can contain an AUR-<7 digits>
        string in the live body AND a different, stale one in a quoted reply
        below a '>' line. Only the live message is the current request, so
        search ONLY the live text (QUOTE_MARKER strips quoted lines first) --
        this is data/README.md's own rule ("A ticket whose only
        policy-shaped string is inside a quoted reply is labelled null"),
        read directly off the annotation guide rather than fitted by
        eyeballing examples, so it should generalise to any ticket built the
        same way. It does not handle two matches within the live body itself
        (no case in dev/test exercises that); the honest caveat belongs in
        the report.

    contains_pii:
        True if the ticket contains a phone number, or an email address that
        is not one of Aurora's own published addresses. aip.guards has the
        patterns. Scans the WHOLE ticket (not just the live text) --
        data/README.md's known limitations section notes a quoted auto-reply
        footer containing Aurora's own support address, which must NOT count,
        but a customer's phone number anywhere in the thread should. A name
        alone never counts for this dataset's labels.
    """
    live = _live_text(ticket)
    policy_match = POLICY_RE.search(live)
    policy_number = policy_match.group(0) if policy_match else None

    has_phone = bool(_PII_PATTERNS["PHONE_IN"].search(ticket))
    emails = [e for e in _PII_PATTERNS["EMAIL"].findall(ticket)
             if e.lower() not in AURORA_EMAILS]
    contains_pii = has_phone or bool(emails)

    return {"policy_number": policy_number, "contains_pii": contains_pii}


def apply_business_rules(rec_fields: dict, ticket: str) -> dict:
    """Compute `escalate` in code.

        escalate = urgency >= 4 or 'ombudsman' appears in the ticket

    This is a business rule. It belongs in code where it can be read by a
    compliance officer, changed without touching a prompt, and unit-tested.
    """
    rec_fields = dict(rec_fields)
    rec_fields["escalate"] = (
        rec_fields.get("urgency", 0) >= 4 or "ombudsman" in ticket.lower()
    )
    return rec_fields


class TicketRecordC(BaseModel):
    """The reduced schema the model sees in Part C.

    TicketRecord with `policy_number` and `contains_pii` deleted -- those are
    now computed by extract_deterministic(). Fewer fields means a shorter
    prompt, fewer output tokens, and those two fields (plus `escalate`, never
    modelled at all) land at 100% accuracy and become auditable.
    """

    evidence: str = Field(
        max_length=200,
        description="The exact span of the message that determined `category`, "
                    "quoted verbatim. One sentence at most."
    )

    category: CATEGORIES = Field(
        description="billing = money in: premium, debits, refunds, invoices, "
                    "the 80D tax certificate, instalment options. claims = an "
                    "actual or intended claim: cashless, reimbursement, "
                    "settlement amount, deduction, rejection. policy_change = "
                    "altering the contract: add/remove a member, upgrade, "
                    "port, change contact details. technical = the app, "
                    "portal, OTP, login, locator, or document upload is "
                    "broken. complaint = Aurora's own CONDUCT is the subject "
                    "-- mis-selling, being kept on hold, an ignored grievance "
                    "-- not just anger about a claim or a bill. information = "
                    "a question with no pending transaction behind it. An "
                    "angry message about a claim is 'claims' if the customer "
                    "still wants it processed; it is 'complaint' only when "
                    "Aurora's conduct itself is what the message is about."
    )

    urgency: int = Field(
        ge=1, le=5,
        description="1 = answerable from general product knowledge or a "
                    "self-service how-to; Aurora need not look anything up. "
                    "2 = requires Aurora to look up THIS customer's account, "
                    "act on it, or fix a defect, or a transaction is already "
                    "in flight. 3 = something has already gone wrong or is "
                    "stuck and the customer is waiting. 4 = repeated failure "
                    "to resolve, money or access at risk right now, or an "
                    "explicit threat to escalate. 5 = an emergency in "
                    "progress, a formal denial demanding immediate reversal, "
                    "or the customer states they ARE escalating to the "
                    "Ombudsman (not merely threatening to). Add 1 (capped at "
                    "5) if the message states a same-day or next-morning "
                    "deadline. Judge the situation, not how loudly it is "
                    "written."
    )

    sentiment: Literal["angry", "frustrated", "neutral", "satisfied"] = Field(
        description="Tone only, independent of urgency. angry = hostile, "
                    "shouting, threatening. frustrated = unhappy and tired of "
                    "trying but still civil -- requires the message to "
                    "reference a PRIOR failure (a repeat attempt, an "
                    "unanswered request, or a delay). neutral = matter-of-"
                    "fact, including a terse first-time request. satisfied = "
                    "thanks or praise."
    )

    product: Literal["bronze", "silver", "gold", "platinum", "unknown"] = Field(
        description="The plan name exactly as stated in the message: bronze, "
                    "silver, gold, or platinum. Use 'unknown' if no plan name "
                    "is mentioned -- never infer it from the sum insured or "
                    "any other context."
    )

    language: Literal["en", "hi-en"] = Field(
        description="'hi-en' if Hindi words are mixed into the English, "
                    "including transliterated Hindi in Latin script (e.g. "
                    "kripya, jaldi, bahut, turant). Otherwise 'en'."
    )

    # Set by our code, never by the model.
    needs_human_review: bool = False
    review_reason: str = ""


def extract_c(ticket: str) -> dict:
    """Part C: model for judgement, code for everything else.

    Returns a plain dict (model fields + deterministic fields + business rules)
    so that run_eval.py can score it against the gold labels directly.
    """
    try:
        rec = structured(
            f"Ticket:\n{ticket}",
            schema=TicketRecordC,
            system=SYSTEM_PROMPT,
            tier="SMALL",
        )
        fields = rec.model_dump()
    except Exception as exc:  # noqa: BLE001 - never raise: degrade any failure (StructuredOutputError, and T1 3 modes 1-3: transport, rate limit, timeout/budget) to a review record
        fields = TicketRecordC(
            evidence="",
            category="information",
            urgency=1,
            sentiment="neutral",
            product="unknown",
            language="en",
            needs_human_review=True,
            review_reason=f"{type(exc).__name__}: {exc}"[:300],
        ).model_dump()

    fields.update(extract_deterministic(ticket))
    return apply_business_rules(fields, ticket)


if __name__ == "__main__":
    import json

    root = Path(__file__).resolve().parents[2]
    sample = json.loads(
        (root / "data/eval/extraction_dev.jsonl").open(encoding="utf-8").readline()
    )
    print("--- ticket ---")
    print(sample["input"][:600])
    print("\n--- gold ---")
    print(sample["expected"])
    print("\n--- yours ---")
    print(extract_c(sample["input"]))