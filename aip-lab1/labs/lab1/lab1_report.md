# Lab 1 — Reliable Extractor Report

## Setup

The experiments used the `gemini-3.5-flash-lite` model from the `SMALL` tier with `temperature=0` and prompt caching enabled. The pricing used was $0.30 per million input tokens and $2.50 per million output tokens. Prompt tuning was performed on the 60-ticket development set, while the 120-ticket test set was evaluated once and stored in `reports/lab1_test.json`.

## Part A — Behaviour of the Naive Extractor

The initial implementation, `v0_naive.py`, sends a single prompt and directly applies `json.loads()` to the model response. Running it on 40 development tickets exposed several consistent output-format and schema issues.

| Failure type | Tickets (of 40) | Taxonomy (T1 §3) |
|---|---:|---|
| Response was not JSON | 0 | #5 |
| JSON enclosed in Markdown fences | 39 | #5 malformed output |
| Additional text around JSON | 1 | #5 |
| Required field missing | 0 | #6 |
| Invalid `category` value | 39 | #6 schema violation |
| `urgency` returned with the wrong type | 38 | #6 wrong type |
| Hallucinated `policy_number` | 1 | #8 hallucination |
| Unhandled crash | 0 | — |

The main weakness of the naive version was not the extraction task itself, but the lack of output control. Most responses contained a Markdown code fence, while category and urgency also frequently failed the expected schema.

Because the raw responses were not cleaned before parsing, only 1 of the 40 responses could be consumed directly. Removing the Markdown fence greatly improved parsing, but schema validation was still necessary before accepting a record. The complete run cost approximately $0.0080, with a p95 latency of about 1310 ms.

This experiment demonstrates that parsing alone is not enough. A production extractor needs both output normalization and explicit schema validation.

## Part B and Part C

Part B performs one model call and validates the complete eight-field record. If validation fails, the system returns a reviewable record instead of terminating.

Part C keeps the same general approach but moves `policy_number` and `contains_pii` out of the model-generated schema. These values are determined programmatically using a regular expression and deterministic logic.

| Metric | v0 | B (dev, 60) | C (dev, 60) | C (test, 120) |
|---|---:|---:|---:|---:|
| Valid records | 1 / 40 | 60 / 60 | 60 / 60 | 120 / 120 |
| Field accuracy | 0.01 | 0.92 | 0.90 | 0.93 |
| Whole-record accuracy | 0.01 | 0.48 | 0.46 | 0.54 |
| Total cost | $0.0080 | $0.047 | $0.039 | $0.077 |
| Cost per ticket | $0.00020 | $0.00078 | $0.00065 | $0.00064 |
| p95 latency | 1310 ms | ~1345 ms | 1485 ms | 1405 ms |
| Review flags / crashes | — / 0 | 1 / 0 | 1 / 0 | 1 / 0 |

On the same 60 development tickets, Part C reduced the amount of information sent to the model while keeping overall quality broadly stable. Input tokens fell by roughly 13%, output tokens by around 21%, and the per-ticket cost decreased by approximately 16%.

The small change in field and record accuracy is within the variation expected from a relatively small development sample. More importantly, the two moved fields are now handled deterministically, making those decisions easier to inspect and reproduce.

For the test set, Part C achieved a valid-record rate of 1.00, field accuracy of approximately 0.93, cost of $0.077, p95 latency of 1405 ms, and essentially no execution failures. Whole-record accuracy was approximately 0.54, which remains slightly below the target of 0.55.

## Per-Field Results

The test-set field results show that most of the extraction task is reliable, while urgency and sentiment remain the most difficult fields.

| Field | Accuracy | Field | Accuracy |
|---|---:|---|---:|
| urgency | 0.68 | category | 0.94 |
| sentiment | 0.82 | contains_pii | 1.00 |
| escalate | 0.92 | language | 1.00 |
| policy_number | 1.00 | product | 1.00 |

The deterministic fields perform perfectly, while category and escalation are also relatively strong. The largest opportunities for improvement are urgency and sentiment.

## Category Confusion

The category errors are concentrated around the distinction between `complaint`, `information`, and `claims`.

```text
              billing claims complaint information policy_change technical
billing          16      .        .          .           .           .
claims            .     21        .          .           .           .
complaint         .      4       12          .           .           .
information       .      3        .         20           .           .
policy_change     .      .        .          .          22           .
technical         .      .        .          .           .          23
```

There are seven category mistakes in total. Almost all occur when the ticket contains terminology associated with claims but the actual intent is either a complaint or a request for information.

This suggests that the model understands the broad taxonomy but still struggles when keyword cues conflict with the underlying purpose of the ticket.

## Main Error Patterns

### 1. Urgency threshold calibration

Urgency remains the largest source of record-level mistakes. Many incorrect predictions are only one level away from the expected answer.

For example, a request for a tax certificate may be treated as a low-priority self-service request even though generating the document requires additional work. Conversely, a simple download request may be rated too urgently. Some technical problems are also treated as emergencies when they should receive a normal priority.

A useful improvement would be to add several examples specifically positioned around the urgency boundaries. The field definition should also make the decision rule explicit, such as asking whether the issue can be resolved without accessing or modifying the customer's file.

This is likely to provide the largest improvement because urgency affects a substantial portion of imperfect records.

### 2. Hinglish, politeness, and emoji

A second recurring issue is that informal Hinglish, polite words, and emojis sometimes influence urgency and sentiment more than they should.

Expressions such as `jaldi karo` or `kripya` can be interpreted as signs of urgency even when the underlying request is routine. Similarly, an angry-looking emoji may cause a neutral request to be classified as frustrated.

The prompt should explicitly state that tone markers are not sufficient evidence of urgency or sentiment. A few examples containing Hinglish and emojis would make the intended distinction clearer.

### 3. Claims versus complaint/information

The final major cluster concerns the boundary between claims and the other categories.

A ticket can mention a claim while actually complaining about how the company handled it. Likewise, a customer can ask for information after a claim has already been settled. In both situations, claim-related vocabulary can incorrectly dominate the classification.

The prompt should therefore include examples where claim terminology appears but the correct category is `complaint` or `information`.

## Cost and Operational Impact

The test run cost approximately $0.077 for 120 tickets, corresponding to about $0.00064 per ticket. At a workload of 10,000 tickets per day, this would be roughly $2,300–$2,400 per year in model costs.

For comparison, if a human agent requires around 40 seconds per ticket at ₹300 per hour, the equivalent manual cost is approximately ₹3.33 per ticket. The automated approach is therefore substantially cheaper on direct processing cost.

The relevant economic condition is:

`c_llm + (1 − r) · c_human < c_human`

which means that automation remains worthwhile when the cost of model processing plus the cost of correcting incorrect records is lower than fully manual processing.

At the observed accuracy level, the financial case is strong. However, cost is not the main operational constraint. The more important issue is identifying uncertain records so that human review can be focused on them.

With roughly 54% whole-record accuracy, a large fraction of records may still contain at least one incorrect field. If the review mechanism cannot reliably identify those cases, the organization cannot safely reduce human oversight by the same proportion.

## Schema Reduction: An Important Observation

Moving `policy_number` and `contains_pii` into deterministic code reduced token usage and processing cost, but it also changed the prompt structure seen by the model.

On the development set, sentiment accuracy changed noticeably after the schema was shortened, even though the sentiment definition itself was not modified. This indicates that the complete schema contributes to the model's context and that removing fields is effectively a prompt change.

Therefore, schema simplification should always be evaluated with before-and-after measurements rather than being treated as a purely mechanical optimization.

## Conclusion

The experiments show a clear progression from an unreliable one-shot JSON extractor to a more robust validated pipeline. The naive implementation failed mainly because it trusted the model's output format and did not enforce the expected schema.

Part B solved the reliability problem by validating every response, while Part C improved efficiency by moving deterministic fields into code. The test results show high validity, strong field-level accuracy, low cost, and comfortably acceptable latency.

The remaining gap is whole-record accuracy. Most of that gap comes from urgency calibration, sentiment interpretation, and the complaint/information versus claims boundary. Adding targeted examples for these cases should provide a better improvement than making the prompt generally longer.

Overall, Part C is the better implementation: it is cheaper, keeps accuracy at a similar level, and places deterministic decisions in code where they are easier to verify. The next iteration should focus on targeted prompt examples and better review-flagging rather than simply increasing model usage.
