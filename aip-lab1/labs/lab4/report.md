# Lab 4 Report — Grounded Answers with Citations

## A–B. Prompt and citation contract

The system prompt requires the assistant to use only numbered sources, cite claims by source index, never invent citation numbers, surface source disagreement, and keep answers concise. It uses the exact refusal string: `I don't have enough information in the provided sources to answer that.` It also directs the assistant to answer supported parts of a multi-part question and decline only unsupported details.

Compared with `aip.rag.ANSWER_SYSTEM`, this prompt makes the partial-answer rule explicit and says the exact refusal is for cases where none of the requested facts is supported. The reference prompt instead applies its refusal rule whenever the answer is not in the sources. Both prompts forbid general knowledge, require indexed citations, address disagreements, and set a short-answer limit.

`validate_answer()` rejects empty or truncated output, out-of-range citations, and uncited non-refusals. On validation failure the pipeline fails closed to the exact refusal; it does not retry or return the invalid answer. Citation validity in the final 45-row JSON is **45/45 (1.000)**.

## C. Refusal results

There are only five unanswerable questions, so each one changes recall by 0.20; interpret these ratios cautiously.

| Setting | Correct full refusals | Total unanswerable | All full refusals | Recall | Precision | Wrongful refusals |
|---|---:|---:|---:|---:|---:|---:|
| Default | 4 | 5 | 6 | 0.800 | 0.667 (4/6) | 2 |
| Strict | 4 | 5 | 6 | 0.800 | 0.667 (4/6) | 2 |

The stricter instruction did not change the measured full-refusal counts. In the final full-set results, Q37 is a correct partial response: Platinum’s international emergency benefit is cited, while Singapore eligibility and the limit are identified as unavailable. Counting that partial response as a successful decline gives 5/5 recall and 5/8 precision (0.625).

For an insurance helpdesk, use the default setting with the partial-answer rule. A blanket refusal can withhold a supported benefit fact; an unsupported coverage limit can mislead a customer about a claim. The system should state the supported part and explicitly decline the missing detail.

## D. Judge calibration

The 20 hand-labelled examples gave:

| Rubric | n | Raw agreement | Cohen's κ |
|---|---:|---:|---:|
| Faithfulness | 20 | 1.00 | 1.00 |
| Correctness | 20 | 0.80 | 0.467 |

Correctness disagreements were Q04, Q05, Q10, and Q19. Correctness κ is above the lab's 0.4 threshold, so the rubric was retained. Generation used Gemini 3.7 Flash and judging used Gemini 3.5 Flash: distinct model versions, though the shared provider means some common bias may remain.

## E. Full evaluation, decomposition, and failure backlog

The corrected retriever indexes `data/corpus` (the Lab 4 insurance documents), not Lab 3's scaled synthetic corpus. The final JSON contains 45 cases, all with correctness and faithfulness scores, and no invalid final citations.

| Metric | Result |
|---|---:|
| Correctness (normalized) | 0.838 |
| Faithfulness | 0.956 |
| Citation validity | 1.000 (45/45) |
| Full-refusal recall / precision | 0.800 (4/5) / 0.667 (4/6) |
| Repair rate | 0.022 |
| Cost per query, fresh corrected run | $0.0091 |
| p95 latency, fresh corrected run | 4,787 ms |

The final refresh was cache-heavy, so its 16 ms p95 and $0.0002 run cost describe cached evaluation, not production. The fresh corrected run figures above are the relevant cost and latency measurements.

On the 40 answerable questions, correctness with gold context was **A = 0.975** and with retrieved context **B = 0.838**. Retrieval-attributable loss was **0.137**; generation-attributable loss was **0.025**. Retrieval is the larger remaining source of error.

The failure classifier tagged nine below-target answers: **mode 6 (generation), 6** and **mode 2 (chunk boundary), 3**. No presentation failures were found because the citation validator enforces valid indices. These automated tags are a triage aid; the three mode-2 cases still warrant chunk inspection.
