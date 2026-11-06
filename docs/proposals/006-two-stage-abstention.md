# Proposal 006: two-stage abstention (amends ADR-011)

Status: **approved by the owner.**
Changes: the abstention bullet of ADR-011 (already amended once by proposal 003), the PLAN M6 row,
and the `[generation]` config. Written **before any calibration was run**: the rules below are
pre-registered. Nothing in them was chosen after looking at a result.

## Problem and evidence

ADR-011 (as amended by proposal 003) refuses when the top dense cosine is below one threshold.
A look at the **dev split only** (no LLM; the pipeline of M5; the owner's gold set) showed that this
signal separates the two kinds of query weakly:

- top dense cosine of the 6 unanswerable dev queries: 0.673 to 0.733;
- top dense cosine of the 54 answerable dev queries: median 0.745, range 0.598 to 0.872;
- AUC (answerable above unanswerable) 0.753;
- best abstain F1 on dev 0.385, at a threshold that refuses 15 of the 54 answerable queries.

Source: an ad-hoc script on dev, not a committed results file. The committed calibration output of M6
(`results/abstention_calibration_dev.json`) supersedes these numbers.
(The same script also printed the cosine ranges of the test split. That was a mistake. The owner was
told, and no number from the test split is used for any decision here or in the code.)

## Decision (the owner's)

Abstention has two stages.

**Stage 1, no LLM call.** Refuse (answer "not in your notes", make no API call) only if both hold:

- top dense cosine similarity `< tau_cos`, and
- normalized top BM25 score `< tau_bm25`.

**Normalization, fixed now:** `bm25_norm = top BM25 score / q`, where the top BM25 score is the score
of the best chunk of the BM25 index (the code-aware tokenizer, k1 = 1.5, b = 0.75, over the indexed
text, header included), and `q` is the number of tokens the same tokenizer produces for the query
(duplicates counted). If `q = 0` or no chunk matches, `bm25_norm = 0`.

**Calibration on dev (pre-registered).**

1. Candidate thresholds for each axis: the midpoints between consecutive distinct dev values of
   that signal, plus one value below the smallest and one above the largest. A value above the largest
   makes that axis's condition always true, so "cosine only" is part of the same grid.
2. A pair is *feasible* if the false-refusal rate on the answerable dev queries is at most 5%, checked
   in integers: `false_refusals * 100 <= 5 * answerable`. (With 54 answerable dev queries that
   allows at most 2.)
3. Among feasible pairs choose the one that refuses the most unanswerable dev queries.
4. Ties: the lowest `tau_cos`, then the lowest `tau_bm25`.
5. The full grid (every pair, with its counts) is written to the results file.

**Stage 2, model side.** The answer schema has a required boolean `insufficient`. If it is true the
answer must have no claims, and the response is a refusal. If it is false the answer needs at least one
claim and every claim needs at least one valid citation (ADR-011 validator, one retry with the error
message, then refuse).

## What is reported

Three rows, on dev and on test (thresholds always fixed on dev):

| Row | Meaning |
|---|---|
| cosine only | stage 1 with the best pair among those with the BM25 axis always true, same 5% rule |
| cosine + BM25 | stage 1 with the chosen pair |
| two-stage (ships) | stage 1, then the model's `insufficient` flag and the validator's refusals |

**Primary metric: the false-refusal rate** on answerable queries. Unanswerable recall is given as raw
counts (for example 5/6). Precision and F1 are given for reference only. With 6 dev and 3 test
unanswerable queries every one of these is coarse, and the report says so.

## What does not change

ADR-011's other rules: at most 6 chunks in context, short IDs, code-enforced citations, one retry.
Cell-level labels (ADR-009), the dev/test split, and the test split being run once per milestone.
The validator checks structure only (known IDs, non-empty citations); whether a cited chunk supports
the claim is the M7 judge's job (ADR-012).

## Impact

`retrieval/pipeline.py` exposes the two stage-1 signals; `generate/abstain.py` holds the pure rule and
the calibration; the answer schema and validator follow this proposal. Config gains `[generation]`;
the thresholds are written there only after the calibration results file exists.

## Not verified

That any threshold pair reaches the 5% bound while catching most unanswerable queries (unknown until
calibrated), and that the model's `insufficient` flag is used sensibly (unknown until the dev run).

## Addendum 1: one answer-prompt revision on dev (pre-registered, written before it was run)

Why: the first full dev run (`results/generation_dev.json`, prompt `answer` version `f715ca2be4cf`)
refused 2 of the 6 unanswerable dev queries and 0 of the 54 answerable ones. Stage 1 refuses nothing
(see the ADR-011 Result line), so the model's `insufficient` flag is the only abstention there is.

**The change (the only one).** In `llm/prompts/answer.md`, the instructions say that the model answers
only if the provided notes directly contain the answer, and otherwise sets `insufficient` to true with
no claims. Nothing else changes: not the schema, the validator, the retry, the retrieval, the
context size, the model, the effort or the output cap.

**Adoption rule.** The revised prompt is adopted if and only if, on the dev split:

1. false refusals on the answerable dev queries are at most 2 of 54, **and**
2. unanswerable dev queries refused are strictly more than 2 of 6.

Otherwise the original prompt stays. Both outcomes are reported; a rejected revision stays in
`results/` and in git history as evidence. This is one revision, one run: no second revision is tried
on dev under this addendum.

**Reported side by side:** the table of the first run (outcomes per kind of query, first-attempt
citation validity, retries, uncited-claim rate, gold-hit rate, rows, tokens and cost) for both prompts.

**Limits that apply to the decision.** Six unanswerable dev queries: one more refusal is a change of
17 points and is not statistical evidence. The rule is a pre-registered choice, not a test. The test
split is not touched by this addendum.

**Result of addendum 1 (dev; `results/generation_dev.json` for the original prompt `f715ca2be4cf`,
`results/generation_dev_prompt2.json` for the revision `ee9346d6e271`; rendered by
`python scripts/render_results.py results/generation_dev.json results/generation_dev_prompt2.json`):**

- answerable queries refused: 0/54 for both (the bound was at most 2/54);
- unanswerable queries refused: 2/6 for the original, 3/6 for the revision (the rule needed strictly more than 2/6);
- by the rule above the revision is **adopted**. The change is one query (q037); six unanswerable
  queries cannot tell this apart from chance, and the decision follows the rule, not a significance test;
- other numbers moved too, outside the rule: the revision writes shorter answers (169 claims against 205),
  first-attempt validity (60/60), retries (0) and gold-hit (45/54) are unchanged, and the cost is
  $0.0522 against $0.0547.

No second revision was tried. The test split has not been run.

## Addendum 2: owner adjudication of three dev queries, and two-view reporting (written before the test run)

**Owner adjudication, not a model change.** The owner read q024, q051 and q081 against the notes the model
cited. All three are **labelling errors in the gold set**: the notes answer q024 and q051, and partly answer
q081. They were labelled unanswerable and should not have been. The gold file was not touched: it is the
owner's, it is frozen, and its split hash is unchanged. Nothing in the pipeline, the prompt or the thresholds
changes because of this, and there are no further abstention changes.

**Two views, from now on.** Every abstention number is reported two ways:

- **as labelled:** the gold set exactly as frozen;
- **adjudicated:** the owner-confirmed label errors, listed by id, are left out of the unanswerable queries
  (out of the count refused and out of the count of unanswerable queries). They are not moved to the
  answerable queries, because they carry no gold cell labels. Answerable counts are the same in both views.

The list lives in config (`[generation] label_errors`) and in this addendum. For the dev split it is
q024, q051, q081, so the adjudicated dev view of the adopted prompt has 3 unanswerable queries (q037, q083,
q084), all three refused, and 0/54 answerable queries refused. The as-labelled view stays 3/6 and 0/54.
Earlier decisions that were made on the as-labelled numbers (proposal 006 addendum 1, proposal 007) are not
revisited.

**Pre-registered for the M6 test run, before it is run.** If any unanswerable test-split query is answered,
the question, the claims and the cited chunks are shown to the owner in chat, the owner adjudicates it, and
both numbers (as labelled, adjudicated) are reported with the owner's decision recorded in this proposal.
Until the owner has decided, the adjudicated test view equals the as-labelled one (no test id is in the list).
The test split is run once, with the adopted answer prompt `ee9346d6e271` and the stage-1 thresholds of the
dev calibration.

**For M7 (faithfulness evaluation).** q081, claim 1 of the adopted-prompt answer (`results/generation_dev_prompt2.json`
has the counts only; the cached answer has the text) is an example of an **unsupported claim** to include
in the M7 judge's calibration material: it states a comparative speed claim against another library that
the cited chunk does not state (the chunk says only that performance is comparable). The claim passes the
structural check of ADR-011 because it cites a chunk that exists. Ids only are recorded here, not text.
