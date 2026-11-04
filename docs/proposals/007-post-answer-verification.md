# Proposal 007: stage 3, post-answer verification (amends ADR-011)

Status: **open. Waiting for the owner's approval. Nothing in this proposal is built yet.**
Changes (if approved): the abstention part of ADR-011 (already amended by proposals 003 and 006),
the PLAN M6 row, `[generation]` config, and the evaluation tooling (a separate abstention set).

## Problem and evidence

After proposal 006 the only abstention left is the model's own `insufficient` flag, because stage 1
refuses nothing on this corpus. With the adopted answer prompt (`ee9346d6e271`) the dev run
(`results/generation_dev_prompt2.json`) refuses 0 of 54 answerable and **3 of 6 unanswerable**
queries. The other three (q024, q051, q081) were answered with citations that pass the validator.
The validator checks structure only (known IDs, non-empty citations), so a fluent answer built from
related notes passes. Six unanswerable queries are also too few to measure small changes: one query
is 17 points.

## Decision (the owner's)

Add **stage 3**, one extra LLM call for every query that the answer call answered:

- Model `gpt-5-mini`, with a **higher reasoning effort than the answer call** (the answer call uses
  `minimal`).
- Input: the question, the draft claims, and the **full text of the chunks the claims cite**.
- Output, strict JSON schema: `answers_question` (do the cited chunks directly answer the question
  that was asked, not a related topic?) and, for each claim, `supported`.
- Refuse if `answers_question` is false. Drop every claim whose `supported` is false. Refuse if no
  claim remains.

Stage 1 and stage 2 are unchanged.

## Choices I made, fixed here before anything is run

| Item | Value | Why |
|---|---|---|
| Reasoning effort of the verifier | `medium` | Above `minimal`, below `high`; reasoning tokens are billed as output, so `high` costs more. Not tuned. |
| Output cap of the verifier | 4000 tokens (reasoning included) | Room for reasoning plus a short JSON answer. Not tuned. |
| What the verifier sees | question (secret-scrubbed), the draft claims numbered 1..n with their cited IDs, and **only the cited chunks**, each under its ID with notebook and headings | As the owner specified. Chunks that were in the answer context but not cited are not shown. |
| Prompt | `llm/prompts/verify.md`, one version, never edited under this proposal | The prompt's content hash is recorded in the results. |
| Malformed verifier output | one retry with the problems listed (not all claims judged exactly once, an out-of-range claim number, no parse), then **refuse** | Fail closed: an answer that could not be verified is not shown. These refusals are counted separately so they cannot hide inside a rate. |
| Outcomes added | `refused_by_verifier` (`answers_question` false), `refused_unsupported` (no claim left), `refused_verifier_invalid` | Each is reported by its own count. |
| Shown answer | only the claims that were kept; the sources list is rebuilt from them | A dropped claim must not leave its sources behind. |
| Verifier is not the validator | The citation validator still runs first and is unchanged | Stage 3 adds a model judgment on top of the code-enforced check; it does not replace it. |

## Pre-registered rule

Evaluate on the **dev split only**, one run, the pipeline of the adopted prompt with stage 3 added.

**Adopt stage 3 if and only if**

1. false refusals on the answerable dev queries are at most **2 of 54**, **and**
2. unanswerable dev queries refused are **strictly more than 3 of 6** (so 4 or more).

Otherwise stage 3 is not adopted: its code and prompt are removed from `src/` (they stay in git
history, as with the M5 blurb and clean), and this proposal and the results file stay as the record.
No second verifier prompt or effort is tried under this proposal.

**Reported alongside the current pipeline** (same queries, same retrieval, same answer prompt, the
answers of the existing run reused through the response cache, so the only difference is stage 3):

- answerable and unanswerable queries refused, as counts (for example 3/6), split by the refusal reason;
- claims dropped by stage 3 in total and per answered query, and the gold-hit rate after dropping;
- verifier failures (invalid output after the retry);
- **cost**: input, cached, output and reasoning tokens and dollars of the stage 3 calls, and the total
  per query for both pipelines;
- **latency**: p50 and p95 of the answer path per query, with and without stage 3, taken from fresh
  calls (a cached call reports the latency of the call that filled the cache).

Cost and latency are reported, not part of the rule.

## The abstention set (tooling, built only after approval)

The owner will write about 20 unanswerable and about 20 answerable queries with the existing label
tool into a separate private file, for example `data/rerag-data/gold/abstention.json`. It is **not**
part of the frozen gold set, has no dev/test split, and is **report-only**.

What I would build:

- `noterecall label --id-prefix a` (new option) so the ids are `a001`, `a002`, ... and cannot be
  confused with the gold ids `q001`...; `--gold-path` already exists. Answerable queries get cell labels
  as today; unanswerable ones have none.
- `noterecall gold validate --kind abstention`: orphan labels as today, plus checks that the set is
  never frozen or split, ids do not collide with the gold set, no query text equals a gold query
  (reported by id only), and the counts per kind are printed. No size or stratification minimum.
- `eval --suite generation --set abstention`: runs the same pipeline on every query of the set and
  writes `results/generation_abstention_<pipeline>.json` (ids and numbers only, like the other
  results files).
- Raw counts with a **Wilson 95% interval** for each rate (a small pure function with hand-computed
  tests). No decision is made on this set.

Rules for the set that keep it honest: it is written and committed to the private repository **before**
any model output for those queries is seen; the queries are new (not variants of the gold queries);
the unanswerable ones should be near the topic of the notes, because those are the ones that get through;
I do not create, edit or reformat the file.

It is reported for both pipelines (current and with stage 3), on its own, next to the dev numbers. If the dev rule adopts stage 3 and this set disagrees, both
are reported as they are and the owner decides what follows; the set cannot change the adoption rule.

## Cost, latency and the order of work (my estimates, to be replaced by dry-run figures)

- The verifier input is the question, the claims and the cited chunks: roughly 1 to 3k tokens. Reasoning
  at `medium` is unknown until measured; at the 4000-token cap a call costs about $0.008 plus input.
  For the 57 answered dev queries that is about $0.1 to $0.5, and the same order of magnitude again for
  the 40 abstention-set queries. M6 has spent about $0.11 of its $2.00 cap.
- Each answered query gets a second sequential call, so the answer path takes roughly twice as long.
- Order: (1) the owner approves; (2) build and test offline with the fake model; (3) smoke test: one real
  verifier call on three invented chunks to confirm the API accepts `medium` and the schema, and show the
  real tokens and cost; (4) `--dry-run` for the dev run; (5) the dev run and the comparison; then stop.
  No test run, and no PR, without the owner's approval.

## Risks and limits

- **A model checking a model.** The verifier is the same model family as the answerer, so it can share
  its blind spots and prefer its own phrasing (ADR-012 cites self-preference, Zheng et al. 2023). The
  owner chose `gpt-5-mini`; this is a limit of the design, not something a threshold fixes. The M7
  judge, calibrated against hand labels, remains the place where faithfulness is measured.
- **False refusals.** A strict `answers_question` can refuse questions the notes answer only partly. In
  the adopted dev run 9 of the 54 answered answerable queries cite no gold-labelled cell; the verifier may
  refuse some of these. The bound of 2/54 is the guard.
- **Weak power.** With 6 unanswerable dev queries, 4/6 against 3/6 is one query. The rule is a
  pre-registered choice, not a significance test, and the abstention set exists to give a larger sample.
- **Verifier failures count as refusals** (fail closed), which can raise false refusals; they are reported.
- **Data.** The question and the cited chunks go to the OpenAI API with `store=False`, as in the answer
  call (consent given earlier). Nothing from the abstention set is committed.

## Proposed wording for ADR-011

> **Abstention** (amended by proposals 003, 006 and 007): stage 1 (no LLM), stage 2 (the model's
> `insufficient` flag) and **stage 3**: after an answer passes the citation validator, a second call with
> a higher reasoning effort receives the question, the claims and the cited chunks, and returns
> `answers_question` and a `supported` flag per claim. The answer is refused if `answers_question` is
> false, unsupported claims are dropped, and it is refused if none remain. Adopted only if it passes the
> pre-registered dev rule of proposal 007.

## Questions for the owner

1. Is `medium` the right effort for the verifier, or do you want `low` or `high`? (Fixed once you answer.)
2. When the M6 test run is approved, should it report the pipeline with and without stage 3 in the same
   single run (as the M5 header row did), or only the shipped pipeline?
3. Should the abstention set also be run through the current pipeline at no extra decision cost? I assumed
   yes, so the two pipelines are compared on the same new queries.
