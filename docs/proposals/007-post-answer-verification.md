# Proposal 007: stage 3, post-answer verification (amends ADR-011)

Status: **approved by the owner. Evaluated on dev, not adopted (the pre-registered rule was not met), and
removed from the code: see Addendum 1.** (The owner's answers to the three questions are recorded below.)
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

## Owner's answers (approval)

1. Reasoning effort of the verifier: **`medium`**, as proposed.
2. The M6 test run reports **both rows** (the current pipeline and the pipeline with stage 3) in the single run.
3. The abstention set is run through **both pipelines**.

Composition of the abstention set, from the owner: mostly near-topic unanswerable queries, plus about 5
clearly off-topic ones. The owner writes and commits the file to `rerag-data` before any model output for
those queries is seen.

Order of work, as agreed: build and test offline, a real smoke call on invented chunks, `--dry-run`, the
dev run; then stop and show the results. No test run and no PR without the owner's approval.

## Result on dev (rule applied; written after the run)

Run: `eval --suite generation --split dev --stage3`, `results/generation_dev_stage3.json`, answer prompt
`ee9346d6e271` (its 60 answers were already in the response cache, so stages 1+2 are the same drafts),
verify prompt `b0d920c8a901`, effort `medium`, cap 4000. Rendered by
`python scripts/render_results.py results/generation_dev_stage3.json`.

| Measure | stages 1+2 (current) | stages 1+2+3 |
|---|---|---|
| answerable queries refused (95% Wilson) | 0/54 [0.00, 0.07] | 0/54 [0.00, 0.07] |
| unanswerable queries refused (95% Wilson) | 3/6 [0.19, 0.81] | 3/6 [0.19, 0.81] |
| answered answerable queries citing a gold chunk | 45/54 | 44/54 |
| claims shown | 169 | 161 |
| model cost USD per query | 0.00087 | 0.00262 |
| answer-path model latency p50 / p95, s | 2.44 / 4.33 | 9.07 / 22.63 |

**Rule: not met.** False refusals are 0/54 (the bound was at most 2/54), but unanswerable refusals did not
strictly improve on 3/6: they are 3/6. Stage 3 is therefore **not adopted** by the pre-registered rule.

What stage 3 did: all 57 answered queries reached it; `answers_question` was true for every one of them,
including the three unanswerable queries that are still answered (q024, q051, q081); it dropped 8 of 169
claims (in 7 queries, none of them emptied), made no verifier-invalid or retried calls, and cost $0.1048
(41,902 output tokens of which 36,800 were reasoning; 735 output tokens per call, against the 1,500 assumed
in the estimate). One answerable query (q020) lost its only gold-hit claim, so the gold-hit rate went from
45/54 to 44/54. The cost per query tripled and the median latency of the model calls of an answered query
went from 2.4 s to 9.1 s.

## Addendum 1: stage 3 rejected on dev and removed from the code

Decision (the owner's): the rule of this proposal says a rejected stage 3 is removed from `src/`, and it is.
The code and the `verify` prompt are in git history: the last commit that has them is `4e877d8`
(`git checkout 4e877d8`; there `python scripts/render_results.py results/generation_dev_stage3.json`
prints the table above). `results/generation_dev_stage3.json` stays as the record, and a test checks that
it still loads and says what this addendum says.

What was measured on dev (60 queries, 54 answerable and 6 unanswerable; all numbers from the results file):

- **Abstention: no change.** Unanswerable queries refused 3/6 before and 3/6 after stage 3. The rule needed
  strictly more than 3/6.
- **False refusals: none.** 0/54 before and after (the bound was at most 2/54).
- **The verifier said `answers_question = true` for all 57 answered queries**, including q024, q051 and
  q081, the three unanswerable queries that are still answered. It dropped 8 of 169 claims (in 7 queries)
  and never emptied an answer, so it never refused anything.
- **Cost about 3 times:** model cost per query 0.00087 USD before, 0.00262 USD after (3.0 times).
- **Latency about 4 times:** the model calls of an answered query took a median of 2.44 s without stage 3
  and 9.07 s with it (3.7 times); the 95th percentile went from 4.33 s to 22.63 s (5.2 times). Most of the
  extra is reasoning: 36,800 of the 41,902 output tokens of stage 3 were reasoning tokens.
- **Gold-hit:** 45/54 before, 44/54 after (one answerable query, q020, lost its only gold-hit claim).

What this does and does not show. Within this corpus and with the same model as verifier, near-topic
unanswerable queries look answerable to a second reading of the same notes. This is one run, one verifier
prompt and one effort level on six unanswerable queries; it says nothing about other verifiers, other
models or other corpora, and it was not tested on the test split or on the abstention set.

State after this addendum: abstention rests on the model's `insufficient` flag (stage 2) alone, as after
proposal 006. ADR-011 is unchanged by this proposal, because stage 3 was not adopted. The abstention set
that the owner is writing remains a report-only set for the current pipeline (stages 1 and 2); the tooling
for it stays. The per-call reasoning effort in the LLM layer and the API-latency record also stay (they
are general features of the layer, tested without stage 3).

Update (owner's decision, after Addendum 1): the abstention set described in "The abstention set" above will
**not be written**, so it was never run through either pipeline. The tooling for it stays, unused on real
data. Proposal 006, addendum 3 records the same.

