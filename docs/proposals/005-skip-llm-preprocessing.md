# Proposal 005: skip the LLM preprocessing steps (blurb and LLM clean) in M5

Status: **approved by the owner.**
Changes: the M5 scope (PLAN section 3), ADR-003 (LLM clean) and ADR-007 (LLM blurb).
This is an **owner decision on cost and benefit. It is not a measured result.**

## Decision

M5 makes no real LLM call. The LLM contextual blurb and the LLM markdown clean are **not run**:

- ADR-003: the LLM clean is **not run** and ships **off**.
- ADR-007: the LLM blurb is **not run**. The deterministic header stays on, as ADR-007 already says.
- The M5 acceptance row now reads "not measured, by owner decision" for both.

Neither step was tested, so nothing can be said about whether they help. The README says
"not tested", never "does not help".

## Reasons (the owner's)

1. The deterministic header already adds context. That is ADR-007's own hypothesis for well-structured notebooks.
2. M4 Recall@10 is already high, so the headroom is small. The owner's figure of 0.832 is the M4 test-split value (`results/retrieval_test_M4.json`); the dev value for the same pipeline is 0.819 (`results/retrieval_ablation_dev.json`).
3. With 54 dev queries, an ADR-010 comparison can only detect gains of about 5 points, so a small
   benefit from either step could not be told apart from noise, while the steps cost money and add code to defend.

## What changes

| Where | Change |
|---|---|
| PLAN M5 row | Header on vs off is measured; blurb and LLM clean are "not measured, by owner decision". |
| ADR-003 | LLM clean: not run, ships off. |
| ADR-007 | LLM blurb: not run. Header: on, with its on/off dev row reported. |
| `src/` | The blurb, clean and verifier features, their CLI commands and their two prompt files are removed. They stay in git history (commits on branch `m5-enrichment`). |
| `src/` (kept) | The LLM layer: `LLMClient` protocol, response cache and spend ledger, cost meter, fake transport, OpenAI transport, prompt loader. M6 and M7 need it. |
| Proposal 004 | Still approved: it chose the provider for the kept layer. Its per-task settings are collapsed: `[llm]` keeps one `model`, and `blurb_model`, `clean_model` and the two `max_output_tokens_*` keys are removed because nothing uses them. M6 sets its own output caps. |
| Evaluation | M5 reports the header on vs off on dev, with its interval. The header stays on whatever the row shows (ADR-007 is locked). |

## Not verified

Everything about the removed features. In particular the layer that stays was tested offline only: no request has ever
been sent to the OpenAI API, so the accepted `reasoning.effort` values, the strict JSON schema handling and the real cost
and cache behaviour are unverified until M6 first uses it.
