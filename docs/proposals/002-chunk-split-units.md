# Proposal 002: chunk split units, oversize flag, and the output policy

Status: **approved by the owner.** Implemented on branch `m2-chunking`. The output policy was
settled by the owner's final decision below (outputs are never chunked).
ADR affected: ADR-004 (structure-aware chunking), amended in `PLAN.md` and `docs/adr/ADR-004.md`.

## Problem

ADR-004 says a single oversized code cell is split "at blank lines / top-level `def`/`class`".
The owner decided on a different rule, and the original text says nothing about oversized
markdown cells, about what happens when a unit is still too large, or about cell outputs.
A survey of the real corpus (53 notebooks, `cl100k_base`, aggregates only) shows the cases are real:

| Quantity | Value |
|---|---:|
| Markdown / code cells over the cap alone at 256 | 253 / 377 |
| ... at 512 | 72 / 120 |
| ... at 1024 | 12 / 16 |
| Single top-level statements over the cap (256 / 512 / 1024) | 37 / 11 / 1 |
| Markdown paragraphs over the cap (256 / 512 / 1024) | 33 / 7 / 2 |
| ... of which still over after a sentence split | 19 / 5 / 2 |
| Code cells that parse as Python once `%` and `!` lines are masked | 2,071 of 2,078 |
| Code cells with output text / outputs over 256 tokens on their own | 1,284 / 246 |

## Decision (the owner's answers 12 and 13)

1. **Smallest unit.** One top-level code statement, or one markdown sentence, is the smallest
   unsplittable piece. A statement keeps its leading comments and decorators; `%` and `!` lines
   count as one-line statements.
2. **Split order.** An oversized code cell is split at statement boundaries. An oversized
   markdown cell is split at paragraphs, then at sentences.
3. **Two separate invariants, both tested.** (a) Code is never split mid-statement. (b) A fenced
   code block inside a markdown cell always stays in one piece. An unclosed fence runs to the end
   of the cell.
4. **Oversize flag.** A chunk may exceed the cap only if it is one indivisible piece. It is then
   flagged with the reason (statement, sentence, fenced block, or unparsed code).
5. **Unparseable code** (7 of 2,078 cells) is indivisible and flagged; no boundaries are invented.
6. **"Exactly one chunk" restated.** Each cell's pieces partition its text exactly, and every
   piece lies in exactly one chunk. A cell appears in several chunks only when it alone exceeded
   the cap, and then its pieces are consecutive.

Also fixed in M2 (no ADR text change needed): sections are flat (any ATX heading level starts one),
small sections are not merged, a cell with several headings is atomic, setext headings are
ignored, and the fixed-token baseline uses no overlap.

## Output policy: final decision (owner)

**Code-cell outputs are excluded from chunk text, permanently.** They stay in the database
(stored by M1) and neither chunker uses them, the fixed-token baseline included. There are no
output modes.

Original principle (owner): an output belongs in chunk text only if it is evidence needed to
understand a concept or conclusion. Outputs were first excluded by default, with a deterministic
"evidence" rule proposed and validated against the owner's labels before any decision to include
some. The measured result (`results/evidence_rule_eval.json`, 50 outputs in two strata, labels
by the owner with borderline counted as noise):

Rendered by `scripts/render_results.py` from that file:

| Evidence rule vs owner labels | Value |
|---|---:|
| Outputs the rule keeps / drops (population) | 434 / 850 |
| Sampled from each | 25 / 25 |
| Labelled evidence among sampled kept | 11 |
| Labelled evidence among sampled dropped | 6 |
| Precision (kept that are evidence) | 44.0% (95% Wilson 26.7% to 62.9%) |
| Recall (evidence that is kept, weighted to the population) | 48.3% |
| Evidence prevalence (weighted) | 30.8% |

Sampling seed 20261029, rule `859eea67ec82`.

Reasoning: no simple rule separates evidence from noise, and an ambiguous output in chunk text
risks the LLM grounding an answer on it. The principle is "only clear evidence; when in doubt,
exclude", and nothing measured can meet it. So the policy is the simplest one that satisfies it.

Consequences:

- The 3-way output ablation (no outputs / all outputs / evidence-only) that this proposal first
  recorded for M4 is **dropped**. M4 compares only chunk caps and the fixed-token baseline.
- The evidence rule, its sampling and its evaluation commands are removed from `src/`
  (the rule is in git history at commit `810e07c`, which also holds the code that produced the
  record below). `OutputMode`, the output piece kind, output-line splitting and the
  "an output follows its code" placement logic that briefly existed during M2 are removed with it:
  with no outputs in chunks the invariant holds trivially, and a property test now checks that no
  output text can reach either chunker's output.
- Chunk IDs no longer depend on an output mode; chunker versions are `structural-3` and `fixed-2`.
- A future change (for example a different rule, or outputs attached as separate metadata) needs
  a new proposal and a fresh labelled sample; the 50 labels used here are spent for tuning.
- Retrieval may still look outputs up in the database later (for example to display them next to
  a retrieved chunk); this proposal only decides what is indexed.

### Record of the negative result

`results/evidence_rule_eval.json` is kept as the record. It holds the metrics, the sampling seed,
the rule fingerprint, the git SHA of the clean tree it was produced from (`9ead3e5`, which still
contains the rule and the commands), the config snapshot, the tokenizer and `tiktoken` version,
and one row per sampled item: stratum, the rule's verdict category and the owner's label. It holds
no output text, no file names and no cell identifiers. The earlier version of the file, committed
in `06da9bf`, had the same metrics but lacked the metadata and per-item scores.

The sample manifest and the labels stay under the git-ignored `data/evidence/`.

## Impact

- ADR-004 text amended (see its heading and note). No other ADR changes.
- `raw_hash`, the gold-set cell identity, is unaffected: labels stay valid under every cap.
