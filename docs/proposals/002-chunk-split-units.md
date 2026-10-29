# Proposal 002: chunk split units, oversize flag, and the output policy

Status: **approved by the owner.** Implemented on branch `m2-chunking`.
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
   flagged with the reason (statement, sentence, fenced block, output line, or unparsed code).
5. **Unparseable code** (7 of 2,078 cells) is indivisible and flagged; no boundaries are invented.
6. **"Exactly one chunk" restated.** Each cell's pieces partition its text exactly, and every
   piece lies in exactly one chunk. A cell appears in several chunks only when it alone exceeded
   the cap, and then its pieces are consecutive.

Also fixed in M2 (no ADR text change needed): sections are flat (any ATX heading level starts one),
small sections are not merged, a cell with several headings is atomic, setext headings are
ignored, and the fixed-token baseline uses no overlap.

## Output policy

Principle (owner): an output belongs in chunk text only if it is evidence needed to understand a
concept or conclusion.

- **Default for M2: outputs are excluded from chunk text.** They stay in the database, and the
  chunker has an output-mode switch.
- **Evidence rule (deterministic, proposed).** Short text (at most 64 tokens) that is not
  table-shaped, not a long array, not a warning, not a progress bar, not an object repr, and not
  truncated. The stored text does not record whether an output came from `print` or from a final
  value, so the rule judges the text alone. It was designed from the principle and aggregate
  pattern counts, without reading individual outputs, and frozen before the validation sample was
  drawn. Its precision and recall against the owner's labels are reported separately.

## M4 ablation to record

Three arms, chosen by the ADR-010 rule on the dev split:

| Arm | Chunk text contains |
|---|---|
| A | no outputs (the M2 default) |
| B | all outputs |
| C | evidence-only outputs (the rule above) |

The arm is a parameter of the chunker, not a code change, so the arms differ only in what text
goes into the index. Chunk caps (256, 512, 1024) and the fixed-token baseline are compared as before.

## Impact

- ADR-004 text amended (see its heading and note). No other ADR changes.
- `raw_hash`, the gold-set cell identity, is unaffected: labels stay valid under every arm and cap.
- Chunk IDs depend on the output mode, since the chunk text differs.
