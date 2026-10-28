# Proposal 001: inline base64 images in markdown cells

Status: **approved by the owner, as written.** Implemented on branch `m1-ingest`.
ADR affected: ADR-003 (two-stage cleaning), amended in `PLAN.md` and `docs/adr/ADR-003.md`.

## Problem

ADR-003 drops "image/HTML/base64 **outputs**". The first real-corpus run showed that the
bulk of the base64 is not in outputs but in **markdown cell source**, as inline data URIs
(`![...](data:image/png;base64,...)`). The cleaner as specified does not touch them, so they
are scrubbed for secrets, stored, tokenised, and would later be chunked, embedded and indexed.

## Evidence

Measured on the owner's 53-notebook corpus after every ADR-003 rule as written had run.
These are aggregate counts from an uncommitted preview run (byte-level tokenizer, database in
a scratch directory), because the real `results/corpus_stats.json` cannot be produced yet: the
`cl100k_base` vocabulary download is blocked by this environment's network policy. They
should be replaced by numbers from a results file once `noterecall stats` runs.

| Quantity | Value |
|---|---:|
| Notebooks / raw cells / cells kept | 53 / 6,432 / 6,340 |
| Characters stored (source + kept output text) | 94,242,942 |
| Inline base64 data URIs (291 PNG, 11 JPEG, 4 GIF) | 306 |
| Markdown cells containing at least one | 300 (code cells: 0) |
| Notebooks containing at least one | 29 of 53 |
| Characters inside those data URIs | 91,868,375 (97.5% of all stored text) |
| Characters stored if they were removed | 2,374,567 |
| Median cell length after removal / longest cell | 166 / 11,475 characters |
| Longest cell as it is today | 1,498,204 characters |
| Long base64-like runs that are *not* inside a data URI | 0 |

Per-notebook cleaned size is dominated by this: the largest notebook is about 11.2 million
characters against a median of about 129 thousand. The database is 97 MB, of which these
payloads are almost all.

## Proposed change

Add one deterministic rule to `config/cleaning_rules.yaml`, markdown cells only: replace the
payload of an inline data URI (`data:<mime>;base64,<payload>`) with a short fixed placeholder
and keep the surrounding markdown (image alt text and link syntax stay). Count it as
`inline_data_uri_removed` (and characters removed) in the removed-by-rule report. Add a
synthetic fixture cell built from the pattern and unit tests, as for the instructor-note rule.

ADR-003's first bullet would read "drop image/HTML/base64 outputs **and inline base64 data
URIs in markdown cells**". The flowchart box in PLAN section 1 already lists "images, base64"
under deterministic cleaning, so this matches the stated intent.

## Alternatives

1. Do nothing: ~97.5% of stored text is image payload; chunk caps, embeddings and BM25 would
   be built on it. Not recommended.
2. Strip at chunking time (M2) instead: the database would still hold and hash the payloads,
   `noterecall stats` would keep reporting inflated token counts, and every consumer would have
   to remember to strip. Not recommended.

## Impact

- `raw_hash` (the gold-set cell identity) is computed before cleaning, so labels are unaffected.
- The pipeline hash changes, so the next `sync` re-processes every notebook once.
- Token and cell-length statistics change by roughly the 97.5% above; M2 chunk-size decisions
  should be made on the cleaned numbers.
- Risk: a data URI that carries non-image content would also be shortened. In this corpus all
  306 are images.

## Decision

Approved as written. Implementation notes:

- Rule `inline_data_uri` in `config/cleaning_rules.yaml`; placeholder `[removed]`; counted as
  `inline_data_uri_removed` and `inline_data_uri_chars_removed`.
- The pattern requires a word boundary before `data:` so that a word such as `metadata:` cannot
  match. This is a precision guard inside the approved rule, not a change to it.
- Data URIs are replaced before secret scrubbing, because the payload is discarded anyway and
  random base64 can look like a secret; everything that remains is still scrubbed.
- Applies to markdown cells only. A data URI inside a code cell is code and is left alone.

## Follow-up decision

The first run on the real corpus showed that many markdown cells are nothing but an image. After
the payload is replaced, such a cell holds only `![alt](data:...;base64,[removed])` (or an
`<img>` tag), which has no retrievable text. The owner decided that a markdown cell whose only
content after cleaning is image placeholders, with or without alt text, counts as empty and is
dropped under the existing empty-cell rule (no ADR text change). It is counted on its own line,
`image_only_cells_dropped`, separate from `cells_dropped_empty`. A cell is only treated this way
if a data URI was actually removed from it; a cell that merely contains the text `[removed]` is
kept. Cells with any other text, such as a caption, are kept.
