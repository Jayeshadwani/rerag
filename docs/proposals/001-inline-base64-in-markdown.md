# Proposal 001: inline base64 images in markdown cells

Status: **open, awaiting owner decision.** Nothing in this proposal is implemented.
ADR affected: ADR-003 (two-stage cleaning). The ADR is locked; this is a request to extend it.

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

## Decision needed

Approve the rule as described, approve it with changes, or reject it. Until then M1 ships
without it, and the stats it would produce are not reported as final.
