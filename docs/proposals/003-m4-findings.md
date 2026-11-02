# Proposal 003: what the M4 dev ablation implies for ADR-004, ADR-005/006 and ADR-011

Status: **open. Nothing here is implemented; no ADR text was changed beyond filling in "Result" lines.**
Evidence: `results/retrieval_ablation_dev.json` (54 dev queries; test split untouched).

The pre-registered rules were applied mechanically, and the decided pipeline is **dense-only
(`BAAI/bge-base-en-v1.5`), chunk cap 512, no reranker**. Three findings touch locked ADRs, so per
`CLAUDE.md` section 1 they are put to the owner instead of being acted on.

## 1. ADR-004: the structural chunker did not beat the fixed-token baseline

ADR-004 says to compare the structural chunker with the fixed baseline at an equal retrieved-token
budget. At a 3,072-token budget (BM25 with the code tokenizer as the probe), structural minus fixed:

| Stage | Comparison | Metric | Mean delta | 95% CI | Free | Decision |
|---|---|---|---|---|---|---|
| 0a structural vs fixed | structural-256 \| bm25-code vs fixed-256 \| bm25-code | recall@tokens=3072 | -0.114 | [-0.204, -0.037] | yes | reject: the interval is entirely below 0 |
| 0a structural vs fixed | structural-512 \| bm25-code vs fixed-512 \| bm25-code | recall@tokens=3072 | -0.065 | [-0.163, +0.027] | yes | reject: free, but the mean is negative |
| 0a structural vs fixed | structural-1024 \| bm25-code vs fixed-1024 \| bm25-code | recall@tokens=3072 | -0.001 | [-0.116, +0.116] | yes | reject: free, but the mean is negative |

Structural is not better at any cap, and clearly worse at 256. Possible reasons, none tested:

- Many structural chunks are far below the cap (`results/chunk_stats.json`), so a fixed budget
  buys fewer, more fragmented pieces than fixed windows give.
- **The metric may favour fixed windows.** A chunk counts as covering a cell if it holds any part
  of it (`cell_hashes` lists every cell a chunk overlaps). Fixed windows cut far more cells than
  the structural chunker does (the "Cells split" column of `results/chunk_stats.json`), so a
  window holding a few tokens of a relevant cell scores a hit, and the same cell can score in
  two neighbouring windows.
- BM25 was the probe. Dense retrieval might behave differently.

Options for the owner (no preference stated by the agent): (a) keep structural, as ADR-004 locks,
and report the negative result; (b) re-score with a stricter coverage rule (for example a window
covers a cell only if it holds at least half of the cell's tokens), which changes the ADR-009
definition of "a chunk is relevant if it contains a relevant cell"; (c) propose fixed windows
with cell-boundary snapping as a new chunker.

## 2. ADR-005: the hybrid was not adopted, though BM25 wins on identifier queries

Hybrid RRF (k=60) against the better single retriever (dense) had a mean Recall@10 gain whose
interval includes 0, and the hybrid is not free, so ADR-010 rejects it:

| Stage | Comparison | Metric | Mean delta | 95% CI | Free | Decision |
|---|---|---|---|---|---|---|
| 3 hybrid | structural-512 \| bm25-code + dense:BAAI/bge-base-en-v1.5 + rrf60 vs structural-512 \| dense:BAAI/bge-base-en-v1.5 | recall@10 | +0.023 | [-0.064, +0.111] | no | reject: not free and the interval includes 0 |
| 5 rerank | structural-512 \| dense:BAAI/bge-base-en-v1.5 + rerank vs structural-512 \| dense:BAAI/bge-base-en-v1.5 | recall@10 | -0.023 | [-0.094, +0.046] | no | reject: not free and the interval includes 0 (p95 total latency 1868 ms vs budget 1000 ms) |

By query type (descriptive, 54 queries, small cells):

| Recall@10 by query type | code_identifier (n=8) | concept (n=36) | cross_notebook (n=4) | how_did_i (n=6) |
|---|---|---|---|---|
| structural-512 \| bm25-code | 1.000 | 0.607 | 0.750 | 0.667 |
| structural-512 \| dense:BAAI/bge-small-en-v1.5 | 0.750 | 0.716 | 0.562 | 0.667 |
| structural-512 \| dense:BAAI/bge-base-en-v1.5 | 0.750 | 0.788 | 0.562 | 1.000 |
| structural-512 \| dense:intfloat/e5-base-v2 | 0.938 | 0.780 | 0.562 | 0.500 |
| structural-512 \| bm25-code + dense:BAAI/bge-base-en-v1.5 + rrf60 | 1.000 | 0.787 | 0.625 | 0.833 |
| structural-512 \| dense:BAAI/bge-base-en-v1.5 + rerank | 0.688 | 0.795 | 0.562 | 0.833 |

BM25 with the code tokenizer is perfect on the identifier queries, and the hybrid keeps that while
matching dense on concept queries. With 8 identifier and 4 cross-notebook queries on dev, the overall
interval cannot separate them. Options: (a) accept dense-only per the rule; (b) keep the hybrid as
the shipped default on the strength of ADR-005's design rationale, which departs from the
pre-registered ADR-010 rule and should be an explicit owner decision; (c) re-test after more gold
queries exist.

## 3. ADR-011: abstention is defined on a reranker score, and the reranker is off

ADR-011 calibrates the abstention threshold tau on the top **reranker** score. The reranker was
rejected (ADR-006 says it then ships off), so M6 needs a different score. Options: (a) calibrate
tau on the top dense cosine similarity; (b) run the reranker only to decide abstention (its
latency is shown in the dev table above); (c) revisit the reranker decision. This is for M6 and
needs the owner's choice before then.

## Impact

None yet. M5 and M6 should wait for the owner's answers to 2 and 3; 1 only affects reporting.
