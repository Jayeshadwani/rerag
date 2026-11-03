# Proposal 003: what the M4 dev ablation implies for ADR-004, ADR-005/006 and ADR-011

Status: **approved by the owner, with the five decisions below.** Evidence: `results/retrieval_ablation_dev.json`
(54 dev queries; the test split has not been run).

The ablation was run with rules fixed before the run. Four of its outcomes touched locked ADRs, so
they were put to the owner. The owner decided as follows.

| # | Decision |
|---|---|
| 1 | **Reranker: off**, as the rule decided (no proven benefit; p95 over the 1 s budget). |
| 2 | **Hybrid RRF: adopted**, recorded here as a **post-hoc reclassification** (see section 2). Stage 4 (RRF k) was then run under the free rule. |
| 3 | **ADR-004: keep the structural chunker.** A secondary token-coverage metric is reported, not decided on. The negative result stays on record. |
| 4 | **ADR-011 amended:** abstention uses the top dense cosine similarity. |
| 5 | **Known limitation:** chunks cut at the embedding model's input limit, noted in the results and the README. |

## 1. Reranker: off

Under the final pipeline (hybrid, bge-base, k=10) the cross-encoder was not better and was far over the latency budget:

| Stage | Comparison | Metric | Mean delta | 95% CI | Free | Decision |
|---|---|---|---|---|---|---|
| 5 rerank | structural-512 \| bm25-code + dense:BAAI/bge-base-en-v1.5 + rrf10 + rerank vs structural-512 \| bm25-code + dense:BAAI/bge-base-en-v1.5 + rrf10 | recall@10 | -0.070 | [-0.168, +0.024] | no | reject: not free and the interval includes 0 (p95 total latency 1630 ms vs budget 1000 ms) |

## 2. Hybrid RRF: adopted (a post-hoc reclassification)

**What was pre-registered.** The M4 plan treated the hybrid as *not free* (it needs both retrievers), so it had to show an interval above 0 under ADR-010. It did not: the mean Recall@10 gain over dense alone was small and its interval included 0. Under the pre-registered rule it was **rejected** (that is the first dev result, commit `d316219`).

**What the owner decided, after seeing the dev results.** The hybrid's measured added latency is a few milliseconds, which meets ADR-010's definition of "free" (no latency or cost), and ADR-005 stated its hypothesis in advance: BM25 and dense fail on different queries, with identifiers favouring BM25 (BM25 1.000 vs dense 0.750 on the 8 identifier queries). Under the free rule the hybrid is adopted. This is a reclassification made **after** looking at the dev data, so the outcome is a decision and not a test; the interval still includes 0.

Measured latency of the hybrid over dense alone, in the two dev runs (single queries, warm, query embedding included, on a shared 4-vCPU container):

| Run | Dense only p50 / p95 (ms) | Hybrid (k=60) p50 / p95 (ms) | Added p50 / p95 (ms) |
|---|---|---|---|
| 1 (results at commit `d316219`) | 138.4 / 172.3 | 142.5 / 175.1 | +4.2 / +2.8 |
| 2 (current file, code at `2b3d953`) | 157.1 / 437.4 | 156.5 / 181.8 | -0.6 / -255.6 |

Run 2 is noisier: its dense-only p95 contains a slow outlier, so its p95 difference is not meaningful. The p50 differences are within about 5 ms in both runs.

Stages 3 and 4 as run under the free rule:

| Stage | Comparison | Metric | Mean delta | 95% CI | Free | Decision |
|---|---|---|---|---|---|---|
| 3 hybrid | structural-512 \| bm25-code + dense:BAAI/bge-base-en-v1.5 + rrf60 vs structural-512 \| dense:BAAI/bge-base-en-v1.5 | recall@10 | +0.023 | [-0.064, +0.111] | yes | ADOPT: free, the interval includes 0 and the mean is not negative (post-hoc: classified free by the owner (proposal 003); measured added latency -0.6 ms p50 / -255.6 ms p95) |
| 4 rrf k | structural-512 \| bm25-code + dense:BAAI/bge-base-en-v1.5 + rrf10 vs structural-512 \| bm25-code + dense:BAAI/bge-base-en-v1.5 + rrf60 | recall@10 | +0.007 | [-0.008, +0.028] | yes | ADOPT: free, the interval includes 0 and the mean is not negative |
| 4 rrf k | structural-512 \| bm25-code + dense:BAAI/bge-base-en-v1.5 + rrf30 vs structural-512 \| bm25-code + dense:BAAI/bge-base-en-v1.5 + rrf60 | recall@10 | +0.000 | [+0.000, +0.000] | yes | ADOPT: free, the interval includes 0 and the mean is not negative |

k=10 has the best mean and is chosen; k=30 returns rankings identical to k=60 on this data. A mean gain of +0.007 with an interval that includes 0 is the literal ADR-010 free rule, not evidence that 10 beats 60.

By query type (descriptive, small cells):

| Recall@10 by query type | code_identifier (n=8) | concept (n=36) | cross_notebook (n=4) | how_did_i (n=6) |
|---|---|---|---|---|
| structural-512 \| bm25-code | 1.000 | 0.607 | 0.750 | 0.667 |
| structural-512 \| dense:BAAI/bge-small-en-v1.5 | 0.750 | 0.716 | 0.562 | 0.667 |
| structural-512 \| dense:BAAI/bge-base-en-v1.5 | 0.750 | 0.788 | 0.562 | 1.000 |
| structural-512 \| dense:intfloat/e5-base-v2 | 0.938 | 0.780 | 0.562 | 0.500 |
| structural-512 \| bm25-code + dense:BAAI/bge-base-en-v1.5 + rrf60 | 1.000 | 0.787 | 0.625 | 0.833 |
| structural-512 \| bm25-code + dense:BAAI/bge-base-en-v1.5 + rrf10 | 1.000 | 0.797 | 0.625 | 0.833 |
| structural-512 \| bm25-code + dense:BAAI/bge-base-en-v1.5 + rrf30 | 1.000 | 0.787 | 0.625 | 0.833 |

## 3. ADR-004: the structural chunker is kept; the negative result stands

At an equal 3,072-token budget the structural chunker did not beat the fixed-token baseline on cell recall:

| Stage | Comparison | Metric | Mean delta | 95% CI | Free | Decision |
|---|---|---|---|---|---|---|
| 0a structural vs fixed | structural-256 \| bm25-code vs fixed-256 \| bm25-code | recall@tokens=3072 | -0.114 | [-0.204, -0.037] | yes | reject: the interval is entirely below 0 |
| 0a structural vs fixed | structural-512 \| bm25-code vs fixed-512 \| bm25-code | recall@tokens=3072 | -0.065 | [-0.163, +0.027] | yes | reject: free, but the mean is negative |
| 0a structural vs fixed | structural-1024 \| bm25-code vs fixed-1024 \| bm25-code | recall@tokens=3072 | -0.001 | [-0.116, +0.116] | yes | reject: free, but the mean is negative |

The secondary metric the owner asked for, token coverage (the share of the relevant cells' tokens that the budgeted chunks hold; a report, not a decision):

| Stage | Comparison | Metric | Mean delta | 95% CI | Free | Decision |
|---|---|---|---|---|---|---|
| 0a token coverage (report only) | structural-256 \| bm25-code vs fixed-256 \| bm25-code | tokcov@tokens=3072 | -0.012 | [-0.097, +0.062] | yes | report only: free, but the mean is negative |
| 0a token coverage (report only) | structural-512 \| bm25-code vs fixed-512 \| bm25-code | tokcov@tokens=3072 | +0.006 | [-0.097, +0.104] | yes | report only: free, the interval includes 0 and the mean is not negative |
| 0a token coverage (report only) | structural-1024 \| bm25-code vs fixed-1024 \| bm25-code | tokcov@tokens=3072 | +0.038 | [-0.084, +0.159] | yes | report only: free, the interval includes 0 and the mean is not negative |

By token coverage the two are indistinguishable at the decided cap, which is consistent with, but does not prove, the suspicion that part of the recall gap is partial credit for fixed windows touching many cells. The structural chunker is kept because statement and cell boundaries are needed for citations and for LLM context. The ADR-004 Result line records both results.

## 4. ADR-011: abstention on the top dense cosine similarity

ADR-011 calibrated the abstention threshold on the top reranker score. The reranker is off, and RRF scores depend only on ranks, so they cannot be thresholded. The amended text (in `PLAN.md` and `docs/adr/ADR-011.md`) says:

> **Abstention** (amended by proposal 003): if the top **dense cosine similarity** < tau, return "not in your notes" without calling the LLM. tau is calibrated on dev in M6 to maximise answer/abstain F1 using the unanswerable queries.

## 5. Known limitation: truncation at the embedding model's input limit

Known limitation: at chunk cap 512, 294 of 2310 chunks (12.7%) are longer than the input limit of BAAI/bge-base-en-v1.5 and are cut, so dense retrieval does not see their tails. The chunker is unchanged.

The same limitation is in the README. The chunker and the cap are unchanged.

## Impact

- `[retrieval.decided]` is now: chunk cap 512; BM25 (code tokenizer) plus `BAAI/bge-base-en-v1.5`; RRF k=10; no reranker.
- ADR-004, ADR-005 and ADR-006 Result lines are updated; ADR-011 is amended.
- `make eval-test` will run once, on the final pipeline, with dense-only and BM25-only as comparison rows in the same run. It has not been run.
