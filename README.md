# NoteRecall

See PLAN.md for the architecture and CLAUDE.md for contributor rules.

## Not tested (owner decision)

The LLM contextual blurb (ADR-007) and the LLM markdown clean (ADR-003) were **not tested**. They are
not run and ship off. This is the owner's decision on cost and benefit, recorded in
`docs/proposals/005-skip-llm-preprocessing.md`; it is not a result, and nothing here says they do or do
not help.

## Context header (M5, dev split)

The deterministic header (`{notebook} > {headings}` in front of each indexed chunk) is on, as ADR-007
requires. Its on/off row on the dev split, rendered from `results/header_ablation_dev.json` by
`scripts/render_results.py`:

| Configuration | Recall@5 | Recall@10 | MRR@10 | nDCG@10 |
|---|---|---|---|---|
| M4 pipeline, header off | 0.670 | 0.819 | 0.613 | 0.584 |
| M4 pipeline, header on | 0.713 | 0.860 | 0.672 | 0.647 |

Recall@10 difference +0.042, 95% CI [-0.029, +0.117] (paired bootstrap, 54 dev queries). The interval
includes 0, so this dev run does not establish a gain; the row is reported, not decided on. The header
also lengthens the indexed text: 344 of 2310 chunks are cut at the dense model's input limit with it, 294
without.

## Grounded Q&A (M6, dev split)

`noterecall ask "<question>"` answers from your notes with numbered sources, or says it is not in them.
Every claim must cite note IDs that were in the context; the validator rejects unknown IDs, claims with no
citation and empty answers, retries once with the problems listed, and otherwise refuses (ADR-011). The
validator checks structure only: whether a cited note really supports the claim is not measured yet (M7).

**Stage 1 abstention refuses nothing on this corpus.** Proposal 006 pre-registered a rule: refuse without
calling the model only if the top dense cosine and the normalized top BM25 score are both below thresholds
chosen on dev, subject to refusing at most 5% of the answerable queries. On the dev split no threshold pair
catches any of the 6 unanswerable queries within that bound (`results/abstention_calibration_dev.json`;
catching one needs at least 5 false refusals out of 54). In a single-domain corpus, unanswerable queries
that are near the topic of the notes look like answerable ones to cosine and to BM25. The selected
thresholds therefore refuse nothing, the bound was kept, and **abstention rests on the model's own
`insufficient` flag**. The ADR-011 Result line records this.

Dev results, rendered from `results/generation_dev.json` by `scripts/render_results.py`:

| Queries | n | Answered | Refused | by stage 1 | by the model | invalid twice |
|---|---|---|---|---|---|---|
| answerable | 54 | 54 | 0/54 | 0 | 0 | 0 |
| unanswerable | 6 | 4 | 2/6 | 0 | 2 | 0 |

First-attempt citation validity was 60/60 with no retries and 0 uncited claims out of 205; 45 of the 54
answered answerable queries cite a chunk that contains a gold-relevant cell. With so few unanswerable
queries (6 dev, 3 test) these counts are coarse. The dev run cost $0.0547.

## Known limitations

- At chunk cap 512, 294 of 2310 chunks (12.7%) are longer than the input limit of BAAI/bge-base-en-v1.5 and are cut, so dense retrieval does not see their tails. The chunker is unchanged. (Source: `results/retrieval_ablation_dev.json`, the `notes` field and the truncation table; rendered by `scripts/render_results.py`.)
