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

One answer-prompt revision was tried on dev under a rule written down first (proposal 006, addendum 1):
adopt it only if answerable queries refused stay at most 2/54 and unanswerable queries refused strictly
improve on 2/6. It met both, so the shipped prompt is the revision (it says to answer only if the notes
directly contain the answer). Both prompts side by side, rendered from `results/generation_dev.json`
(original) and `results/generation_dev_prompt2.json` (shipped) by `scripts/render_results.py`:

| Measure | original `f715ca2be4cf` | shipped `ee9346d6e271` |
|---|---|---|
| answerable queries refused | 0/54 | 0/54 |
| unanswerable queries refused | 2/6 | 3/6 |
| first attempt valid | 60/60 | 60/60 |
| retried | 0/60 | 0/60 |
| uncited claims, first attempt | 0/205 | 0/169 |
| gold-hit (answered answerable citing a gold chunk) | 45/54 | 45/54 |
| claims in all answers, first attempt | 205 | 169 |
| input / output tokens | 119804 / 12386 | 120344 / 11080 |
| cost USD | 0.0547 | 0.0522 |

The improvement is one unanswerable query out of six, so it is a pre-registered choice and not statistical
evidence; the revision also gives shorter answers (fewer claims). The other 3 unanswerable dev queries
are still answered, with citations that pass the structural check. Whether those answers are supported by
the cited notes is not measured yet (M7). Gold-hit says that 9 of the 54 answered answerable queries cite
no gold-labelled cell, which does not by itself make an answer wrong. The test split has not been run.

## Known limitations

- At chunk cap 512, 294 of 2310 chunks (12.7%) are longer than the input limit of BAAI/bge-base-en-v1.5 and are cut, so dense retrieval does not see their tails. The chunker is unchanged. (Source: `results/retrieval_ablation_dev.json`, the `notes` field and the truncation table; rendered by `scripts/render_results.py`.)
