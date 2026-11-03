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

## Known limitations

- At chunk cap 512, 294 of 2310 chunks (12.7%) are longer than the input limit of BAAI/bge-base-en-v1.5 and are cut, so dense retrieval does not see their tails. The chunker is unchanged. (Source: `results/retrieval_ablation_dev.json`, the `notes` field and the truncation table; rendered by `scripts/render_results.py`.)
