# NoteRecall

See PLAN.md for the architecture and CLAUDE.md for contributor rules.

## Known limitations

- At chunk cap 512, 294 of 2310 chunks (12.7%) are longer than the input limit of BAAI/bge-base-en-v1.5 and are cut, so dense retrieval does not see their tails. The chunker is unchanged. (Source: `results/retrieval_ablation_dev.json`, the `notes` field and the truncation table; rendered by `scripts/render_results.py`.)
