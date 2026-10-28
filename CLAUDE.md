# Instructions for the coding agent

You are implementing **NoteRecall** exactly as specified in `PLAN.md`. The owner will defend every line of this repo in a technical interview. Code that looks plausible but is not verified is worse than no code.

## 1. Authority

1. `PLAN.md` §2 (ADRs) is **locked**. You may not change a decision, a decision rule, a module boundary, or the dependency list.
2. If you find evidence that an ADR is wrong, write `docs/proposals/NNN-short-title.md` (problem, evidence with numbers or links, proposed change, impact), **stop work on that part**, and tell the owner. Do not implement the change.
3. Parameters marked **data-decided** may only be set by running the named experiment. The commit that changes the config value must reference the `results/*.json` file that justifies it, and the ADR's "Result" line must be filled in from that file.
4. Work one milestone (PLAN §3) per branch/PR, in order. Do not start M(n+1) until M(n)'s acceptance criteria pass. **M3 ends with a hard stop**: wait for the owner to write the gold set.

## 2. No convincing junk

- **No stubs in `src/`.** CI fails on `TODO`, `FIXME`, `XXX`, `NotImplementedError`, `...` as a function body, `pass` as a function body, and `except Exception:` / bare `except:`. If something can't be finished, it doesn't get merged.
- **No silent fallbacks.** Never catch an error and return a default that hides it. Fail loudly with a typed exception and a message.
- **No invented APIs.** Before using any third-party function, confirm it exists in the pinned version (read its source or run `python -c "import pkg, inspect; print(inspect.signature(pkg.X))"`). Libraries in this project have changed APIs between major versions (`fsrs` notably).
- **No invented numbers.** Any metric, latency, or cost in README/docs/PR text must come from a file in `results/`, rendered by `scripts/render_results.py`. Never type a number by hand.
- **No invented model names.** Confirm Hugging Face models exist and record the revision hash in config.
- **No frameworks** beyond PLAN §5 (no LangChain, LlamaIndex, Haystack, DSPy, vector DBs). New dependency → proposal.

## 3. Tests are the definition of done

- Every public function in `src/` has tests. Coverage ≥ 85% on `src/` (CI-enforced); `retrieval/`, `eval/`, `generate/citations.py`, `srs/` aim for ~100%.
- Tests must be able to **fail**. A test that passes against a broken implementation is junk. For core math (RRF, metrics, bootstrap, planner), include hand-computed expected values in the test with the arithmetic in a comment.
- Use property-based tests (hypothesis) for the chunker invariants listed in M2.
- **No network in the default test run.** LLM calls go through `llm/fake.py`. Live tests are marked `@pytest.mark.live` and skipped unless `NOTERECALL_LIVE=1`.
- Time goes through `srs/clock.py`; tests use a fixed UTC clock. FSRS fuzzing is disabled in tests.
- **Never weaken a test, threshold, or acceptance criterion to make it pass.** If a criterion can't be met, report the measured value honestly in the PR and stop.

## 4. Evaluation integrity

- Never create, edit, reformat, or "fix" files under `eval_data/gold/`. The owner writes them. You may build `noterecall label` and `noterecall gold validate`.
- All tuning uses the **dev** split. The **test** split is run once per milestone via `make eval-test`, and its output is committed unchanged.
- Apply the ADR-010 decision rule mechanically. Report every ablation row, including ones that hurt.
- `results/*.json` must contain: metric values, per-query scores, CI, config snapshot, model revisions, prompt versions, git SHA. Metrics only — never note content or query text from private data.

## 5. Code standards

- Python 3.12, `uv`, `ruff` (lint + format), `mypy --strict`. Type every function.
- Pydantic models at every boundary (config, DB rows ↔ domain, LLM I/O). LLM outputs are parsed into a schema or rejected.
- Pure functions for ranking, fusion, metrics, planning. I/O at the edges only.
- Prompts live in `src/noterecall/llm/prompts/*.md`, never inline strings. Their content hash is the prompt version.
- Structured logging (`structlog`): every LLM call logs model, prompt version, input/output tokens, latency, cache hit.
- Functions ≤ ~40 lines; modules have a one-paragraph docstring stating which ADR they implement.

## 6. Data safety

- `data/`, `.env`, and any real notebooks are git-ignored. Only synthetic fixtures under `tests/fixtures/` are committed.
- Secret scrubbing (ADR-003) runs before anything is stored or sent to an LLM. There is a test proving a planted fake key never reaches SQLite or the LLM fake's recorded inputs.

## 7. Every PR description must include

1. Milestone and which acceptance criteria are met (checklist).
2. Pasted output of `make check` (lint, types, tests, coverage).
3. For eval milestones: the rendered results table and the ADR-010 decision for each ablation.
4. "Things I could not verify" — be explicit.
5. A short "how this works" paragraph per new module, written so the owner can explain it aloud.

## 8. Make targets (create in M0)

`make setup` · `make check` (ruff + mypy + pytest + coverage + banned-patterns) · `make sync` · `make eval` (dev) · `make eval-test` · `make results` (render README tables) · `make mutation`
