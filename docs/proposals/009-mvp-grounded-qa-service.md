# Proposal 009: the MVP is grounded Q&A as a served service; M7 and M8 are deferred

Status: **approved by the owner.**
Changes: the product scope (PLAN section 0 pitch, milestone table, module layout, section 5
dependencies), the way the dense model is served (ONNX instead of torch), and adds a FastAPI service,
a Streamlit UI and the deployment files. No locked ADR decision changes; ADR-012, ADR-013 and ADR-014 are
**deferred**, not altered.

## Decision

The MVP is **grounded Q&A over the owner's notes: sync and ask**, answered by `gpt-5-mini` (config only,
as before). It is served by two web services (an API and a UI) on Render's free tier, because the owner
wants to show it. M7 (concepts, flashcards, the calibrated judge) and M8 (FSRS scheduler, planner) are
deferred: nothing of them is built now, and nothing of the MVP depends on them.

## What changes

| Where | Change |
|---|---|
| Scope | The pipeline of M0 to M6 plus a served API and UI. The only LLM calls: the grounded answer, and (only when there is history) a small rewrite of a follow-up question into a standalone one. |
| PLAN section 3 | M7 and M8 are marked deferred; M9 is re-scoped as the MVP (API, UI, deployment, generated README, mutation testing on `fusion.py`, `metrics.py`, `citations.py` and the rate limiter). |
| Dense model at serve time | `BAAI/bge-base-en-v1.5` at the same pinned revision, same query instruction, run with `onnxruntime` (CPU, fp32, **no quantization**) from the `onnx/model.onnx` of that same revision. torch and sentence-transformers stay for offline evaluation only. |
| Images | The API image has no torch and no sentence-transformers. The UI image has no torch, no onnxruntime and no notes. |
| ADR-012, ADR-013, ADR-014 | Deferred, text unchanged otherwise. The known cases for the future judge (q081 claim 1, q033) are recorded in proposal 006. |

## Dependency changes, exact pins

Dependencies move from one list into PEP 735 groups in `pyproject.toml`, so that each image installs only
its own group (`uv export --only-group ...`). Versions are in `uv.lock`; these are the pins.

| Group | Package | Pin | Why |
|---|---|---|---|
| `core` | bm25s, numpy, openai, pydantic, pydantic-settings, pyyaml, structlog | unchanged pins (0.3.16, 2.5.4, 3.42.1, 2.14.1, 2.16.2, 6.0.3, 26.1.0) | retrieval, generation, config, logging |
| `ingest` | nbformat, tiktoken, typer | unchanged (5.11.1, 0.14.0, 0.27.3) | sync, chunking, CLI; build step only, not in the API image |
| `onnx` | **onnxruntime** | **1.31.1** | runs the ONNX embedder (CPU) |
| `onnx` | **tokenizers** | **0.23.3** | the model's own `tokenizer.json`, no transformers needed (already a transitive dependency of the torch stack) |
| `build` | **huggingface-hub** | **2.4.0** | fetches the pinned ONNX files when an image is built; not in the running API |
| `api` | **fastapi** | **0.162.0** | the service |
| `api` | **uvicorn** | **0.54.0** | the server (no extras) |
| `ui` | **streamlit** | **1.67.0** | the UI |
| `ui` | **requests** | **2.34.2** | the UI's HTTP client to the API (already a dependency of streamlit; pinned explicitly) |
| `eval` | sentence-transformers, torch | unchanged (6.2.0, 2.15.0) | offline evaluation and the parity check only; never in an image |
| `dev` | **httpx** | **0.28.1** | starlette's test client for the API tests |
| `dev` | **mutmut** | **3.8.1** | mutation testing (it was in the PLAN's dev list from the start) |
| `dev` | hypothesis, mypy, pytest, pytest-cov, ruff | unchanged (6.170.13, 2.4.0, 9.2.0, 7.1.0, 0.17.3) | |

No HTTP client is added to the API for its own use: the OpenAI SDK carries its own. Starlette 1.10.0 comes
in with fastapi. `fsrs` is not added (M8 is deferred).

## ONNX parity check (a gate, no LLM calls)

The M5 dev retrieval evaluation (the decided hybrid with the context header) is run with the torch
embedder and with the ONNX embedder on the dev split (`noterecall eval --suite onnx-parity`,
`results/onnx_parity_dev.json`). It reports Recall@5, Recall@10, MRR@10 and nDCG@10 for both, a paired
bootstrap interval on the Recall@10 difference, and the maximum absolute difference between the two
runtimes' cosine similarities over every dev query and every chunk.
**Rule: if the ONNX dev Recall@10 is lower than the torch one by more than 0.01, work stops and the
result is reported; the ONNX embedder is not deployed.**

## The service

- `POST /ask {question, history}` returns `{status, refusal_reason, standalone_question, claims[{text,
  citations}], sources[{id, notebook, heading_path, cell_index, text}]}`; `GET /health`. The indexes (chunks,
  embeddings, BM25) are loaded once at startup. `OPENAI_API_KEY` comes only from the environment and is never
  logged or returned.
- **Conversation memory** lives in the browser session only (the last 5 turns; nothing is stored on the
  server). With history, one small call rewrites the follow-up into a standalone question; then the normal
  pipeline runs (retrieve, grounded answer, refusal). History is never evidence: only retrieved chunks can be
  cited. Without history there is no rewrite call. An unusable rewrite is an error, not a silent fallback.
- **Rate limit:** 15 questions per end-user IP per UTC day, in memory (it resets when the service restarts).
  The UI forwards the end user's IP in a header together with a shared secret (`UI_API_SECRET`); the API
  trusts that header only when the secret matches (constant-time comparison), otherwise it limits by the
  caller's own IP (the first `X-Forwarded-For` entry from Render's proxy). The 16th question gets HTTP 429.
- The UI holds no business logic: it sends the question and history, and renders the answer.

## Deployment (Render free, two services)

Two multi-stage Dockerfiles and one `render.yaml`. The notes and the built index exist only in the **API
image**, built from the private `rerag-data` repository at build time with a token that is never copied into
the final stage. They are in no public image, registry or repository. The owner does the deployment from
written step-by-step instructions. Image size, peak memory and cold-start time are measured locally and
reported against the 512 MB limit; the 750 free instance hours are shared by the two services.

## Risks and limits, stated now

- **The public API exposes note content.** Every answer returns the cited chunks' text. The only protection
  is the rate limit (15 per IP per day); there is no authentication. Anyone with the URL can read parts of the
  notes. Rotating IPs also lets someone spend the owner's OpenAI budget (about $0.001 per question). Both are
  accepted by the owner for the MVP and written in the README.
- **Memory.** The fp32 ONNX model file is 436 MB; Render's free limit is 512 MB. Whether the API fits is a
  measurement, reported either way; no quantization (the owner's rule), so a failure is reported, not worked
  around.
- **Not verified here:** the Render build itself (build secrets, memory limit, cold start), the exact
  behaviour of Render's proxy headers, and how the pinned Streamlit exposes request headers (checked in the
  code of the pinned version and tested, but not on Render).
- The rate limit and the in-memory counters reset on restart or redeploy, and are per instance.

## Deferred

M7 (concept extraction with `concepts.yaml` review, grounded flashcards, the judge calibrated against the
owner's labels) and M8 (FSRS scheduling, the daily planner, the review CLI). The plan for M7 stays in the
conversation record and in proposal 006's notes (q081 claim 1 and q033 as known cases for the judge).

## Result: ONNX parity (dev split, no LLM call)

From `results/onnx_parity_dev.json` (`noterecall eval --suite onnx-parity --split dev`): the decided
pipeline (structural cap 512, header, BM25 + dense, RRF k=10) with the torch embedder and with the ONNX
embedder, same chunks, same queries.

| metric | torch | ONNX |
|---|---|---|
| Recall@10 | 0.8604 | 0.8604 |
| MRR@10 | 0.6720 | 0.6720 |
| nDCG@10 | 0.6470 | 0.6470 |

Maximum absolute cosine difference between the torch and ONNX vectors of the same texts: 0.000001.
Decision: **pass** (the rule was: stop if dev Recall@10 drops by more than 0.01; it did not drop).
