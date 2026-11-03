# Proposal 004: the LLM provider is OpenAI, not Anthropic

Status: **approved by the owner.**
Changes: PLAN assumption A2, the PLAN section 5 dependency list, the name of the real client module,
and one phrase of ADR-007. The `LLMClient` protocol and the blurb method are unchanged.

## Decision

All API-driven LLM tasks use the **OpenAI API**. For M5 (contextual blurbs and the markdown clean):

- Model: `gpt-5-mini` for both tasks, with the lowest reasoning effort the API allows. Model names stay in config only.
- Caching: OpenAI caches a shared prompt prefix automatically. The blurb prompt therefore puts the
  notebook first and the chunk last, and the cost report shows the cached-token counts from the usage fields.
- Prices (owner-supplied, USD per 1M tokens): input 0.25, cached input 0.025, output 2.00. Reasoning
  tokens are billed as output and are logged separately in the cost report.
- Spend cap: `max_usd = 10`, enforced from the usage of the calls actually made.
- Consent: the owner agreed to send scrubbed notebook text to the OpenAI API (ADR-003 scrubbing has
  already run). `OPENAI_API_KEY` is read from the environment and is never printed, logged or committed.

## What changes

| Where | Change |
|---|---|
| PLAN A2 | "LLM = OpenAI API", amended by this proposal. |
| PLAN section 5 | `anthropic` replaced by `openai`. The pin is `openai==3.42.1`. |
| PLAN module list | `llm/anthropic_client.py` becomes `llm/openai_client.py`. |
| ADR-007 | The phrase "Anthropic's method, with prompt caching on the notebook" becomes "the Contextual Retrieval method, with the notebook as a shared prefix that the provider caches". The citation of Anthropic's report as evidence in ADR-006 and the reference list stays: it is where the method comes from. |
| Config | New `[llm]` section (models, reasoning effort, token caps, timeout, retries, concurrency, prices, `max_usd`, cache path). |
| `CLAUDE.md` | No change needed: it names no provider (it refers to `llm/fake.py` and PLAN section 5). |

## What does not change

- `LLMClient.complete_json(prompt, schema) -> model` and the rest of the `llm/` layout (fake, cache, versioned prompt files).
- ADR-007's method: the blurb is a few sentences situating the chunk, generated with the whole notebook as a shared prefix, prepended to the indexed text.
- Grounding, citation validation and abstention (ADR-011) are provider-independent.

## Verified before use (2026-11-03)

- The model exists: `models.retrieve("gpt-5-mini")` returns it (a metadata call; no generation, no tokens).
- The installed SDK (`openai==3.42.1`) offers `responses.parse` for structured output, a `reasoning.effort`
  field (its type lists `none`, `minimal`, `low`, `medium`, `high`, `xhigh`, `max` and says not every model supports every
  value), `prompt_cache_key`, and a usage object with `input_tokens_details.cached_tokens`,
  `input_tokens_details.cache_write_tokens` and `output_tokens_details.reasoning_tokens`.
- `tiktoken` maps `gpt-5-mini` to `o200k_base`, which the `--dry-run` token estimate uses.

## Not yet verified

- Which `reasoning.effort` values `gpt-5-mini` accepts. The default in config is `minimal`; the first real call checks it
  and a rejection fails loudly. Nothing is silently replaced.
- Whether tokens written to the cache (`cache_write_tokens`) are billed differently from ordinary input. The owner's price list has no
  such price, so the cost report treats them as ordinary input and also reports their count.
- Real cost and cache hit rates; those come from the usage fields of the real run.
- Sampling settings are not set: reproducibility comes from the response cache, not from the model.
