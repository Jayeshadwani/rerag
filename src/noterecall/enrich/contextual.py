"""LLM contextual blurbs. Implements ADR-007 (the blurb ablation; the method is provider-agnostic).

For each chunk the model sees the whole notebook (the shared prefix, identical for every chunk of
that notebook so the provider can cache it) and then the chunk, and writes one or two sentences
situating the chunk in the notebook. The sentences are indexed in front of the chunk text. The
prompt contains only notebook text: never a query and never a gold label. A blurb is a function of
(model, prompt version, notebook text, chunk text), so answers are looked up in the response cache
by recomputing the prompt; a missing one is an error that says how to generate it.
"""

from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict, Field

from noterecall.llm.cache import ResponseCache, cache_key
from noterecall.llm.client import LLMOutputError, Prompt
from noterecall.llm.prompts import PromptTemplate
from noterecall.models import Cell, Chunk

MAX_BLURB_CHARS = 600


class Blurb(BaseModel):
    """The model's answer for one chunk."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    context: str = Field(min_length=1, max_length=MAX_BLURB_CHARS)


class MissingEnrichmentError(Exception):
    """Raised when an enrichment the evaluation needs has not been generated yet."""


def notebook_text(cells: Sequence[Cell]) -> str:
    """The notebook as the model sees it: every non-blank cell source in order, no outputs."""
    return "\n\n".join(c.source.strip("\n") for c in cells if c.source.strip())


def blurb_prompt(
    template: PromptTemplate, model: str, max_output_tokens: int, notebook: str, chunk: Chunk
) -> Prompt:
    """The blurb prompt for one chunk: notebook first, chunk last."""
    return template.render(model, max_output_tokens, notebook=notebook, chunk=chunk.text)


def read_blurb(cache: ResponseCache, effort: str, prompt: Prompt) -> str | None:
    """The cached blurb for a prompt, or None if it was never generated."""
    stored = cache.get(cache_key(prompt, effort))
    if stored is None:
        return None
    try:
        return Blurb.model_validate_json(stored.text).context
    except ValueError as exc:
        raise LLMOutputError(f"a cached blurb no longer parses: {exc}") from exc
