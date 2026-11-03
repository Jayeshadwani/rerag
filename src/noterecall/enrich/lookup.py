"""Blurbs for the evaluation, read back from the response cache. Implements ADR-007 (blurbs).

The evaluation never calls the model. It rebuilds, for each chunk, the exact prompt that
`noterecall enrich blurbs` sent and looks its answer up in the response cache. A chunk whose
answer is missing is an error (`MissingEnrichmentError`) that says how to generate it, so a
half-finished enrichment can never be scored as if it were complete.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from noterecall.enrich.contextual import MissingEnrichmentError, notebook_text
from noterecall.enrich.run import Job, blurb_groups, read_blurbs
from noterecall.llm.cache import ResponseCache
from noterecall.llm.prompts import PromptTemplate
from noterecall.models import Cell, Chunk


@dataclass(frozen=True)
class BlurbSetup:
    """Everything that, with the notebook and chunk, determines a blurb prompt."""

    template: PromptTemplate
    model: str
    effort: str
    max_output_tokens: int


def blurb_job_groups(
    setup: BlurbSetup, cells_by_path: Mapping[str, Sequence[Cell]], chunks: Sequence[Chunk]
) -> list[list[Job[str]]]:
    """One group per notebook (in order of first appearance), with that notebook's chunks."""
    order: list[str] = []
    per_path: dict[str, list[Chunk]] = {}
    for c in chunks:
        if c.notebook_path not in per_path:
            order.append(c.notebook_path)
            per_path[c.notebook_path] = []
        per_path[c.notebook_path].append(c)
    by_notebook = [(p, notebook_text(cells_by_path[p]), per_path[p]) for p in order]
    return blurb_groups(by_notebook, setup.template, setup.model, setup.max_output_tokens)


class CachedBlurbs:
    """A `BlurbSource`: chunk id -> blurb, read from the cache for the original or cleaned text."""

    def __init__(
        self,
        cache: ResponseCache,
        setup: BlurbSetup,
        original: Mapping[str, Sequence[Cell]],
        cleaned: Mapping[str, Sequence[Cell]] | None,
    ) -> None:
        self._cache = cache
        self._setup = setup
        self._original = original
        self._cleaned = cleaned

    def __call__(self, clean: bool, chunks: Sequence[Chunk]) -> dict[str, str]:
        cells = self._cleaned if clean else self._original
        if cells is None:
            raise MissingEnrichmentError(
                "the cleaned notebooks were not supplied; run `noterecall enrich clean`"
            )
        groups = blurb_job_groups(self._setup, cells, chunks)
        jobs = [job for group in groups for job in group]
        return read_blurbs(jobs, self._cache, self._setup.effort)
