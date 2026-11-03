"""Running the LLM steps: blurbs for chunks and the clean for markdown cells (ADR-003, ADR-007).

Jobs are grouped by notebook. Within a group the first job runs alone, so it writes the provider's
prefix cache, and the rest run on a small thread pool. A job whose answer is unusable
(`LLMOutputError`) is recorded as a failure with its reason; any other error (a failed call, the
spend cap) stops the run, because continuing would hide it. Finished answers are in the response
cache, so a stopped run resumes where it left off without paying again.
"""

import threading
from collections import Counter
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import NamedTuple

from pydantic import BaseModel, ConfigDict

from noterecall.enrich.contextual import Blurb, MissingEnrichmentError, blurb_prompt
from noterecall.ingest.llm_clean import CleanedCell
from noterecall.llm.cache import ResponseCache, cache_key
from noterecall.llm.client import LLMClient, LLMOutputError, Prompt
from noterecall.llm.prompts import PromptTemplate
from noterecall.models import Cell, CellType, Chunk


class Job[K](NamedTuple):
    """One LLM call: what it is for, and its prompt."""

    key: K
    prompt: Prompt


class RunSummary(BaseModel):
    """Counts for the report: calls answered by the cache vs the API, failures by reason."""

    model_config = ConfigDict(frozen=True)

    jobs: int
    from_cache: int
    from_api: int
    failures: dict[str, int]


def run_groups[K, T: BaseModel](
    groups: Sequence[Sequence[Job[K]]],
    client: LLMClient,
    schema: type[T],
    concurrency: int,
) -> tuple[dict[K, T], RunSummary]:
    """Run every job; answers by key, plus a summary."""
    answers: dict[K, T] = {}
    failures: Counter[str] = Counter()
    counts = {"cache": 0, "api": 0}
    lock = threading.Lock()

    def call(job: Job[K]) -> None:
        try:
            result = client.complete_json(job.prompt, schema)
        except LLMOutputError as exc:
            with lock:
                failures[str(exc).split(":")[0][:80]] += 1
            return
        with lock:
            answers[job.key] = result.parsed
            counts["cache" if result.from_cache else "api"] += 1

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        finished = False
        try:
            for group in groups:
                if not group:
                    continue
                call(group[0])
                for future in [pool.submit(call, job) for job in group[1:]]:
                    future.result()
            finished = True
        finally:
            if not finished:  # an error: do not start the jobs still waiting (they cost money)
                pool.shutdown(wait=True, cancel_futures=True)
    total = sum(len(g) for g in groups)
    summary = RunSummary(
        jobs=total, from_cache=counts["cache"], from_api=counts["api"], failures=dict(failures)
    )
    return answers, summary


def blurb_groups(
    by_notebook: Sequence[tuple[str, str, Sequence[Chunk]]],
    template: PromptTemplate,
    model: str,
    max_output_tokens: int,
) -> list[list[Job[str]]]:
    """One group per notebook: (path, notebook text, its chunks) -> blurb jobs keyed by chunk id."""
    return [
        [Job(c.id, blurb_prompt(template, model, max_output_tokens, text, c)) for c in chunks]
        for _path, text, chunks in by_notebook
    ]


def clean_groups(
    notebooks: Sequence[tuple[str, list[Cell]]],
    template: PromptTemplate,
    model: str,
    max_output_tokens: int,
) -> list[list[Job[tuple[str, int]]]]:
    """One group per notebook: a job for every non-empty markdown cell, keyed (path, cell index)."""
    return [
        [
            Job(
                (path, cell.index),
                template.render(model, max_output_tokens, cell=cell.source.strip("\n")),
            )
            for cell in cells
            if cell.cell_type is CellType.MARKDOWN and cell.source.strip()
        ]
        for path, cells in notebooks
    ]


def read_answers[K, T: BaseModel](
    jobs: Sequence[Job[K]], cache: ResponseCache, effort: str, schema: type[T], what: str
) -> dict[K, T]:
    """Parsed answers for jobs from the cache; a missing one is an error naming how many."""
    found: dict[K, T] = {}
    missing = 0
    for job in jobs:
        stored = cache.get(cache_key(job.prompt, effort))
        if stored is None:
            missing += 1
            continue
        try:
            found[job.key] = schema.model_validate_json(stored.text)
        except ValueError as exc:
            raise LLMOutputError(f"a cached {what} no longer parses: {exc}") from exc
    if missing:
        raise MissingEnrichmentError(
            f"{missing} of {len(jobs)} {what}s are not in the cache; "
            f"run `noterecall enrich {what_command(what)}`"
        )
    return found


def what_command(what: str) -> str:
    """The CLI sub-command that generates an answer kind."""
    return {"blurb": "blurbs", "clean": "clean"}[what]


def read_blurbs(jobs: Sequence[Job[str]], cache: ResponseCache, effort: str) -> dict[str, str]:
    """Blurbs for jobs from the cache; a missing one is an error naming how many are missing."""
    return {k: v.context for k, v in read_answers(jobs, cache, effort, Blurb, "blurb").items()}


def read_cleaned(
    jobs: Sequence[Job[tuple[str, int]]], cache: ResponseCache, effort: str
) -> dict[tuple[str, int], str]:
    """Cleaned markdown cells from the cache, keyed (notebook path, cell index)."""
    return {k: v.text for k, v in read_answers(jobs, cache, effort, CleanedCell, "clean").items()}
