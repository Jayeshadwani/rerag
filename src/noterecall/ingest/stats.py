"""Corpus statistics report. Implements ADR-015 (results carry config, revisions and git SHA).

Builds `results/corpus_stats.json` from the database. The report holds aggregate numbers only:
no note text, no file names, no per-notebook rows. It refuses to run on an empty database or
one whose notebooks were synced with a different pipeline than the current one.
"""

import shutil
import sqlite3
import statistics
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any, Literal

import tiktoken
from pydantic import BaseModel, ConfigDict

from noterecall.config import Settings
from noterecall.index.store import (
    cell_type_counts,
    duplicate_counts,
    notebook_summaries,
    pipeline_hashes,
)
from noterecall.ingest.clean import RULE_KEYS


class StatsError(Exception):
    """Raised when statistics cannot be produced honestly."""


class GitState(BaseModel):
    """Commit the numbers were produced at, and whether the tree had local changes."""

    model_config = ConfigDict(frozen=True)

    sha: str
    dirty: bool


class Distribution(BaseModel):
    """Min / median / max of a per-notebook quantity."""

    model_config = ConfigDict(frozen=True)

    min: int
    median: float
    max: int


class TokenTotals(BaseModel):
    """Token counts before and after cleaning, with the tokenizer that produced them."""

    model_config = ConfigDict(frozen=True)

    tokenizer: str
    tiktoken_version: str
    raw: int
    clean: int


class CorpusStats(BaseModel):
    """The contents of `results/corpus_stats.json`."""

    model_config = ConfigDict(frozen=True)

    schema_version: Literal[1]
    git: GitState
    config: dict[str, Any]
    rules_sha256: str
    pipeline_hash: str
    notebooks: int
    cells_raw: int
    cells_kept: dict[str, int]
    tokens: TokenTotals
    removed_by_rule: dict[str, int]
    secrets_scrubbed_by_type: dict[str, int]
    duplicate_extra_copies: dict[str, int]
    cells_per_notebook: Distribution
    clean_tokens_per_notebook: Distribution


def git_state() -> GitState:
    """Current commit and dirtiness. Fails loudly if git is unavailable."""
    git = shutil.which("git")
    if git is None:
        raise StatsError("git executable not found; cannot record the commit in the report")
    sha = subprocess.run(  # noqa: S603 - fixed arguments
        [git, "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()
    status = subprocess.run(  # noqa: S603 - fixed arguments
        [git, "status", "--porcelain"], capture_output=True, text=True, check=True
    ).stdout
    return GitState(sha=sha, dirty=bool(status.strip()))


def _distribution(values: list[int]) -> Distribution:
    return Distribution(min=min(values), median=float(statistics.median(values)), max=max(values))


def _split_counts(totals: Counter[str]) -> tuple[dict[str, int], dict[str, int]]:
    removed = {key: totals.get(key, 0) for key in RULE_KEYS}
    secrets_by_type = {
        key.removeprefix("secrets_"): value
        for key, value in sorted(totals.items())
        if key.startswith("secrets_")
    }
    return removed, secrets_by_type


class Provenance(BaseModel):
    """What produced the numbers: cleaning pipeline, rules file, tokenizer and commit."""

    model_config = ConfigDict(frozen=True)

    pipeline_hash: str
    rules_sha256: str
    tokenizer_name: str
    git: GitState


def compute_stats(
    conn: sqlite3.Connection, settings: Settings, provenance: Provenance
) -> CorpusStats:
    """Aggregate the database into a `CorpusStats`."""
    summaries = notebook_summaries(conn)
    if not summaries:
        raise StatsError("the database has no notebooks; run `noterecall sync` first")
    if pipeline_hashes(conn) != {provenance.pipeline_hash}:
        raise StatsError("the database was synced with a different pipeline; re-run `sync`")
    totals: Counter[str] = Counter()
    for summary in summaries:
        totals.update(summary.counts)
    removed, secrets_by_type = _split_counts(totals)
    types = cell_type_counts(conn)
    dups = duplicate_counts(conn)
    return CorpusStats(
        schema_version=1,
        git=provenance.git,
        config=settings.model_dump(mode="json"),
        rules_sha256=provenance.rules_sha256,
        pipeline_hash=provenance.pipeline_hash,
        notebooks=len(summaries),
        cells_raw=totals["cells_raw"],
        cells_kept={"markdown": types.get("markdown", 0), "code": types.get("code", 0)},
        tokens=TokenTotals(
            tokenizer=provenance.tokenizer_name,
            tiktoken_version=tiktoken.__version__,
            raw=sum(s.raw_tokens for s in summaries),
            clean=sum(s.clean_tokens for s in summaries),
        ),
        removed_by_rule=removed,
        secrets_scrubbed_by_type=secrets_by_type,
        duplicate_extra_copies={
            "within_notebook": dups.within_notebook,
            "across_notebooks": dups.across_notebooks,
        },
        cells_per_notebook=_distribution([s.cells for s in summaries]),
        clean_tokens_per_notebook=_distribution([s.clean_tokens for s in summaries]),
    )


def write_stats(stats: CorpusStats, path: Path) -> None:
    """Write the report as stable, indented JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(stats.model_dump_json(indent=2) + "\n", encoding="utf-8")
