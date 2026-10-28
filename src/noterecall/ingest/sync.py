"""Incremental sync. Implements ADR-015 (content hashes) and ADR-003 (cleaning at ingest).

A notebook is re-processed only when its file hash or the pipeline hash changed. The
pipeline hash covers the rules file, the output-line limit, the tokenizer name and the
source of the parser, cleaner and scrubber, so changing any cleaning behaviour re-syncs
everything. An unchanged corpus therefore performs zero database writes. Logs carry counts
and short hashes only: never note text, and never file names.
"""

import hashlib
import sqlite3
from dataclasses import dataclass
from pathlib import Path

import structlog
from pydantic import BaseModel, ConfigDict

from noterecall.chunking.tokens import Tokenizer
from noterecall.index.store import (
    StoredNotebook,
    delete_notebooks,
    replace_notebook,
    stored_notebooks,
)
from noterecall.ingest import clean, ipynb, secrets
from noterecall.ingest.clean import CleaningRules, clean_notebook
from noterecall.ingest.loader import SourceLoader

log = structlog.get_logger()

_HASH_CHUNK = 1 << 20


class SyncError(Exception):
    """Raised when the raw directory cannot be synced safely."""


@dataclass(frozen=True)
class Pipeline:
    """Everything the cleaner needs, fixed for one sync run."""

    rules: CleaningRules
    max_output_lines: int
    tokenizer: Tokenizer
    pipeline_hash: str


class SyncReport(BaseModel):
    """What a sync run did."""

    model_config = ConfigDict(frozen=True)

    added: int
    updated: int
    unchanged: int
    removed: int


def file_sha256(path: Path) -> str:
    """sha256 of a file's bytes, read in chunks."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(_HASH_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def compute_pipeline_hash(rules_path: Path, max_output_lines: int, tokenizer_name: str) -> str:
    """Hash of every input that changes what the cleaner produces."""
    digest = hashlib.sha256()
    digest.update(rules_path.read_bytes())
    digest.update(f"\0{max_output_lines}\0{tokenizer_name}\0".encode())
    digest.update(secrets.patterns_fingerprint().encode())
    for module in (ipynb, clean, secrets):
        digest.update(Path(str(module.__file__)).read_bytes())
    return digest.hexdigest()


def discover(raw_dir: Path, loader: SourceLoader) -> list[Path]:
    """Sorted source files under `raw_dir`, skipping hidden directories such as checkpoints."""
    if not raw_dir.is_dir():
        raise SyncError(f"raw directory does not exist: {raw_dir}")
    found = [
        p
        for p in raw_dir.rglob("*")
        if p.is_file()
        and p.suffix in loader.suffixes
        and not any(part.startswith(".") for part in p.relative_to(raw_dir).parts)
    ]
    if not found:
        raise SyncError(f"no {'/'.join(loader.suffixes)} files under {raw_dir}; refusing to sync")
    return sorted(found)


@dataclass(frozen=True)
class _Run:
    """State shared by every file in one sync run."""

    raw_dir: Path
    conn: sqlite3.Connection
    loader: SourceLoader
    pipeline: Pipeline
    known: dict[str, StoredNotebook]


def _sync_one(run: _Run, path: Path) -> str:
    """Sync one file; returns 'unchanged', 'added' or 'updated'."""
    relative = path.relative_to(run.raw_dir).as_posix()
    file_hash = file_sha256(path)
    previous = run.known.get(relative)
    pipeline = run.pipeline
    if previous == StoredNotebook(file_hash, pipeline.pipeline_hash):
        return "unchanged"
    notebook = clean_notebook(
        run.loader.load(path, run.raw_dir, file_hash),
        pipeline.rules,
        pipeline.max_output_lines,
        pipeline.tokenizer,
        pipeline.pipeline_hash,
    )
    replace_notebook(run.conn, notebook)
    action = "added" if previous is None else "updated"
    log.info("sync.notebook", action=action, file=file_hash[:8], cells=len(notebook.cells))
    return action


def sync(
    raw_dir: Path, conn: sqlite3.Connection, loader: SourceLoader, pipeline: Pipeline
) -> SyncReport:
    """Bring the database in line with `raw_dir`; unchanged notebooks are not touched."""
    files = discover(raw_dir, loader)
    run = _Run(raw_dir, conn, loader, pipeline, stored_notebooks(conn))
    actions = [_sync_one(run, p) for p in files]
    present = {p.relative_to(raw_dir).as_posix() for p in files}
    stale = sorted(set(run.known) - present)
    if stale:
        delete_notebooks(conn, stale)
    report = SyncReport(
        added=actions.count("added"),
        updated=actions.count("updated"),
        unchanged=actions.count("unchanged"),
        removed=len(stale),
    )
    log.info("sync.done", **report.model_dump())
    return report
