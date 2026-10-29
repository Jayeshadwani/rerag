"""SQLite store. Implements ADR-008 (SQLite for everything except vectors) and ADR-003.

Schema version 1 holds cleaned notebooks and cells. `PRAGMA user_version` records the
schema version; a database from a newer version is refused instead of being modified.
Duplicate cells are *marked* by the `duplicate_cells` view (every copy after the first),
never deleted. Reading never writes: opening an up-to-date database performs no changes.
"""

import json
import sqlite3
from collections.abc import Iterator
from pathlib import Path
from typing import NamedTuple

from pydantic import BaseModel, ConfigDict

from noterecall.models import Cell, CellType, Notebook

SCHEMA_VERSION = 1

_SCHEMA = """
CREATE TABLE notebooks (
    path TEXT PRIMARY KEY,
    file_hash TEXT NOT NULL,
    pipeline_hash TEXT NOT NULL,
    raw_tokens INTEGER NOT NULL,
    clean_tokens INTEGER NOT NULL,
    counts_json TEXT NOT NULL
);
CREATE TABLE cells (
    notebook_path TEXT NOT NULL REFERENCES notebooks(path) ON DELETE CASCADE,
    idx INTEGER NOT NULL,
    cell_type TEXT NOT NULL CHECK (cell_type IN ('markdown', 'code')),
    raw_hash TEXT NOT NULL,
    clean_hash TEXT NOT NULL,
    source TEXT NOT NULL,
    output_text TEXT NOT NULL,
    PRIMARY KEY (notebook_path, idx)
);
CREATE INDEX cells_clean_hash ON cells (clean_hash);
CREATE VIEW duplicate_cells AS
SELECT notebook_path, idx, clean_hash, cell_type, canonical_notebook, canonical_idx
FROM (
    SELECT notebook_path, idx, clean_hash, cell_type,
           FIRST_VALUE(notebook_path) OVER w AS canonical_notebook,
           FIRST_VALUE(idx) OVER w AS canonical_idx
    FROM cells
    WINDOW w AS (PARTITION BY clean_hash ORDER BY notebook_path, idx)
)
WHERE NOT (notebook_path = canonical_notebook AND idx = canonical_idx);
"""


class StoreError(Exception):
    """Raised when the database cannot be used (unknown schema, foreign database)."""


class StoredNotebook(NamedTuple):
    """The hashes that decide whether a notebook needs re-processing."""

    file_hash: str
    pipeline_hash: str


class NotebookSummary(BaseModel):
    """Per-notebook aggregates. Deliberately carries no path or text."""

    model_config = ConfigDict(frozen=True)

    cells: int
    raw_tokens: int
    clean_tokens: int
    counts: dict[str, int]


class DuplicateCounts(BaseModel):
    """Extra copies of cells (all copies after the first of each identical group)."""

    model_config = ConfigDict(frozen=True)

    within_notebook: int
    across_notebooks: int


def _create_schema(conn: sqlite3.Connection) -> None:
    with conn:
        conn.executescript(f"BEGIN;{_SCHEMA}PRAGMA user_version = {SCHEMA_VERSION};COMMIT;")


def _migrate(conn: sqlite3.Connection) -> None:
    version = int(conn.execute("PRAGMA user_version").fetchone()[0])
    if version == SCHEMA_VERSION:
        return
    if version > SCHEMA_VERSION:
        raise StoreError(f"database schema v{version} is newer than supported v{SCHEMA_VERSION}")
    if conn.execute("SELECT count(*) FROM sqlite_master").fetchone()[0]:
        raise StoreError("database has tables but no schema version; refusing to modify it")
    _create_schema(conn)


def connect(db_path: Path) -> sqlite3.Connection:
    """Open (creating if needed) the database and make sure the schema is current."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = ON")
    _migrate(conn)
    return conn


def stored_notebooks(conn: sqlite3.Connection) -> dict[str, StoredNotebook]:
    """Map notebook path to the hashes it was synced with."""
    rows = conn.execute("SELECT path, file_hash, pipeline_hash FROM notebooks")
    return {path: StoredNotebook(file_hash, pipeline) for path, file_hash, pipeline in rows}


def replace_notebook(conn: sqlite3.Connection, notebook: Notebook) -> None:
    """Insert or fully replace one notebook and its cells in a single transaction."""
    with conn:
        conn.execute("DELETE FROM notebooks WHERE path = ?", (notebook.path,))
        conn.execute(
            "INSERT INTO notebooks VALUES (?, ?, ?, ?, ?, ?)",
            (
                notebook.path,
                notebook.file_hash,
                notebook.pipeline_hash,
                notebook.raw_tokens,
                notebook.clean_tokens,
                json.dumps(notebook.counts, sort_keys=True),
            ),
        )
        conn.executemany(
            "INSERT INTO cells VALUES (?, ?, ?, ?, ?, ?, ?)",
            [
                (
                    notebook.path,
                    cell.index,
                    cell.cell_type.value,
                    cell.raw_hash,
                    cell.clean_hash,
                    cell.source,
                    cell.output_text,
                )
                for cell in notebook.cells
            ],
        )


def delete_notebooks(conn: sqlite3.Connection, paths: list[str]) -> None:
    """Remove notebooks (and, by cascade, their cells)."""
    with conn:
        conn.executemany("DELETE FROM notebooks WHERE path = ?", [(p,) for p in paths])


def notebook_summaries(conn: sqlite3.Connection) -> list[NotebookSummary]:
    """One aggregate row per notebook, without paths."""
    rows = conn.execute(
        "SELECT (SELECT count(*) FROM cells c WHERE c.notebook_path = n.path),"
        " n.raw_tokens, n.clean_tokens, n.counts_json FROM notebooks n ORDER BY n.path"
    )
    return [
        NotebookSummary(cells=cells, raw_tokens=raw, clean_tokens=clean, counts=json.loads(counts))
        for cells, raw, clean, counts in rows
    ]


def pipeline_hashes(conn: sqlite3.Connection) -> set[str]:
    """Distinct pipeline hashes present; more than one means the corpus is partly stale."""
    return {row[0] for row in conn.execute("SELECT DISTINCT pipeline_hash FROM notebooks")}


def cell_type_counts(conn: sqlite3.Connection) -> dict[str, int]:
    """Number of stored cells per cell type."""
    rows = conn.execute("SELECT cell_type, count(*) FROM cells GROUP BY cell_type")
    return dict(rows)


def duplicate_counts(conn: sqlite3.Connection) -> DuplicateCounts:
    """Count marked duplicates inside the canonical copy's notebook and in other notebooks."""
    row = conn.execute(
        "SELECT coalesce(sum(notebook_path = canonical_notebook), 0),"
        " coalesce(sum(notebook_path != canonical_notebook), 0) FROM duplicate_cells"
    ).fetchone()
    return DuplicateCounts(within_notebook=int(row[0]), across_notebooks=int(row[1]))


def notebook_cells(conn: sqlite3.Connection) -> Iterator[tuple[str, list[Cell]]]:
    """Each notebook's stored cells in cell order, notebooks in path order."""
    current: str | None = None
    cells: list[Cell] = []
    query = (
        "SELECT notebook_path, idx, cell_type, raw_hash, clean_hash, source, output_text"
        " FROM cells ORDER BY notebook_path, idx"
    )
    for path, idx, cell_type, raw_hash, clean_hash, source, output_text in conn.execute(query):
        if current is not None and path != current:
            yield current, cells
            cells = []
        current = path
        cells.append(
            Cell(
                index=idx,
                cell_type=CellType(cell_type),
                raw_hash=raw_hash,
                clean_hash=clean_hash,
                source=source,
                output_text=output_text,
            )
        )
    if current is not None:
        yield current, cells
