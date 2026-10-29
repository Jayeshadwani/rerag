"""Chunk statistics report for `results/chunk_stats.json`. Supports ADR-004 and proposal 002.

For each candidate cap (from config) it chunks the whole stored corpus with the structural
chunker and with the fixed-token baseline (outputs are never chunked), and records aggregate
counts: how many chunks, how big, how many were flagged oversized and why, how many cells had to
be split. Descriptive only: it does not choose the cap (that is the M4 experiment). No note text,
file names or per-notebook rows.
"""

import math
import sqlite3
from collections import Counter
from collections.abc import Sequence
from typing import Any, Literal

import tiktoken
from pydantic import BaseModel, ConfigDict

from noterecall.chunking.fixed import fixed_chunks
from noterecall.chunking.structural import chunk_notebook
from noterecall.chunking.tokens import Tokenizer
from noterecall.config import Settings
from noterecall.index.store import notebook_cells, pipeline_hashes
from noterecall.ingest.stats import GitState, Provenance, StatsError
from noterecall.models import Chunk, OversizeReason, PieceUnit


class ChunkSizes(BaseModel):
    """Chunk token counts: min, median, nearest-rank 90th percentile, max."""

    model_config = ConfigDict(frozen=True)

    min: int
    median: float
    p90: int
    max: int


class ChunkRow(BaseModel):
    """Aggregates for one chunker configuration."""

    model_config = ConfigDict(frozen=True)

    chunker: Literal["structural", "fixed"]
    cap: int
    chunks: int
    tokens: ChunkSizes
    under_quarter_cap: int
    flagged_by_reason: dict[str, int]
    cells_split: int
    unparsed_code_cells: int


class ChunkStats(BaseModel):
    """The contents of `results/chunk_stats.json`."""

    model_config = ConfigDict(frozen=True)

    schema_version: Literal[2]
    git: GitState
    config: dict[str, Any]
    pipeline_hash: str
    tokenizer: str
    tiktoken_version: str
    notebooks: int
    cells: int
    rows: list[ChunkRow]


def sizes(values: Sequence[int]) -> ChunkSizes:
    """Summary of chunk sizes; the 90th percentile uses the nearest-rank method."""
    ordered = sorted(values)
    rank = max(1, math.ceil(0.9 * len(ordered)))
    middle = len(ordered) // 2
    median = ordered[middle] if len(ordered) % 2 else (ordered[middle - 1] + ordered[middle]) / 2
    return ChunkSizes(min=ordered[0], median=float(median), p90=ordered[rank - 1], max=ordered[-1])


class _Accumulator:
    """Collects one configuration's chunks across all notebooks."""

    def __init__(self) -> None:
        self.tokens: list[int] = []
        self.flags: Counter[str] = Counter()
        self.cells_split = 0
        self.unparsed_cells = 0

    def add(self, chunks: Sequence[Chunk]) -> None:
        chunks_of_cell: dict[int, set[str]] = {}
        unparsed: set[int] = set()
        for chunk in chunks:
            self.tokens.append(chunk.tokens)
            self.flags.update(flag.value for flag in chunk.flags)
            for piece in chunk.pieces:
                chunks_of_cell.setdefault(piece.cell_index, set()).add(chunk.id)
                if piece.unit is PieceUnit.UNPARSED_CODE:
                    unparsed.add(piece.cell_index)
        self.cells_split += sum(1 for ids in chunks_of_cell.values() if len(ids) > 1)
        self.unparsed_cells += len(unparsed)

    def row(self, chunker: Literal["structural", "fixed"], cap: int) -> ChunkRow:
        flagged = {reason.value: self.flags.get(reason.value, 0) for reason in OversizeReason}
        return ChunkRow(
            chunker=chunker,
            cap=cap,
            chunks=len(self.tokens),
            tokens=sizes(self.tokens),
            under_quarter_cap=sum(1 for t in self.tokens if t < cap / 4),
            flagged_by_reason=flagged,
            cells_split=self.cells_split,
            unparsed_code_cells=self.unparsed_cells,
        )


def compute_chunk_stats(
    conn: sqlite3.Connection, settings: Settings, provenance: Provenance, tokenizer: Tokenizer
) -> ChunkStats:
    """Chunk the stored corpus under every configuration and aggregate."""
    notebooks = list(notebook_cells(conn))
    if not notebooks:
        raise StatsError("the database has no notebooks; run `noterecall sync` first")
    if pipeline_hashes(conn) != {provenance.pipeline_hash}:
        raise StatsError("the database was synced with a different pipeline; re-run `sync`")
    rows: list[ChunkRow] = []
    for chunker in ("structural", "fixed"):
        for cap in settings.chunking.candidate_caps:
            acc = _Accumulator()
            for path, cells in notebooks:
                make = chunk_notebook if chunker == "structural" else fixed_chunks
                acc.add(make(path, cells, cap, tokenizer))
            rows.append(acc.row(chunker, cap))
    return ChunkStats(
        schema_version=2,
        git=provenance.git,
        config=settings.model_dump(mode="json"),
        pipeline_hash=provenance.pipeline_hash,
        tokenizer=provenance.tokenizer_name,
        tiktoken_version=tiktoken.__version__,
        notebooks=len(notebooks),
        cells=sum(len(cells) for _path, cells in notebooks),
        rows=rows,
    )
