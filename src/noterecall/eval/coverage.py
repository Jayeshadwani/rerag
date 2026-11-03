"""Token coverage of relevant cells within a token budget. A secondary metric (report only).

Recall@token-budget counts a relevant cell as found if any part of it is in a retrieved chunk.
Token coverage asks how much of it is: over the chunks that fit in the budget (the same
rank-order prefix as `metrics.budget_prefix`), the share of the relevant cells' tokens that those
chunks actually hold. Tokens are counted with the project tokenizer on the text of each piece a
chunk holds and on each cell's source; a cell's tokens are counted once. It is reported next to
Recall@token-budget and never used to decide anything (owner decision on proposal 003).
"""

from collections.abc import Collection, Sequence

from noterecall.chunking.tokens import Tokenizer
from noterecall.eval.gold import CellKey
from noterecall.eval.metrics import MetricsError, budget_prefix
from noterecall.models import Cell, Chunk

CellId = tuple[str, int]


class CoverageError(Exception):
    """Raised when a relevant cell is not in the corpus the index was built from."""


class CoverageIndex:
    """Token counts of every cell, and a cache of how many tokens of which cell each chunk holds."""

    def __init__(self, notebooks: Sequence[tuple[str, list[Cell]]], tokenizer: Tokenizer) -> None:
        self._tokenizer = tokenizer
        self._cell_tokens: dict[CellId, int] = {}
        self._indices: dict[CellKey, list[int]] = {}
        for path, cells in notebooks:
            for cell in cells:
                self._cell_tokens[(path, cell.index)] = tokenizer.count(cell.source.strip("\n"))
                self._indices.setdefault((path, cell.raw_hash), []).append(cell.index)
        self._held: dict[str, dict[CellId, int]] = {}

    def _chunk_holds(self, chunk: Chunk) -> dict[CellId, int]:
        if chunk.id not in self._held:
            held: dict[CellId, int] = {}
            for piece in chunk.pieces:
                cell = (chunk.notebook_path, piece.cell_index)
                held[cell] = held.get(cell, 0) + self._tokenizer.count(piece.text.strip("\n"))
            self._held[chunk.id] = held
        return self._held[chunk.id]

    def at_budget(
        self, ranked: Sequence[Chunk], relevant: Collection[CellKey], budget: int
    ) -> float:
        """Tokens of the relevant cells held by the budgeted chunks / tokens of those cells."""
        if not relevant:
            raise MetricsError("a query with no relevant cells has no retrieval score")
        held: dict[CellId, int] = {}
        for chunk in budget_prefix(ranked, budget):
            for cell, tokens in self._chunk_holds(chunk).items():
                held[cell] = held.get(cell, 0) + tokens
        covered = total = 0
        for key in set(relevant):
            if key not in self._indices:
                raise CoverageError("a relevant cell is not in the corpus")
            cell_id = max(
                ((key[0], i) for i in self._indices[key]),
                key=lambda c: held.get(c, 0) / max(self._cell_tokens[c], 1),
            )
            size = self._cell_tokens[cell_id]
            covered += min(held.get(cell_id, 0), size)
            total += size
        if total == 0:
            raise CoverageError("the relevant cells have no tokens")
        return covered / total
