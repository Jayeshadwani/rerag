"""Exact dense retrieval. Implements ADR-008 (NumPy exact search, no vector database).

All chunk vectors sit in one matrix; a query is one matrix-vector product (cosine, since the
vectors are unit length). Exact search over a few thousand vectors takes milliseconds and has
perfect recall. Ties break by chunk order, so results are deterministic.
"""

from collections.abc import Mapping, Sequence

import numpy as np

from noterecall.index.embeddings import EmbeddingService
from noterecall.index.lexical import indexed_text
from noterecall.models import Chunk, RetrievalHit


class DenseError(Exception):
    """Raised when the index cannot be built or queried."""


class DenseIndex:
    """Chunk vectors plus the service that embeds queries."""

    def __init__(
        self, chunk_ids: Sequence[str], matrix: np.ndarray, service: EmbeddingService
    ) -> None:
        self._chunk_ids = list(chunk_ids)
        self._matrix = matrix
        self._service = service

    @classmethod
    def build(
        cls,
        chunks: Sequence[Chunk],
        service: EmbeddingService,
        texts: Mapping[str, str] | None = None,
    ) -> "DenseIndex":
        """Embed the chunks' text, or `texts[chunk.id]` where an enriched text is supplied."""
        if not chunks:
            raise DenseError("cannot index zero chunks")
        matrix = service.embed_passages([indexed_text(c, texts) for c in chunks])
        return cls([c.id for c in chunks], matrix, service)

    def search(self, query: str, k: int) -> list[RetrievalHit]:
        """Top-k chunks by cosine similarity, best first."""
        if k < 1:
            raise DenseError(f"k must be at least 1, got {k}")
        scores = self._matrix @ self._service.embed_query(query)
        order = np.lexsort((np.arange(len(scores)), -scores))
        return [
            RetrievalHit(chunk_id=self._chunk_ids[index], score=float(scores[index]), rank=rank)
            for rank, index in enumerate(order[:k], start=1)
        ]
