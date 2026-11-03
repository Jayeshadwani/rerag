"""BM25 index over chunk text. Implements ADR-005 (lexical side of the hybrid).

An optional mapping supplies the enriched text to index in place of the chunk text (ADR-007).
Wraps `bm25s` (Lucene-variant BM25, k1=1.5, b=0.75). Chunks are tokenized by a function passed
in, so the plain and the code-aware tokenizer are interchangeable in the ablation. Ranking is
done here from the full score vector so that ties break deterministically (by chunk order), and
chunks that match no query token are not returned.
"""

from collections.abc import Callable, Mapping, Sequence

import bm25s
import numpy as np

from noterecall.models import Chunk, RetrievalHit

Tokenize = Callable[[str], list[str]]


class LexicalError(Exception):
    """Raised when the index cannot be built or queried."""


def indexed_text(chunk: Chunk, texts: Mapping[str, str] | None) -> str:
    """The text to index for a chunk: the supplied one (it must exist) or the chunk's own."""
    if texts is None:
        return chunk.text
    if chunk.id not in texts:
        raise LexicalError(f"no indexed text was supplied for chunk {chunk.id}")
    return texts[chunk.id]


class LexicalIndex:
    """A BM25 index; build once per (chunk set, tokenizer)."""

    def __init__(self, chunk_ids: Sequence[str], bm25: bm25s.BM25, tokenize: Tokenize) -> None:
        self._chunk_ids = list(chunk_ids)
        self._bm25 = bm25
        self._tokenize = tokenize

    @classmethod
    def build(
        cls, chunks: Sequence[Chunk], tokenize: Tokenize, texts: Mapping[str, str] | None = None
    ) -> "LexicalIndex":
        """Index the chunks' text, or `texts[chunk.id]` where an enriched text is supplied."""
        if not chunks:
            raise LexicalError("cannot index zero chunks")
        bm25 = bm25s.BM25()
        bm25.index([tokenize(indexed_text(c, texts)) for c in chunks], show_progress=False)
        return cls([c.id for c in chunks], bm25, tokenize)

    def search(self, query: str, k: int) -> list[RetrievalHit]:
        """Top-k chunks with a positive BM25 score, best first (ties by chunk order)."""
        if k < 1:
            raise LexicalError(f"k must be at least 1, got {k}")
        tokens = self._tokenize(query)
        if not tokens:
            return []
        scores = np.asarray(self._bm25.get_scores(tokens), dtype=np.float64)
        order = np.lexsort((np.arange(len(scores)), -scores))
        hits: list[RetrievalHit] = []
        for index in order[:k]:
            if scores[index] <= 0.0:
                break
            hits.append(
                RetrievalHit(
                    chunk_id=self._chunk_ids[index], score=float(scores[index]), rank=len(hits) + 1
                )
            )
        return hits
