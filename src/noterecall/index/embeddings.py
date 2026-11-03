"""Embedding wrapper with a content-hash cache. Implements ADR-005 and ADR-008.

An `Embedder` turns the exact strings it is given into L2-normalized float32 vectors, so cosine
similarity is a dot product. `EmbeddingService` adds what each model's card documents: a query
prefix or instruction and a passage prefix (BGE: an instruction on queries only; E5: `query: `
and `passage: `). It caches vectors in SQLite under sha256(model id, revision, exact text),
so re-indexing an unchanged corpus never calls the model. `HashEmbedder` is the deterministic
offline stand-in used by the default tests (feature hashing of word tokens), the way
`llm/fake.py` stands in for the LLM.
"""

import hashlib
import sqlite3
from collections.abc import Sequence
from typing import Protocol

import numpy as np
import structlog
from pydantic import BaseModel, ConfigDict

from noterecall.index.code_tokenizer import tokenize_plain
from noterecall.index.store import embeddings_get, embeddings_put

log = structlog.get_logger()
BATCH = 32


class EmbeddingError(Exception):
    """Raised when a model returns vectors that cannot be used."""


class EmbeddingSpec(BaseModel):
    """A candidate embedding model: where it comes from and how its card says to use it."""

    model_config = ConfigDict(frozen=True)

    id: str
    revision: str
    query_prefix: str
    passage_prefix: str


class Embedder(Protocol):
    """Encodes exact strings to normalized float32 vectors of shape (n, dim)."""

    def encode(self, texts: Sequence[str]) -> np.ndarray: ...

    def count_truncated(self, texts: Sequence[str]) -> int:
        """How many of the strings the model would cut because they exceed its input limit."""
        ...


class HashEmbedder:
    """Deterministic offline embedder: signed feature hashing of word tokens, normalized."""

    def __init__(self, dim: int = 64) -> None:
        self._dim = dim

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        out = np.zeros((len(texts), self._dim), dtype=np.float32)
        out[:, -1] = (
            0.01  # a small constant keeps colliding tokens from cancelling to a zero vector
        )
        for row, text in enumerate(texts):
            for token in tokenize_plain(text):
                digest = hashlib.sha256(token.encode()).digest()
                index = int.from_bytes(digest[:4], "big") % self._dim
                out[row, index] += 1.0 if digest[4] % 2 == 0 else -1.0
        result: np.ndarray = out / np.linalg.norm(out, axis=1, keepdims=True)
        return result

    def count_truncated(self, texts: Sequence[str]) -> int:
        return 0


def cache_key(spec: EmbeddingSpec, text: str) -> str:
    """Cache key: the model, its revision and the exact string that is embedded."""
    return hashlib.sha256(f"{spec.id}\0{spec.revision}\0{text}".encode()).hexdigest()


class EmbeddingService:
    """Prefix handling, batching and caching around an `Embedder`."""

    def __init__(
        self, embedder: Embedder, spec: EmbeddingSpec, conn: sqlite3.Connection | None = None
    ) -> None:
        self.embedder = embedder
        self.spec = spec
        self._conn = conn
        self.hits = 0
        self.misses = 0

    def _embed(self, texts: Sequence[str], use_cache: bool = True) -> np.ndarray:
        conn = self._conn if use_cache else None
        keys = [cache_key(self.spec, t) for t in texts]
        cached = embeddings_get(conn, keys) if conn is not None else {}
        missing = [i for i, key in enumerate(keys) if key not in cached]
        self.hits += len(texts) - len(missing)
        self.misses += len(missing)
        fresh: dict[str, np.ndarray] = {}
        for start in range(0, len(missing), BATCH):
            batch = missing[start : start + BATCH]
            vectors = self.embedder.encode([texts[i] for i in batch])
            self._check(vectors, len(batch))
            fresh.update({keys[i]: vectors[n] for n, i in enumerate(batch)})
        if fresh and conn is not None:
            embeddings_put(conn, fresh)
        cached.update(fresh)
        return np.stack([cached[key] for key in keys]) if keys else np.zeros((0, 0), np.float32)

    @staticmethod
    def _check(vectors: np.ndarray, expected: int) -> None:
        if vectors.ndim != 2 or vectors.shape[0] != expected:
            raise EmbeddingError(f"expected {expected} vectors, got shape {vectors.shape}")
        if not np.isfinite(vectors).all():
            raise EmbeddingError("the model returned non-finite values")
        if not np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-3):
            raise EmbeddingError("the model returned vectors that are not unit length")

    def embed_passages(self, texts: Sequence[str]) -> np.ndarray:
        """Vectors for passages (chunk text), with the passage prefix applied."""
        vectors = self._embed([self.spec.passage_prefix + t for t in texts])
        log.info("embed.passages", model=self.spec.id, hits=self.hits, misses=self.misses)
        return vectors

    def embed_query(self, text: str) -> np.ndarray:
        """Vector for one query (prefix applied). Never cached: queries are one-offs, and the
        retrieval latency we report must include this model call."""
        vector: np.ndarray = self._embed([self.spec.query_prefix + text], use_cache=False)[0]
        return vector

    def count_truncated(self, texts: Sequence[str]) -> int:
        """Passages the model would cut (prefix included)."""
        return self.embedder.count_truncated([self.spec.passage_prefix + t for t in texts])
