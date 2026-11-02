"""The real models behind the `Embedder` and `Reranker` protocols. Implements ADR-005 and ADR-006.

`SentenceTransformerEmbedder` wraps a `sentence_transformers.SentenceTransformer` pinned to an
exact revision (CPU, unit-length float32 output). `CrossEncoderReranker` wraps a
`sentence_transformers.CrossEncoder` the same way; its scores are raw logits, which is all a
ranking needs. Both can say how many inputs the model would cut at its input limit (512
tokens for the models used here), so truncation is counted and reported, never silent.
The classes take an already loaded model, so their logic is tested offline with stand-ins;
`load_embedder` and `load_cross_encoder` are the thin loaders that download and pin.
"""

from collections.abc import Callable, Mapping, Sequence
from typing import Protocol

import numpy as np
from sentence_transformers import CrossEncoder, SentenceTransformer

from noterecall.index.embeddings import EmbeddingSpec
from noterecall.retrieval.rerank import RerankSpec

ENCODE_BATCH = 32
PREDICT_BATCH = 16


Tokenizer = Callable[..., Mapping[str, Sequence[Sequence[int]]]]


class _EncoderModel(Protocol):
    tokenizer: Tokenizer
    max_seq_length: int | None

    def encode(
        self,
        inputs: Sequence[str],
        *,
        batch_size: int,
        normalize_embeddings: bool,
        convert_to_numpy: bool,
        show_progress_bar: bool,
    ) -> np.ndarray: ...


class _PairModel(Protocol):
    tokenizer: Tokenizer
    max_length: int | None

    def predict(
        self, inputs: Sequence[tuple[str, str]], *, batch_size: int, show_progress_bar: bool
    ) -> np.ndarray: ...


class ModelError(Exception):
    """Raised when a loaded model lacks what the wrappers need."""


def _limit(value: int | None, what: str) -> int:
    if value is None:
        raise ModelError(f"the {what} has no input length limit to count truncation against")
    return value


class SentenceTransformerEmbedder:
    """Encodes strings to unit-length float32 vectors on the CPU."""

    def __init__(self, model: _EncoderModel) -> None:
        self._model = model
        self._limit = _limit(model.max_seq_length, "embedding model")

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        vectors: np.ndarray = self._model.encode(
            list(texts),
            batch_size=ENCODE_BATCH,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return vectors.astype(np.float32, copy=False)

    def count_truncated(self, texts: Sequence[str]) -> int:
        """Texts whose token count (special tokens included) exceeds the model's limit."""
        if not texts:
            return 0
        ids = self._model.tokenizer(list(texts), add_special_tokens=True, truncation=False)
        return sum(1 for item in ids["input_ids"] if len(item) > self._limit)


class CrossEncoderReranker:
    """Scores (query, passage) pairs; higher is more relevant."""

    def __init__(self, model: _PairModel) -> None:
        self._model = model
        self._limit = _limit(model.max_length, "cross-encoder")

    def score(self, query: str, passages: Sequence[str]) -> list[float]:
        pairs = [(query, passage) for passage in passages]
        scores = self._model.predict(pairs, batch_size=PREDICT_BATCH, show_progress_bar=False)
        return [float(s) for s in scores]

    def count_truncated(self, query: str, passages: Sequence[str]) -> int:
        """Pairs whose joint token count exceeds the model's limit."""
        if not passages:
            return 0
        ids = self._model.tokenizer(
            [query] * len(passages), list(passages), add_special_tokens=True, truncation=False
        )
        return sum(1 for item in ids["input_ids"] if len(item) > self._limit)


def load_embedder(spec: EmbeddingSpec) -> SentenceTransformerEmbedder:
    """Download (or reuse) the pinned revision and wrap it."""
    model = SentenceTransformer(spec.id, revision=spec.revision, device="cpu")
    return SentenceTransformerEmbedder(model)  # type: ignore[arg-type]  # structural match


def load_cross_encoder(spec: RerankSpec) -> CrossEncoderReranker:
    """Download (or reuse) the pinned revision and wrap it."""
    model = CrossEncoder(spec.id, revision=spec.revision, device="cpu")
    return CrossEncoderReranker(model)
