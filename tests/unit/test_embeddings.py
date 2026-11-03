"""Embedding service: prefixes, the content-hash cache, validation, and the SQLite cache table."""

import sqlite3
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pytest

from noterecall.index.embeddings import (
    EmbeddingError,
    EmbeddingService,
    EmbeddingSpec,
    HashEmbedder,
    cache_key,
)
from noterecall.index.store import connect, embeddings_get, embeddings_put

SPEC = EmbeddingSpec(id="m", revision="r1", query_prefix="query: ", passage_prefix="passage: ")


class Recording:
    """An embedder that records exactly what it was asked to encode."""

    def __init__(self) -> None:
        self.seen: list[list[str]] = []
        self._inner = HashEmbedder()

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        self.seen.append(list(texts))
        return self._inner.encode(texts)

    def count_truncated(self, texts: Sequence[str]) -> int:
        return sum(1 for t in texts if len(t) > 20)


@pytest.fixture
def conn(tmp_path: Path) -> sqlite3.Connection:
    return connect(tmp_path / "db.sqlite")


def test_hash_embedder_is_deterministic_unit_length_and_overlap_sensitive() -> None:
    embedder = HashEmbedder()
    a = embedder.encode(["train test split", "train test split", "numpy array"])
    assert a.dtype == np.float32
    assert np.allclose(np.linalg.norm(a, axis=1), 1.0, atol=1e-6)
    assert np.array_equal(a[0], a[1])
    assert float(a[0] @ a[1]) == pytest.approx(1.0, abs=1e-6)
    near = embedder.encode(["train test"])[0]
    assert float(a[0] @ near) > float(a[0] @ a[2])
    assert embedder.count_truncated(["x" * 10_000]) == 0


def test_text_whose_tokens_cancel_still_gets_a_unit_vector() -> None:
    embedder = HashEmbedder(dim=1)  # every token lands on the one coordinate; signs can cancel
    vectors = embedder.encode(["", "a b c d e f g h", "x"])
    assert np.isfinite(vectors).all()
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-6)


def test_prefixes_are_applied_to_the_right_side() -> None:
    rec = Recording()
    service = EmbeddingService(rec, SPEC)
    service.embed_passages(["alpha", "beta"])
    service.embed_query("gamma")
    assert rec.seen == [["passage: alpha", "passage: beta"], ["query: gamma"]]


def test_cache_key_depends_on_model_revision_and_exact_text() -> None:
    other_revision = SPEC.model_copy(update={"revision": "r2"})
    other_model = SPEC.model_copy(update={"id": "n"})
    keys = {
        cache_key(SPEC, "t"),
        cache_key(other_revision, "t"),
        cache_key(other_model, "t"),
        cache_key(SPEC, "t "),
    }
    assert len(keys) == 4
    assert cache_key(SPEC, "t") == cache_key(SPEC, "t")


def test_passages_are_cached_and_a_second_pass_never_calls_the_model(
    conn: sqlite3.Connection,
) -> None:
    rec = Recording()
    service = EmbeddingService(rec, SPEC, conn)
    first = service.embed_passages(["a", "b", "c"])
    assert (service.hits, service.misses) == (0, 3)
    again = EmbeddingService(rec, SPEC, conn)
    second = again.embed_passages(["a", "b", "c", "d"])
    assert (again.hits, again.misses) == (3, 1)
    assert rec.seen[-1] == ["passage: d"]
    assert np.array_equal(first, second[:3])
    assert conn.execute("SELECT count(*) FROM embeddings").fetchone()[0] == 4


def test_a_new_revision_does_not_reuse_old_vectors(conn: sqlite3.Connection) -> None:
    EmbeddingService(Recording(), SPEC, conn).embed_passages(["a"])
    rec = Recording()
    EmbeddingService(rec, SPEC.model_copy(update={"revision": "r2"}), conn).embed_passages(["a"])
    assert rec.seen == [["passage: a"]]


def test_queries_bypass_the_cache(conn: sqlite3.Connection) -> None:
    rec = Recording()
    service = EmbeddingService(rec, SPEC, conn)
    service.embed_query("q")
    service.embed_query("q")
    assert rec.seen == [["query: q"], ["query: q"]]
    assert conn.execute("SELECT count(*) FROM embeddings").fetchone()[0] == 0


def test_batches_of_32(conn: sqlite3.Connection) -> None:
    rec = Recording()
    EmbeddingService(rec, SPEC, conn).embed_passages([f"t{i}" for i in range(70)])
    assert [len(batch) for batch in rec.seen] == [32, 32, 6]


def test_no_texts_gives_an_empty_matrix() -> None:
    assert EmbeddingService(Recording(), SPEC).embed_passages([]).shape[0] == 0


def test_service_without_a_connection_still_works() -> None:
    service = EmbeddingService(Recording(), SPEC)
    assert service.embed_passages(["a"]).shape == (1, 64)
    assert service.misses == 1


def test_truncation_count_includes_the_prefix() -> None:
    service = EmbeddingService(Recording(), SPEC)
    # "passage: " is 9 characters: a 12-character text becomes 21 > 20, an 11-character one is 20
    assert service.count_truncated(["x" * 12, "x" * 11]) == 1


class Broken:
    def __init__(self, result: np.ndarray) -> None:
        self._result = result

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        return self._result

    def count_truncated(self, texts: Sequence[str]) -> int:
        return 0


@pytest.mark.parametrize(
    ("result", "message"),
    [
        (np.ones((2, 4), dtype=np.float32) / 2, "expected 1 vectors"),
        (np.ones(4, dtype=np.float32), "expected 1 vectors"),
        (np.array([[np.nan, 0.0]], dtype=np.float32), "non-finite"),
        (np.array([[3.0, 4.0]], dtype=np.float32), "not unit length"),
    ],
)
def test_unusable_model_output_is_rejected(result: np.ndarray, message: str) -> None:
    with pytest.raises(EmbeddingError, match=message):
        EmbeddingService(Broken(result), SPEC).embed_passages(["a"])


def test_store_round_trip_keeps_float32_and_ignores_duplicates(conn: sqlite3.Connection) -> None:
    vec = np.array([0.6, 0.8], dtype=np.float64)
    embeddings_put(conn, {"k1": vec})
    embeddings_put(conn, {"k1": np.array([1.0, 0.0])})  # existing key is left untouched
    got = embeddings_get(conn, ["k1", "missing"])
    assert set(got) == {"k1"}
    assert got["k1"].dtype == np.float32
    assert np.allclose(got["k1"], [0.6, 0.8])
    assert embeddings_get(conn, []) == {}


def test_store_lookup_handles_more_keys_than_one_batch(conn: sqlite3.Connection) -> None:
    items = {f"k{i}": np.array([1.0, 0.0], dtype=np.float32) for i in range(1200)}
    embeddings_put(conn, items)
    assert len(embeddings_get(conn, [*items, "absent"])) == 1200
