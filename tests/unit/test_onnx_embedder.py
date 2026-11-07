"""The ONNX embedder with a stand-in session (offline), and its cache namespace."""

import sqlite3
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from tokenizers import Tokenizer, models, pre_tokenizers, processors

from noterecall.index.embeddings import EmbeddingService, EmbeddingSpec, HashEmbedder, cache_key
from noterecall.index.onnx_embedder import (
    BATCH,
    INPUT_NAMES,
    OnnxEmbedder,
    OnnxEmbedderError,
    load_onnx_embedder,
)
from noterecall.index.store import connect

VOCAB = {"[PAD]": 0, "[CLS]": 1, "[SEP]": 2, "[UNK]": 3, "a": 4, "b": 5, "c": 6, "d": 7}


def tiny_tokenizer(with_pad: bool = True) -> Tokenizer:
    vocab = dict(VOCAB) if with_pad else {k: v for k, v in VOCAB.items() if k != "[PAD]"}
    tok = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    tok.post_processor = processors.TemplateProcessing(
        single="[CLS] $A [SEP]",
        special_tokens=[("[CLS]", 1), ("[SEP]", 2)],
    )
    return tok


class FakeSession:
    """Returns a hidden state whose [CLS] vector is (n_real_tokens, 1, 0): so the unit vector of a
    text with n tokens (special tokens included) is (n, 1, 0) / sqrt(n^2 + 1)."""

    def __init__(self) -> None:
        self.feeds: list[dict[str, np.ndarray]] = []

    def run(self, _outputs: None, feed: dict[str, np.ndarray]) -> list[np.ndarray]:
        self.feeds.append(feed)
        mask = feed["attention_mask"]
        batch, length = mask.shape
        hidden = np.zeros((batch, length, 3), dtype=np.float32)
        hidden[:, 0, 0] = mask.sum(axis=1)
        hidden[:, 0, 1] = 1.0
        hidden[:, 1:, 2] = 99.0  # must be ignored: only [CLS] is pooled
        return [hidden]


def test_the_embedding_is_the_normalized_cls_vector() -> None:
    embedder = OnnxEmbedder(FakeSession(), tiny_tokenizer())
    vectors = embedder.encode(["a b", "a"])
    # "a b" -> [CLS] a b [SEP]: n = 4, vector (4, 1, 0) / sqrt(17);
    # "a" -> n = 3, (3, 1, 0) / sqrt(10)
    assert vectors.dtype == np.float32
    assert vectors.shape == (2, 3)
    np.testing.assert_allclose(vectors[0], np.array([4, 1, 0]) / np.sqrt(17), atol=1e-6)
    np.testing.assert_allclose(vectors[1], np.array([3, 1, 0]) / np.sqrt(10), atol=1e-6)
    np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-6)


def test_padding_does_not_change_a_vector_and_inputs_are_what_the_model_takes() -> None:
    session = FakeSession()
    embedder = OnnxEmbedder(session, tiny_tokenizer())
    together = embedder.encode(["a", "a b c d"])
    alone = embedder.encode(["a"])
    np.testing.assert_allclose(together[0], alone[0], atol=1e-6)
    feed = session.feeds[0]
    assert set(feed) == set(INPUT_NAMES)
    assert feed["input_ids"].shape == feed["attention_mask"].shape == (2, 6)  # padded to 6
    assert feed["input_ids"].dtype == np.int64
    assert (feed["token_type_ids"] == 0).all()
    assert feed["attention_mask"].tolist() == [[1, 1, 1, 0, 0, 0], [1, 1, 1, 1, 1, 1]]
    assert feed["input_ids"][0].tolist() == [1, 4, 2, 0, 0, 0]  # [CLS] a [SEP] then [PAD]


def test_long_texts_are_cut_at_the_limit_and_counted() -> None:
    embedder = OnnxEmbedder(FakeSession(), tiny_tokenizer(), max_length=5)
    texts = ["a b c", "a b c d", "a"]  # 5, 6 and 3 tokens with the special tokens
    assert embedder.count_truncated(texts) == 1
    vectors = embedder.encode(["a b c d a b"])
    np.testing.assert_allclose(vectors[0], np.array([5, 1, 0]) / np.sqrt(26), atol=1e-6)  # n = 5


def test_batches_are_split_and_order_is_kept() -> None:
    session = FakeSession()
    embedder = OnnxEmbedder(session, tiny_tokenizer())
    texts = ["a"] * BATCH + ["a b"] * 3
    vectors = embedder.encode(texts)
    assert [f["input_ids"].shape[0] for f in session.feeds] == [BATCH, 3]
    assert vectors.shape == (BATCH + 3, 3)
    np.testing.assert_allclose(vectors[-1], np.array([4, 1, 0]) / np.sqrt(17), atol=1e-6)
    assert embedder.encode([]).shape == (0, 0)


def test_a_tokenizer_without_a_pad_token_is_rejected() -> None:
    with pytest.raises(OnnxEmbedderError, match=r"no .PAD. token"):
        OnnxEmbedder(FakeSession(), tiny_tokenizer(with_pad=False))


def test_missing_model_files_are_a_loud_error(tmp_path: Path) -> None:
    with pytest.raises(OnnxEmbedderError, match=r"missing .*model\.onnx"):
        load_onnx_embedder(tmp_path)
    (tmp_path / "model.onnx").write_bytes(b"x")
    with pytest.raises(OnnxEmbedderError, match=r"missing .*tokenizer\.json"):
        load_onnx_embedder(tmp_path)


SPEC = EmbeddingSpec(id="m", revision="r1", query_prefix="query: ", passage_prefix="")


def test_the_cache_namespace_keeps_another_runtime_apart_and_the_old_keys_unchanged() -> None:
    plain = cache_key(SPEC, "text")
    assert plain == cache_key(SPEC, "text", "")  # existing keys are unchanged
    assert cache_key(SPEC, "text", "onnx") != plain
    assert cache_key(SPEC, "text", "onnx") == cache_key(SPEC, "text", "onnx")
    assert cache_key(SPEC, "text", "onnx") != cache_key(SPEC, "text", "other")


def test_a_namespaced_service_never_reads_the_vectors_of_another_runtime() -> None:
    conn: sqlite3.Connection = connect(Path(":memory:"))
    torch_like = EmbeddingService(HashEmbedder(dim=8), SPEC, conn)
    torch_like.embed_passages(["one", "two"])
    onnx_like = EmbeddingService(HashEmbedder(dim=8), SPEC, conn, namespace="onnx")
    onnx_like.embed_passages(["one", "two"])
    assert (onnx_like.hits, onnx_like.misses) == (0, 2)  # nothing was reused from the first
    onnx_like.embed_passages(["one"])
    assert onnx_like.hits == 1


@pytest.mark.live
def test_the_real_model_agrees_with_torch_to_within_1e_4(tmp_path: Path) -> None:
    """Needs the network and torch: cosine of the two runtimes' vectors for a few sentences."""
    from sentence_transformers import SentenceTransformer  # noqa: PLC0415

    from noterecall.index.onnx_embedder import fetch_onnx_files  # noqa: PLC0415

    repo, revision = "BAAI/bge-base-en-v1.5", "a5beb1e3e68b9ab74eb54cfd186867f64f240e1a"
    embedder = load_onnx_embedder(fetch_onnx_files(repo, revision, tmp_path))
    torch_model: Any = SentenceTransformer(repo, revision=revision, device="cpu")
    texts = ["groupby mean of each key", "train_test_split import", "Naive Bayes in one line"]
    a = embedder.encode(texts)
    b = torch_model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)
    assert np.abs((a * b).sum(axis=1) - 1.0).max() < 1e-4
