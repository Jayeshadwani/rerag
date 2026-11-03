"""The model wrappers: logic tested with stand-in models; real downloads only under `live`."""

from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import pytest

from noterecall.config import load_settings
from noterecall.index.st_models import (
    CrossEncoderReranker,
    ModelError,
    SentenceTransformerEmbedder,
    Tokenizer,
    load_cross_encoder,
    load_embedder,
)


def fake_tokenizer(*texts: Sequence[str], **kwargs: bool) -> Mapping[str, Sequence[Sequence[int]]]:
    """Token ids = one per word, plus two special tokens; a pair adds both texts."""
    assert kwargs["add_special_tokens"] is True
    assert kwargs["truncation"] is False
    joined = list(zip(*texts, strict=True))
    return {"input_ids": [[0] * (2 + sum(len(part.split()) for part in row)) for row in joined]}


class StandInEncoder:
    max_seq_length: int | None = 6

    def __init__(self) -> None:
        self.tokenizer: Tokenizer = fake_tokenizer
        self.calls: list[dict[str, object]] = []

    def encode(
        self,
        inputs: Sequence[str],
        *,
        batch_size: int,
        normalize_embeddings: bool,
        convert_to_numpy: bool,
        show_progress_bar: bool,
    ) -> np.ndarray:
        self.calls.append(
            {
                "n": len(inputs),
                "batch_size": batch_size,
                "normalize": normalize_embeddings,
                "numpy": convert_to_numpy,
                "progress": show_progress_bar,
            }
        )
        return np.tile(np.array([0.6, 0.8]), (len(inputs), 1))  # float64 on purpose


class StandInPairModel:
    max_length: int | None = 6

    def __init__(self) -> None:
        self.tokenizer: Tokenizer = fake_tokenizer
        self.seen: list[tuple[str, str]] = []

    def predict(
        self, inputs: Sequence[tuple[str, str]], *, batch_size: int, show_progress_bar: bool
    ) -> np.ndarray:
        assert batch_size == 16
        assert show_progress_bar is False
        self.seen = list(inputs)
        return np.array([float(len(p.split())) for _, p in inputs], dtype=np.float32)


def test_embedder_asks_for_normalized_numpy_vectors_in_float32() -> None:
    model = StandInEncoder()
    vectors = SentenceTransformerEmbedder(model).encode(["a", "b", "c"])
    assert vectors.dtype == np.float32
    assert vectors.shape == (3, 2)
    assert model.calls == [
        {"n": 3, "batch_size": 32, "normalize": True, "numpy": True, "progress": False}
    ]


def test_embedder_counts_texts_over_the_limit_special_tokens_included() -> None:
    embedder = SentenceTransformerEmbedder(StandInEncoder())
    # limit 6 = 4 words + 2 special tokens: "a b c d" fits exactly, "a b c d e" does not
    assert embedder.count_truncated(["a b c d", "a b c d e", "x", "a b c d e f g"]) == 2
    assert embedder.count_truncated([]) == 0


def test_cross_encoder_scores_pairs_in_order() -> None:
    model = StandInPairModel()
    scores = CrossEncoderReranker(model).score("q", ["one", "one two three"])
    assert scores == [1.0, 3.0]
    assert model.seen == [("q", "one"), ("q", "one two three")]
    assert CrossEncoderReranker(model).score("q", []) == []


def test_cross_encoder_counts_pairs_over_the_limit() -> None:
    reranker = CrossEncoderReranker(StandInPairModel())
    # query 1 word + passage words + 2 special: limit 6 allows passages of up to 3 words
    assert reranker.count_truncated("q", ["a b c", "a b c d", "x"]) == 1
    assert reranker.count_truncated("q", []) == 0


def test_a_model_without_a_limit_is_refused_loudly() -> None:
    encoder = StandInEncoder()
    encoder.max_seq_length = None
    with pytest.raises(ModelError, match="no input length limit"):
        SentenceTransformerEmbedder(encoder)
    pair = StandInPairModel()
    pair.max_length = None
    with pytest.raises(ModelError, match="cross-encoder has no input length limit"):
        CrossEncoderReranker(pair)


@pytest.mark.live
def test_real_models_load_at_their_pinned_revisions_and_behave() -> None:
    """Downloads the configured models (needs the Hugging Face hosts)."""
    settings = load_settings(Path(__file__).resolve().parents[2] / "config" / "default.toml")
    long_text = "word " * 700
    for spec in settings.embeddings.candidates:
        embedder = load_embedder(spec)
        query = spec.query_prefix + "how do I split data for testing?"
        related = spec.passage_prefix + "train_test_split splits arrays into train and test sets"
        unrelated = spec.passage_prefix + "the cat sat on the warm windowsill all afternoon"
        q, a, b = embedder.encode([query, related, unrelated])
        assert np.allclose(np.linalg.norm([q, a, b], axis=1), 1.0, atol=1e-4), spec.id
        assert float(q @ a) > float(q @ b), spec.id
        assert embedder.count_truncated([long_text, "short"]) == 1, spec.id
    reranker = load_cross_encoder(settings.rerank.model)
    good, bad = reranker.score(
        "how many people live in berlin",
        ["Berlin has 3.5 million inhabitants.", "Cats sleep a lot."],
    )
    assert good > bad
    assert reranker.count_truncated("q", [long_text, "short"]) == 1
