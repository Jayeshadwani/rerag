"""The torch-against-ONNX comparison: hand-computed differences, the stop rule, an offline run."""

from collections.abc import Sequence

import numpy as np
import pytest

from helpers_retrieval import CONTEXT, MARKER, decided_settings, make_gold, make_resources
from noterecall.chunking.tokens import Tokenizer
from noterecall.config import load_settings
from noterecall.eval.onnx_parity import TOLERANCE, cosine_difference, parity_decision, run_parity
from noterecall.eval.run_retrieval import RunError
from noterecall.index.embeddings import EmbeddingService, EmbeddingSpec, HashEmbedder
from noterecall.models import Cell

Notebooks = list[tuple[str, list[Cell]]]


class Perturbed:
    """The hash embedder with a small deterministic disturbance: a stand-in for another runtime."""

    def __init__(self) -> None:
        self._inner = HashEmbedder()

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        vectors = self._inner.encode(texts)
        noise = np.sin(np.arange(vectors.size, dtype=np.float32).reshape(vectors.shape)) * 0.01
        mixed = vectors + noise
        return (mixed / np.linalg.norm(mixed, axis=1, keepdims=True)).astype(np.float32)

    def count_truncated(self, texts: Sequence[str]) -> int:
        return 0


def test_cosine_difference_is_hand_computed() -> None:
    q_t, q_o = np.array([[1.0, 0.0]]), np.array([[0.6, 0.8]])
    c_t, c_o = np.array([[1.0, 0.0], [0.0, 1.0]]), np.array([[1.0, 0.0], [0.6, 0.8]])
    result = cosine_difference((q_t, q_o), (c_t, c_o))
    # torch cosines: q_t . c_t = [1, 0]. onnx cosines: q_o . (1, 0) = 0.6 and
    # q_o . (0.6, 0.8) = 0.36 + 0.64 = 1.0, so the differences are |1 - 0.6| = 0.4 and
    # |0 - 1.0| = 1.0: max 1.0, mean 0.7
    assert result.pairs == 2
    assert result.max_abs_cosine_difference == pytest.approx(1.0)
    assert result.mean_abs_cosine_difference == pytest.approx(0.7)
    # chunk 1 is identical in both (self cosine 1); chunk 2: (0, 1) . (0.6, 0.8) = 0.8
    assert result.min_self_cosine_chunks == pytest.approx(0.8)
    assert result.min_self_cosine_queries == pytest.approx(0.6)  # (1, 0) . (0.6, 0.8)


def test_identical_runtimes_have_no_difference() -> None:
    q = np.array([[0.6, 0.8], [1.0, 0.0]])
    c = np.array([[0.0, 1.0], [0.8, 0.6]])
    result = cosine_difference((q, q), (c, c))
    assert result.max_abs_cosine_difference == 0.0
    assert result.min_self_cosine_chunks == pytest.approx(1.0)


def test_the_stop_rule_allows_a_drop_of_up_to_the_tolerance_only() -> None:
    assert TOLERANCE == 0.01
    assert parity_decision(0.8, 0.8)[0] == "pass"
    assert parity_decision(0.8, 0.85)[0] == "pass"  # better is fine
    assert parity_decision(0.8, 0.79)[0] == "pass"  # exactly the tolerance passes
    assert parity_decision(0.8, 0.789)[0] == "stop"
    decision, reason = parity_decision(0.83, 0.80)
    assert decision == "stop"
    assert "do not deploy" in reason
    assert "0.0300" in reason


def test_an_offline_run_reports_both_runtimes_and_never_query_text(
    config: object, notebooks: Notebooks, tiny_tokenizer: Tokenizer
) -> None:
    settings = decided_settings(load_settings(config))  # type: ignore[arg-type]
    gold = make_gold(notebooks)
    resources = make_resources(settings, notebooks, tiny_tokenizer)
    spec = next(c for c in settings.embeddings.candidates if c.id == "small")
    other = EmbeddingService(Perturbed(), spec, None, namespace="onnx")
    results = run_parity(settings, gold, resources, other, CONTEXT)
    assert results.torch.key == results.onnx.key  # the same configuration on both runtimes
    assert set(results.torch.per_query) == set(results.onnx.per_query)
    assert results.recall10_comparison.report_only
    assert results.cosine.pairs > 0
    assert results.cosine.max_abs_cosine_difference >= 0.0
    want, _ = parity_decision(results.torch.means["recall@10"], results.onnx.means["recall@10"])
    assert results.decision == want
    assert MARKER not in results.model_dump_json()
    # an identical runtime is a perfect pass
    same = EmbeddingService(HashEmbedder(), spec, None, namespace="onnx")
    perfect = run_parity(settings, gold, resources, same, CONTEXT)
    assert perfect.decision == "pass"
    assert perfect.cosine.max_abs_cosine_difference == 0.0
    assert perfect.recall10_comparison.ci.mean == 0.0


def test_a_run_needs_the_decided_dense_model(
    config: object, notebooks: Notebooks, tiny_tokenizer: Tokenizer
) -> None:
    settings = load_settings(config)  # type: ignore[arg-type]
    resources = make_resources(settings, notebooks, tiny_tokenizer)
    spec = EmbeddingSpec(id="small", revision="r", query_prefix="", passage_prefix="")
    with pytest.raises(RunError, match=r"no .retrieval.decided. with a dense model"):
        run_parity(
            settings,
            make_gold(notebooks),
            resources,
            EmbeddingService(HashEmbedder(), spec),
            CONTEXT,
        )
