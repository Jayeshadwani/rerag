"""Reciprocal rank fusion against hand-computed scores."""

import pytest

from noterecall.retrieval.fusion import FusionError, rrf


def scores(hits) -> dict[str, float]:  # type: ignore[no-untyped-def]  # noqa: ANN001
    return {h.chunk_id: h.score for h in hits}


def test_two_lists_with_k_60() -> None:
    hits = rrf([["a", "b", "c"], ["b", "d", "a"]], k=60)
    # a: 1/61 + 1/63 = 0.016393 + 0.015873 = 0.032266
    # b: 1/62 + 1/61 = 0.016129 + 0.016393 = 0.032522
    # c: 1/63 = 0.015873;  d: 1/62 = 0.016129
    got = scores(hits)
    assert got["a"] == pytest.approx(1 / 61 + 1 / 63)
    assert got["b"] == pytest.approx(1 / 62 + 1 / 61)
    assert got["c"] == pytest.approx(1 / 63)
    assert got["d"] == pytest.approx(1 / 62)
    assert [h.chunk_id for h in hits] == ["b", "a", "d", "c"]
    assert [h.rank for h in hits] == [1, 2, 3, 4]


def test_a_small_k_rewards_the_top_ranks_more() -> None:
    lists = [["x", "p", "q", "r"], ["s", "t", "u", "p"]]
    # k = 1: x = 1/2 = 0.5 but p = 1/3 + 1/5 = 0.5333, so p wins
    first = rrf(lists, k=1)[0]
    assert first.chunk_id == "p"
    assert first.score == pytest.approx(1 / 3 + 1 / 5)
    # k = 100: x = 1/101 = 0.0099 and p = 1/102 + 1/104 = 0.0194: p still wins; but s (1/101) ties x
    big = {h.chunk_id: h.score for h in rrf(lists, k=100)}
    assert big["p"] == pytest.approx(1 / 102 + 1 / 104)
    assert big["x"] == pytest.approx(big["s"])


def test_ties_break_by_chunk_id_whatever_the_list_order() -> None:
    first = rrf([["b"], ["a"]], k=60)  # both 1/61
    second = rrf([["a"], ["b"]], k=60)
    assert [h.chunk_id for h in first] == ["a", "b"] == [h.chunk_id for h in second]


def test_single_list_keeps_its_order_and_empty_input_is_empty() -> None:
    assert [h.chunk_id for h in rrf([["z", "y", "x"]], k=10)] == ["z", "y", "x"]
    assert rrf([], k=60) == []
    assert rrf([[], []], k=60) == []


def test_invalid_inputs_fail_loudly() -> None:
    with pytest.raises(FusionError, match="k must be at least 1"):
        rrf([["a"]], k=0)
    with pytest.raises(FusionError, match="more than once"):
        rrf([["a", "a"]], k=60)
