"""Metrics against hand-computed cases (arithmetic in comments)."""

import math

import pytest

from noterecall.eval.metrics import (
    MetricsError,
    chunk_cells,
    count_relevant_chunks,
    is_relevant,
    mrr_at_k,
    ndcg_at_k,
    recall_at_k,
    recall_at_token_budget,
    score_query,
)
from noterecall.models import Chunk


def chunk(cid: str, notebook: str, hashes: tuple[str, ...], tokens: int = 10) -> Chunk:
    return Chunk(
        id=cid, notebook_path=notebook, heading_path=(), pieces=(), text="", tokens=tokens,
        cell_hashes=hashes,
    )  # fmt: skip


RELEVANT = {("a", "h1"), ("a", "h2"), ("b", "h3")}  # 3 relevant cells
C1 = chunk("c1", "a", ("x1",), 10)  # irrelevant
C2 = chunk("c2", "a", ("h1", "x2"), 20)  # covers h1
C3 = chunk("c3", "b", ("h9",), 5)  # irrelevant
C4 = chunk("c4", "b", ("h3",), 30)  # covers h3
C5 = chunk("c5", "a", ("h2",), 15)  # covers h2
RANKED = [C1, C2, C3, C4, C5]
UNIVERSE = [*RANKED, chunk("c6", "a", ("x3",)), chunk("c7", "c", ("h1",))]  # c7: h1 in notebook c


def test_a_chunk_covers_cells_of_its_own_notebook_only() -> None:
    assert chunk_cells(C2) == {("a", "h1"), ("a", "x2")}
    assert is_relevant(C2, RELEVANT)
    assert not is_relevant(chunk("z", "b", ("h1",)), RELEVANT)  # h1 is relevant in a, not in b


@pytest.mark.parametrize(
    ("k", "expected"),
    [(1, 0.0), (2, 1 / 3), (3, 1 / 3), (4, 2 / 3), (5, 1.0), (50, 1.0)],
)
def test_recall_at_k(k: int, expected: float) -> None:
    # top-1 = c1: 0 of 3; top-2 adds h1: 1/3; top-4 adds h3: 2/3; top-5 adds h2: 3/3
    assert recall_at_k(RANKED, RELEVANT, k) == pytest.approx(expected)


def test_recall_counts_each_cell_once_even_if_two_chunks_cover_it() -> None:
    twice = [chunk("p", "a", ("h1",)), chunk("q", "a", ("h1", "h2"))]
    assert recall_at_k(twice, RELEVANT, 2) == pytest.approx(2 / 3)  # h1, h2 of {h1, h2, h3}


@pytest.mark.parametrize(("k", "expected"), [(1, 0.0), (2, 0.5), (10, 0.5)])
def test_mrr_at_k(k: int, expected: float) -> None:
    assert mrr_at_k(RANKED, RELEVANT, k) == expected  # first relevant chunk is c2 at rank 2


def test_mrr_is_zero_when_nothing_relevant_is_retrieved() -> None:
    assert mrr_at_k([C1, C3], RELEVANT, 10) == 0.0


def test_ndcg_at_5() -> None:
    assert count_relevant_chunks(UNIVERSE, RELEVANT) == 3  # c2, c4, c5 (c7 is notebook c)
    # relevant at ranks 2, 4, 5: DCG = 1/log2(3) + 1/log2(5) + 1/log2(6)
    #   = 0.63093 + 0.43068 + 0.38685 = 1.44846
    # ideal (R=3, k=5): ranks 1..3 = 1 + 0.63093 + 0.5 = 2.13093; nDCG = 1.44846 / 2.13093 = 0.67973
    assert ndcg_at_k(RANKED, RELEVANT, 5, 3) == pytest.approx(0.67973, abs=1e-5)


def test_ndcg_at_2_caps_the_ideal_at_k() -> None:
    # DCG@2 = 1/log2(3) = 0.63093; ideal@2 = min(2, 3) = 2 chunks = 1 + 0.63093 = 1.63093
    assert ndcg_at_k(RANKED, RELEVANT, 2, 3) == pytest.approx(0.38685, abs=1e-5)


def test_ndcg_of_a_perfect_ranking_is_one_and_ideal_uses_available_relevant_chunks() -> None:
    perfect = [C2, C4, C5, C1]
    assert ndcg_at_k(perfect, RELEVANT, 10, 3) == pytest.approx(1.0)
    # only one relevant chunk exists (R=1) and it is first: ideal = 1, DCG = 1
    assert ndcg_at_k([C2, C1], RELEVANT, 10, 1) == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("budget", "expected"),
    [
        (9, 0.0),  # c1 alone is 10 tokens: nothing fits
        (34, 1 / 3),  # c1+c2 = 30; c3 would make 35 > 34: stop. covers h1
        (35, 1 / 3),  # c1+c2+c3 = 35 fits; c4 would make 65: stop
        (62, 1 / 3),  # stops at c4 (65 > 62); c5 is never considered, though 35 + 15 would fit
        (79, 2 / 3),  # c1..c4 = 65 fits, c5 makes 80 > 79: h1 and h3
        (80, 1.0),  # all five = 80
    ],
)
def test_recall_at_token_budget(budget: int, expected: float) -> None:
    assert recall_at_token_budget(RANKED, RELEVANT, budget) == pytest.approx(expected)


def test_score_query_names_and_values() -> None:
    scores = score_query(RANKED, RELEVANT, UNIVERSE, ks=(2, 5), budgets=(35,))
    assert set(scores) == {
        "recall@2", "mrr@2", "ndcg@2", "recall@5", "mrr@5", "ndcg@5", "recall@tokens=35",
    }  # fmt: skip
    assert scores["recall@5"] == pytest.approx(1.0)
    assert scores["ndcg@5"] == pytest.approx(0.67973, abs=1e-5)
    assert scores["recall@tokens=35"] == pytest.approx(1 / 3)
    assert math.isclose(scores["mrr@2"], 0.5)


def test_undefined_inputs_fail_loudly() -> None:
    with pytest.raises(MetricsError, match="no relevant cells"):
        recall_at_k(RANKED, set(), 5)
    with pytest.raises(MetricsError, match="k must be at least 1"):
        recall_at_k(RANKED, RELEVANT, 0)
    with pytest.raises(MetricsError, match="k must be at least 1"):
        mrr_at_k(RANKED, RELEVANT, 0)
    with pytest.raises(MetricsError, match="more than once"):
        mrr_at_k([C1, C1], RELEVANT, 5)
    with pytest.raises(MetricsError, match="no chunk of the corpus"):
        ndcg_at_k(RANKED, RELEVANT, 5, 0)
    with pytest.raises(MetricsError, match="at least 1 token"):
        recall_at_token_budget(RANKED, RELEVANT, 0)
