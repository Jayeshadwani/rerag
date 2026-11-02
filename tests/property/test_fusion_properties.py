"""Properties of reciprocal rank fusion."""

from hypothesis import example, given
from hypothesis import strategies as st

from noterecall.retrieval.fusion import rrf

IDS = st.lists(st.sampled_from(list("abcdefgh")), unique=True, max_size=8)
RANKINGS = st.lists(IDS, min_size=0, max_size=4)


@given(RANKINGS, st.integers(1, 100))
@example(
    [["a", "c", "b"], ["a", "b"], ["a", "b"]], 1
)  # a plain running sum differs in the last bit
def test_result_is_independent_of_list_order(rankings: list[list[str]], k: int) -> None:
    assert rrf(rankings, k) == rrf(list(reversed(rankings)), k)


@given(RANKINGS, st.integers(1, 100))
def test_every_input_chunk_appears_once_with_descending_scores(
    rankings: list[list[str]], k: int
) -> None:
    hits = rrf(rankings, k)
    assert sorted(h.chunk_id for h in hits) == sorted({c for r in rankings for c in r})
    assert [h.score for h in hits] == sorted((h.score for h in hits), reverse=True)
    assert [h.rank for h in hits] == list(range(1, len(hits) + 1))


@given(IDS, IDS, st.integers(1, 100))
def test_a_chunk_ranked_first_in_any_list_scores_at_least_one_over_k_plus_one(
    left: list[str], right: list[str], k: int
) -> None:
    hits = {h.chunk_id: h.score for h in rrf([left, right], k)}
    for ranking in (left, right):
        if ranking:
            assert hits[ranking[0]] >= 1 / (k + 1) - 1e-12


@given(RANKINGS, st.integers(1, 100))
def test_score_is_the_sum_of_one_over_k_plus_rank(rankings: list[list[str]], k: int) -> None:
    hits = {h.chunk_id: h.score for h in rrf(rankings, k)}
    for chunk_id, score in hits.items():
        expected = sum(1 / (k + r.index(chunk_id) + 1) for r in rankings if chunk_id in r)
        assert abs(score - expected) < 1e-12
