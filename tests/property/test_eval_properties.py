"""Properties of the metrics and the split, on generated rankings and gold sets."""

import random

from hypothesis import given
from hypothesis import strategies as st

from noterecall.eval.gold import GoldQuery, QueryType, Split, assign_splits, held_out_count
from noterecall.eval.metrics import (
    chunk_cells,
    count_relevant_chunks,
    is_relevant,
    mrr_at_k,
    ndcg_at_k,
    recall_at_k,
    recall_at_token_budget,
)
from noterecall.models import Chunk

Cells = set[tuple[str, str]]
Corpus = tuple[list[Chunk], Cells]
Ranking = tuple[list[Chunk], list[Chunk], Cells]

NOTEBOOKS = ["a", "b", "c"]
HASHES = ["h0", "h1", "h2", "h3", "h4", "h5"]


@st.composite
def corpora(draw: st.DrawFn) -> Corpus:
    """A chunked corpus (unique chunk ids) and a non-empty relevant set drawn from its cells."""
    size = draw(st.integers(1, 12))
    chunks = [
        Chunk(
            id=f"c{i}",
            notebook_path=draw(st.sampled_from(NOTEBOOKS)),
            heading_path=(),
            pieces=(),
            text="",
            tokens=draw(st.integers(1, 40)),
            cell_hashes=tuple(
                draw(st.lists(st.sampled_from(HASHES), min_size=1, max_size=3, unique=True))
            ),
        )
        for i in range(size)
    ]
    cells = sorted({cell for c in chunks for cell in chunk_cells(c)})
    relevant = set(draw(st.lists(st.sampled_from(cells), min_size=1, max_size=4, unique=True)))
    return chunks, relevant


@st.composite
def rankings(draw: st.DrawFn) -> Ranking:
    chunks, relevant = draw(corpora())
    order = draw(st.permutations(chunks))
    return chunks, list(order[: draw(st.integers(0, len(order)))]), relevant


@given(rankings(), st.integers(1, 15))
def test_scores_stay_in_range(data: Ranking, k: int) -> None:
    universe, ranked, relevant = data
    assert 0.0 <= recall_at_k(ranked, relevant, k) <= 1.0
    assert 0.0 <= mrr_at_k(ranked, relevant, k) <= 1.0
    r = count_relevant_chunks(universe, relevant)
    assert 0.0 <= ndcg_at_k(ranked, relevant, k, r) <= 1.0 + 1e-9


@given(rankings())
def test_recall_and_budget_recall_never_decrease_as_the_cutoff_grows(data: Ranking) -> None:
    _, ranked, relevant = data
    recalls = [recall_at_k(ranked, relevant, k) for k in range(1, len(ranked) + 2)]
    assert recalls == sorted(recalls)
    budgets = [recall_at_token_budget(ranked, relevant, b) for b in range(1, 200, 7)]
    assert budgets == sorted(budgets)


@given(rankings())
def test_a_budget_covering_every_chunk_equals_full_recall(data: Ranking) -> None:
    _, ranked, relevant = data
    total = sum(c.tokens for c in ranked) or 1
    assert recall_at_token_budget(ranked, relevant, total) == recall_at_k(
        ranked, relevant, len(ranked) + 1
    )


@given(corpora(), st.integers(1, 15), st.randoms(use_true_random=False))
def test_putting_relevant_chunks_first_gives_ndcg_one(
    data: Corpus, k: int, rng: random.Random
) -> None:
    chunks, relevant = data
    good = [c for c in chunks if is_relevant(c, relevant)]
    bad = [c for c in chunks if not is_relevant(c, relevant)]
    rng.shuffle(good)
    rng.shuffle(bad)
    r = len(good)
    assert abs(ndcg_at_k([*good, *bad], relevant, k, r) - 1.0) < 1e-9
    assert mrr_at_k([*good, *bad], relevant, k) == 1.0


@given(corpora())
def test_all_cells_covered_means_recall_one(data: Corpus) -> None:
    chunks, relevant = data
    assert recall_at_k(chunks, relevant, len(chunks)) == 1.0  # relevant cells come from the corpus


QUERY_TYPES = st.sampled_from(list(QueryType))


@st.composite
def query_sets(draw: st.DrawFn) -> tuple[GoldQuery, ...]:
    types = draw(st.lists(QUERY_TYPES, max_size=80))
    return tuple(GoldQuery(id=f"q{i:03d}", text="t", type=t) for i, t in enumerate(types))


@given(query_sets(), st.integers(0, 10**6))
def test_split_is_a_stratified_partition_of_the_right_size(
    queries: tuple[GoldQuery, ...], seed: int
) -> None:
    assignment = assign_splits(queries, seed)
    assert set(assignment) == {q.id for q in queries}
    for qtype in QueryType:
        stratum = [assignment[q.id] for q in queries if q.type is qtype]
        assert stratum.count(Split.TEST) == held_out_count(len(stratum))
        assert stratum.count(Split.TEST) + stratum.count(Split.DEV) == len(stratum)


@given(query_sets(), st.integers(0, 10**6), st.randoms(use_true_random=False))
def test_split_does_not_depend_on_file_order(
    queries: tuple[GoldQuery, ...], seed: int, rng: random.Random
) -> None:
    shuffled = list(queries)
    rng.shuffle(shuffled)
    assert assign_splits(tuple(shuffled), seed) == assign_splits(queries, seed)
