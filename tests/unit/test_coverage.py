"""Token coverage within a budget, hand-computed (byte tokenizer: one byte is one token)."""

import pytest

from noterecall.chunking.tokens import Tokenizer
from noterecall.eval.coverage import CoverageError, CoverageIndex
from noterecall.eval.metrics import MetricsError, budget_prefix
from noterecall.models import Cell, CellType, Chunk, Piece, PieceKind, PieceUnit


def cell(index: int, source: str, raw: str) -> Cell:
    return Cell(index=index, cell_type=CellType.CODE, raw_hash=raw, clean_hash=raw, source=source)


def chunk(cid: str, tokens: int, *held: tuple[int, str]) -> Chunk:
    pieces = tuple(
        Piece(cell_index=i, kind=PieceKind.CODE, part=n, unit=PieceUnit.STATEMENT, text=t)
        for n, (i, t) in enumerate(held)
    )
    return Chunk(
        id=cid, notebook_path="n", heading_path=(), pieces=pieces, text="", tokens=tokens,
        cell_hashes=(),
    )  # fmt: skip


NOTEBOOKS = [
    (
        "n",
        [
            cell(0, "aaaa", "h0"),  # 4 tokens
            cell(1, "bbbbbbbb", "h1"),  # 8 tokens
            cell(2, "aaaa", "h0"),  # an identical cell in the same notebook
            cell(3, "\n", "blank"),  # no tokens
        ],
    )
]
C1 = chunk("c1", 4, (0, "aaaa\n"))  # all of cell 0
C2A = chunk("c2a", 4, (1, "bbbb"))  # half of cell 1
C2B = chunk("c2b", 4, (1, "bbbb"))  # the other half
C4 = chunk("c4", 4, (2, "aaaa"))  # all of the identical cell 2
RELEVANT = {("n", "h0"), ("n", "h1")}  # 4 + 8 = 12 tokens


def index(tiny_tokenizer: Tokenizer) -> CoverageIndex:
    return CoverageIndex(NOTEBOOKS, tiny_tokenizer)


@pytest.mark.parametrize(
    ("budget", "expected"),
    [
        (3, 0.0),  # the first chunk (4 tokens) does not fit: nothing is held
        (4, 4 / 12),  # c1: all of cell 0
        (8, 8 / 12),  # c1 + c2a: cell 0 (4) + half of cell 1 (4)
        (12, 1.0),  # all three: 4 + 8 = 12 of 12
        (100, 1.0),
    ],
)
def test_coverage_is_the_share_of_relevant_tokens_inside_the_budgeted_chunks(
    tiny_tokenizer: Tokenizer, budget: int, expected: float
) -> None:
    assert index(tiny_tokenizer).at_budget([C1, C2A, C2B], RELEVANT, budget) == pytest.approx(
        expected
    )


def test_a_partly_held_cell_scores_less_than_a_wholly_held_one(tiny_tokenizer: Tokenizer) -> None:
    idx = index(tiny_tokenizer)
    assert idx.at_budget([C2A], {("n", "h1")}, 10) == pytest.approx(0.5)  # 4 of 8 tokens
    assert idx.at_budget([C2A, C2B], {("n", "h1")}, 10) == pytest.approx(1.0)


def test_budget_stops_at_the_first_chunk_that_does_not_fit(tiny_tokenizer: Tokenizer) -> None:
    big = chunk("big", 50, (0, "aaaa"))
    # big does not fit in 10, and the later small chunk is not skipped to
    assert index(tiny_tokenizer).at_budget([big, C2A], {("n", "h1")}, 10) == 0.0


def test_identical_cells_in_one_notebook_count_once_and_either_copy_may_hold_the_label(
    tiny_tokenizer: Tokenizer,
) -> None:
    # the label (n, h0) names cells 0 and 2; the chunk holding cell 2 covers it fully, once
    assert index(tiny_tokenizer).at_budget([C4], {("n", "h0")}, 10) == pytest.approx(1.0)
    assert index(tiny_tokenizer).at_budget([C1, C4], {("n", "h0")}, 10) == pytest.approx(1.0)


def test_pieces_are_counted_by_their_text_without_surrounding_newlines(
    tiny_tokenizer: Tokenizer,
) -> None:
    padded = chunk("p", 6, (0, "\naaaa\n\n"))
    assert index(tiny_tokenizer).at_budget([padded], {("n", "h0")}, 10) == pytest.approx(1.0)


def test_coverage_never_exceeds_one_when_pieces_overlap_a_cell_twice(
    tiny_tokenizer: Tokenizer,
) -> None:
    twice = [chunk("x", 4, (0, "aaaa")), chunk("y", 4, (0, "aaaa"))]
    assert index(tiny_tokenizer).at_budget(twice, {("n", "h0")}, 10) == 1.0


def test_undefined_inputs_fail_loudly(tiny_tokenizer: Tokenizer) -> None:
    idx = index(tiny_tokenizer)
    with pytest.raises(MetricsError, match="no relevant cells"):
        idx.at_budget([C1], set(), 10)
    with pytest.raises(MetricsError, match="at least 1 token"):
        idx.at_budget([C1], RELEVANT, 0)
    with pytest.raises(CoverageError, match="not in the corpus"):
        idx.at_budget([C1], {("n", "unknown")}, 10)
    with pytest.raises(CoverageError, match="have no tokens"):
        idx.at_budget([C1], {("n", "blank")}, 10)


def test_budget_prefix_is_the_shared_rank_order_prefix() -> None:
    assert budget_prefix([C1, C2A, C2B], 8) == [C1, C2A]
    assert budget_prefix([C1, C2A, C2B], 3) == []
    assert budget_prefix([], 5) == []


def test_newlines_around_a_piece_are_not_counted_as_held_tokens(tiny_tokenizer: Tokenizer) -> None:
    padded = chunk("p", 6, (1, "bbbb\n\n"))  # 4 tokens of cell 1 (8 tokens), plus padding
    assert index(tiny_tokenizer).at_budget([padded], {("n", "h1")}, 10) == pytest.approx(0.5)
