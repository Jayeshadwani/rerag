"""Fixed-token baseline: windows partition the text, boundaries never cut a character."""

import pytest

from noterecall.chunking.fixed import fixed_chunks
from noterecall.chunking.tokens import Tokenizer
from noterecall.models import Cell, CellType, PieceKind, PieceUnit


def cell(index: int, cell_type: CellType, source: str, output: str = "") -> Cell:
    return Cell(
        index=index,
        cell_type=cell_type,
        raw_hash=f"raw{index}",
        clean_hash=f"clean{index}",
        source=source,
        output_text=output,
    )


CELLS = [
    cell(0, CellType.MARKDOWN, "# Title é€\nIntro."),
    cell(1, CellType.CODE, "x = 1\ny = 2\n"),
]
STREAM = (
    "# Title é€\nIntro.\n\nx = 1\ny = 2"  # cell texts joined by a blank line, trailing \n trimmed
)


def test_windows_of_seven_tokens(tiny_tokenizer: Tokenizer) -> None:
    chunks = fixed_chunks("nb", CELLS, 7, tiny_tokenizer)
    # UTF-8 bytes: "# Title" 7 | " é€\n" 1+2+3+1 = 7 | "Intro.\n" 7 | "\nx = 1\n" 7 | "y = 2" 5
    assert [c.text for c in chunks] == ["# Title", " é€\n", "Intro.\n", "\nx = 1\n", "y = 2"]
    assert [c.tokens for c in chunks] == [7, 7, 7, 7, 5]
    assert "".join(c.text for c in chunks) == STREAM


def test_pieces_attribute_each_window_to_the_cells_it_overlaps(tiny_tokenizer: Tokenizer) -> None:
    chunks = fixed_chunks("nb", CELLS, 7, tiny_tokenizer)
    assert [[p.cell_index for p in c.pieces] for c in chunks] == [[0], [0], [0], [1], [1]]
    # window 3 ("\nx = 1\n") starts in the separator, so only "x = 1\n" belongs to cell 1
    assert chunks[3].pieces[0].text == "x = 1\n"
    assert chunks[3].cell_hashes == ("raw1",)
    assert all(p.unit is PieceUnit.WINDOW for c in chunks for p in c.pieces)
    assert all(c.heading_path == () and c.flags == () for c in chunks)


def test_a_window_spanning_two_cells_has_two_pieces(tiny_tokenizer: Tokenizer) -> None:
    (chunk,) = fixed_chunks("nb", CELLS, 1000, tiny_tokenizer)
    assert chunk.text == STREAM
    assert [(p.cell_index, p.kind) for p in chunk.pieces] == [
        (0, PieceKind.MARKDOWN),
        (1, PieceKind.CODE),
    ]
    assert chunk.cell_hashes == ("raw0", "raw1")


def test_boundary_inside_a_character_moves_to_its_start(tiny_tokenizer: Tokenizer) -> None:
    chunks = fixed_chunks("nb", [cell(0, CellType.CODE, "aaé")], 3, tiny_tokenizer)
    # "aaé" is 4 tokens (a, a, 2 bytes of é); the 3-token boundary falls inside é and moves back
    assert [c.text for c in chunks] == ["aa", "é"]
    assert [c.tokens for c in chunks] == [2, 2]


def test_windows_made_only_of_one_moved_character_are_skipped(tiny_tokenizer: Tokenizer) -> None:
    chunks = fixed_chunks("nb", [cell(0, CellType.CODE, "€")], 1, tiny_tokenizer)
    assert [c.text for c in chunks] == [
        "€"
    ]  # the three one-token windows all start at the same character


def test_outputs_are_never_part_of_the_stream(tiny_tokenizer: Tokenizer) -> None:
    cells = [cell(0, CellType.CODE, "x = 6 * 7", output="42")]
    (chunk,) = fixed_chunks("nb", cells, 1000, tiny_tokenizer)
    assert chunk.text == "x = 6 * 7"
    assert [p.kind for p in chunk.pieces] == [PieceKind.CODE]
    assert fixed_chunks("nb", [cell(0, CellType.CODE, "", output="42")], 1000, tiny_tokenizer) == []


def test_ids_are_deterministic_and_sensitive(tiny_tokenizer: Tokenizer) -> None:
    first = fixed_chunks("nb", CELLS, 7, tiny_tokenizer)
    again = fixed_chunks("nb", CELLS, 7, tiny_tokenizer)
    assert [c.id for c in first] == [c.id for c in again]
    assert len({c.id for c in first}) == len(first)
    assert {c.id for c in first}.isdisjoint(
        c.id for c in fixed_chunks("nb", CELLS, 8, tiny_tokenizer)
    )
    assert {c.id for c in first}.isdisjoint(
        c.id for c in fixed_chunks("other", CELLS, 7, tiny_tokenizer)
    )


def test_empty_notebooks_and_bad_windows(tiny_tokenizer: Tokenizer) -> None:
    assert fixed_chunks("nb", [], 7, tiny_tokenizer) == []
    assert fixed_chunks("nb", [cell(0, CellType.CODE, "  \n")], 7, tiny_tokenizer) == []
    with pytest.raises(ValueError, match="window must be at least 1"):
        fixed_chunks("nb", CELLS, 0, tiny_tokenizer)
