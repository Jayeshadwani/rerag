"""Structural chunker: hand-computed chunks. The offline tokenizer counts one token per UTF-8 byte,
so every expected size below is plain arithmetic on string lengths."""

import pytest

from noterecall.chunking.structural import (
    ChunkingError,
    _flags,
    build_sections,
    chunk_id,
    chunk_notebook,
    pack,
    render,
)
from noterecall.chunking.tokens import Tokenizer
from noterecall.models import (
    Cell,
    CellType,
    OversizeReason,
    Piece,
    PieceKind,
    PieceUnit,
)

MD, CODE = CellType.MARKDOWN, CellType.CODE


def cell(index: int, cell_type: CellType, source: str, output: str = "") -> Cell:
    return Cell(
        index=index,
        cell_type=cell_type,
        raw_hash=f"raw{index}",
        clean_hash=f"clean{index}",
        source=source,
        output_text=output,
    )


def units(chunks: list) -> list[list[tuple[int, int, str]]]:  # type: ignore[type-arg]
    return [[(p.cell_index, p.part, p.unit.value) for p in c.pieces] for c in chunks]


def test_everything_fits_in_one_chunk(tiny_tokenizer: Tokenizer) -> None:
    cells = [cell(0, MD, "# T\nintro"), cell(1, CODE, "x = 1")]
    (chunk,) = chunk_notebook("nb.ipynb", cells, 100, tiny_tokenizer)
    # "# T\nintro" is 9 bytes, "x = 1" is 5, joined by a 2-byte blank line: 9 + 2 + 5 = 16
    assert chunk.text == "# T\nintro\n\nx = 1"
    assert chunk.tokens == 16
    assert chunk.heading_path == ("T",)
    assert chunk.cell_hashes == ("raw0", "raw1")
    assert chunk.flags == ()


def test_cells_are_not_split_when_the_chunk_is_full(tiny_tokenizer: Tokenizer) -> None:
    cells = [cell(0, MD, "# T\nintro"), cell(1, CODE, "x = 1")]
    chunks = chunk_notebook("nb.ipynb", cells, 12, tiny_tokenizer)  # 16 > 12, so a cell boundary
    assert [c.text for c in chunks] == ["# T\nintro", "x = 1"]
    assert [c.tokens for c in chunks] == [9, 5]
    assert all(c.heading_path == ("T",) for c in chunks)  # both chunks belong to the section


def test_new_section_at_each_heading_cell(tiny_tokenizer: Tokenizer) -> None:
    cells = [cell(0, MD, "# A"), cell(1, CODE, "a=1"), cell(2, MD, "## B"), cell(3, CODE, "b=2")]
    chunks = chunk_notebook("nb.ipynb", cells, 1000, tiny_tokenizer)
    assert [c.text for c in chunks] == [
        "# A\n\na=1",
        "## B\n\nb=2",
    ]  # never merged across a heading
    assert [c.heading_path for c in chunks] == [("A",), ("A", "B")]


def test_heading_path_pops_to_the_right_level() -> None:
    cells = [
        cell(0, MD, "# C"),
        cell(1, MD, "## D"),
        cell(2, MD, "### E"),
        cell(3, MD, "## F"),
        cell(4, MD, "# G"),
    ]
    assert [s.path for s in build_sections(cells)] == [
        ("C",),
        ("C", "D"),
        ("C", "D", "E"),
        ("C", "F"),  # an H2 closes the H3 and the earlier H2
        ("G",),
    ]


def test_cells_before_the_first_heading_form_a_preamble_section() -> None:
    sections = build_sections([cell(0, CODE, "import os"), cell(1, MD, "# A"), cell(2, CODE, "x")])
    assert [(s.path, [c.index for c in s.cells]) for s in sections] == [((), [0]), (("A",), [1, 2])]


def test_a_cell_with_several_headings_is_atomic() -> None:
    cells = [cell(0, MD, "# A\ntext\n## B\nmore"), cell(1, CODE, "x"), cell(2, MD, "### C")]
    sections = build_sections(cells)
    assert [(s.path, [c.index for c in s.cells]) for s in sections] == [
        (("A",), [0, 1]),  # the section path is the cell's first heading
        (("A", "B", "C"), [2]),  # but later headings in the cell still count for what follows
    ]


def test_headings_inside_fences_do_not_start_sections() -> None:
    sections = build_sections([cell(0, MD, "intro\n```\n# not a heading\n```"), cell(1, CODE, "x")])
    assert [s.path for s in sections] == [()]


def test_oversized_code_cell_is_cut_at_statements(tiny_tokenizer: Tokenizer) -> None:
    chunks = chunk_notebook("nb", [cell(0, CODE, "a = 1\nb = 2\nc = 3\n")], 12, tiny_tokenizer)
    # the cell is 17 bytes (> 12); statements are 5 bytes each once the newline is trimmed.
    # a+b = "a = 1\nb = 2" = 11 <= 12; adding c = 17 > 12.
    assert [c.text for c in chunks] == ["a = 1\nb = 2", "c = 3"]
    assert units(chunks) == [[(0, 0, "statement"), (0, 1, "statement")], [(0, 2, "statement")]]
    assert all(c.flags == () for c in chunks)
    assert [c.cell_hashes for c in chunks] == [("raw0",), ("raw0",)]


def test_a_single_long_statement_is_flagged(tiny_tokenizer: Tokenizer) -> None:
    source = "x = 'aaaaaaaaaaaaaaaaaaaa'"  # 26 bytes
    (chunk,) = chunk_notebook("nb", [cell(0, CODE, source)], 10, tiny_tokenizer)
    assert chunk.tokens == 26
    assert chunk.flags == (OversizeReason.STATEMENT,)


def test_unparseable_code_over_the_cap_is_one_flagged_piece(tiny_tokenizer: Tokenizer) -> None:
    (chunk,) = chunk_notebook("nb", [cell(0, CODE, "def f(:\n  aaaaaaaa")], 5, tiny_tokenizer)
    assert units([chunk]) == [[(0, 0, "unparsed_code")]]
    assert chunk.flags == (OversizeReason.UNPARSED_CODE,)


def test_oversized_markdown_is_cut_at_paragraphs(tiny_tokenizer: Tokenizer) -> None:
    text = "Alpha one. Beta two.\n\nGamma three. Delta four."  # 20 + 2 + 24 = 46 bytes
    chunks = chunk_notebook("nb", [cell(0, MD, text)], 25, tiny_tokenizer)
    # each paragraph fits (20, 24); together 46 > 25
    assert [c.text for c in chunks] == ["Alpha one. Beta two.", "Gamma three. Delta four."]
    assert units(chunks) == [[(0, 0, "paragraph")], [(0, 1, "paragraph")]]


def test_oversized_paragraphs_are_cut_at_sentences(tiny_tokenizer: Tokenizer) -> None:
    text = "Alpha one. Beta two.\n\nGamma three. Delta four."
    chunks = chunk_notebook("nb", [cell(0, MD, text)], 15, tiny_tokenizer)
    # sentences are 11, 9 (trimmed), 13 and 11 bytes; any two together exceed 15,
    # so each sentence is its own chunk.
    assert [c.text for c in chunks] == ["Alpha one. ", "Beta two.", "Gamma three. ", "Delta four."]
    assert all(u[0][2] == "sentence" for u in units(chunks))
    assert all(c.flags == () for c in chunks)


def test_a_sentence_that_is_too_long_is_flagged(tiny_tokenizer: Tokenizer) -> None:
    (chunk,) = chunk_notebook("nb", [cell(0, MD, "## Sub\nBody text here.")], 20, tiny_tokenizer)
    assert chunk.tokens == 22  # "## Sub" (6) + newline (1) + "Body text here." (15)
    assert chunk.flags == (OversizeReason.SENTENCE,)


def test_fenced_block_is_never_split_even_when_too_large(tiny_tokenizer: Tokenizer) -> None:
    text = "intro\n\n```\nline1\nline2\nline3\n```\n"
    chunks = chunk_notebook("nb", [cell(0, MD, text)], 10, tiny_tokenizer)
    fenced = [c for c in chunks if c.pieces[0].unit is PieceUnit.FENCED_BLOCK]
    (chunk,) = fenced
    assert chunk.pieces[0].text == "```\nline1\nline2\nline3\n```\n"  # whole block, one piece
    assert chunk.flags == (OversizeReason.FENCED_BLOCK,)  # 3+1+5+1+5+1+5+1+3 = 25 > 10


def test_outputs_are_never_part_of_chunk_text(tiny_tokenizer: Tokenizer) -> None:
    """Proposal 002 final decision: outputs stay in the database and are never chunked."""
    cells = [
        cell(0, CODE, "x = 6 * 7", output="42"),
        cell(1, CODE, "", output="bare-output"),
        cell(2, CODE, "print(x)", output="UserWarning: something is deprecated"),
    ]
    chunks = chunk_notebook("nb", cells, 1000, tiny_tokenizer)
    assert [c.text for c in chunks] == ["x = 6 * 7\n\nprint(x)"]
    assert {p.kind for c in chunks for p in c.pieces} == {PieceKind.CODE}


def test_empty_inputs_give_no_chunks(tiny_tokenizer: Tokenizer) -> None:
    assert chunk_notebook("nb", [], 100, tiny_tokenizer) == []
    assert chunk_notebook("nb", [cell(0, CODE, "  \n")], 100, tiny_tokenizer) == []


def test_ids_are_deterministic_and_sensitive(tiny_tokenizer: Tokenizer) -> None:
    cells = [cell(0, MD, "# T\nintro"), cell(1, CODE, "x = 1", output="1")]
    first = chunk_notebook("nb", cells, 100, tiny_tokenizer)
    assert [c.id for c in first] == [c.id for c in chunk_notebook("nb", cells, 100, tiny_tokenizer)]
    assert len(first[0].id) == 16
    other_cap = chunk_notebook("nb", cells, 101, tiny_tokenizer)
    other_path = chunk_notebook("other", cells, 100, tiny_tokenizer)
    edited = chunk_notebook("nb", [cells[0], cell(1, CODE, "x = 2")], 100, tiny_tokenizer)
    ids = {c[0].id for c in (first, other_cap, other_path, edited)}
    assert len(ids) == 4


def test_chunk_id_depends_on_piece_order_and_text() -> None:
    a = Piece(cell_index=0, kind=PieceKind.CODE, part=0, unit=PieceUnit.CELL, text="a")
    b = Piece(cell_index=1, kind=PieceKind.CODE, part=0, unit=PieceUnit.CELL, text="b")
    assert chunk_id("nb", 10, [a, b]) != chunk_id("nb", 10, [b, a])


def test_render_joins_parts_of_one_cell_exactly_and_cells_with_a_blank_line() -> None:
    p0 = Piece(cell_index=0, kind=PieceKind.CODE, part=0, unit=PieceUnit.STATEMENT, text="a = 1\n")
    p1 = Piece(
        cell_index=0, kind=PieceKind.CODE, part=1, unit=PieceUnit.STATEMENT, text="\nb = 2\n"
    )
    p2 = Piece(cell_index=1, kind=PieceKind.MARKDOWN, part=0, unit=PieceUnit.CELL, text="text")
    assert render([p0, p1]) == "a = 1\n\nb = 2"
    assert render([p0, p1, p2]) == "a = 1\n\nb = 2\n\ntext"


def test_pack_never_exceeds_the_cap_for_multi_piece_groups(tiny_tokenizer: Tokenizer) -> None:
    pieces = [
        Piece(cell_index=i, kind=PieceKind.CODE, part=0, unit=PieceUnit.CELL, text="x" * n)
        for i, n in enumerate([4, 4, 9, 2, 2])
    ]
    groups = pack(pieces, 10, tiny_tokenizer)
    # 4 + 2 + 4 = 10 fits; adding 9 does not; 9 alone; then 2 + 2 + 2 = 6
    assert [[len(p.text) for p in g] for g in groups] == [[4, 4], [9], [2, 2]]


def test_a_chunk_over_the_cap_that_is_not_a_lone_indivisible_piece_is_a_bug() -> None:
    piece = Piece(cell_index=0, kind=PieceKind.CODE, part=0, unit=PieceUnit.PARAGRAPH, text="x")
    with pytest.raises(ChunkingError):
        _flags([piece], tokens=99, cap=10)
    with pytest.raises(ChunkingError):
        _flags([piece, piece], tokens=99, cap=10)


def test_identical_cells_in_one_chunk_list_their_hash_once(tiny_tokenizer: Tokenizer) -> None:
    # two cells with the same content have the same raw_hash; labels match on hash, so list it once
    twin_a = cell(0, CODE, "y = [1,\n     2]\n")
    twin_b = Cell(**{**twin_a.model_dump(), "index": 1})
    (chunk,) = chunk_notebook("nb", [twin_a, twin_b], 1000, tiny_tokenizer)
    assert [p.cell_index for p in chunk.pieces] == [0, 1]  # both cells are in the chunk
    assert chunk.cell_hashes == ("raw0",)  # but the shared hash appears once
