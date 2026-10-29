"""Property tests for the chunkers (M2 acceptance criteria).

The checks use test-local code (a fence scanner, a statement counter, a heading finder) written
independently of the implementation, so they can catch real chunker bugs instead of echoing them.
The offline tokenizer counts one token per UTF-8 byte.
"""

import ast
import hashlib
import itertools
import re
import warnings

import tiktoken
from hypothesis import example, given
from hypothesis import strategies as st

from noterecall.chunking.fixed import fixed_chunks
from noterecall.chunking.structural import chunk_notebook, render
from noterecall.chunking.tokens import Tokenizer
from noterecall.models import (
    Cell,
    CellType,
    Chunk,
    OversizeReason,
    PieceKind,
    PieceUnit,
)

TOKENIZER = Tokenizer(
    tiktoken.Encoding(
        "tiny-bytes",
        pat_str=r"[\s\S]",
        mergeable_ranks={bytes([i]): i for i in range(256)},
        special_tokens={},
    )
)

# ----------------------------------------------------------------------------- strategies

WORD = st.text(alphabet="abcdefgé€", min_size=1, max_size=8)
SENTENCE = st.builds(lambda w, end: w.capitalize() + end, WORD, st.sampled_from([".", "!", "?"]))
PARAGRAPH = st.lists(SENTENCE, min_size=1, max_size=4).map(" ".join)
HEADING = st.builds(lambda n, w: "#" * n + " " + w, st.integers(1, 4), WORD)
FENCE_LINES = st.lists(st.sampled_from(["x = 1", "# not a heading", "", "text."]), max_size=4)
FENCE = st.builds(
    lambda marker, lines, closed: "\n".join([marker, *lines, *([marker] if closed else [])]),
    st.sampled_from(["```", "~~~", "````"]),
    FENCE_LINES,
    st.booleans(),
)
SEPARATOR = st.sampled_from(["\n", "\n\n", "\n\n\n"])


def _joined(blocks: list[str]) -> st.SearchStrategy[str]:
    seps = st.lists(SEPARATOR, min_size=len(blocks), max_size=len(blocks))
    return seps.map(lambda s: "".join(b + sep for b, sep in zip(blocks, s, strict=True)))


MARKDOWN = st.lists(st.one_of(PARAGRAPH, HEADING, FENCE), min_size=1, max_size=5).flatmap(_joined)

STATEMENTS = [
    "x = 1",
    "y = [1,\n     2]",
    "def f():\n    return 1",
    "@dec\ndef g():\n    pass",
    "class C:\n    a = 1",
    "if x:\n    y\nelse:\n    z",
    "# comment",
    "s = '''a\n\nb'''",
    "%matplotlib inline",
    "!ls",
    "a = 1; b = 2",
    "for i in r:\n    pass",
    "import os",
    "def (:",  # does not parse: the whole cell is then one unparsed piece
]
CODE = st.lists(st.sampled_from(STATEMENTS), min_size=1, max_size=6).flatmap(_joined)

OUTPUT_LINES = [
    "42",
    "Accuracy: 0.9",
    "UserWarning: x",
    "a  b\nc  d",
    "[1 2 3 4 5 6 7 8 9 10]",
    "some output text",
    "50%|##| 1/1 [00:00<00:00]",
]
OUTPUT = st.lists(st.sampled_from(OUTPUT_LINES), max_size=4).map("\n".join)


def _cell(index: int, is_code: bool, source: str, output: str) -> Cell:
    return Cell(
        index=index,
        cell_type=CellType.CODE if is_code else CellType.MARKDOWN,
        raw_hash=hashlib.sha256(source.encode()).hexdigest()[:12],  # equal sources share a hash
        clean_hash="c",
        source=source,
        output_text=output if is_code else "",
    )


@st.composite
def notebooks(draw: st.DrawFn) -> list[Cell]:
    entries = draw(
        st.lists(
            st.tuples(st.booleans(), st.integers(0, 2), MARKDOWN, CODE, OUTPUT),
            max_size=8,
        )
    )
    cells, index = [], 0
    for is_code, gap, markdown, code, output in entries:
        index += gap
        cells.append(_cell(index, is_code, code if is_code else markdown, output))
        index += 1
    return cells


CAPS = st.integers(min_value=6, max_value=120)

# ----------------------------------------------------------------------- independent helpers

_OPEN = re.compile(r"^ {0,3}(`{3,}|~{3,})(.*)$")
_CLOSE = re.compile(r"^ {0,3}(`{3,}|~{3,})[ \t]*$")
_ATX = re.compile(r"^ {0,3}#{1,6}[ \t]+\S")


def fenced_ranges(text: str) -> list[tuple[int, int]]:
    ranges, pos, opener, start = [], 0, None, 0
    for line in text.split("\n"):
        end = min(pos + len(line) + 1, len(text))
        if opener is None:
            m = _OPEN.match(line)
            if m and not (m.group(1)[0] == "`" and "`" in m.group(2)):
                opener, start = m.group(1), pos
        else:
            m = _CLOSE.match(line)
            if m and m.group(1)[0] == opener[0] and len(m.group(1)) >= len(opener):
                ranges.append((start, end))
                opener = None
        pos += len(line) + 1
    if opener is not None:
        ranges.append((start, len(text)))
    return ranges


def has_heading(text: str) -> bool:
    inside = fenced_ranges(text)
    pos = 0
    for line in text.split("\n"):
        in_fence = any(a <= pos < b for a, b in inside)
        if not in_fence and _ATX.match(line):
            return True
        pos += len(line) + 1
    return False


def statement_count(code: str) -> int | None:
    masked = "\n".join(
        re.match(r"\s*", ln).group(0) + "pass" if re.match(r"\s*[%!]", ln) else ln  # type: ignore[union-attr]
        for ln in code.split("\n")
    )
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SyntaxWarning)
            tree = ast.parse(masked)
    except (SyntaxError, ValueError):
        return None
    # two statements on one line cannot be separated into pieces, so count lines they start on
    return len({node.lineno for node in tree.body})


def expected_parts(cell: Cell) -> dict[PieceKind, str]:
    """What the chunkers should cover: a cell's source only (outputs are never chunked)."""
    if not cell.source.strip():
        return {}
    kind = PieceKind.CODE if cell.cell_type is CellType.CODE else PieceKind.MARKDOWN
    return {kind: cell.source}


def pieces_of(chunks: list[Chunk]):  # type: ignore[no-untyped-def]  # noqa: ANN201
    return [p for c in chunks for p in c.pieces]


# ------------------------------------------------------------------ structural: criteria


@given(notebooks(), CAPS)
@example([_cell(0, False, "```\nopen fence only", ""), _cell(1, True, "x = 1", "")], 6)
@example([_cell(0, False, "# A\ntext\n## B\nmore", ""), _cell(1, True, "y = 2", "")], 8)
def test_every_cell_is_covered_exactly_once(cells: list[Cell], cap: int) -> None:
    """Pieces reconstruct each cell exactly, and every piece is in exactly one chunk, in order."""
    chunks = chunk_notebook("nb", cells, cap, TOKENIZER)
    pieces = pieces_of(chunks)
    keys = [(p.cell_index, p.kind, p.part) for p in pieces]
    assert len(keys) == len(set(keys))  # no piece appears twice
    for cell in cells:
        for kind, text in expected_parts(cell).items():
            mine = [p for p in pieces if p.cell_index == cell.index and p.kind is kind]
            assert [p.part for p in mine] == list(range(len(mine)))
            assert "".join(p.text for p in mine) == text
    covered = {(p.cell_index, p.kind) for p in pieces}
    wanted = {(c.index, k) for c in cells for k in expected_parts(c)}
    assert covered == wanted  # nothing extra, nothing missing
    order = [(p.cell_index, list(PieceKind).index(p.kind), p.part) for p in pieces]
    assert order == sorted(order)  # document order


@given(notebooks(), CAPS)
def test_no_chunk_is_over_the_cap_unless_it_is_one_flagged_indivisible_piece(
    cells: list[Cell], cap: int
) -> None:
    reasons = {
        PieceUnit.STATEMENT: OversizeReason.STATEMENT,
        PieceUnit.SENTENCE: OversizeReason.SENTENCE,
        PieceUnit.FENCED_BLOCK: OversizeReason.FENCED_BLOCK,
        PieceUnit.UNPARSED_CODE: OversizeReason.UNPARSED_CODE,
    }
    for chunk in chunk_notebook("nb", cells, cap, TOKENIZER):
        assert chunk.tokens == TOKENIZER.count(chunk.text) == TOKENIZER.count(render(chunk.pieces))
        if chunk.tokens > cap:
            assert len(chunk.pieces) == 1
            assert chunk.flags == (reasons[chunk.pieces[0].unit],)
        else:
            assert chunk.flags == ()  # flagged if and only if over the cap


@given(notebooks(), CAPS)
def test_same_input_gives_the_same_chunks_and_unique_ids(cells: list[Cell], cap: int) -> None:
    first = chunk_notebook("nb", cells, cap, TOKENIZER)
    again = chunk_notebook("nb", list(cells), cap, TOKENIZER)
    assert first == again
    ids = [c.id for c in first]
    assert len(ids) == len(set(ids))
    assert all(re.fullmatch(r"[0-9a-f]{16}", i) for i in ids)


@given(notebooks(), CAPS)
def test_code_is_never_cut_mid_statement(cells: list[Cell], cap: int) -> None:
    chunks = chunk_notebook("nb", cells, cap, TOKENIZER)
    for cell in cells:
        if cell.cell_type is not CellType.CODE:
            continue
        mine = [
            p for p in pieces_of(chunks) if p.cell_index == cell.index and p.kind is PieceKind.CODE
        ]
        if len(mine) < 2:
            continue
        counts = [statement_count(p.text) for p in mine]
        assert None not in counts  # every piece is valid Python on its own
        assert sum(c for c in counts if c is not None) == statement_count(cell.source)


@given(notebooks(), CAPS)
def test_unparseable_code_is_never_cut(cells: list[Cell], cap: int) -> None:
    chunks = chunk_notebook("nb", cells, cap, TOKENIZER)
    for cell in cells:
        if (
            cell.cell_type is CellType.CODE
            and cell.source.strip()
            and statement_count(cell.source) is None
        ):
            mine = [
                p
                for p in pieces_of(chunks)
                if p.cell_index == cell.index and p.kind is PieceKind.CODE
            ]
            assert len(mine) == 1


@given(notebooks(), CAPS)
def test_a_fenced_block_is_never_cut(cells: list[Cell], cap: int) -> None:
    chunks = chunk_notebook("nb", cells, cap, TOKENIZER)
    for cell in cells:
        if cell.cell_type is not CellType.MARKDOWN:
            continue
        mine = [p for p in pieces_of(chunks) if p.cell_index == cell.index]
        boundaries = [sum(len(p.text) for p in mine[:i]) for i in range(1, len(mine))]
        for start, end in fenced_ranges(cell.source):
            assert not any(start < b < end for b in boundaries)


TWIN = _cell(0, True, "y = [1,\n     2]\n", "")
TWINS = [TWIN, _cell(1, True, "y = [1,\n     2]\n", "")]  # equal sources share a raw_hash


@given(notebooks(), CAPS)
@example(TWINS, 32)  # found by this test: identical cells in one chunk
def test_a_chunk_never_crosses_a_section_start(cells: list[Cell], cap: int) -> None:
    heading_cells = {
        c.index for c in cells if c.cell_type is CellType.MARKDOWN and has_heading(c.source)
    }
    for chunk in chunk_notebook("nb", cells, cap, TOKENIZER):
        touched = list(dict.fromkeys(p.cell_index for p in chunk.pieces))
        assert not (set(touched[1:]) & heading_cells)  # only the first cell may start a section
        by_hash = {c.index: c.raw_hash for c in cells}
        assert chunk.cell_hashes == tuple(dict.fromkeys(by_hash[i] for i in touched))


@given(notebooks(), CAPS)
def test_greedy_packing_leaves_no_mergeable_neighbours(cells: list[Cell], cap: int) -> None:
    heading_cells = {
        c.index for c in cells if c.cell_type is CellType.MARKDOWN and has_heading(c.source)
    }
    chunks = chunk_notebook("nb", cells, cap, TOKENIZER)
    for left, right in itertools.pairwise(chunks):
        first = right.pieces[0]
        if first.cell_index in heading_cells and first.part == 0:
            continue  # the next chunk starts a new section
        assert TOKENIZER.count(render([*left.pieces, first])) > cap


@given(notebooks(), CAPS, st.integers(1, 60))
def test_outputs_never_reach_chunk_text(cells: list[Cell], cap: int, window: int) -> None:
    """Proposal 002 final decision: neither chunker ever emits an output."""
    marked = [
        c.model_copy(update={"output_text": "OUTPUT-SENTINEL"})
        if c.cell_type is CellType.CODE
        else c
        for c in cells
    ]
    for chunks in (
        chunk_notebook("nb", marked, cap, TOKENIZER),
        fixed_chunks("nb", marked, window, TOKENIZER),
    ):
        assert all("SENTINEL" not in c.text for c in chunks)
        assert all(p.kind in (PieceKind.CODE, PieceKind.MARKDOWN) for c in chunks for p in c.pieces)
    assert chunk_notebook("nb", marked, cap, TOKENIZER) == chunk_notebook(
        "nb", cells, cap, TOKENIZER
    )


# ---------------------------------------------------------------------- fixed-token baseline


def expected_stream(cells: list[Cell]) -> str:
    parts: list[str] = []
    for cell in cells:
        parts.extend(text.strip("\n") for text in expected_parts(cell).values())
    return "\n\n".join(parts)


@given(notebooks(), st.integers(1, 60))
def test_fixed_windows_partition_the_text(cells: list[Cell], window: int) -> None:
    chunks = fixed_chunks("nb", cells, window, TOKENIZER)
    assert "".join(c.text for c in chunks) == expected_stream(cells)
    ids = [c.id for c in chunks]
    assert len(ids) == len(set(ids))
    assert chunks == fixed_chunks("nb", cells, window, TOKENIZER)


@given(notebooks(), st.integers(1, 60))
def test_fixed_windows_have_the_requested_size_up_to_a_split_character(
    cells: list[Cell], window: int
) -> None:
    chunks = fixed_chunks("nb", cells, window, TOKENIZER)
    for chunk in chunks:
        assert 1 <= chunk.tokens <= window + 2  # a 3-byte character moves at most 2 tokens
    for chunk in chunks[:-1]:
        assert chunk.tokens >= window - 2


@given(notebooks(), st.integers(1, 60))
def test_every_cell_with_text_is_touched_by_some_fixed_window(
    cells: list[Cell], window: int
) -> None:
    chunks = fixed_chunks("nb", cells, window, TOKENIZER)
    touched = {p.cell_index for c in chunks for p in c.pieces}
    assert touched == {c.index for c in cells if c.source.strip()}
    for chunk in chunks:
        for piece in chunk.pieces:
            assert piece.text in chunk.text
