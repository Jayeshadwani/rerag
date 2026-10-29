"""Structure-aware chunker. Implements ADR-004 as amended by proposal 002.

A section starts at every markdown cell that contains an ATX heading (any level, outside fenced
blocks); a cell is never split across sections. Within a section, cells are packed greedily, in
order, into chunks of at most `cap` tokens, counted on the rendered chunk text. A cell that alone
exceeds the cap is cut into pieces (statements, or paragraphs then sentences, with fenced blocks
kept whole). A chunk may exceed the cap only if it is one indivisible piece; it is then flagged.
Outputs are included according to `OutputMode` (excluded by default).
"""

import hashlib
import json
from collections.abc import Sequence
from typing import NamedTuple

from noterecall.chunking.evidence import is_evidence_output
from noterecall.chunking.splitters import (
    Heading,
    code_statement_spans,
    line_spans,
    markdown_blocks,
    markdown_headings,
    sentence_spans,
)
from noterecall.chunking.tokens import Tokenizer
from noterecall.models import (
    Cell,
    CellType,
    Chunk,
    OutputMode,
    OversizeReason,
    Piece,
    PieceKind,
    PieceUnit,
)

CHUNKER = "structural-1"

_REASONS = {
    PieceUnit.STATEMENT: OversizeReason.STATEMENT,
    PieceUnit.SENTENCE: OversizeReason.SENTENCE,
    PieceUnit.FENCED_BLOCK: OversizeReason.FENCED_BLOCK,
    PieceUnit.OUTPUT_LINE: OversizeReason.OUTPUT_LINE,
    PieceUnit.UNPARSED_CODE: OversizeReason.UNPARSED_CODE,
}


class ChunkingError(Exception):
    """Raised when an internal invariant of the chunker is violated (a bug, not bad input)."""


class Section(NamedTuple):
    """Consecutive cells that share a heading path."""

    path: tuple[str, ...]
    cells: tuple[Cell, ...]


def render(pieces: Sequence[Piece]) -> str:
    """Chunk text: pieces of one cell part are joined as-is, parts and cells by a blank line."""
    runs: list[str] = []
    previous: tuple[int, PieceKind] | None = None
    for piece in pieces:
        key = (piece.cell_index, piece.kind)
        if key == previous:
            runs[-1] += piece.text
        else:
            runs.append(piece.text)
            previous = key
    return "\n\n".join(run.strip("\n") for run in runs)


def _push(stack: list[Heading], heading: Heading) -> None:
    while stack and stack[-1].level >= heading.level:
        stack.pop()
    stack.append(heading)


def build_sections(cells: Sequence[Cell]) -> list[Section]:
    """Group cells into flat sections; the path is the one in effect at the cell's first heading."""
    sections: list[Section] = []
    stack: list[Heading] = []
    path: tuple[str, ...] = ()
    current: list[Cell] = []
    for cell in cells:
        headings = markdown_headings(cell.source) if cell.cell_type is CellType.MARKDOWN else []
        if headings:
            if current:
                sections.append(Section(path, tuple(current)))
                current = []
            _push(stack, headings[0])
            path = tuple(h.title for h in stack)
            for heading in headings[1:]:
                _push(stack, heading)
        current.append(cell)
    if current:
        sections.append(Section(path, tuple(current)))
    return sections


class _Splitter:
    """Cuts cells into pieces for one cap and tokenizer."""

    def __init__(self, cap: int, tokenizer: Tokenizer) -> None:
        self.cap = cap
        self.tokenizer = tokenizer

    def fits(self, text: str) -> bool:
        return self.tokenizer.count(text.strip("\n")) <= self.cap

    def _piece(self, cell: Cell, kind: PieceKind, part: int, unit: PieceUnit, text: str) -> Piece:
        return Piece(cell_index=cell.index, kind=kind, part=part, unit=unit, text=text)

    def code(self, cell: Cell) -> list[Piece]:
        text, kind = cell.source, PieceKind.CODE
        if self.fits(text):
            return [self._piece(cell, kind, 0, PieceUnit.CELL, text)]
        spans = code_statement_spans(text)
        if spans is None:
            return [self._piece(cell, kind, 0, PieceUnit.UNPARSED_CODE, text)]
        return [
            self._piece(cell, kind, i, PieceUnit.STATEMENT, text[s.start : s.end])
            for i, s in enumerate(spans)
        ]

    def markdown(self, cell: Cell) -> list[Piece]:
        text, kind = cell.source, PieceKind.MARKDOWN
        if self.fits(text):
            return [self._piece(cell, kind, 0, PieceUnit.CELL, text)]
        pieces: list[Piece] = []
        for block in markdown_blocks(text):
            body = text[block.span.start : block.span.end]
            if block.fenced:
                units = [(PieceUnit.FENCED_BLOCK, body)]
            elif self.fits(body):
                units = [(PieceUnit.PARAGRAPH, body)]
            else:
                units = [
                    (PieceUnit.SENTENCE, text[s.start : s.end])
                    for s in sentence_spans(text, block.span)
                ]
            base = len(pieces)
            pieces.extend(
                self._piece(cell, kind, base + i, unit, part_text)
                for i, (unit, part_text) in enumerate(units)
            )
        return pieces

    def output(self, cell: Cell) -> list[Piece]:
        text, kind = cell.output_text, PieceKind.OUTPUT
        if self.fits(text):
            return [self._piece(cell, kind, 0, PieceUnit.CELL, text)]
        return [
            self._piece(cell, kind, i, PieceUnit.OUTPUT_LINE, text[s.start : s.end])
            for i, s in enumerate(line_spans(text))
        ]


def wants_output(cell: Cell, mode: OutputMode, tokenizer: Tokenizer) -> bool:
    if mode is OutputMode.NONE or not cell.output_text.strip():
        return False
    return mode is OutputMode.ALL or is_evidence_output(cell.output_text, tokenizer)


def cell_pieces(cell: Cell, splitter: _Splitter, mode: OutputMode) -> list[Piece]:
    """All pieces of one cell, in order: its source, then (per `mode`) its output."""
    pieces: list[Piece] = []
    if cell.source.strip():
        cut = splitter.code if cell.cell_type is CellType.CODE else splitter.markdown
        pieces.extend(cut(cell))
    if wants_output(cell, mode, splitter.tokenizer):
        pieces.extend(splitter.output(cell))
    return pieces


def pack(pieces: Sequence[Piece], cap: int, tokenizer: Tokenizer) -> list[list[Piece]]:
    """Greedy, order-preserving packing. Only a lone piece can exceed the cap."""
    groups: list[list[Piece]] = []
    current: list[Piece] = []
    for piece in pieces:
        if current and tokenizer.count(render([*current, piece])) > cap:
            groups.append(current)
            current = []
        current.append(piece)
    if current:
        groups.append(current)
    return groups


def chunk_id(notebook_path: str, cap: int, mode: OutputMode, pieces: Sequence[Piece]) -> str:
    """Deterministic id from the chunker, cap, output mode, notebook and the exact pieces."""
    payload = {
        "chunker": CHUNKER,
        "cap": cap,
        "outputs": mode.value,
        "notebook": notebook_path,
        "pieces": [
            [p.cell_index, p.kind.value, p.part, hashlib.sha256(p.text.encode()).hexdigest()]
            for p in pieces
        ],
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


def _flags(pieces: Sequence[Piece], tokens: int, cap: int) -> tuple[OversizeReason, ...]:
    if tokens <= cap:
        return ()
    if len(pieces) != 1 or pieces[0].unit not in _REASONS:
        raise ChunkingError("a chunk over the cap must be one indivisible piece")
    return (_REASONS[pieces[0].unit],)


def _build_chunk(group: list[Piece], section: Section, ctx: "_Context") -> Chunk:
    text = render(group)
    tokens = ctx.tokenizer.count(text)
    return Chunk(
        id=chunk_id(ctx.notebook_path, ctx.cap, ctx.mode, group),
        notebook_path=ctx.notebook_path,
        heading_path=section.path,
        pieces=tuple(group),
        text=text,
        tokens=tokens,
        flags=_flags(group, tokens, ctx.cap),
        cell_hashes=tuple(dict.fromkeys(ctx.raw_hashes[p.cell_index] for p in group)),
    )


class _Context(NamedTuple):
    notebook_path: str
    cap: int
    mode: OutputMode
    tokenizer: Tokenizer
    raw_hashes: dict[int, str]


def chunk_notebook(
    notebook_path: str,
    cells: Sequence[Cell],
    cap: int,
    tokenizer: Tokenizer,
    outputs: OutputMode = OutputMode.NONE,
) -> list[Chunk]:
    """Chunk one notebook's cleaned cells (in cell order) with the given token cap."""
    ctx = _Context(notebook_path, cap, outputs, tokenizer, {c.index: c.raw_hash for c in cells})
    splitter = _Splitter(cap, tokenizer)
    chunks: list[Chunk] = []
    for section in build_sections(cells):
        pieces = [p for cell in section.cells for p in cell_pieces(cell, splitter, outputs)]
        chunks.extend(_build_chunk(group, section, ctx) for group in pack(pieces, cap, tokenizer))
    return chunks
