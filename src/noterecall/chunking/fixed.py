"""Fixed-token chunker, the baseline for the ADR-004 ablation. Not used in production.

The notebook's cell texts (the same ones the structural chunker sees, in the same order, joined
by a blank line) form one stream. It is cut into windows of `window` tokens with no overlap,
ignoring all structure. A window boundary that would fall inside a multi-byte character moves
to that character's start, so a window can differ from `window` tokens by a few tokens there.
Each window records which cells it overlaps so cell-level relevance labels still apply.
"""

import hashlib
import itertools
import json
from collections.abc import Sequence
from typing import NamedTuple

from noterecall.chunking.structural import wants_output
from noterecall.chunking.tokens import Tokenizer
from noterecall.models import Cell, Chunk, OutputMode, Piece, PieceKind, PieceUnit

CHUNKER = "fixed-1"


class _Segment(NamedTuple):
    cell_index: int
    kind: PieceKind
    start: int
    end: int


def _stream(
    cells: Sequence[Cell], mode: OutputMode, tokenizer: Tokenizer
) -> tuple[str, list[_Segment]]:
    """The concatenated text and where each cell part sits in it."""
    parts: list[tuple[int, PieceKind, str]] = []
    for cell in cells:
        if cell.source.strip():
            kind = PieceKind.CODE if cell.cell_type.value == "code" else PieceKind.MARKDOWN
            parts.append((cell.index, kind, cell.source.strip("\n")))
        if wants_output(cell, mode, tokenizer):
            parts.append((cell.index, PieceKind.OUTPUT, cell.output_text.strip("\n")))
    segments: list[_Segment] = []
    offset = 0
    for index, kind, text in parts:
        segments.append(_Segment(index, kind, offset, offset + len(text)))
        offset += len(text) + 2
    return "\n\n".join(text for _, _, text in parts), segments


def _pieces(
    stream: str, segments: Sequence[_Segment], start: int, end: int, ordinal: int
) -> list[Piece]:
    pieces: list[Piece] = []
    for seg in segments:
        lo, hi = max(start, seg.start), min(end, seg.end)
        if lo < hi:
            pieces.append(
                Piece(
                    cell_index=seg.cell_index,
                    kind=seg.kind,
                    part=ordinal,
                    unit=PieceUnit.WINDOW,
                    text=stream[lo:hi],
                )
            )
    return pieces


def _window_id(path: str, window: int, mode: OutputMode, span: tuple[int, int], text: str) -> str:
    payload = {
        "chunker": CHUNKER,
        "window": window,
        "outputs": mode.value,
        "notebook": path,
        "start": span[0],
        "end": span[1],
        "sha": hashlib.sha256(text.encode()).hexdigest(),
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


def fixed_chunks(
    notebook_path: str,
    cells: Sequence[Cell],
    window: int,
    tokenizer: Tokenizer,
    outputs: OutputMode = OutputMode.NONE,
) -> list[Chunk]:
    """Cut one notebook into consecutive `window`-token chunks (the last may be shorter)."""
    if window < 1:
        raise ValueError(f"window must be at least 1, got {window}")
    stream, segments = _stream(cells, outputs, tokenizer)
    if not stream:
        return []
    starts = tokenizer.token_starts(stream)
    bounds = [*starts[::window], len(stream)]
    raw_hashes = {c.index: c.raw_hash for c in cells}
    chunks: list[Chunk] = []
    for ordinal, (start, end) in enumerate(itertools.pairwise(bounds)):
        if end <= start:
            continue  # a window made only of bytes of one character that moved to the next window
        text = stream[start:end]
        pieces = _pieces(stream, segments, start, end, ordinal)
        hashes = tuple(dict.fromkeys(raw_hashes[p.cell_index] for p in pieces))
        chunks.append(
            Chunk(
                id=_window_id(notebook_path, window, outputs, (start, end), text),
                notebook_path=notebook_path,
                heading_path=(),
                pieces=tuple(pieces),
                text=text,
                tokens=tokenizer.count(text),
                cell_hashes=hashes,
            )
        )
    return chunks
