"""Jupyter notebook parsing. Implements ADR-003 (input side of deterministic cleaning).

Reads `.ipynb` files with nbformat, normalises newlines, and classifies each output so the
cleaner can drop image/HTML/error outputs. Only markdown and code cells are accepted;
anything else raises `NotebookParseError` instead of being skipped silently.
"""

from pathlib import Path
from typing import Any

import nbformat
from nbformat import NotebookNode, ValidationError
from nbformat.reader import NotJSONError

from noterecall.models import CellType, OutputKind, RawCell, RawNotebook, RawOutput


class NotebookParseError(Exception):
    """Raised when a notebook cannot be read or contains something the pipeline rejects."""


def _normalise(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n")


def _classify_data(data: dict[str, Any]) -> RawOutput:
    """Classify a display_data / execute_result payload by its MIME types."""
    if any(mime.startswith("image/") for mime in data):
        return RawOutput(kind=OutputKind.IMAGE)
    if "text/plain" in data:
        return RawOutput(
            kind=OutputKind.TEXT,
            text=_normalise(str(data["text/plain"])),
            has_html="text/html" in data,
        )
    if "text/html" in data:
        return RawOutput(kind=OutputKind.HTML)
    return RawOutput(kind=OutputKind.OTHER)


def _parse_output(output: NotebookNode) -> RawOutput:
    output_type = output.get("output_type")
    if output_type == "stream":
        return RawOutput(kind=OutputKind.TEXT, text=_normalise(str(output.get("text", ""))))
    if output_type == "error":
        return RawOutput(kind=OutputKind.ERROR)
    if output_type in ("display_data", "execute_result"):
        return _classify_data(dict(output.get("data", {})))
    raise NotebookParseError(f"unknown output type: {output_type!r}")


def _parse_cell(index: int, cell: NotebookNode) -> RawCell:
    try:
        cell_type = CellType(str(cell.get("cell_type")))
    except ValueError as exc:
        raise NotebookParseError(
            f"cell {index} has unsupported type {cell.get('cell_type')!r}"
        ) from exc
    outputs = tuple(_parse_output(o) for o in cell.get("outputs", []))
    return RawCell(
        index=index, cell_type=cell_type, source=_normalise(str(cell.source)), outputs=outputs
    )


class IpynbLoader:
    """`SourceLoader` for Jupyter notebooks."""

    suffixes: tuple[str, ...] = (".ipynb",)

    def load(self, path: Path, root: Path, file_hash: str) -> RawNotebook:
        """Parse one notebook; `path` in the result is relative to `root`."""
        try:
            notebook = nbformat.reads(path.read_text(encoding="utf-8"), as_version=4)
        except (OSError, UnicodeDecodeError, NotJSONError, ValidationError) as exc:
            raise NotebookParseError(f"cannot read notebook {path}: {exc}") from exc
        cells = tuple(_parse_cell(i, c) for i, c in enumerate(notebook.cells))
        return RawNotebook(path=path.relative_to(root).as_posix(), file_hash=file_hash, cells=cells)
