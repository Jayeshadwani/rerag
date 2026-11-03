"""Deterministic context header. Implements ADR-007 ("deterministic always").

A chunk is indexed (BM25 and dense) with `"{notebook title} > {heading} > {heading}..."` in front
of its text. The title is the notebook's file stem; the headings are the chunk's heading path,
all levels, with HTML tags, asterisks and backticks removed from each (underscores stay: they
are part of identifiers). The header is part of the indexed text only: the chunk text that is
cited and shown to a reader is unchanged, and the header does not count toward token budgets.
"""

import re
from pathlib import PurePosixPath

from noterecall.models import Chunk

_TAG = re.compile(r"<[^>]+>")
_MARKS = re.compile(r"[*`]+")
_SPACE = re.compile(r"\s+")


def clean_heading(title: str) -> str:
    """A heading as plain words: tags, asterisks and backticks removed, whitespace collapsed."""
    plain = _MARKS.sub("", _TAG.sub("", title))
    return _SPACE.sub(" ", plain).strip()


def notebook_title(notebook_path: str) -> str:
    """The notebook's file stem."""
    return PurePosixPath(notebook_path).stem


def chunk_header(chunk: Chunk) -> str:
    """`title > heading > heading ...`, skipping headings that are empty after cleaning."""
    parts = [notebook_title(chunk.notebook_path), *(clean_heading(h) for h in chunk.heading_path)]
    return " > ".join(p for p in parts if p)


def index_text(chunk: Chunk, header: bool) -> str:
    """The text that is indexed for a chunk: the header (if on), then the chunk text."""
    if not header:
        return chunk.text
    return f"{chunk_header(chunk)}\n\n{chunk.text}"
