"""Cell splitters. Implements ADR-004 as amended by proposal 002 (split units).

Every function returns `Span`s that tile the text exactly: the first starts at 0, each starts
where the previous ends, and the last ends at `len(text)`. Joining the slices therefore gives
back the original text, which is what the chunker's "pieces reconstruct the cell" invariant
relies on. Blank lines between two paragraphs attach to the paragraph before them; blank lines
after a closing fence start the next block.

Units: a top-level code statement (with its leading comments and decorators), a markdown
fenced block (never split), a markdown paragraph, a sentence, an output line.
"""

import ast
import itertools
import re
import warnings
from typing import NamedTuple

_MAGIC = re.compile(r"^(\s*)[%!]")
_FENCE_OPEN = re.compile(r"^ {0,3}(`{3,}|~{3,})(.*)$")
_FENCE_CLOSE = re.compile(r"^ {0,3}(`{3,}|~{3,})[ \t]*$")
_HEADING = re.compile(r"^ {0,3}(#{1,6})[ \t]+(\S.*?)[ \t]*$")
_CLOSING_HASHES = re.compile(r"[ \t]+#+$")
_SENTENCE_END = re.compile(r"(?<=[.!?])[\"')\]]*\s+(?=[A-Z0-9\"'(\[`*_<])")


class Span(NamedTuple):
    """A half-open character range `[start, end)`."""

    start: int
    end: int


class MarkdownBlock(NamedTuple):
    """A paragraph or a fenced code block, with its trailing blank lines."""

    span: Span
    fenced: bool


class Heading(NamedTuple):
    """An ATX heading found outside any fenced block."""

    level: int
    title: str


def _line_starts(text: str) -> list[int]:
    starts = [0]
    for line in text.split("\n")[:-1]:
        starts.append(starts[-1] + len(line) + 1)
    return starts


def _statement_line_ranges(masked: str) -> list[tuple[int, int]] | None:
    """1-based (first, last) line of each top-level statement, or None if it does not parse."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SyntaxWarning)
            tree = ast.parse(masked)
    except (SyntaxError, ValueError, RecursionError):
        return None
    ranges: list[tuple[int, int]] = []
    for node in tree.body:
        decorators = getattr(node, "decorator_list", [])
        first = min([node.lineno, *(d.lineno for d in decorators)])
        last = node.end_lineno or node.lineno
        if ranges and first <= ranges[-1][1]:
            ranges[-1] = (ranges[-1][0], max(ranges[-1][1], last))  # `a = 1; b = 2` on one line
        else:
            ranges.append((first, last))
    return ranges


def code_statement_spans(source: str) -> list[Span] | None:
    """Split code at top-level statement boundaries; None if the cell does not parse.

    IPython `%` and `!` lines are treated as one-line statements. A statement's piece also holds
    the comments and blank lines before it; comments after the last statement stay with it.
    """
    lines = source.split("\n")
    masked = "\n".join(f"{m.group(1)}pass" if (m := _MAGIC.match(ln)) else ln for ln in lines)
    ranges = _statement_line_ranges(masked)
    if ranges is None:
        return None
    starts = _line_starts(source)
    cuts = [starts[last] for _, last in ranges[:-1] if last < len(starts)]
    bounds = [0, *cuts, len(source)]
    return [Span(a, b) for a, b in itertools.pairwise(bounds)] or [Span(0, 0)]


def _fence_closes(line: str, opener: str) -> bool:
    match = _FENCE_CLOSE.match(line)
    return (
        match is not None and match.group(1)[0] == opener[0] and len(match.group(1)) >= len(opener)
    )


def _opens_fence(line: str) -> str | None:
    match = _FENCE_OPEN.match(line)
    if match is None:
        return None
    marker, info = match.groups()
    return None if marker[0] == "`" and "`" in info else marker


def markdown_blocks(text: str) -> list[MarkdownBlock]:
    """Split markdown into paragraphs and fenced blocks. An unclosed fence runs to the end."""
    starts = _line_starts(text)
    lines = text.split("\n")
    blocks: list[MarkdownBlock] = []
    block_start = 0
    has_content = trailing_blank = False
    opener: str | None = None
    for i, line in enumerate(lines):
        line_end = starts[i + 1] if i + 1 < len(starts) else len(text)
        if opener is not None:
            if _fence_closes(line, opener):
                blocks.append(MarkdownBlock(Span(block_start, line_end), True))
                block_start, opener = line_end, None
                has_content = trailing_blank = False
            continue
        fence = _opens_fence(line)
        if fence is not None:
            if has_content:
                blocks.append(MarkdownBlock(Span(block_start, starts[i]), False))
                block_start = starts[i]
            opener, has_content, trailing_blank = fence, False, False
        elif not line.strip():
            trailing_blank = trailing_blank or has_content
        else:
            if trailing_blank:
                blocks.append(MarkdownBlock(Span(block_start, starts[i]), False))
                block_start, trailing_blank = starts[i], False
            has_content = True
    if opener is not None:
        blocks.append(MarkdownBlock(Span(block_start, len(text)), True))
    elif block_start < len(text) or not blocks:
        blocks.append(MarkdownBlock(Span(block_start, len(text)), False))
    return blocks


def markdown_headings(text: str) -> list[Heading]:
    """ATX headings outside fenced blocks, in order. Setext and HTML headings are ignored."""
    found: list[Heading] = []
    for block in markdown_blocks(text):
        if block.fenced:
            continue
        for line in text[block.span.start : block.span.end].split("\n"):
            match = _HEADING.match(line)
            if match:
                title = _CLOSING_HASHES.sub("", match.group(2)).strip()
                found.append(Heading(len(match.group(1)), title))
    return found


def sentence_spans(text: str, span: Span) -> list[Span]:
    """Split `span` of `text` into sentences. Breaks inside inline code are not taken."""
    segment = text[span.start : span.end]
    cuts = [
        span.start + m.end()
        for m in _SENTENCE_END.finditer(segment)
        if segment.count("`", 0, m.start()) % 2 == 0 and m.end() < len(segment)
    ]
    bounds = [span.start, *cuts, span.end]
    return [Span(a, b) for a, b in itertools.pairwise(bounds)]
