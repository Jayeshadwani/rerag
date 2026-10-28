"""Deterministic cleaning. Implements ADR-003 (rule-based, never LLM), as amended by proposal 001.

Per notebook: replace the payload of inline base64 data URIs in markdown cells, scrub secrets,
drop image/HTML/error outputs, truncate text outputs, remove instructor notes (rules in
`config/cleaning_rules.yaml`, markdown cells only), drop empty cells, and hash each cell.
Every removal is counted by rule so `noterecall stats` can report it. Data URIs go first
because their payload is discarded anyway and random base64 can look like a secret; every
other piece of text is scrubbed before any later step sees it.
"""

import hashlib
import re
from collections import Counter
from pathlib import Path
from typing import Annotated, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from noterecall.chunking.tokens import Tokenizer
from noterecall.ingest.secrets import scrub
from noterecall.models import Cell, CellType, Notebook, OutputKind, RawCell, RawNotebook, RawOutput

RULE_KEYS: tuple[str, ...] = (
    "cells_dropped_empty",
    "instructor_notes",
    "instructor_note_lines_removed",
    "instructor_note_cells_dropped",
    "inline_data_uri_removed",
    "inline_data_uri_chars_removed",
    "output_image_dropped",
    "output_html_dropped",
    "output_other_dropped",
    "output_error_dropped",
    "output_html_representation_dropped",
    "outputs_truncated",
    "output_lines_truncated",
)

_LIST_ITEM = re.compile(r"^\s*(?:[-*+]|\d+\.)\s")
_TAG = re.compile(r"</?(?:span|font)\b[^>\n]*>", re.IGNORECASE)
_LABEL_NOISE = re.compile(r"[\s*_`\[\]:.-]|&nbsp;")
_BLANK_RUNS = re.compile(r"\n{3,}")


class RulesError(Exception):
    """Raised when the cleaning rules file is missing, malformed, or has an invalid regex."""


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class _Extended(_Strict):
    when_any: list[str | dict[Literal["rest_of_line_matches"], str]]
    stop_before: list[str]


class _Extent(_Strict):
    default: Literal["paragraph"]
    extended: _Extended


class _InstructorNoteSpec(_Strict):
    id: Literal["instructor_note"]
    cell_type: Literal["markdown"]
    marker: str
    extent: _Extent
    list_item_scope: bool
    drop_cell_if_empty: bool


class _InlineDataUriSpec(_Strict):
    id: Literal["inline_data_uri"]
    cell_type: Literal["markdown"]
    pattern: str
    replacement: str


_RuleSpec = Annotated[_InstructorNoteSpec | _InlineDataUriSpec, Field(discriminator="id")]


class _RulesFile(_Strict):
    version: Literal[1]
    rules: Annotated[list[_RuleSpec], Field(min_length=2, max_length=2)]


class InstructorNoteRule(BaseModel):
    """The compiled `instructor_note` rule."""

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    marker: re.Pattern[str]
    extend_bare_label: bool
    optional_rest: re.Pattern[str] | None
    stop_before: tuple[re.Pattern[str], ...]
    list_item_scope: bool
    drop_cell_if_empty: bool


def _compile(pattern: str, what: str) -> re.Pattern[str]:
    try:
        return re.compile(pattern)
    except re.error as exc:
        raise RulesError(f"invalid regex for {what}: {exc}") from exc


class InlineDataUriRule(BaseModel):
    """The compiled `inline_data_uri` rule. The pattern has groups `head` and `payload`."""

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    pattern: re.Pattern[str]
    replacement: str


class CleaningRules(BaseModel):
    """Every rule loaded from the rules file."""

    model_config = ConfigDict(frozen=True)

    instructor_note: InstructorNoteRule
    inline_data_uri: InlineDataUriRule


def _build_note_rule(spec: _InstructorNoteSpec) -> InstructorNoteRule:
    bare = False
    optional: re.Pattern[str] | None = None
    for condition in spec.extent.extended.when_any:
        if condition == "bare_label":
            bare = True
        elif isinstance(condition, dict):
            optional = _compile(condition["rest_of_line_matches"], "rest_of_line_matches")
        else:
            raise RulesError(f"unknown extended condition: {condition!r}")
    return InstructorNoteRule(
        marker=_compile(spec.marker, "marker"),
        extend_bare_label=bare,
        optional_rest=optional,
        stop_before=tuple(_compile(p, "stop_before") for p in spec.extent.extended.stop_before),
        list_item_scope=spec.list_item_scope,
        drop_cell_if_empty=spec.drop_cell_if_empty,
    )


def _build_uri_rule(spec: _InlineDataUriSpec) -> InlineDataUriRule:
    pattern = _compile(spec.pattern, "inline_data_uri pattern")
    missing = {"head", "payload"} - set(pattern.groupindex)
    if missing:
        raise RulesError(f"inline_data_uri pattern lacks groups: {sorted(missing)}")
    return InlineDataUriRule(pattern=pattern, replacement=spec.replacement)


def _build_rules(parsed: _RulesFile) -> CleaningRules:
    notes = [spec for spec in parsed.rules if isinstance(spec, _InstructorNoteSpec)]
    uris = [spec for spec in parsed.rules if isinstance(spec, _InlineDataUriSpec)]
    if len(notes) != 1 or len(uris) != 1:
        raise RulesError(
            "the cleaning rules file must define instructor_note and inline_data_uri exactly once"
        )
    return CleaningRules(
        instructor_note=_build_note_rule(notes[0]), inline_data_uri=_build_uri_rule(uris[0])
    )


def load_rules(path: Path) -> CleaningRules:
    """Read and validate the cleaning rules file."""
    try:
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise RulesError(f"cleaning rules file not found: {path}") from exc
    except yaml.YAMLError as exc:
        raise RulesError(f"cleaning rules file {path} is not valid YAML: {exc}") from exc
    try:
        parsed = _RulesFile.model_validate(document)
    except ValidationError as exc:
        raise RulesError(f"cleaning rules file {path} is invalid: {exc}") from exc
    return _build_rules(parsed)


class NoteRemoval(BaseModel):
    """Result of removing instructor notes from one markdown cell."""

    model_config = ConfigDict(frozen=True)

    text: str
    notes: int
    lines_removed: int


def _paragraph_ends(lines: list[str]) -> list[int]:
    """For each non-blank line, the index just past the last line of its paragraph."""
    ends = [0] * len(lines)
    end = len(lines)
    for k in range(len(lines) - 1, -1, -1):
        if not lines[k].strip():
            end = k
        ends[k] = end
    return ends


def _is_bare_label(rest_of_line: str) -> bool:
    return not _LABEL_NOISE.sub("", _TAG.sub("", rest_of_line))


def _indent(line: str) -> int:
    return len(line) - len(line.lstrip())


def _list_scope_end(lines: list[str], hit: int, end: int) -> int:
    """Shorten `end` so a bulleted note does not take its sibling list items with it."""
    marker_indent = _indent(lines[hit])
    for k in range(hit + 1, end):
        if _LIST_ITEM.match(lines[k]) and _indent(lines[k]) <= marker_indent:
            return k
    return end


def _note_end(
    lines: list[str], hit: int, para_end: int, rest: str, rule: InstructorNoteRule
) -> int:
    """Index just past the last line of the note whose marker is `lines[hit]`.

    `rest` is the text of the marker line after the marker itself.
    """
    bare = rule.extend_bare_label and para_end == hit + 1 and _is_bare_label(rest)
    optional = rule.optional_rest is not None and rule.optional_rest.search(rest) is not None
    end = para_end
    if bare or optional:
        end = next(
            (
                k
                for k in range(hit + 1, len(lines))
                if any(stop.match(lines[k]) for stop in rule.stop_before)
            ),
            len(lines),
        )
    if _LIST_ITEM.match(lines[hit]) and rule.list_item_scope:
        end = _list_scope_end(lines, hit, end)
    return end


def _trim_blank_edges(lines: list[str]) -> list[str]:
    start, stop = 0, len(lines)
    while start < stop and not lines[start].strip():
        start += 1
    while stop > start and not lines[stop - 1].strip():
        stop -= 1
    return lines[start:stop]


def remove_instructor_notes(text: str, rule: InstructorNoteRule) -> NoteRemoval:
    """Remove every instructor note from `text`; returns `text` unchanged if there is none."""
    lines = text.splitlines()
    para_ends = _paragraph_ends(lines)
    dropped: set[int] = set()
    notes = 0
    for hit, line in enumerate(lines):
        match = rule.marker.match(line)
        if hit in dropped or match is None:
            continue
        end = _note_end(lines, hit, para_ends[hit], line[match.end() :], rule)
        dropped.update(range(hit, end))
        notes += 1
    if not notes:
        return NoteRemoval(text=text, notes=0, lines_removed=0)
    kept = _trim_blank_edges([ln for k, ln in enumerate(lines) if k not in dropped])
    cleaned = _BLANK_RUNS.sub("\n\n", "\n".join(kept))
    return NoteRemoval(text=cleaned, notes=notes, lines_removed=len(dropped))


class DataUriRemoval(BaseModel):
    """Result of replacing inline data URI payloads in one markdown cell."""

    model_config = ConfigDict(frozen=True)

    text: str
    removed: int
    chars: int


def remove_inline_data_uris(text: str, rule: InlineDataUriRule) -> DataUriRemoval:
    """Replace every data URI payload with the placeholder; the rest of the text is untouched."""
    removed = 0
    chars = 0

    def replace(match: re.Match[str]) -> str:
        nonlocal removed, chars
        removed += 1
        chars += len(match.group("payload"))
        return match.group("head") + rule.replacement

    return DataUriRemoval(text=rule.pattern.sub(replace, text), removed=removed, chars=chars)


def _drop_reason(output: RawOutput) -> str | None:
    return {
        OutputKind.IMAGE: "output_image_dropped",
        OutputKind.HTML: "output_html_dropped",
        OutputKind.OTHER: "output_other_dropped",
        OutputKind.ERROR: "output_error_dropped",
    }.get(output.kind)


def _truncate(text: str, max_lines: int, counts: Counter[str]) -> str:
    lines = text.split("\n")
    if len(lines) <= max_lines:
        return text
    hidden = len(lines) - max_lines
    counts["outputs_truncated"] += 1
    counts["output_lines_truncated"] += hidden
    return "\n".join([*lines[:max_lines], f"[... {hidden} more lines truncated]"])


def clean_outputs(outputs: tuple[RawOutput, ...], max_lines: int, counts: Counter[str]) -> str:
    """Scrub, filter and truncate a cell's outputs into one text block ('' if nothing is kept)."""
    parts: list[str] = []
    for output in outputs:
        reason = _drop_reason(output)
        if reason:
            counts[reason] += 1
            continue
        scrubbed = scrub(output.text)
        for name, found in scrubbed.counts.items():
            counts[f"secrets_{name}"] += found
        if output.has_html:
            counts["output_html_representation_dropped"] += 1
        if scrubbed.text.strip():
            parts.append(_truncate(scrubbed.text.rstrip("\n"), max_lines, counts))
    return "\n".join(parts)


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _clean_source(cell: RawCell, rules: CleaningRules, counts: Counter[str]) -> tuple[str, bool]:
    """Clean a cell's source. Returns the text and whether instructor notes emptied it."""
    text = cell.source
    markdown = cell.cell_type is CellType.MARKDOWN
    if markdown:
        uris = remove_inline_data_uris(text, rules.inline_data_uri)
        counts["inline_data_uri_removed"] += uris.removed
        counts["inline_data_uri_chars_removed"] += uris.chars
        text = uris.text
    scrubbed = scrub(text)
    for name, found in scrubbed.counts.items():
        counts[f"secrets_{name}"] += found
    if not markdown:
        return scrubbed.text, False
    removal = remove_instructor_notes(scrubbed.text, rules.instructor_note)
    counts["instructor_notes"] += removal.notes
    counts["instructor_note_lines_removed"] += removal.lines_removed
    return removal.text, bool(removal.notes) and not removal.text.strip()


def _clean_cell(
    cell: RawCell, rules: CleaningRules, max_lines: int, counts: Counter[str]
) -> Cell | None:
    """Clean one cell; None if it is dropped."""
    source, emptied_by_notes = _clean_source(cell, rules, counts)
    output_text = clean_outputs(cell.outputs, max_lines, counts)
    keep_empty = emptied_by_notes and not rules.instructor_note.drop_cell_if_empty
    if not source.strip() and not output_text.strip() and not keep_empty:
        counts["instructor_note_cells_dropped" if emptied_by_notes else "cells_dropped_empty"] += 1
        return None
    return Cell(
        index=cell.index,
        cell_type=cell.cell_type,
        raw_hash=_sha256(cell.source),
        clean_hash=_sha256(f"{cell.cell_type}\0{source}"),
        source=source,
        output_text=output_text,
    )


def _raw_tokens(cells: tuple[RawCell, ...], tokenizer: Tokenizer) -> int:
    total = 0
    for cell in cells:
        total += tokenizer.count(cell.source)
        total += sum(tokenizer.count(o.text) for o in cell.outputs)
    return total


def clean_notebook(
    raw: RawNotebook,
    rules: CleaningRules,
    max_output_lines: int,
    tokenizer: Tokenizer,
    pipeline_hash: str,
) -> Notebook:
    """Apply every deterministic cleaning step to a parsed notebook."""
    counts: Counter[str] = Counter({"cells_raw": len(raw.cells)})
    cells = [_clean_cell(c, rules, max_output_lines, counts) for c in raw.cells]
    kept = tuple(c for c in cells if c is not None)
    clean_tokens = sum(tokenizer.count(c.source) + tokenizer.count(c.output_text) for c in kept)
    return Notebook(
        path=raw.path,
        file_hash=raw.file_hash,
        pipeline_hash=pipeline_hash,
        raw_tokens=_raw_tokens(raw.cells, tokenizer),
        clean_tokens=clean_tokens,
        counts={k: v for k, v in sorted(counts.items()) if v},
        cells=kept,
    )
