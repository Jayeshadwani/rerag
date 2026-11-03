"""Optional LLM clean of markdown cells and its verifier. Implements ADR-003 (LLM stage).

The model is asked to fix spelling and broken heading markup in one markdown cell and nothing
else. Its answer is trusted only after `verify_clean` passes, which rejects it if:

- the length (characters) leaves 0.7 to 1.3 times the original;
- any inline-code span, or any identifier-shaped word (snake_case, camelCase, a name with digits,
  or a dotted path) of the original is missing from the answer;
- the sequence of fenced code blocks is not identical, block for block.

A rejected cell keeps its original text, and the reason is counted, never hidden. The raw model
answer is kept in the response cache and the original text stays in the database, so both versions
exist. Verifying happens when answers are applied, so a stricter verifier needs no new API calls.
"""

import re
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import NamedTuple

from pydantic import BaseModel, ConfigDict, Field

from noterecall.chunking.splitters import markdown_blocks
from noterecall.models import Cell, CellType

RATIO_MIN = 0.7
RATIO_MAX = 1.3

_INLINE_CODE = re.compile(r"(`+)(.+?)\1", re.DOTALL)
_WORD = re.compile(r"[^\W\d]\w*(?:\.[^\W\d]\w*)*|\w+")
_CAMEL = re.compile(r"[a-z0-9][A-Z]")


class CleanedCell(BaseModel):
    """The model's answer for one cell."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    text: str = Field(min_length=1)


class Verdict(NamedTuple):
    """Whether a cleaned cell may replace the original, and why not."""

    ok: bool
    reasons: tuple[str, ...]


def _is_identifier_shaped(word: str) -> bool:
    has_digit = any(c.isdigit() for c in word)
    has_alpha = any(c.isalpha() for c in word)
    return "_" in word or "." in word or bool(_CAMEL.search(word)) or (has_digit and has_alpha)


def protected_tokens(text: str) -> set[str]:
    """Inline-code spans and identifier-shaped words that must survive cleaning."""
    spans = {match.group(2).strip() for match in _INLINE_CODE.finditer(text)}
    outside = _INLINE_CODE.sub(" ", text)
    words = {w for w in _WORD.findall(outside) if _is_identifier_shaped(w)}
    return {t for t in spans | words if t}


def fenced_blocks(text: str) -> list[str]:
    """The text of every fenced code block, in order."""
    return [text[b.span.start : b.span.end] for b in markdown_blocks(text) if b.fenced]


def verify_clean(original: str, cleaned: str) -> Verdict:
    """Apply the three ADR-003 checks; all failing reasons are returned."""
    reasons: list[str] = []
    if not original.strip() or not RATIO_MIN <= len(cleaned) / len(original) <= RATIO_MAX:
        reasons.append("length_ratio")
    if any(token not in cleaned for token in protected_tokens(original)):
        reasons.append("missing_token")
    if fenced_blocks(original) != fenced_blocks(cleaned):
        reasons.append("code_fence_changed")
    return Verdict(not reasons, tuple(reasons))


class CleanReport(BaseModel):
    """What happened to the markdown cells: counts only."""

    model_config = ConfigDict(frozen=True)

    cells: int
    changed: int
    unchanged: int
    rejected_cells: int
    rejected_by_reason: dict[str, int]


def apply_clean(
    cells: Sequence[Cell], answers: Mapping[int, str]
) -> tuple[list[Cell], CleanReport]:
    """Substitute verified answers (by cell index) into the cells. Everything else is kept."""
    out: list[Cell] = []
    changed = unchanged = rejected_cells = 0
    rejected: Counter[str] = Counter()
    for cell in cells:
        answer = answers.get(cell.index)
        if cell.cell_type is not CellType.MARKDOWN or answer is None:
            out.append(cell)
            continue
        if answer == cell.source:
            unchanged += 1
            out.append(cell)
            continue
        verdict = verify_clean(cell.source, answer)
        if verdict.ok:
            changed += 1
            out.append(cell.model_copy(update={"source": answer}))
        else:
            rejected_cells += 1
            rejected.update(verdict.reasons)
            out.append(cell)
    report = CleanReport(
        cells=len(answers),
        changed=changed,
        unchanged=unchanged,
        rejected_cells=rejected_cells,
        rejected_by_reason=dict(rejected),
    )
    return out, report


def apply_all(
    notebooks: Sequence[tuple[str, list[Cell]]], answers: Mapping[tuple[str, int], str]
) -> tuple[list[tuple[str, list[Cell]]], CleanReport]:
    """`apply_clean` over every notebook (answers keyed by path and cell index), with the
    reports added together."""
    out: list[tuple[str, list[Cell]]] = []
    total = Counter[str]()
    rejected: Counter[str] = Counter()
    for path, cells in notebooks:
        mine = {i: text for (p, i), text in answers.items() if p == path}
        new_cells, report = apply_clean(cells, mine)
        out.append((path, new_cells))
        total.update(
            cells=report.cells,
            changed=report.changed,
            unchanged=report.unchanged,
            rejected_cells=report.rejected_cells,
        )
        rejected.update(report.rejected_by_reason)
    return out, CleanReport(
        cells=total["cells"],
        changed=total["changed"],
        unchanged=total["unchanged"],
        rejected_cells=total["rejected_cells"],
        rejected_by_reason=dict(rejected),
    )
