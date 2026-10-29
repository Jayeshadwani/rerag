"""Evidence-output rule. Implements the output policy of proposal 002 (ADR-004 amendment).

Principle (owner): an output belongs in chunk text only if it is evidence needed to understand a
concept or conclusion. The rule is a pure function of the output text: short, and not a table,
a long array, a warning, a progress bar, an object repr, or a truncated listing. The stored text
does not record whether an output was printed or a final value, so that distinction is not used.

The rule was written from the principle and from aggregate pattern counts, without reading
individual outputs, and frozen in its own commit before the validation sample was drawn.
"""

import hashlib
import re
from enum import StrEnum
from pathlib import Path

from noterecall.chunking.tokens import Tokenizer

MAX_EVIDENCE_TOKENS = 64
MAX_NUMBERS_PER_LINE = 8
MIN_TABLE_ROWS = 2

_TRUNCATED = re.compile(r"\[\.\.\. \d+ more lines truncated\]")
_PROGRESS = re.compile(r"\d+%\||\bit/s\b|\bs/it\b|[█▉▊▋▌▍▎▏━]|\bETA\b|\d+/\d+ \[")
_WARNING = re.compile(r"warning|deprecat", re.IGNORECASE)
_OBJECT_REPR = re.compile(r"<[\w.]+(?: [\w.]+)* at 0x[0-9a-fA-F]+>|<Figure size|<matplotlib")
_NUMBER = re.compile(r"-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")
_ARRAY_CALL = re.compile(r"\barray\(")
_COLUMN_GAP = re.compile(r"\S {2,}\S")


class EvidenceVerdict(StrEnum):
    """The first failing check, or EVIDENCE if the output passes them all."""

    EVIDENCE = "evidence"
    EMPTY = "empty"
    TRUNCATED = "truncated"
    TOO_LONG = "too_long"
    PROGRESS = "progress"
    WARNING = "warning"
    OBJECT_REPR = "object_repr"
    ARRAY = "array"
    TABLE = "table"


def _is_array(text: str) -> bool:
    if _ARRAY_CALL.search(text):
        return True
    return any(len(_NUMBER.findall(line)) > MAX_NUMBERS_PER_LINE for line in text.split("\n"))


def _is_table(text: str) -> bool:
    rows = sum(1 for line in text.split("\n") if _COLUMN_GAP.search(line))
    return rows >= MIN_TABLE_ROWS


def judge_output(text: str, tokenizer: Tokenizer) -> EvidenceVerdict:
    """Apply the rule to one output; returns why it was rejected, or EVIDENCE."""
    if not text.strip():
        return EvidenceVerdict.EMPTY
    if _TRUNCATED.search(text):
        return EvidenceVerdict.TRUNCATED
    if tokenizer.count(text) > MAX_EVIDENCE_TOKENS:
        return EvidenceVerdict.TOO_LONG
    checks = (
        (_PROGRESS.search(text) is not None, EvidenceVerdict.PROGRESS),
        (_WARNING.search(text) is not None, EvidenceVerdict.WARNING),
        (_OBJECT_REPR.search(text) is not None, EvidenceVerdict.OBJECT_REPR),
        (_is_array(text), EvidenceVerdict.ARRAY),
        (_is_table(text), EvidenceVerdict.TABLE),
    )
    return next((verdict for failed, verdict in checks if failed), EvidenceVerdict.EVIDENCE)


def is_evidence_output(text: str, tokenizer: Tokenizer) -> bool:
    """True if the output should be part of chunk text in the `evidence` output mode."""
    return judge_output(text, tokenizer) is EvidenceVerdict.EVIDENCE


def rule_fingerprint() -> str:
    """Hash of this module's source, recorded with any evaluation of the rule."""
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
