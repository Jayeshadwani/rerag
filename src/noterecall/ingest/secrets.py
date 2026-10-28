"""Secret scrubbing. Implements ADR-003 (scrub secrets before anything is stored or sent).

Each pattern replaces the secret with `[REDACTED:<name>]` and is counted by name only.
Matched values are never logged or returned. There is deliberately no entropy-based
detector: it flags hashes and ids in code. Patterns are ordered so that a more specific
pattern runs before a general one (`sk-ant-` before `sk-`).
"""

import hashlib
import re
from collections import Counter
from typing import NamedTuple

from pydantic import BaseModel, ConfigDict


class SecretPattern(NamedTuple):
    """A named regex. If the regex has a group called `secret`, only that group is replaced."""

    name: str
    regex: re.Pattern[str]


class ScrubResult(BaseModel):
    """Scrubbed text plus how many secrets of each type were removed."""

    model_config = ConfigDict(frozen=True)

    text: str
    counts: dict[str, int]


def _p(name: str, pattern: str, flags: int = 0) -> SecretPattern:
    return SecretPattern(name, re.compile(pattern, flags))


PATTERNS: tuple[SecretPattern, ...] = (
    _p(
        "private_key",
        r"-----BEGIN [A-Z ]*PRIVATE KEY-----(?:.*?-----END [A-Z ]*PRIVATE KEY-----|.*\Z)",
        re.DOTALL,
    ),
    _p("anthropic_api_key", r"\bsk-ant-[A-Za-z0-9_-]{10,}"),
    _p("openai_api_key", r"\bsk-(?:proj-)?[A-Za-z0-9_-]{20,}"),
    _p("aws_access_key_id", r"\b(?:AKIA|ASIA|AGPA|AIDA|AROA|AIPA|ANPA|ANVA)[A-Z0-9]{16}\b"),
    _p(
        "aws_secret_access_key",
        r"(?i:aws_?secret_?access_?key|secret_?access_?key)['\"]?\s*[=:]\s*['\"]?"
        r"(?P<secret>[A-Za-z0-9/+=]{40})(?![A-Za-z0-9/+=])",
    ),
    _p("github_token", r"\b(?:gh[pousr]_[A-Za-z0-9]{36,}|github_pat_[A-Za-z0-9_]{22,})"),
    _p("slack_token", r"\bxox[abprs]-[A-Za-z0-9-]{10,}"),
    _p("google_api_key", r"\bAIza[0-9A-Za-z_-]{35}"),
    _p("bearer_token", r"\bBearer\s+(?P<secret>[A-Za-z0-9._~+/=-]{20,})", re.IGNORECASE),
)


def patterns_fingerprint() -> str:
    """Hash of every pattern, so a pattern change invalidates previously synced notebooks."""
    digest = hashlib.sha256()
    for pattern in PATTERNS:
        digest.update(f"{pattern.name}\0{pattern.regex.pattern}\0{pattern.regex.flags}\n".encode())
    return digest.hexdigest()


def _replacement(pattern: SecretPattern) -> str:
    return f"[REDACTED:{pattern.name}]"


def _scrub_one(pattern: SecretPattern, text: str) -> tuple[str, int]:
    """Replace every match of `pattern`; returns the new text and the match count."""
    if "secret" not in pattern.regex.groupindex:
        return pattern.regex.subn(_replacement(pattern), text)

    def substitute(match: re.Match[str]) -> str:
        start, end = match.span("secret")
        offset = match.start()
        whole = match.group(0)
        return whole[: start - offset] + _replacement(pattern) + whole[end - offset :]

    return pattern.regex.subn(substitute, text)


def scrub(text: str) -> ScrubResult:
    """Remove all known secret shapes from `text`."""
    counts: Counter[str] = Counter()
    for pattern in PATTERNS:
        text, found = _scrub_one(pattern, text)
        if found:
            counts[pattern.name] += found
    return ScrubResult(text=text, counts=dict(counts))
