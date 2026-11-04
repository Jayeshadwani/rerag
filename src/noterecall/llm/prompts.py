"""Versioned prompt files. Implements the CLAUDE.md rule: prompts live in `llm/prompts/*.md`.

A prompt file has up to three sections, each started by a `## instructions`, `## shared` or
`## variable` line. `instructions` is the system text, `shared` is the part that is the same for
every call that reuses a prefix (the notebook), and `variable` is the tail that changes per call
(the chunk or cell). `{{notebook}}`, `{{chunk}}` and `{{cell}}` are replaced when the prompt is
rendered. The prompt's version is the first 12 hex characters of the sha256 of the file's bytes,
so editing a prompt changes the cache key of every call that used it.
"""

import hashlib
import re
from pathlib import Path

from noterecall.llm.client import LLMError, Prompt

PROMPT_DIR = Path(__file__).resolve().parent / "prompts"
SECTIONS = ("instructions", "shared", "variable")
_PLACEHOLDER = re.compile(r"\{\{(\w+)\}\}")


class PromptFileError(LLMError):
    """Raised when a prompt file is missing or malformed."""


class PromptTemplate:
    """A parsed prompt file."""

    def __init__(self, name: str, version: str, parts: dict[str, str]) -> None:
        self.name = name
        self.version = version
        self._parts = parts

    def render(
        self,
        model: str,
        max_output_tokens: int,
        *,
        reasoning_effort: str | None = None,
        **values: str,
    ) -> Prompt:
        """Fill the placeholders and return a ready `Prompt` (with its own effort, if given)."""
        filled = {key: self._fill(text, values) for key, text in self._parts.items()}
        return Prompt(
            name=self.name,
            version=self.version,
            model=model,
            instructions=filled["instructions"],
            shared=filled.get("shared", ""),
            variable=filled.get("variable", ""),
            max_output_tokens=max_output_tokens,
            reasoning_effort=reasoning_effort,
        )

    @staticmethod
    def _fill(text: str, values: dict[str, str]) -> str:
        """Replace every known placeholder in one pass, so a value that itself contains a
        placeholder (a note quoting `{{question}}`) is never substituted a second time."""
        return _PLACEHOLDER.sub(lambda m: values.get(m.group(1), m.group(0)), text)


def load_prompt(name: str, directory: Path = PROMPT_DIR) -> PromptTemplate:
    """Read `<directory>/<name>.md`."""
    path = directory / f"{name}.md"
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise PromptFileError(f"cannot read the prompt file {path}: {exc}") from exc
    parts: dict[str, list[str]] = {}
    current: str | None = None
    for line in raw.decode("utf-8").splitlines():
        if line.startswith("## ") and line[3:].strip() in SECTIONS:
            current = line[3:].strip()
            parts[current] = []
        elif current is not None:
            parts[current].append(line)
    if "instructions" not in parts:
        raise PromptFileError(f"the prompt file {path} has no '## instructions' section")
    version = hashlib.sha256(raw).hexdigest()[:12]
    return PromptTemplate(name, version, {k: "\n".join(v).strip("\n") for k, v in parts.items()})
