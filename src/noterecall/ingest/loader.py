"""Source loader protocol. Supports ADR-003; later sources (PDF, handwritten) plug in here.

A loader turns one file into a `RawNotebook`. It must not clean, scrub or store anything:
the cleaner and the store do that for every source in the same way.
"""

from pathlib import Path
from typing import Protocol

from noterecall.models import RawNotebook


class SourceLoader(Protocol):
    """Reads one kind of source file."""

    suffixes: tuple[str, ...]

    def load(self, path: Path, root: Path, file_hash: str) -> RawNotebook:
        """Parse `path`; the result's `path` is relative to `root` (POSIX separators)."""
        ...
