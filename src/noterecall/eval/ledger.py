"""The test-split ledger. Implements ADR-009/ADR-010 (the test split is run once per milestone).

`results/test_ledger.json` lists every milestone whose test split has been run, with the commit
and the sha256 of the results file. A second test run for the same milestone is refused, and so
is a run from a dirty tree or against a gold set that is not frozen.
"""

import hashlib
import json
from pathlib import Path

from pydantic import BaseModel, ConfigDict, ValidationError


class LedgerError(Exception):
    """Raised when the test split must not be run, or the ledger cannot be read or written."""


class LedgerEntry(BaseModel):
    """One test run: which milestone, at which commit, and the results file it produced."""

    model_config = ConfigDict(frozen=True)

    milestone: str
    git_sha: str
    results_file: str
    results_sha256: str


class Ledger(BaseModel):
    """All test runs so far."""

    model_config = ConfigDict(frozen=True)

    schema_version: int = 1
    entries: tuple[LedgerEntry, ...] = ()


def read_ledger(path: Path) -> Ledger:
    """The ledger at `path`, or an empty one if the file does not exist yet."""
    if not path.exists():
        return Ledger()
    try:
        return Ledger.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValidationError) as exc:
        raise LedgerError(f"cannot read the test ledger at {path}: {exc}") from exc


def check_test_allowed(ledger: Ledger, milestone: str, git_dirty: bool, gold_frozen: bool) -> None:
    """Raise unless the test split may be run now for this milestone."""
    if any(e.milestone == milestone for e in ledger.entries):
        raise LedgerError(f"the test split was already run for milestone {milestone}")
    if git_dirty:
        raise LedgerError("the working tree has uncommitted changes; commit before the test run")
    if not gold_frozen:
        raise LedgerError("the gold set is not frozen; run `noterecall gold freeze` first")


def file_sha256(path: Path) -> str:
    """sha256 of a file's bytes."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def record_run(path: Path, entry: LedgerEntry) -> None:
    """Append an entry and write the ledger (stable formatting)."""
    ledger = read_ledger(path)
    updated = Ledger(entries=(*ledger.entries, entry))
    text = json.dumps(updated.model_dump(mode="json"), indent=2, sort_keys=True)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text + "\n", encoding="utf-8")
    except OSError as exc:
        raise LedgerError(f"cannot write the test ledger at {path}: {exc}") from exc
