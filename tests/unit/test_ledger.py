"""The test-split ledger: once per milestone, clean tree, frozen gold set."""

from pathlib import Path

import pytest

from noterecall.eval.ledger import (
    Ledger,
    LedgerEntry,
    LedgerError,
    check_test_allowed,
    read_ledger,
    record_run,
)


def entry(milestone: str = "M4") -> LedgerEntry:
    return LedgerEntry(
        milestone=milestone, git_sha="a" * 40, results_file="r.json", results_sha256="b" * 64
    )


def test_a_missing_ledger_is_empty_and_allows_the_first_run() -> None:
    ledger = read_ledger(Path("/nonexistent/ledger.json"))
    assert ledger == Ledger()
    check_test_allowed(ledger, "M4", git_dirty=False, gold_frozen=True)  # no error


def test_a_second_run_for_the_same_milestone_is_refused_but_another_milestone_is_not() -> None:
    ledger = Ledger(entries=(entry("M4"),))
    with pytest.raises(LedgerError, match="already run for milestone M4"):
        check_test_allowed(ledger, "M4", git_dirty=False, gold_frozen=True)
    check_test_allowed(ledger, "M5", git_dirty=False, gold_frozen=True)


def test_a_dirty_tree_or_unfrozen_gold_is_refused() -> None:
    with pytest.raises(LedgerError, match="uncommitted changes"):
        check_test_allowed(Ledger(), "M4", git_dirty=True, gold_frozen=True)
    with pytest.raises(LedgerError, match="not frozen"):
        check_test_allowed(Ledger(), "M4", git_dirty=False, gold_frozen=False)


def test_recording_appends_and_round_trips(tmp_path: Path) -> None:
    path = tmp_path / "results" / "test_ledger.json"
    record_run(path, entry("M4"))
    record_run(path, entry("M5"))
    ledger = read_ledger(path)
    assert [e.milestone for e in ledger.entries] == ["M4", "M5"]
    with pytest.raises(LedgerError, match="already run"):
        check_test_allowed(ledger, "M4", git_dirty=False, gold_frozen=True)


def test_unreadable_ledger_and_unwritable_path_fail_loudly(tmp_path: Path) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    with pytest.raises(LedgerError, match="cannot read the test ledger"):
        read_ledger(bad)
    blocker = tmp_path / "file"
    blocker.write_text("x")
    with pytest.raises(LedgerError, match="cannot write the test ledger"):
        record_run(blocker / "ledger.json", entry())
