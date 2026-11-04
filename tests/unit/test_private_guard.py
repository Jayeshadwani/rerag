"""The session guard that keeps tests away from the owner's private gold files."""

from pathlib import Path

import pytest

import conftest


def test_the_guard_notices_a_file_that_appears_or_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    watched = tmp_path / "abstention.json"
    monkeypatch.setattr(conftest, "PRIVATE_GOLD_FILES", [watched])
    absent = conftest._private_state()
    assert absent == [(False, None)]
    watched.write_text("{}")
    created = conftest._private_state()
    assert created != absent
    assert created[0][0] is True
    watched.write_text('{"changed": true}')
    assert conftest._private_state() != created


def test_the_real_private_files_are_watched() -> None:
    names = {p.name for p in conftest.PRIVATE_GOLD_FILES}
    assert names == {"gold.json", "abstention.json"}
