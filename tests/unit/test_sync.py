"""Incremental sync: hash-based skipping, zero writes, secrets never stored, privacy of logs."""

import hashlib
import json
import os
import shutil
import sqlite3
import types
from pathlib import Path

import pytest
import structlog
from structlog.testing import capture_logs

from make_fixtures import (
    FAKE_ANTHROPIC_KEY,
    FAKE_AWS_ACCESS_KEY,
    FAKE_AWS_SECRET_KEY,
    FIXTURE_DIR,
)
from noterecall.chunking.tokens import Tokenizer
from noterecall.index.store import connect, duplicate_counts, stored_notebooks
from noterecall.ingest import sync as sync_module
from noterecall.ingest.clean import load_rules
from noterecall.ingest.ipynb import IpynbLoader, NotebookParseError
from noterecall.ingest.sync import (
    Pipeline,
    SyncError,
    SyncReport,
    compute_pipeline_hash,
    discover,
    file_sha256,
    sync,
)

RULES_PATH = Path(__file__).resolve().parents[2] / "config" / "cleaning_rules.yaml"
N_FIXTURES = 7


def _pipeline(tokenizer: Tokenizer, max_lines: int = 20, rules: Path = RULES_PATH) -> Pipeline:
    return Pipeline(
        rules=load_rules(rules),
        max_output_lines=max_lines,
        tokenizer=tokenizer,
        pipeline_hash=compute_pipeline_hash(rules, max_lines, tokenizer.name),
    )


@pytest.fixture
def raw_dir(tmp_path: Path) -> Path:
    target = tmp_path / "raw"
    shutil.copytree(FIXTURE_DIR, target)
    return target


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    return tmp_path / "state" / "notes.sqlite"


def _run(raw_dir: Path, db_path: Path, pipeline: Pipeline) -> tuple[sqlite3.Connection, SyncReport]:
    conn = connect(db_path)
    return conn, sync(raw_dir, conn, IpynbLoader(), pipeline)


def _db_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_first_sync_adds_everything(
    raw_dir: Path, db_path: Path, tiny_tokenizer: Tokenizer
) -> None:
    conn, report = _run(raw_dir, db_path, _pipeline(tiny_tokenizer))
    assert report.model_dump() == {"added": N_FIXTURES, "updated": 0, "unchanged": 0, "removed": 0}
    assert len(stored_notebooks(conn)) == N_FIXTURES
    assert "01_no_headings.ipynb" in stored_notebooks(conn)


def test_resync_of_unchanged_corpus_does_zero_writes(
    raw_dir: Path, db_path: Path, tiny_tokenizer: Tokenizer
) -> None:
    pipeline = _pipeline(tiny_tokenizer)
    first, _ = _run(raw_dir, db_path, pipeline)
    first.close()
    before = _db_sha(db_path)
    second, report = _run(raw_dir, db_path, pipeline)
    assert report.model_dump() == {"added": 0, "updated": 0, "unchanged": N_FIXTURES, "removed": 0}
    assert second.total_changes == 0  # SQLite counted no row changes on this connection
    second.close()
    assert _db_sha(db_path) == before  # and the file is byte-identical


def test_touching_a_file_without_changing_bytes_is_unchanged(
    raw_dir: Path, db_path: Path, tiny_tokenizer: Tokenizer
) -> None:
    pipeline = _pipeline(tiny_tokenizer)
    _run(raw_dir, db_path, pipeline)[0].close()
    os.utime(raw_dir / "01_no_headings.ipynb", (1_000_000_000, 1_000_000_000))
    conn, report = _run(raw_dir, db_path, pipeline)
    assert report.unchanged == N_FIXTURES
    assert conn.total_changes == 0


def test_changed_notebook_is_replaced_and_others_untouched(
    raw_dir: Path, db_path: Path, tiny_tokenizer: Tokenizer
) -> None:
    pipeline = _pipeline(tiny_tokenizer)
    _run(raw_dir, db_path, pipeline)[0].close()
    target = raw_dir / "01_no_headings.ipynb"
    target.write_text(
        target.read_text(encoding="utf-8").replace("Remember that", "Do not forget that")
    )
    conn, report = _run(raw_dir, db_path, pipeline)
    assert report.model_dump() == {
        "added": 0,
        "updated": 1,
        "unchanged": N_FIXTURES - 1,
        "removed": 0,
    }
    sources = [
        r[0] for r in conn.execute("SELECT source FROM cells WHERE notebook_path LIKE '01_%'")
    ]
    assert any("Do not forget that" in s for s in sources)
    assert not any("Remember that" in s for s in sources)


def test_removed_file_is_deleted_with_its_cells(
    raw_dir: Path, db_path: Path, tiny_tokenizer: Tokenizer
) -> None:
    pipeline = _pipeline(tiny_tokenizer)
    _run(raw_dir, db_path, pipeline)[0].close()
    (raw_dir / "05_fake_secret.ipynb").unlink()
    conn, report = _run(raw_dir, db_path, pipeline)
    assert (report.removed, report.unchanged) == (1, N_FIXTURES - 1)
    assert "05_fake_secret.ipynb" not in stored_notebooks(conn)
    assert (
        conn.execute("SELECT count(*) FROM cells WHERE notebook_path LIKE '05_%'").fetchone()[0]
        == 0
    )


def test_new_file_is_added(raw_dir: Path, db_path: Path, tiny_tokenizer: Tokenizer) -> None:
    pipeline = _pipeline(tiny_tokenizer)
    _run(raw_dir, db_path, pipeline)[0].close()
    shutil.copy(raw_dir / "01_no_headings.ipynb", raw_dir / "07_copy.ipynb")
    _, report = _run(raw_dir, db_path, pipeline)
    assert (report.added, report.unchanged) == (1, N_FIXTURES)


def test_changed_pipeline_resyncs_every_notebook(
    raw_dir: Path, db_path: Path, tiny_tokenizer: Tokenizer
) -> None:
    _run(raw_dir, db_path, _pipeline(tiny_tokenizer, max_lines=20))[0].close()
    _, report = _run(raw_dir, db_path, _pipeline(tiny_tokenizer, max_lines=10))
    assert (report.updated, report.unchanged) == (N_FIXTURES, 0)


def test_secrets_never_reach_the_database_file(
    raw_dir: Path, db_path: Path, tiny_tokenizer: Tokenizer
) -> None:
    conn, _ = _run(raw_dir, db_path, _pipeline(tiny_tokenizer))
    conn.close()
    blobs = [p.read_bytes() for p in db_path.parent.iterdir()]  # database plus any journal/wal
    data = b"".join(blobs)
    assert b"[REDACTED:" in data  # positive control: we are searching a database with content
    assert b"Calling the API" in data
    for secret in (FAKE_ANTHROPIC_KEY, FAKE_AWS_ACCESS_KEY, FAKE_AWS_SECRET_KEY):
        assert secret.encode() not in data
    assert b"FAKEFAKEFAKE" not in data  # no fragment of the planted key either


def test_duplicates_across_fixtures_are_marked(
    raw_dir: Path, db_path: Path, tiny_tokenizer: Tokenizer
) -> None:
    conn, _ = _run(raw_dir, db_path, _pipeline(tiny_tokenizer))
    # The split cell: 02 (canonical, first by path) + two copies in 04 -> 2 extra copies.
    counts = duplicate_counts(conn)
    assert (counts.within_notebook, counts.across_notebooks) == (0, 2)


def test_duplicates_within_a_single_notebook_are_marked(
    tmp_path: Path, db_path: Path, tiny_tokenizer: Tokenizer
) -> None:
    raw = tmp_path / "solo"
    raw.mkdir()
    cell: dict[str, object] = {
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": "x = 1",
    }
    document = {"nbformat": 4, "nbformat_minor": 0, "metadata": {}, "cells": [cell, cell]}
    (raw / "only.ipynb").write_text(json.dumps(document), encoding="utf-8")
    conn, _ = _run(raw, db_path, _pipeline(tiny_tokenizer))
    counts = duplicate_counts(conn)
    assert (counts.within_notebook, counts.across_notebooks) == (1, 0)


def test_logs_contain_no_file_names_or_note_text(
    raw_dir: Path, db_path: Path, tiny_tokenizer: Tokenizer
) -> None:
    structlog.reset_defaults()
    with capture_logs() as logs:
        _run(raw_dir, db_path, _pipeline(tiny_tokenizer))[0].close()
    text = repr(logs)
    assert logs, "sync should log"
    for forbidden in ("no_headings", "deep_headings", ".ipynb", "Sorting", "groupby", "FAKE"):
        assert forbidden not in text


def test_invalid_notebook_fails_loudly_and_is_not_stored(
    raw_dir: Path, db_path: Path, tiny_tokenizer: Tokenizer
) -> None:
    (raw_dir / "99_broken.ipynb").write_text("not json", encoding="utf-8")
    conn = connect(db_path)
    with pytest.raises(NotebookParseError, match=r"99_broken\.ipynb"):
        sync(raw_dir, conn, IpynbLoader(), _pipeline(tiny_tokenizer))
    assert "99_broken.ipynb" not in stored_notebooks(conn)


def test_discover_skips_hidden_dirs_and_other_files(tmp_path: Path) -> None:
    (tmp_path / "sub").mkdir()
    (tmp_path / ".ipynb_checkpoints").mkdir()
    for name in ("a.ipynb", "sub/b.ipynb", ".ipynb_checkpoints/a-checkpoint.ipynb", "c.txt"):
        (tmp_path / name).write_text("{}", encoding="utf-8")
    found = [p.relative_to(tmp_path).as_posix() for p in discover(tmp_path, IpynbLoader())]
    assert found == ["a.ipynb", "sub/b.ipynb"]


def test_missing_directory_is_a_sync_error(tmp_path: Path) -> None:
    with pytest.raises(SyncError, match="does not exist"):
        discover(tmp_path / "nope", IpynbLoader())


def test_empty_directory_refuses_and_keeps_existing_rows(
    raw_dir: Path, db_path: Path, tmp_path: Path, tiny_tokenizer: Tokenizer
) -> None:
    pipeline = _pipeline(tiny_tokenizer)
    conn, _ = _run(raw_dir, db_path, pipeline)
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(SyncError, match="refusing to sync"):
        sync(empty, conn, IpynbLoader(), pipeline)
    assert len(stored_notebooks(conn)) == N_FIXTURES


def test_file_sha256_matches_hashlib(tmp_path: Path) -> None:
    path = tmp_path / "f.bin"
    data = os.urandom(3 * (1 << 20) + 17)  # spans several read chunks
    path.write_bytes(data)
    assert file_sha256(path) == hashlib.sha256(data).hexdigest()


def test_pipeline_hash_reacts_to_every_input(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    base = compute_pipeline_hash(RULES_PATH, 20, "enc")
    assert base == compute_pipeline_hash(RULES_PATH, 20, "enc")
    assert base != compute_pipeline_hash(RULES_PATH, 21, "enc")
    assert base != compute_pipeline_hash(RULES_PATH, 20, "other")
    edited = tmp_path / "rules.yaml"
    edited.write_text(RULES_PATH.read_text(encoding="utf-8") + "\n# edited\n", encoding="utf-8")
    assert base != compute_pipeline_hash(edited, 20, "enc")
    stub = tmp_path / "stub.py"
    stub.write_text("changed = True\n", encoding="utf-8")
    monkeypatch.setattr(sync_module, "clean", types.SimpleNamespace(__file__=str(stub)))
    assert base != compute_pipeline_hash(RULES_PATH, 20, "enc")


def test_inline_image_payloads_never_reach_the_database_file(
    raw_dir: Path, db_path: Path, tiny_tokenizer: Tokenizer
) -> None:
    conn, _ = _run(raw_dir, db_path, _pipeline(tiny_tokenizer))
    conn.close()
    data = b"".join(p.read_bytes() for p in db_path.parent.iterdir())
    assert b"data:image/png;base64,[removed]" in data  # the placeholder is stored
    assert b"QUJD" * 10 not in data  # the 400-character payload is not
    assert b"AAAA" * 10 not in data  # nor the 200-character one
