"""Corpus stats: hand-derived totals for the fixture corpus, refusal cases, privacy."""

import shutil
import sqlite3
import subprocess
from pathlib import Path

import pytest

from make_fixtures import FIXTURE_DIR
from noterecall.chunking.tokens import Tokenizer
from noterecall.config import Settings, load_settings
from noterecall.index.store import connect, notebook_summaries
from noterecall.ingest.clean import load_rules
from noterecall.ingest.ipynb import IpynbLoader
from noterecall.ingest.stats import (
    CorpusStats,
    GitState,
    Provenance,
    StatsError,
    compute_stats,
    git_state,
    write_stats,
)
from noterecall.ingest.sync import Pipeline, compute_pipeline_hash, file_sha256, sync

RULES_PATH = Path(__file__).resolve().parents[2] / "config" / "cleaning_rules.yaml"
GIT = GitState(sha="a" * 40, dirty=False)


def _pipeline(tokenizer: Tokenizer) -> Pipeline:
    return Pipeline(
        rule=load_rules(RULES_PATH),
        max_output_lines=20,
        tokenizer=tokenizer,
        pipeline_hash=compute_pipeline_hash(RULES_PATH, 20, tokenizer.name),
    )


def _provenance(tokenizer: Tokenizer) -> Provenance:
    return Provenance(
        pipeline_hash=_pipeline(tokenizer).pipeline_hash,
        rules_sha256=file_sha256(RULES_PATH),
        tokenizer_name=tokenizer.name,
        git=GIT,
    )


@pytest.fixture
def settings() -> Settings:
    return load_settings()


@pytest.fixture
def synced(tmp_path: Path, tiny_tokenizer: Tokenizer) -> sqlite3.Connection:
    raw = tmp_path / "raw"
    shutil.copytree(FIXTURE_DIR, raw)
    conn = connect(tmp_path / "notes.sqlite")
    sync(raw, conn, IpynbLoader(), _pipeline(tiny_tokenizer))
    return conn


def test_fixture_corpus_totals(
    synced: sqlite3.Connection, settings: Settings, tiny_tokenizer: Tokenizer
) -> None:
    stats = compute_stats(synced, settings, _provenance(tiny_tokenizer))
    assert stats.notebooks == 6
    # raw cells: 6 + 9 + 3 + 6 + 4 + 15 = 43; dropped: 1 empty (01), 2 emptied notes (06)
    assert stats.cells_raw == 43
    # kept markdown 3+7+2+2+2+12 = 28, kept code 2+2+1+4+2+1 = 12
    assert stats.cells_kept == {"markdown": 28, "code": 12}
    assert stats.removed_by_rule == {
        "cells_dropped_empty": 1,
        "instructor_notes": 8,
        "instructor_note_lines_removed": 24,
        "instructor_note_cells_dropped": 2,
        "output_image_dropped": 1,
        "output_html_dropped": 0,
        "output_other_dropped": 0,
        "output_error_dropped": 0,
        "output_html_representation_dropped": 1,
        "outputs_truncated": 1,
        "output_lines_truncated": 4980,
    }
    assert stats.secrets_scrubbed_by_type == {
        "anthropic_api_key": 2,
        "aws_access_key_id": 2,
        "aws_secret_access_key": 1,
    }
    assert stats.duplicate_extra_copies == {"within_notebook": 0, "across_notebooks": 2}


def test_distributions_and_token_totals(
    synced: sqlite3.Connection, settings: Settings, tiny_tokenizer: Tokenizer
) -> None:
    stats = compute_stats(synced, settings, _provenance(tiny_tokenizer))
    # kept cells per notebook: 5, 9, 3, 6, 4, 13 -> sorted 3 4 5 6 9 13, median (5+6)/2
    dist = stats.cells_per_notebook
    assert (dist.min, dist.median, dist.max) == (3, 5.5, 13)
    summaries = notebook_summaries(synced)
    assert stats.tokens.raw == sum(s.raw_tokens for s in summaries)
    assert stats.tokens.clean == sum(s.clean_tokens for s in summaries)
    assert stats.tokens.clean < stats.tokens.raw
    assert stats.tokens.tokenizer == "tiny-bytes"
    assert stats.git == GIT


def test_report_contains_no_text_or_file_names(
    synced: sqlite3.Connection, settings: Settings, tiny_tokenizer: Tokenizer, tmp_path: Path
) -> None:
    out = tmp_path / "results" / "corpus_stats.json"
    write_stats(compute_stats(synced, settings, _provenance(tiny_tokenizer)), out)
    text = out.read_text(encoding="utf-8")
    for forbidden in ("no_headings", ".ipynb", "Sorting", "groupby", "FAKEFAKE", "AKIA"):
        assert forbidden not in text
    assert CorpusStats.model_validate_json(text).notebooks == 6
    assert text.endswith("}\n")


def test_empty_database_is_refused(
    tmp_path: Path, settings: Settings, tiny_tokenizer: Tokenizer
) -> None:
    conn = connect(tmp_path / "empty.sqlite")
    with pytest.raises(StatsError, match="no notebooks"):
        compute_stats(conn, settings, _provenance(tiny_tokenizer))


def test_stale_pipeline_is_refused(
    synced: sqlite3.Connection, settings: Settings, tiny_tokenizer: Tokenizer
) -> None:
    stale = _provenance(tiny_tokenizer).model_copy(update={"pipeline_hash": "different"})
    with pytest.raises(StatsError, match="different pipeline"):
        compute_stats(synced, settings, stale)


def test_git_state_reads_this_repository() -> None:
    state = git_state()
    assert len(state.sha) == 40
    assert set(state.sha) <= set("0123456789abcdef")
    git = shutil.which("git")
    assert git is not None
    head = subprocess.run(  # noqa: S603 - fixed arguments
        [git, "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    )
    assert state.sha == head.stdout.strip()


def test_git_state_without_git_fails_loudly(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("noterecall.ingest.stats.shutil.which", lambda _name: None)
    with pytest.raises(StatsError, match="git executable not found"):
        git_state()
