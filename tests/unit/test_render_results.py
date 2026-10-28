"""render_results.py prints only numbers that are in the results file."""

from pathlib import Path

import pytest

from noterecall.ingest.stats import (
    CorpusStats,
    Distribution,
    GitState,
    TokenTotals,
)
from render_results import main, render_corpus_stats


def _stats(raw: int = 200, clean: int = 150, secrets: dict[str, int] | None = None) -> CorpusStats:
    return CorpusStats(
        schema_version=1,
        git=GitState(sha="abcdef0123456789" * 2 + "abcdef01", dirty=True),
        config={},
        rules_sha256="r" * 64,
        pipeline_hash="p" * 64,
        notebooks=1234,
        cells_raw=5000,
        cells_kept={"markdown": 3000, "code": 1500},
        tokens=TokenTotals(tokenizer="enc", tiktoken_version="0", raw=raw, clean=clean),
        removed_by_rule={"instructor_notes": 7, "cells_dropped_empty": 0},
        secrets_scrubbed_by_type=secrets or {},
        duplicate_extra_copies={"within_notebook": 2, "across_notebooks": 11},
        cells_per_notebook=Distribution(min=1, median=20.5, max=9000),
        clean_tokens_per_notebook=Distribution(min=10, median=2500.0, max=123456),
    )


def test_numbers_are_formatted_from_the_file() -> None:
    text = render_corpus_stats(_stats())
    assert "| Notebooks | 1,234 |" in text
    assert "| Cells kept: markdown | 3,000 |" in text
    assert "| Tokens before cleaning (enc) | 200 |" in text
    assert "| Tokens after cleaning (enc) | 150 |" in text
    assert "| Tokens removed | 25.0% |" in text  # (200 - 150) / 200
    assert "| Cells per notebook (min / median / max) | 1 / 20.5 / 9,000 |" in text
    assert "| instructor notes" not in text  # rule ids are printed as stored
    assert "| instructor_notes | 7 |" in text
    assert "| across notebooks | 11 |" in text


def test_provenance_line_shows_short_hashes_and_dirty_flag() -> None:
    text = render_corpus_stats(_stats())
    assert "Commit `abcdef012345` (dirty tree)" in text
    assert f"pipeline `{'p' * 12}`" in text


def test_secrets_section_lists_types_or_says_none() -> None:
    assert "| aws_access_key_id | 3 |" in render_corpus_stats(
        _stats(secrets={"aws_access_key_id": 3})
    )
    assert "| (none found) | 0 |" in render_corpus_stats(_stats())


def test_zero_raw_tokens_does_not_divide_by_zero() -> None:
    assert "| Tokens removed | 0.0% |" in render_corpus_stats(_stats(raw=0, clean=0))


def test_main_renders_a_valid_file(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    path = tmp_path / "s.json"
    path.write_text(_stats().model_dump_json(), encoding="utf-8")
    assert main([str(path)]) == 0
    assert "| Notebooks | 1,234 |" in capsys.readouterr().out


def test_main_rejects_missing_and_invalid_files(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main([str(tmp_path / "missing.json")]) == 1
    bad = tmp_path / "bad.json"
    bad.write_text('{"schema_version": 1}', encoding="utf-8")
    assert main([str(bad)]) == 1
    assert capsys.readouterr().err.count("error: cannot render") == 2
