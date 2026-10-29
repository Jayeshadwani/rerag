"""render_results.py prints only numbers that are in the results file."""

from pathlib import Path

import pytest

from noterecall.chunking.evidence_check import EvidenceEval, Interval
from noterecall.chunking.report import ChunkRow, ChunkSizes, ChunkStats
from noterecall.ingest.stats import (
    CorpusStats,
    Distribution,
    GitState,
    TokenTotals,
)
from noterecall.models import OutputMode
from render_results import main, render_chunk_stats, render_corpus_stats, render_evidence_eval


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


def _chunk_stats() -> ChunkStats:
    row = ChunkRow(
        chunker="structural",
        cap=256,
        outputs=OutputMode.NONE,
        chunks=3300,
        tokens=ChunkSizes(min=3, median=157.5, p90=250, max=2999),
        under_quarter_cap=705,
        flagged_by_reason={"statement": 37, "sentence": 19, "fenced_block": 0},
        cells_split=630,
        unparsed_code_cells=7,
    )
    return ChunkStats(
        schema_version=1,
        git=GitState(sha="f" * 40, dirty=False),
        config={},
        pipeline_hash="p" * 64,
        tokenizer="cl100k_base",
        tiktoken_version="0",
        evidence_rule_fingerprint="e" * 64,
        notebooks=53,
        cells=6061,
        outputs_by_verdict={"evidence": 100, "too_long": 900},
        rows=[row],
    )


def test_chunk_stats_table_shows_every_number_from_the_file() -> None:
    text = render_chunk_stats(_chunk_stats())
    row = "| structural | 256 | none | 3,300 | 3 / 157.5 / 250 / 2,999 | 705 |"
    assert row + " statement 37, sentence 19 | 630 | 7 |" in text
    assert "| evidence | 100 |" in text
    assert "| too_long | 900 |" in text
    assert "53 notebooks, 6,061 cells. Commit `ffffffffffff`, rule `eeeeeeeeeeee`." in text


def _evaluation(precision: float | None = 0.8) -> EvidenceEval:
    interval = Interval(low=0.6087, high=0.9116) if precision is not None else None
    return EvidenceEval(
        seed=7,
        rule_fingerprint="a" * 64,
        population_kept=100,
        population_dropped=300,
        sampled_kept=25,
        sampled_dropped=25,
        labelled_evidence_in_kept=20,
        labelled_evidence_in_dropped=5,
        precision=precision,
        precision_wilson95=interval,
        recall_weighted=80 / 140,
        evidence_prevalence_weighted=0.35,
    )


def test_evidence_eval_table() -> None:
    text = render_evidence_eval(_evaluation())
    assert "| Precision (kept that are evidence) | 80.0% (95% Wilson 60.9% to 91.2%) |" in text
    assert "| Recall (evidence that is kept, weighted to the population) | 57.1% |" in text
    assert "| Evidence prevalence (weighted) | 35.0% |" in text
    assert "| Outputs the rule keeps / drops (population) | 100 / 300 |" in text
    assert "Sampling seed 7, rule `aaaaaaaaaaaa`." in text
    assert "| Precision (kept that are evidence) | n/a |" in render_evidence_eval(_evaluation(None))


def test_main_dispatches_on_the_kind_of_results_file(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    chunks = tmp_path / "chunks.json"
    chunks.write_text(_chunk_stats().model_dump_json(), encoding="utf-8")
    evaluation = tmp_path / "eval.json"
    evaluation.write_text(_evaluation().model_dump_json(), encoding="utf-8")
    assert main([str(chunks)]) == 0
    assert "Flagged oversize" in capsys.readouterr().out
    assert main([str(evaluation)]) == 0
    assert "Evidence rule vs owner labels" in capsys.readouterr().out
