"""Command line: sync and stats end to end on the synthetic fixtures (tokenizer is offline)."""

import json
import shutil
from collections.abc import Iterator
from pathlib import Path

import pytest
import structlog
from typer.testing import CliRunner

from make_fixtures import FIXTURE_DIR
from noterecall import cli
from noterecall.chunking.evidence import rule_fingerprint
from noterecall.chunking.evidence_check import EvidenceEval, read_manifest
from noterecall.chunking.report import ChunkStats
from noterecall.chunking.tokens import Tokenizer, TokenizerError
from noterecall.ingest.stats import CorpusStats

RULES_PATH = Path(__file__).resolve().parents[2] / "config" / "cleaning_rules.yaml"
runner = CliRunner()


@pytest.fixture(autouse=True)
def _offline(monkeypatch: pytest.MonkeyPatch, tiny_tokenizer: Tokenizer) -> Iterator[None]:
    for name in ("NOTERECALL_LOGGING__LEVEL", "NOTERECALL_PATHS__RAW_DIR"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(cli, "load_tokenizer", lambda _name: tiny_tokenizer)
    yield
    structlog.reset_defaults()


@pytest.fixture
def config(tmp_path: Path) -> Path:
    raw = tmp_path / "raw"
    shutil.copytree(FIXTURE_DIR, raw)
    path = tmp_path / "config.toml"
    path.write_text(
        f"""
[paths]
raw_dir = "{raw}"
db_path = "{tmp_path / "state" / "notes.sqlite"}"
results_dir = "{tmp_path / "results"}"

[logging]
level = "INFO"

[cleaning]
rules_path = "{RULES_PATH}"
max_output_lines = 20

[tokenizer]
encoding = "tiny-bytes"

[chunking]
candidate_caps = [64, 128]
""",
        encoding="utf-8",
    )
    return path


def test_help_lists_both_commands() -> None:
    result = runner.invoke(cli.app, ["--help"])
    assert result.exit_code == 0
    assert "sync" in result.stdout
    assert "stats" in result.stdout


def test_sync_then_resync(config: Path) -> None:
    first = runner.invoke(cli.app, ["sync", "--config", str(config)])
    assert first.exit_code == 0, first.stderr
    assert first.stdout.strip() == "sync: 7 added, 0 updated, 0 unchanged, 0 removed"
    second = runner.invoke(cli.app, ["sync", "--config", str(config)])
    assert second.stdout.strip() == "sync: 0 added, 0 updated, 7 unchanged, 0 removed"


def test_logs_go_to_stderr_without_file_names(config: Path) -> None:
    result = runner.invoke(cli.app, ["sync", "--config", str(config)])
    assert "sync.done" in result.stderr
    assert "sync.done" not in result.stdout
    assert ".ipynb" not in result.stderr
    assert "no_headings" not in result.stderr


def test_stats_writes_the_report_to_the_default_path(config: Path, tmp_path: Path) -> None:
    runner.invoke(cli.app, ["sync", "--config", str(config)])
    result = runner.invoke(cli.app, ["stats", "--config", str(config)])
    assert result.exit_code == 0, result.stderr
    target = tmp_path / "results" / "corpus_stats.json"
    assert result.stdout.strip() == f"stats: wrote {target}"
    stats = CorpusStats.model_validate_json(target.read_text(encoding="utf-8"))
    assert (stats.notebooks, stats.cells_raw) == (7, 54)


def test_stats_honours_out(config: Path, tmp_path: Path) -> None:
    runner.invoke(cli.app, ["sync", "--config", str(config)])
    out = tmp_path / "elsewhere" / "s.json"
    result = runner.invoke(cli.app, ["stats", "--config", str(config), "--out", str(out)])
    assert result.exit_code == 0
    assert out.is_file()


def test_stats_before_sync_fails_with_a_clear_message(config: Path) -> None:
    result = runner.invoke(cli.app, ["stats", "--config", str(config)])
    assert result.exit_code == 1
    assert "error: the database has no notebooks" in result.stderr


def test_sync_with_missing_raw_dir_fails(config: Path, tmp_path: Path) -> None:
    shutil.rmtree(tmp_path / "raw")
    result = runner.invoke(cli.app, ["sync", "--config", str(config)])
    assert result.exit_code == 1
    assert "error: raw directory does not exist" in result.stderr


def test_tokenizer_failure_is_reported_not_raised(
    config: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def offline(_name: str) -> Tokenizer:
        raise TokenizerError("cannot load tokenizer 'x': proxy said no")

    monkeypatch.setattr(cli, "load_tokenizer", offline)
    result = runner.invoke(cli.app, ["sync", "--config", str(config)])
    assert result.exit_code == 1
    assert "proxy said no" in result.stderr


def test_missing_config_file_fails(tmp_path: Path) -> None:
    result = runner.invoke(cli.app, ["sync", "--config", str(tmp_path / "nope.toml")])
    assert result.exit_code == 1
    assert "error: config file not found" in result.stderr


def test_unexpected_errors_are_not_swallowed(config: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(*_args: object) -> None:
        raise RuntimeError("unexpected")

    monkeypatch.setattr(cli, "run_sync", boom)
    result = runner.invoke(cli.app, ["sync", "--config", str(config)])
    assert result.exit_code == 1
    assert isinstance(result.exception, RuntimeError)


def test_log_level_filters_events(config: Path) -> None:
    text = config.read_text(encoding="utf-8").replace('level = "INFO"', 'level = "WARNING"')
    config.write_text(text, encoding="utf-8")
    result = runner.invoke(cli.app, ["sync", "--config", str(config)])
    assert result.exit_code == 0
    assert "sync.done" not in result.stderr


def _synced(config: Path) -> None:
    assert runner.invoke(cli.app, ["sync", "--config", str(config)]).exit_code == 0


def test_chunk_stats_writes_rows_for_every_cap_and_mode(config: Path, tmp_path: Path) -> None:
    _synced(config)
    result = runner.invoke(cli.app, ["chunk-stats", "--config", str(config)])
    assert result.exit_code == 0, result.stderr
    target = tmp_path / "results" / "chunk_stats.json"
    assert result.stdout.strip() == f"chunk-stats: wrote {target}"
    stats = ChunkStats.model_validate_json(target.read_text(encoding="utf-8"))
    assert len(stats.rows) == 8  # caps [64, 128] x (3 output modes + the fixed baseline)
    # kept cells per fixture notebook: 5 + 9 + 3 + 6 + 4 + 13 + 6
    assert (stats.notebooks, stats.cells) == (7, 46)
    assert stats.evidence_rule_fingerprint == rule_fingerprint()


def test_chunk_stats_honours_out_and_needs_a_synced_database(config: Path, tmp_path: Path) -> None:
    before = runner.invoke(cli.app, ["chunk-stats", "--config", str(config)])
    assert before.exit_code == 1
    assert "error: the database has no notebooks" in before.stderr
    _synced(config)
    out = tmp_path / "elsewhere" / "c.json"
    assert (
        runner.invoke(
            cli.app, ["chunk-stats", "--config", str(config), "--out", str(out)]
        ).exit_code
        == 0
    )
    assert out.is_file()


def test_evidence_sample_prints_a_blind_sample_and_stores_it_privately(
    config: Path, tmp_path: Path
) -> None:
    _synced(config)
    result = runner.invoke(cli.app, ["evidence-sample", "--config", str(config), "--n", "4"])
    assert result.exit_code == 0, result.stderr
    assert all(f"#{i}  (" in result.stdout for i in (1, 2, 3, 4))
    for private in ("ipynb", "no_headings", "caching", "True", "False"):
        assert private not in result.stdout
    manifest = read_manifest(tmp_path / "state" / "evidence" / "sample.json")
    # kept by the rule: 04's short table-less output and 05's "loaded key ..."; dropped: 01's table
    # and 03's truncated listing
    assert (manifest.population_kept, manifest.population_dropped) == (2, 2)
    assert len(manifest.items) == 4
    assert "drawn blind" in result.stderr


def _label_everything_like_the_rule(tmp_path: Path) -> Path:
    manifest = read_manifest(tmp_path / "state" / "evidence" / "sample.json")
    labels = {str(i.sample_id): "E" if i.rule_keeps else "N" for i in manifest.items}
    path = tmp_path / "state" / "evidence" / "labels.json"
    path.write_text(json.dumps({"labels": labels}), encoding="utf-8")
    return path


def test_evidence_eval_reports_against_the_labels(config: Path, tmp_path: Path) -> None:
    _synced(config)
    runner.invoke(cli.app, ["evidence-sample", "--config", str(config), "--n", "4"])
    _label_everything_like_the_rule(tmp_path)
    result = runner.invoke(cli.app, ["evidence-eval", "--config", str(config)])
    assert result.exit_code == 0, result.stderr
    target = tmp_path / "results" / "evidence_rule_eval.json"
    evaluation = EvidenceEval.model_validate_json(target.read_text(encoding="utf-8"))
    assert (evaluation.precision, evaluation.recall_weighted) == (1.0, 1.0)
    text = target.read_text(encoding="utf-8")
    assert "ipynb" not in text  # no paths in the results file


def test_evidence_eval_refuses_after_the_rule_changed(config: Path, tmp_path: Path) -> None:
    _synced(config)
    runner.invoke(cli.app, ["evidence-sample", "--config", str(config), "--n", "4"])
    labels = _label_everything_like_the_rule(tmp_path)
    sample = tmp_path / "state" / "evidence" / "sample.json"
    data = json.loads(sample.read_text(encoding="utf-8"))
    data["rule_fingerprint"] = "0" * 64
    sample.write_text(json.dumps(data), encoding="utf-8")
    result = runner.invoke(
        cli.app, ["evidence-eval", "--config", str(config), "--labels", str(labels)]
    )
    assert result.exit_code == 1
    assert "rule changed after this sample was drawn" in result.stderr


def test_evidence_eval_without_a_sample_or_labels_fails_clearly(
    config: Path, tmp_path: Path
) -> None:
    result = runner.invoke(cli.app, ["evidence-eval", "--config", str(config)])
    assert result.exit_code == 1
    assert "cannot read the sample" in result.stderr
    _synced(config)
    runner.invoke(cli.app, ["evidence-sample", "--config", str(config), "--n", "4"])
    missing = runner.invoke(cli.app, ["evidence-eval", "--config", str(config)])
    assert missing.exit_code == 1
    assert "cannot read the labels" in missing.stderr


def test_evidence_sample_on_an_empty_database_fails_loudly(config: Path) -> None:
    result = runner.invoke(cli.app, ["evidence-sample", "--config", str(config)])
    assert result.exit_code == 1
    assert "no code-cell outputs to sample" in result.stderr


@pytest.mark.parametrize("command", ["chunk-stats", "evidence-sample", "evidence-eval"])
def test_new_commands_report_a_missing_config_file(command: str, tmp_path: Path) -> None:
    result = runner.invoke(cli.app, [command, "--config", str(tmp_path / "nope.toml")])
    assert result.exit_code == 1
    assert "error: config file not found" in result.stderr
