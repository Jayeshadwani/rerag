"""`noterecall enrich ...` and `noterecall eval --suite enrichment` end to end, offline."""

from pathlib import Path

import pytest
import structlog
from test_enrich_integration import generate, respond
from typer.testing import CliRunner, Result

from helpers_retrieval import GIT, MARKER, make_gold
from noterecall import cli
from noterecall.chunking.tokens import Tokenizer
from noterecall.config import load_settings
from noterecall.enrich import steps
from noterecall.enrich.steps import model_tokenizer
from noterecall.eval.gold import save_gold
from noterecall.eval.ledger import read_ledger
from noterecall.eval.run_enrichment import EnrichmentDevResults
from noterecall.eval.run_retrieval import TestResults
from noterecall.llm.cache import ResponseCache
from noterecall.llm.fake import FakeTransport
from noterecall.models import Cell

Notebooks = list[tuple[str, list[Cell]]]
DECIDED = """
[retrieval.decided]
chunk_cap = 64

[retrieval.decided.pipeline]
lexical = "code"
dense_model = "small"
rrf_k = 60
rerank = false
"""


@pytest.fixture(autouse=True)
def _offline_steps(monkeypatch: pytest.MonkeyPatch, tiny_tokenizer: Tokenizer) -> None:
    monkeypatch.setattr(steps, "model_tokenizer", lambda _model: tiny_tokenizer)
    monkeypatch.setattr(cli, "git_state", lambda: GIT)

    def no_api(*_args: object) -> object:
        raise AssertionError("a test must never create the real API transport")

    monkeypatch.setattr(steps, "create_transport", no_api)


def run(config: Path, *args: str) -> Result:
    result = CliRunner().invoke(cli.app, [*args, "--config", str(config)])
    structlog.reset_defaults()
    return result


def decide(config: Path) -> None:
    config.write_text(config.read_text(encoding="utf-8") + DECIDED, encoding="utf-8")


def test_enrich_needs_the_m4_decisions_for_the_chunk_cap(
    config: Path, notebooks: Notebooks
) -> None:
    result = run(config, "enrich", "blurbs", "--dry-run")
    assert result.exit_code == 1
    assert "config has no [retrieval.decided]" in result.stderr


def test_a_dry_run_prints_the_estimate_and_changes_nothing(
    config: Path, notebooks: Notebooks
) -> None:
    decide(config)
    for step in ("blurbs", "clean"):
        result = run(config, "enrich", step, "--dry-run")
        assert result.exit_code == 0, result.output
        assert f"enrich {step}: model test-model, reasoning effort minimal" in result.stdout
        assert "estimated cost, prefix caching works" in result.stdout
    cache = ResponseCache(load_settings(config).llm.cache_path)
    assert cache.spend() == []  # nothing was stored, nothing was spent


def test_a_real_run_is_refused_without_yes_and_says_how_many_calls_it_would_make(
    config: Path, notebooks: Notebooks
) -> None:
    decide(config)
    result = run(config, "enrich", "blurbs")
    assert result.exit_code == 1
    assert "re-run with --yes" in result.stderr
    assert "no call was made" in result.stderr


def test_a_confirmed_run_calls_the_transport_and_fills_the_cache(
    config: Path, notebooks: Notebooks, monkeypatch: pytest.MonkeyPatch
) -> None:
    decide(config)
    transport = FakeTransport(respond)
    monkeypatch.setattr(steps, "create_transport", lambda *_a: transport)
    result = run(config, "enrich", "clean", "--yes")
    assert result.exit_code == 0, result.output
    assert "answered by the API" in result.stdout
    assert transport.prompts
    cache = ResponseCache(load_settings(config).llm.cache_path)
    assert {r.prompt_name for r in cache.spend()} == {"clean_markdown"}
    blurbs = run(config, "enrich", "blurbs", "--cleaned", "--yes")
    assert blurbs.exit_code == 0, blurbs.output
    assert {r.prompt_name for r in cache.spend()} == {"clean_markdown", "contextual_blurb"}


def test_blurbs_for_cleaned_notebooks_need_the_clean_to_have_run(
    config: Path, notebooks: Notebooks
) -> None:
    decide(config)
    result = run(config, "enrich", "blurbs", "--cleaned", "--dry-run")
    assert result.exit_code == 1
    assert "cleans are not in the cache" in result.stderr


def test_the_enrichment_eval_runs_dev_then_test_once(
    config: Path,
    notebooks: Notebooks,
    tiny_tokenizer: Tokenizer,
    fake_models: list[str],
) -> None:
    decide(config)
    settings = load_settings(config)
    save_gold(make_gold(notebooks), settings.paths.gold_path)
    missing = run(config, "eval", "--suite", "enrichment")
    assert missing.exit_code == 1
    assert "not in the cache" in missing.stderr  # nothing was generated yet: no silent baseline
    cache = ResponseCache(settings.llm.cache_path)
    generate(settings, notebooks, tiny_tokenizer, cache)
    dev = run(config, "eval", "--suite", "enrichment")
    assert dev.exit_code == 0, dev.output
    target = settings.paths.results_dir / "enrichment_ablation_dev.json"
    text = target.read_text(encoding="utf-8")
    parsed = EnrichmentDevResults.model_validate_json(text)
    assert set(parsed.meta.prompt_versions) == {"contextual_blurb", "clean_markdown"}
    assert MARKER not in text
    stale = run(config, "eval", "--suite", "enrichment", "--split", "test", "--milestone", "M5")
    assert stale.exit_code == 1  # the config's decided values lack the variant the dev run chose
    assert "differs from the decisions" in stale.stderr
    variant = parsed.decided.variant
    assert variant.header  # the header is carried whatever its row says
    flags = "\n".join(f"{k} = {str(v).lower()}" for k, v in variant.model_dump().items())
    config.write_text(
        config.read_text(encoding="utf-8") + f"\n[retrieval.decided.variant]\n{flags}\n",
        encoding="utf-8",
    )
    done = run(config, "eval", "--suite", "enrichment", "--split", "test", "--milestone", "M5")
    assert done.exit_code == 0, done.output
    result_file = settings.paths.results_dir / "enrichment_test_M5.json"
    test = TestResults.model_validate_json(result_file.read_text(encoding="utf-8"))
    assert test.milestone == "M5"
    assert [
        e.milestone for e in read_ledger(settings.paths.results_dir / "test_ledger.json").entries
    ] == ["M5"]
    again = run(config, "eval", "--suite", "enrichment", "--split", "test", "--milestone", "M5")
    assert again.exit_code == 1
    assert "already run" in again.stderr


def test_enrich_needs_a_synced_database(config: Path) -> None:
    decide(config)
    for step in ("blurbs", "clean"):
        result = run(config, "enrich", step, "--dry-run")
        assert result.exit_code == 1
        assert "the database has no notebooks" in result.stderr


def test_the_enrichment_test_run_needs_decisions_and_the_dev_results(
    config: Path, notebooks: Notebooks, fake_models: list[str]
) -> None:
    settings = load_settings(config)
    save_gold(make_gold(notebooks), settings.paths.gold_path)
    args = ("eval", "--suite", "enrichment", "--split", "test", "--milestone", "M5")
    no_decisions = run(config, *args)
    assert no_decisions.exit_code == 1
    assert "config has no [retrieval.decided]" in no_decisions.stderr
    decide(config)
    no_dev = run(config, *args)
    assert no_dev.exit_code == 1
    assert "cannot read the enrichment dev results" in no_dev.stderr


@pytest.mark.live
def test_the_model_tokenizer_is_the_encoding_tiktoken_maps_to_the_model() -> None:
    assert model_tokenizer("gpt-5-mini").name == "o200k_base"
