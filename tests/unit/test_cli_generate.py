"""`noterecall ask`, `eval --suite abstention|generation` and `smoke-answer`, offline."""

from pathlib import Path

import pytest
import structlog
from test_generation_flow import answer_with_first_note, verdict_json
from typer.testing import CliRunner, Result

from helpers_retrieval import GIT, MARKER, make_gold
from noterecall import cli
from noterecall.chunking.tokens import Tokenizer
from noterecall.config import load_settings
from noterecall.eval.gold import GoldSet, Split, save_gold
from noterecall.eval.ledger import read_ledger
from noterecall.eval.run_generation import CalibrationResults, GenerationResults
from noterecall.generate import runtime
from noterecall.llm.cache import ResponseCache
from noterecall.llm.client import Prompt
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


def respond(prompt: Prompt) -> str:
    """The fake model: answers cite the first note; the verifier supports every claim."""
    return verdict_json(prompt) if prompt.name == "verify" else answer_with_first_note(prompt)


@pytest.fixture
def transport(monkeypatch: pytest.MonkeyPatch, tiny_tokenizer: Tokenizer) -> FakeTransport:
    fake = FakeTransport(respond)
    monkeypatch.setattr(runtime, "create_transport", lambda *_args: fake)
    monkeypatch.setattr(cli, "model_tokenizer", lambda _model: tiny_tokenizer)
    monkeypatch.setattr(cli, "git_state", lambda: GIT)
    return fake


def run(config: Path, *args: str) -> Result:
    result = CliRunner().invoke(cli.app, [*args, "--config", str(config)])
    structlog.reset_defaults()
    return result


def prepare(config: Path, notebooks: Notebooks, thresholds: str | None = None) -> None:
    text = config.read_text(encoding="utf-8") + DECIDED
    if thresholds is not None:
        text = text.replace(
            "max_false_refusal_pct = 5\n", "max_false_refusal_pct = 5\ncosine_only = -1.0\n", 1
        )
        text += f"\n[generation.decided]\n{thresholds}\n"
    config.write_text(text, encoding="utf-8")
    save_gold(make_gold(notebooks), load_settings(config).paths.gold_path)


NEVER = "cosine = -1.0\nbm25 = -1.0"
ALWAYS = "cosine = 2.0\nbm25 = 1000000.0"


def test_the_abstention_calibration_writes_its_file_and_calls_no_model(
    config: Path, notebooks: Notebooks, fake_models: list[str], transport: FakeTransport
) -> None:
    prepare(config, notebooks)
    result = run(config, "eval", "--suite", "abstention")
    assert result.exit_code == 0, result.output
    assert "cosine + BM25: tau_cos" in result.stdout
    assert "cosine only: tau_cos" in result.stdout
    target = load_settings(config).paths.results_dir / "abstention_calibration_dev.json"
    parsed = CalibrationResults.model_validate_json(target.read_text(encoding="utf-8"))
    assert parsed.meta.split is Split.DEV
    assert MARKER not in target.read_text(encoding="utf-8")
    assert transport.prompts == []
    assert fake_models == ["small"]  # only the decided dense model is loaded
    test = run(config, "eval", "--suite", "abstention", "--split", "test", "--milestone", "M6")
    assert test.exit_code == 1
    assert "dev report" in test.stderr


def test_a_dry_run_prints_the_estimate_and_a_real_run_needs_yes(
    config: Path, notebooks: Notebooks, fake_models: list[str], transport: FakeTransport
) -> None:
    prepare(config, notebooks, NEVER)
    dry = run(config, "eval", "--suite", "generation", "--dry-run")
    assert dry.exit_code == 0, dry.output
    assert "0 refused by stage 1" in dry.stdout
    assert "estimated cost" in dry.stdout
    assert "of the $1.00 cap" in dry.stdout  # the fixture config's cap
    refused = run(config, "eval", "--suite", "generation")
    assert refused.exit_code == 1
    assert "re-run with --yes" in refused.stderr
    assert transport.prompts == []  # neither command called the model


def test_a_confirmed_dev_run_writes_the_results_and_the_cost(
    config: Path, notebooks: Notebooks, fake_models: list[str], transport: FakeTransport
) -> None:
    prepare(config, notebooks, NEVER)
    result = run(config, "eval", "--suite", "generation", "--yes")
    assert result.exit_code == 0, result.output
    target = load_settings(config).paths.results_dir / "generation_dev.json"
    assert f"eval: wrote {target} (cost $" in result.stdout
    parsed = GenerationResults.model_validate_json(target.read_text(encoding="utf-8"))
    assert parsed.validity.answered == len(parsed.queries)
    assert MARKER not in target.read_text(encoding="utf-8")
    assert transport.prompts  # the fake was called


def test_the_test_split_runs_once_per_milestone_and_needs_a_clean_tree(
    config: Path,
    notebooks: Notebooks,
    fake_models: list[str],
    transport: FakeTransport,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepare(config, notebooks, NEVER)
    args = ("eval", "--suite", "generation", "--split", "test", "--yes")
    assert (
        run(config, *args)
        .stderr.strip()
        .endswith("needs --milestone (the test split runs once per milestone)")
    )
    done = run(config, *args, "--milestone", "M6")
    assert done.exit_code == 0, done.output
    settings = load_settings(config)
    assert (settings.paths.results_dir / "generation_test_M6.json").is_file()
    ledger = read_ledger(settings.paths.results_dir / "test_ledger.json")
    assert [e.milestone for e in ledger.entries] == ["M6"]
    again = run(config, *args, "--milestone", "M6")
    assert again.exit_code == 1
    assert "already run" in again.stderr


def test_a_dirty_tree_blocks_the_test_run(
    config: Path,
    notebooks: Notebooks,
    fake_models: list[str],
    transport: FakeTransport,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepare(config, notebooks, NEVER)
    dirty = GIT.model_copy(update={"dirty": True})
    monkeypatch.setattr(cli, "git_state", lambda: dirty)
    result = run(config, "eval", "--suite", "generation", "--split", "test", "--yes",
                 "--milestone", "M6")  # fmt: skip
    assert result.exit_code == 1
    assert "uncommitted changes" in result.stderr
    assert transport.prompts == []


def test_ask_prints_an_answer_with_sources_or_says_it_is_not_in_the_notes(
    config: Path, notebooks: Notebooks, fake_models: list[str], transport: FakeTransport
) -> None:
    prepare(config, notebooks, NEVER)
    answered = run(config, "ask", "groupby mean of each key", "--show-context")
    assert answered.exit_code == 0, answered.output
    assert "- an answer [1]" in answered.stdout
    assert "Sources:" in answered.stdout
    assert "top cosine" in answered.stdout
    sent = len(transport.prompts)
    config.write_text(config.read_text(encoding="utf-8").replace(NEVER, ALWAYS), encoding="utf-8")
    refused = run(config, "ask", "groupby mean of each key")
    assert refused.exit_code == 0
    assert "Not in your notes" in refused.stdout
    assert len(transport.prompts) == sent  # the refusal made no call


def test_ask_and_the_generation_run_need_the_calibrated_thresholds(
    config: Path, notebooks: Notebooks, fake_models: list[str], transport: FakeTransport
) -> None:
    prepare(config, notebooks)
    for args in (("ask", "anything"), ("eval", "--suite", "generation", "--dry-run")):
        result = run(config, *args)
        assert result.exit_code == 1
        assert "no [generation.decided]" in result.stderr


def test_smoke_a_needs_no_notes_and_b_picks_three_dev_queries_with_a_spend_cap(
    config: Path, notebooks: Notebooks, fake_models: list[str], transport: FakeTransport
) -> None:
    prepare(config, notebooks, NEVER)
    a = run(config, "smoke-answer", "--step", "a")
    assert a.exit_code == 0, a.output
    assert "step A (invented chunks)" in a.stdout
    assert "total cost: $" in a.stdout
    b = run(config, "smoke-answer", "--step", "b")
    assert b.exit_code == 0, b.output
    assert b.stdout.count("| q") == 3  # three dev queries, shown by id only
    assert MARKER not in b.stdout
    bad = run(config, "smoke-answer", "--step", "d")
    assert bad.exit_code == 1
    assert "--step must be a, b or c" in bad.stderr
    cache = ResponseCache(load_settings(config).llm.cache_path)
    assert cache.spend()  # the calls were recorded in the ledger


def test_the_abstention_calibration_needs_the_m4_decisions(
    config: Path, notebooks: Notebooks, fake_models: list[str], transport: FakeTransport
) -> None:
    save_gold(make_gold(notebooks), load_settings(config).paths.gold_path)
    result = run(config, "eval", "--suite", "abstention")
    assert result.exit_code == 1
    assert "config has no [retrieval.decided]" in result.stderr


@pytest.mark.live
def test_the_model_tokenizer_is_the_encoding_tiktoken_maps_to_the_model() -> None:
    assert runtime.model_tokenizer("gpt-5-mini").name == "o200k_base"


def test_a_stage_three_dry_run_prices_the_verifier_and_a_real_run_writes_its_own_file(
    config: Path, notebooks: Notebooks, fake_models: list[str], transport: FakeTransport
) -> None:
    prepare(config, notebooks, NEVER)
    dry = run(config, "eval", "--suite", "generation", "--stage3", "--dry-run")
    assert dry.exit_code == 0, dry.output
    assert "stage 3 calls:" in dry.stdout
    assert "total estimated cost" in dry.stdout
    assert transport.prompts == []
    done = run(config, "eval", "--suite", "generation", "--stage3", "--yes")
    assert done.exit_code == 0, done.output
    target = load_settings(config).paths.results_dir / "generation_dev_stage3.json"
    parsed = GenerationResults.model_validate_json(target.read_text(encoding="utf-8"))
    assert parsed.stage3 is not None
    assert parsed.rows[-1].name == "stages 1+2+3"
    assert parsed.verify_prompt_version is not None
    assert MARKER not in target.read_text(encoding="utf-8")
    assert not (load_settings(config).paths.results_dir / "generation_dev.json").exists()


def test_the_abstention_set_is_run_without_a_split_and_without_the_test_ledger(
    config: Path,
    notebooks: Notebooks,
    fake_models: list[str],
    transport: FakeTransport,
) -> None:
    prepare(config, notebooks, NEVER)
    settings = load_settings(config)
    missing = run(config, "eval", "--suite", "generation", "--set", "abstention", "--dry-run")
    assert missing.exit_code == 1
    assert "abstention set was not found" in missing.stderr
    gold = make_gold(notebooks)
    unfrozen = GoldSet(queries=tuple(q.model_copy(update={"split": None}) for q in gold.queries))
    save_gold(unfrozen, settings.paths.abstention_path)
    args = ("eval", "--suite", "generation", "--set", "abstention", "--stage3", "--yes")
    done = run(config, *args)
    assert done.exit_code == 0, done.output
    target = settings.paths.results_dir / "generation_abstention_stage3.json"
    parsed = GenerationResults.model_validate_json(target.read_text(encoding="utf-8"))
    assert parsed.query_set == "abstention"
    assert parsed.meta.split is None
    assert len(parsed.queries) == len(gold.queries)  # every query of the set, both splits' worth
    assert not (settings.paths.results_dir / "test_ledger.json").exists()
    test = run(config, *args, "--split", "test")  # a split means nothing for this set
    assert test.exit_code == 0


def test_with_stage_three_on_in_config_ask_verifies_before_showing_an_answer(
    config: Path, notebooks: Notebooks, fake_models: list[str], transport: FakeTransport
) -> None:
    prepare(config, notebooks, NEVER)
    config.write_text(
        config.read_text(encoding="utf-8").replace(
            "verify_effort", "stage3 = true\nverify_effort", 1
        ),
        encoding="utf-8",
    )
    result = run(config, "ask", "groupby mean of each key")
    assert result.exit_code == 0, result.output
    assert "Sources:" in result.stdout
    assert [p.name for p in transport.prompts] == ["answer", "verify"]
    assert transport.prompts[1].reasoning_effort == "medium"


def test_smoke_step_c_runs_stage_three_on_invented_chunks(
    config: Path, notebooks: Notebooks, fake_models: list[str], transport: FakeTransport
) -> None:
    prepare(config, notebooks, NEVER)
    result = run(config, "smoke-answer", "--step", "c")
    assert result.exit_code == 0, result.output
    assert "step C (stage 3, invented chunks)" in result.stdout
    assert "answers_question=True; kept 2 of 2 claims" in result.stdout
    assert [p.name for p in transport.prompts] == ["verify"]
