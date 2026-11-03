"""Header indexes, resources and the M5 runner end to end on the synthetic corpus (offline)."""

from pathlib import Path

import pytest
import structlog
from typer.testing import CliRunner, Result

from helpers_retrieval import CONTEXT, GIT, MARKER, make_gold, make_resources
from noterecall import cli
from noterecall.chunking.tokens import Tokenizer
from noterecall.config import DecidedConfig, Settings, load_settings
from noterecall.enrich.headers import chunk_header
from noterecall.eval.ablation import RunKey
from noterecall.eval.gold import Split, save_gold
from noterecall.eval.ledger import read_ledger
from noterecall.eval.run_header import HeaderDevResults, run_header_dev
from noterecall.eval.run_retrieval import (
    DevResults,
    RunError,
    TestResults,
    comparison_rows,
    run_test,
    select_queries,
)
from noterecall.index.code_tokenizer import tokenize_plain
from noterecall.index.embeddings import EmbeddingService, EmbeddingSpec, HashEmbedder
from noterecall.index.lexical import LexicalError, LexicalIndex
from noterecall.models import Cell, Chunk
from noterecall.retrieval.dense import DenseIndex
from noterecall.retrieval.pipeline import PipelineSpec, Variant
from render_results import main

SPEC = PipelineSpec(lexical="code", dense_model="small", rrf_k=60)
HEADER_ON = Variant(header=True)
Notebooks = list[tuple[str, list[Cell]]]


def chunk(cid: str, text: str) -> Chunk:
    return Chunk(
        id=cid,
        notebook_path="n.ipynb",
        heading_path=(),
        pieces=(),
        text=text,
        tokens=1,
        cell_hashes=(),
    )


def with_decided(settings: Settings, variant: Variant) -> Settings:
    decided = DecidedConfig(chunk_cap=64, pipeline=SPEC, variant=variant)
    return settings.model_copy(
        update={"retrieval": settings.retrieval.model_copy(update={"decided": decided})}
    )


# ------------------------------------------------------------------------------ indexed text


def test_a_lexical_index_searches_the_supplied_text_not_the_chunk_text() -> None:
    chunks = [chunk("a", "alpha beta"), chunk("b", "gamma delta")]
    assert LexicalIndex.build(chunks, tokenize_plain).search("zebra", 5) == []
    enriched = LexicalIndex.build(chunks, tokenize_plain, {"a": "alpha beta", "b": "gamma zebra"})
    assert [h.chunk_id for h in enriched.search("zebra", 5)] == ["b"]
    assert [h.chunk_id for h in enriched.search("alpha", 5)] == ["a"]


def test_a_missing_supplied_text_is_an_error_not_a_fallback() -> None:
    with pytest.raises(LexicalError, match="no indexed text was supplied for chunk b"):
        LexicalIndex.build([chunk("a", "x"), chunk("b", "y")], tokenize_plain, {"a": "x"})


def test_a_dense_index_embeds_the_supplied_text() -> None:
    chunks = [chunk("a", "alpha beta"), chunk("b", "gamma delta")]
    spec = EmbeddingSpec(id="m", revision="r", query_prefix="", passage_prefix="")
    service = EmbeddingService(HashEmbedder(), spec)
    enriched = DenseIndex.build(chunks, service, {"a": "gamma delta", "b": "alpha beta"})
    assert enriched.search("gamma delta", 1)[0].chunk_id == "a"  # a's indexed text is now this
    assert DenseIndex.build(chunks, service).search("gamma delta", 1)[0].chunk_id == "b"


def test_resources_index_the_header_plus_text_and_nothing_else_changes(
    config: Path, notebooks: Notebooks, tiny_tokenizer: Tokenizer
) -> None:
    resources = make_resources(load_settings(config), notebooks, tiny_tokenizer)
    off = RunKey(cap=64, chunker="structural", spec=SPEC)
    on = off.model_copy(update={"variant": HEADER_ON})
    assert resources.indexed_texts(off) is None
    texts = resources.indexed_texts(on)
    chunks = resources.chunks(64, "structural")
    assert texts is not None
    assert all(texts[c.id] == f"{chunk_header(c)}\n\n{c.text}" for c in chunks)
    assert resources.pipeline(on) is not resources.pipeline(off)  # separate indexes per variant
    assert chunks == resources.chunks(64, "structural")  # the shown text is the same chunks


def test_a_title_only_in_the_header_is_found_with_the_header_and_not_without(
    config: Path, notebooks: Notebooks, tiny_tokenizer: Tokenizer
) -> None:
    resources = make_resources(load_settings(config), notebooks, tiny_tokenizer)
    lexical = PipelineSpec(lexical="plain")
    off = RunKey(cap=64, chunker="structural", spec=lexical)
    on = off.model_copy(update={"variant": HEADER_ON})
    chunks = resources.chunks(64, "structural")
    stem = "02_deep_headings"  # a notebook file stem, absent from every cell's own text
    assert all(stem not in c.text for c in chunks)
    assert resources.pipeline(off).retrieve(stem)[0] == []
    hits = resources.pipeline(on).retrieve(stem)[0]
    assert hits
    by_id = {c.id: c for c in chunks}
    assert all(stem in by_id[h.chunk_id].notebook_path for h in hits)


# ------------------------------------------------------------------------------------ runners


def test_the_dev_runner_reports_header_on_vs_off_and_keeps_the_header_on(
    config: Path, notebooks: Notebooks, tiny_tokenizer: Tokenizer
) -> None:
    settings = with_decided(load_settings(config), Variant())
    gold = make_gold(notebooks)
    resources = make_resources(settings, notebooks, tiny_tokenizer)
    results = run_header_dev(settings, gold, resources, CONTEXT)
    dev_ids = {q.id for q in select_queries(gold, Split.DEV)}
    ablation = results.ablation
    assert set(ablation.with_header.per_query) == set(ablation.without_header.per_query) == dev_ids
    assert ablation.comparison.report_only
    assert results.decided == DecidedConfig(chunk_cap=64, pipeline=SPEC, variant=HEADER_ON)
    assert [(t.variant, t.truncated) for t in results.truncation] == [
        (Variant(), 0),
        (HEADER_ON, 0),
    ]
    assert results.meta.split is Split.DEV
    assert MARKER not in results.model_dump_json()
    again = run_header_dev(settings, gold, resources, CONTEXT)
    assert again.ablation.comparison.ci == ablation.comparison.ci


def test_the_dev_runner_needs_the_m4_decisions(
    config: Path, notebooks: Notebooks, tiny_tokenizer: Tokenizer
) -> None:
    settings = load_settings(config)
    resources = make_resources(settings, notebooks, tiny_tokenizer)
    with pytest.raises(RunError, match=r"no \[retrieval.decided\]"):
        run_header_dev(settings, make_gold(notebooks), resources, CONTEXT)


def test_comparison_rows_start_with_the_header_off_row_only_when_the_header_is_on() -> None:
    off = DecidedConfig(chunk_cap=64, pipeline=SPEC)
    on = DecidedConfig(chunk_cap=64, pipeline=SPEC, variant=HEADER_ON)
    parts = [
        (PipelineSpec(dense_model="small"), off.variant),
        (PipelineSpec(lexical="code"), off.variant),
    ]
    assert comparison_rows(off) == parts
    assert comparison_rows(on) == [
        (SPEC, Variant()),
        (PipelineSpec(dense_model="small"), HEADER_ON),
        (PipelineSpec(lexical="code"), HEADER_ON),
    ]


def test_the_test_run_scores_the_header_pipeline_against_the_same_pipeline_without_it(
    config: Path, notebooks: Notebooks, tiny_tokenizer: Tokenizer
) -> None:
    settings = with_decided(load_settings(config), HEADER_ON)
    gold = make_gold(notebooks)
    resources = make_resources(settings, notebooks, tiny_tokenizer)
    results = run_test(settings, gold, resources, CONTEXT, "M5")
    assert results.chosen.key.variant == HEADER_ON
    first = results.references[0]
    assert (first.key.spec, first.key.variant) == (SPEC, Variant())  # the M4 pipeline
    assert results.comparisons[0].baseline == first.name
    assert all(c.report_only for c in results.comparisons)
    test_ids = {q.id for q in select_queries(gold, Split.TEST)}
    assert set(results.chosen.per_query) == test_ids


# ---------------------------------------------------------------------------------------- CLI


def run(config: Path, *args: str) -> Result:
    result = CliRunner().invoke(cli.app, ["eval", *args, "--config", str(config)])
    structlog.reset_defaults()
    return result


def decide(config: Path, header: bool = False) -> None:
    text = """
[retrieval.decided]
chunk_cap = 64

[retrieval.decided.pipeline]
lexical = "code"
dense_model = "small"
rrf_k = 60
rerank = false
"""
    if header:
        text += "\n[retrieval.decided.variant]\nheader = true\n"
    config.write_text(config.read_text(encoding="utf-8") + text, encoding="utf-8")


def test_eval_header_dev_writes_its_results_file(
    config: Path,
    notebooks: Notebooks,
    fake_models: list[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(cli, "git_state", lambda: GIT)
    decide(config)
    settings = load_settings(config)
    save_gold(make_gold(notebooks), settings.paths.gold_path)
    result = run(config, "--suite", "header")
    assert result.exit_code == 0, result.output
    target = settings.paths.results_dir / "header_ablation_dev.json"
    parsed = HeaderDevResults.model_validate_json(target.read_text(encoding="utf-8"))
    assert parsed.decided.variant == HEADER_ON
    assert MARKER not in target.read_text(encoding="utf-8")
    assert fake_models == ["small"]  # only the decided dense model is loaded


def test_eval_header_needs_decisions_and_is_dev_only(config: Path, fake_models: list[str]) -> None:
    missing = run(config, "--suite", "header")
    assert missing.exit_code == 1
    assert "config has no [retrieval.decided]" in missing.stderr
    test = run(config, "--suite", "header", "--split", "test", "--milestone", "M5")
    assert test.exit_code == 1
    assert "dev report" in test.stderr


def test_the_test_run_needs_the_header_dev_results_when_the_header_is_on(
    config: Path,
    notebooks: Notebooks,
    fake_models: list[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(cli, "git_state", lambda: GIT)
    settings = load_settings(config)
    save_gold(make_gold(notebooks), settings.paths.gold_path)
    assert run(config).exit_code == 0  # the M4 dev ablation needs no decisions
    dev_path = settings.paths.results_dir / "retrieval_ablation_dev.json"
    spec = DevResults.model_validate_json(dev_path.read_text(encoding="utf-8")).decided.pipeline
    lines = [f'lexical = "{spec.lexical}"'] if spec.lexical else []
    lines += [f'dense_model = "{spec.dense_model}"'] if spec.dense_model else []
    lines += [f"rrf_k = {spec.rrf_k}"] if spec.rrf_k is not None else []
    pipeline = "\n".join([*lines, f"rerank = {str(spec.rerank).lower()}"])
    decided = f"\n[retrieval.decided]\nchunk_cap = 64\n\n[retrieval.decided.pipeline]\n{pipeline}\n"
    config.write_text(
        config.read_text(encoding="utf-8")
        + decided
        + "\n[retrieval.decided.variant]\nheader = true\n",
        encoding="utf-8",
    )
    args = ("--split", "test", "--milestone", "M5")
    missing = run(config, *args)
    assert missing.exit_code == 1
    assert "cannot read the header dev results" in missing.stderr
    assert run(config, "--suite", "header").exit_code == 0
    done = run(config, *args)
    assert done.exit_code == 0, done.output
    result_file = settings.paths.results_dir / "retrieval_test_M5.json"
    parsed = TestResults.model_validate_json(result_file.read_text(encoding="utf-8"))
    assert parsed.chosen.key.variant == HEADER_ON
    ledger = read_ledger(settings.paths.results_dir / "test_ledger.json")
    assert [e.milestone for e in ledger.entries] == ["M5"]


def test_the_header_results_render_to_markdown_through_the_results_script(
    config: Path,
    notebooks: Notebooks,
    tiny_tokenizer: Tokenizer,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    settings = with_decided(load_settings(config), Variant())
    resources = make_resources(settings, notebooks, tiny_tokenizer)
    results = run_header_dev(settings, make_gold(notebooks), resources, CONTEXT)
    path = tmp_path / "header.json"
    path.write_text(results.model_dump_json(), encoding="utf-8")
    assert main([str(path)]) == 0
    out = capsys.readouterr().out
    assert "| Stage | Comparison | Metric | Mean delta | 95% CI | Free | Decision |" in out
    assert "header on vs off (report only)" in out
    assert "| structural-64 \\| bm25-code + dense:small + rrf60 +header |" in out
    assert "were not tested (proposal 005, owner decision)" in out
    assert "| header off |" in out
