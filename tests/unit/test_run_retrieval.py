"""The runner end to end on the synthetic fixture corpus, with the offline hash embedder and
overlap reranker. Gold queries here are synthetic and live only in memory."""

from pathlib import Path

import pytest
import structlog
from typer.testing import CliRunner

from noterecall import cli
from noterecall.chunking.tokens import Tokenizer
from noterecall.config import DecidedConfig, Settings, load_settings
from noterecall.eval.ablation import RunKey
from noterecall.eval.gold import (
    CellRef,
    GoldQuery,
    GoldSet,
    QueryType,
    Split,
    freeze_gold,
    save_gold,
)
from noterecall.eval.ledger import read_ledger
from noterecall.eval.run_retrieval import (
    SEED,
    DevResults,
    Resources,
    RunContext,
    RunError,
    TestResults,
    ablation_params,
    build_chunks,
    run_dev,
    run_test,
    score_params,
    select_queries,
)
from noterecall.index.embeddings import EmbeddingService, EmbeddingSpec, HashEmbedder
from noterecall.index.store import connect, notebook_cells
from noterecall.ingest.stats import GitState
from noterecall.ingest.sync import file_sha256
from noterecall.models import Cell
from noterecall.retrieval.pipeline import Depths, PipelineSpec
from noterecall.retrieval.rerank import OverlapReranker
from render_results import main
from render_retrieval import render_dev, render_test

MARKER = "ZEBRAMARK"
GIT = GitState(sha="c" * 40, dirty=False)
CONTEXT = RunContext(git=GIT, versions={"numpy": "0"})
# (query text, notebook, cell index): each query's answer is one cell of the fixtures
TARGETS = [
    ("groupby mean of each key", "01_no_headings.ipynb", 3),
    ("cross_val_score k-fold estimates", "02_deep_headings.ipynb", 5),
    ("train_test_split import", "02_deep_headings.ipynb", 4),
    ("groupby drops rows whose key is NaN", "01_no_headings.ipynb", 5),
]


@pytest.fixture
def notebooks(config: Path) -> list[tuple[str, list[Cell]]]:
    assert CliRunner().invoke(cli.app, ["sync", "--config", str(config)]).exit_code == 0
    structlog.reset_defaults()  # the CLI pointed logging at a stream that is now closed
    conn = connect(load_settings(config).paths.db_path)
    try:
        return list(notebook_cells(conn))
    finally:
        conn.close()


def make_gold(notebooks: list[tuple[str, list[Cell]]], frozen: bool = True) -> GoldSet:
    hashes = {(p, c.index): c.raw_hash for p, cells in notebooks for c in cells}
    queries = []
    for n in range(24):
        text, path, index = TARGETS[n % len(TARGETS)]
        if n % 6 == 5:
            queries.append(
                GoldQuery(id=f"q{n:02d}", text=f"{MARKER} nothing", type=QueryType.UNANSWERABLE)
            )
            continue
        qtype = [QueryType.CONCEPT, QueryType.CODE_IDENTIFIER, QueryType.HOW_DID_I][n % 3]
        ref = CellRef(notebook_path=path, raw_hash=hashes[(path, index)])
        queries.append(
            GoldQuery(id=f"q{n:02d}", text=f"{text} {MARKER}", type=qtype, relevant=(ref,))
        )
    gold = GoldSet(queries=tuple(queries))
    return freeze_gold(gold) if frozen else gold


def make_resources(
    settings: Settings, notebooks: list[tuple[str, list[Cell]]], tok: Tokenizer
) -> Resources:
    services = {
        spec.id: EmbeddingService(HashEmbedder(), spec) for spec in settings.embeddings.candidates
    }
    depths = Depths(
        candidate=settings.retrieval.candidate_depth, fused=settings.retrieval.fused_depth
    )
    return Resources(notebooks, tok, services, OverlapReranker(), depths)


def test_select_queries_returns_answerable_queries_of_the_split(
    notebooks: list[tuple[str, list[Cell]]],
) -> None:
    gold = make_gold(notebooks)
    for split in (Split.DEV, Split.TEST):
        refs = select_queries(gold, split)
        expected = [
            q.id for q in gold.queries if q.split is split and q.type is not QueryType.UNANSWERABLE
        ]
        assert [r.id for r in refs] == expected
        assert all(r.relevant for r in refs)
    assert refs[0].text.endswith(MARKER) or refs[0].text.endswith("ZEBRAMARK")


def test_select_queries_refuses_an_unfrozen_or_empty_split(
    notebooks: list[tuple[str, list[Cell]]],
) -> None:
    with pytest.raises(RunError, match="not frozen"):
        select_queries(make_gold(notebooks, frozen=False), Split.DEV)
    only_unanswerable = freeze_gold(
        GoldSet(queries=(GoldQuery(id="u", text="t", type=QueryType.UNANSWERABLE),))
    )
    with pytest.raises(RunError, match="no answerable queries"):
        select_queries(only_unanswerable, Split.DEV)


def test_build_chunks_covers_every_notebook_for_both_chunkers(
    notebooks: list[tuple[str, list[Cell]]], tiny_tokenizer: Tokenizer
) -> None:
    structural = build_chunks(notebooks, 64, "structural", tiny_tokenizer)
    fixed = build_chunks(notebooks, 64, "fixed", tiny_tokenizer)
    assert {c.notebook_path for c in structural} == {p for p, _ in notebooks}
    assert {c.notebook_path for c in fixed} == {p for p, _ in notebooks}
    assert len(fixed) != len(structural)


def test_duplicate_chunk_ids_are_refused(
    notebooks: list[tuple[str, list[Cell]]], tiny_tokenizer: Tokenizer
) -> None:
    doubled = [*notebooks, *notebooks]
    with pytest.raises(RunError, match="share an id"):
        build_chunks(doubled, 64, "structural", tiny_tokenizer)


def test_run_dev_produces_a_complete_leak_free_result(
    config: Path, notebooks: list[tuple[str, list[Cell]]], tiny_tokenizer: Tokenizer
) -> None:
    settings = load_settings(config)
    gold = make_gold(notebooks)
    results = run_dev(settings, gold, make_resources(settings, notebooks, tiny_tokenizer), CONTEXT)
    dev_ids = {r.id for r in select_queries(gold, Split.DEV)}
    assert results.meta.split is Split.DEV
    assert results.meta.queries == len(dev_ids)
    assert set(results.meta.query_types) == dev_ids
    assert results.meta.git == GIT
    assert results.meta.seed == SEED
    assert results.meta.prompt_versions == {}
    assert results.meta.config["retrieval"]["final_k"] == 2  # type: ignore[index]
    assert [s.stage for s in results.ablation.stages] == [
        "0 chunking", "1 tokenizer", "2 embedding", "3 hybrid", "4 rrf k", "5 rerank",
    ]  # fmt: skip
    for config_result in (*results.ablation.configs, *results.sensitivity):
        assert set(config_result.per_query) == dev_ids
        assert "recall@10" in config_result.means
        assert "recall@tokens=100" in config_result.means  # config budget_tokens in the fixture
    assert results.decided == DecidedConfig(
        chunk_cap=results.ablation.chunk_cap, pipeline=results.ablation.pipeline
    )
    assert [r.key.cap for r in results.sensitivity] == [64, 128]
    assert len(results.truncation) == 2 * 2  # two caps, two candidate models in the fixture config
    assert all(row.truncated == 0 for row in results.truncation)  # the hash embedder never cuts
    dumped = results.model_dump_json()
    assert MARKER not in dumped  # no query text anywhere
    assert "nothing" not in dumped


def test_run_dev_decisions_are_deterministic(
    config: Path, notebooks: list[tuple[str, list[Cell]]], tiny_tokenizer: Tokenizer
) -> None:
    settings = load_settings(config)
    gold = make_gold(notebooks)
    first = run_dev(settings, gold, make_resources(settings, notebooks, tiny_tokenizer), CONTEXT)
    second = run_dev(settings, gold, make_resources(settings, notebooks, tiny_tokenizer), CONTEXT)
    assert [
        (c.candidate, c.ci, c.decision) for s in first.ablation.stages for c in s.comparisons
    ] == [(c.candidate, c.ci, c.decision) for s in second.ablation.stages for c in s.comparisons]
    assert first.decided == second.decided
    assert {c.name: c.per_query for c in first.ablation.configs} == {
        c.name: c.per_query for c in second.ablation.configs
    }


def test_run_test_needs_decided_values_and_a_frozen_gold_set(
    config: Path, notebooks: list[tuple[str, list[Cell]]], tiny_tokenizer: Tokenizer
) -> None:
    settings = load_settings(config)
    resources = make_resources(settings, notebooks, tiny_tokenizer)
    with pytest.raises(RunError, match=r"no \[retrieval.decided\]"):
        run_test(settings, make_gold(notebooks), resources, CONTEXT, "M4")
    decided = DecidedConfig(
        chunk_cap=64, pipeline=PipelineSpec(lexical="code", dense_model="small", rrf_k=60)
    )
    retrieval = settings.retrieval.model_copy(update={"decided": decided})
    with_decided = settings.model_copy(update={"retrieval": retrieval})
    with pytest.raises(RunError, match="not frozen"):
        run_test(with_decided, make_gold(notebooks, frozen=False), resources, CONTEXT, "M4")


def test_run_test_scores_the_decided_pipeline_and_the_baseline_on_test_queries_only(
    config: Path, notebooks: list[tuple[str, list[Cell]]], tiny_tokenizer: Tokenizer
) -> None:
    settings = load_settings(config)
    decided = DecidedConfig(
        chunk_cap=64,
        pipeline=PipelineSpec(lexical="code", dense_model="small", rrf_k=60, rerank=True),
    )
    retrieval = settings.retrieval.model_copy(update={"decided": decided})
    with_decided = settings.model_copy(update={"retrieval": retrieval})
    gold = make_gold(notebooks)
    results = run_test(
        with_decided, gold, make_resources(settings, notebooks, tiny_tokenizer), CONTEXT, "M4"
    )
    test_ids = {r.id for r in select_queries(gold, Split.TEST)}
    dev_ids = {r.id for r in select_queries(gold, Split.DEV)}
    assert results.meta.split is Split.TEST
    assert results.milestone == "M4"
    assert set(results.chosen.per_query) == test_ids
    assert test_ids.isdisjoint(dev_ids)
    assert results.chosen.key.spec == decided.pipeline
    assert results.reference.key.spec == PipelineSpec(lexical="code")
    assert results.reference.key.cap == 64
    assert results.comparison.candidate == results.chosen.name
    assert results.comparison.baseline == results.reference.name
    assert MARKER not in results.model_dump_json()


def test_params_come_from_config(config: Path) -> None:
    settings = load_settings(config)
    params = ablation_params(settings)
    assert params.caps == (64, 128)
    assert params.baseline_cap == 128  # the middle of two sorted caps is index 1
    assert params.budget_metric == "recall@tokens=100"
    assert params.embedding_models == ("small", "base")
    assert params.rrf_challengers == (10, 30)
    scoring = score_params(settings)
    assert scoring.ks == (5, 10)
    assert scoring.budgets == (100, 50, 200)


def test_regression_thresholds_on_the_fixture_corpus(
    config: Path, notebooks: list[tuple[str, list[Cell]]], tiny_tokenizer: Tokenizer
) -> None:
    """Fixed floors (a little under today's values) so a retrieval regression fails CI."""
    settings = load_settings(config)
    results = run_dev(
        settings, make_gold(notebooks), make_resources(settings, notebooks, tiny_tokenizer), CONTEXT
    )
    by_name = {c.name: c.means for c in results.ablation.configs}
    bm25 = by_name["structural-64 | bm25-code"]
    assert bm25["recall@10"] == 1.0
    assert bm25["mrr@10"] >= 0.80  # 0.833 when written
    assert bm25["recall@tokens=100"] >= 0.90  # 1.0 when written
    assert by_name["structural-64 | bm25-code + dense:small + rrf60"]["mrr@10"] >= 0.70  # 0.778
    assert by_name["structural-64 | dense:small"]["mrr@10"] >= 0.55  # 0.644
    # at an equal 100-token budget the structural chunker finds more than the fixed windows
    assert by_name["fixed-128 | bm25-code"]["recall@tokens=100"] < bm25["recall@tokens=100"]
    assert results.decided.chunk_cap == 64
    assert results.decided.pipeline == PipelineSpec(lexical="code")


def test_results_render_to_markdown_and_dispatch_by_file_kind(
    config: Path,
    notebooks: list[tuple[str, list[Cell]]],
    tiny_tokenizer: Tokenizer,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    settings = load_settings(config)
    gold = make_gold(notebooks)
    resources = make_resources(settings, notebooks, tiny_tokenizer)
    dev = run_dev(settings, gold, resources, CONTEXT)
    text = render_dev(dev)
    assert "| Stage | Comparison | Metric | Mean delta | 95% CI | Free | Decision |" in text
    assert "| Configuration | Recall@5 | Recall@10 | MRR@10 | nDCG@10 | Recall@tokens=100 |" in text
    assert (
        "| structural-64 \\| bm25-code | 1.000 | 1.000 |" in text
    )  # a row; the | in the name is escaped
    assert "skipped: the hybrid was not adopted" in text
    assert "| Recall@10 by query type | code_identifier (n=" in text
    assert "Decided: chunk cap 64, lexical code, dense None, RRF k None, rerank False." in text
    assert f"{dev.meta.queries} dev queries" in text
    decided = settings.retrieval.model_copy(update={"decided": dev.decided})
    test = run_test(
        settings.model_copy(update={"retrieval": decided}), gold, resources, CONTEXT, "M4"
    )
    assert "milestone M4" in render_test(test)
    for name, result in (("dev.json", dev), ("test.json", test)):
        (tmp_path / name).write_text(result.model_dump_json(), encoding="utf-8")
    assert main([str(tmp_path / "dev.json")]) == 0
    assert "Decided: chunk cap 64" in capsys.readouterr().out
    assert main([str(tmp_path / "test.json")]) == 0
    assert "test queries, milestone M4" in capsys.readouterr().out


def test_an_unknown_embedding_model_is_a_loud_error(
    config: Path, notebooks: list[tuple[str, list[Cell]]], tiny_tokenizer: Tokenizer
) -> None:
    settings = load_settings(config)
    resources = make_resources(settings, notebooks, tiny_tokenizer)
    with pytest.raises(RunError, match="no embedding model 'missing'"):
        resources.pipeline(
            RunKey(cap=64, chunker="structural", spec=PipelineSpec(dense_model="missing"))
        )


@pytest.fixture
def fake_models(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Replace the real model loaders; returns the ids of the embedding models that were loaded."""
    loaded: list[str] = []

    def embedder(spec: EmbeddingSpec) -> HashEmbedder:
        loaded.append(spec.id)
        return HashEmbedder()

    monkeypatch.setattr(cli, "load_embedder", embedder)
    monkeypatch.setattr(cli, "load_cross_encoder", lambda _spec: OverlapReranker())
    return loaded


def _eval(config: Path, *args: str):  # type: ignore[no-untyped-def]  # noqa: ANN202
    result = CliRunner().invoke(cli.app, ["eval", "--config", str(config), *args])
    structlog.reset_defaults()
    return result


def _prepare(config: Path, notebooks: list[tuple[str, list[Cell]]]) -> Settings:
    settings = load_settings(config)
    save_gold(make_gold(notebooks), settings.paths.gold_path)
    return settings


def test_eval_dev_writes_the_results_file(
    config: Path, notebooks: list[tuple[str, list[Cell]]], fake_models: list[str]
) -> None:
    settings = _prepare(config, notebooks)
    result = _eval(config)
    assert result.exit_code == 0, result.output
    target = settings.paths.results_dir / "retrieval_ablation_dev.json"
    assert result.stdout.strip() == f"eval: wrote {target}"
    parsed = DevResults.model_validate_json(target.read_text(encoding="utf-8"))
    assert parsed.meta.split is Split.DEV
    assert set(parsed.meta.versions) == {
        "numpy", "bm25s", "tiktoken", "torch", "sentence-transformers", "transformers",
    }  # fmt: skip
    assert sorted(fake_models) == ["base", "small"]
    assert MARKER not in target.read_text(encoding="utf-8")


def test_eval_dev_honours_out_and_needs_a_synced_database(
    config: Path, notebooks: list[tuple[str, list[Cell]]], fake_models: list[str], tmp_path: Path
) -> None:
    _prepare(config, notebooks)
    elsewhere = tmp_path / "x" / "dev.json"
    assert _eval(config, "--out", str(elsewhere)).exit_code == 0
    assert elsewhere.is_file()


def test_eval_test_requires_a_milestone(config: Path, fake_models: list[str]) -> None:
    result = _eval(config, "--split", "test")
    assert result.exit_code == 1
    assert "needs --milestone" in result.stderr


def _clean_git(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli, "git_state", lambda: GIT)


def _dirty_git(monkeypatch: pytest.MonkeyPatch) -> None:
    dirty = GitState(sha=GIT.sha, dirty=True)
    monkeypatch.setattr(cli, "git_state", lambda: dirty)


def _with_decided(config: Path, pipeline_toml: str = 'lexical = "code"\n') -> None:
    text = config.read_text(encoding="utf-8")
    config.write_text(
        text
        + f"\n[retrieval.decided]\nchunk_cap = 64\n\n[retrieval.decided.pipeline]\n{pipeline_toml}",
        encoding="utf-8",
    )


def test_eval_test_refuses_a_dirty_tree_and_missing_decisions_and_stale_dev_results(
    config: Path,
    notebooks: list[tuple[str, list[Cell]]],
    fake_models: list[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _prepare(config, notebooks)
    assert _eval(config).exit_code == 0
    _dirty_git(monkeypatch)  # the result must not depend on the checkout the tests run in
    dirty = _eval(config, "--split", "test", "--milestone", "M4")
    assert dirty.exit_code == 1
    assert "uncommitted changes" in dirty.stderr
    _clean_git(monkeypatch)
    undecided = _eval(config, "--split", "test", "--milestone", "M4")
    assert undecided.exit_code == 1
    assert "no [retrieval.decided]" in undecided.stderr
    _with_decided(config, 'lexical = "plain"\n')  # differs from what the dev run decided
    stale = _eval(config, "--split", "test", "--milestone", "M4")
    assert stale.exit_code == 1
    assert "differs from the decisions in the dev results" in stale.stderr


def test_eval_test_without_dev_results_fails_clearly(
    config: Path,
    notebooks: list[tuple[str, list[Cell]]],
    fake_models: list[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _prepare(config, notebooks)
    _clean_git(monkeypatch)
    _with_decided(config)
    result = _eval(config, "--split", "test", "--milestone", "M4")
    assert result.exit_code == 1
    assert "cannot read the dev results" in result.stderr


def test_eval_test_runs_once_records_the_ledger_and_loads_only_needed_models(
    config: Path,
    notebooks: list[tuple[str, list[Cell]]],
    fake_models: list[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _prepare(config, notebooks)
    assert _eval(config).exit_code == 0  # dev: the decided pipeline is BM25 (code) alone
    _clean_git(monkeypatch)
    _with_decided(config)
    fake_models.clear()
    first = _eval(config, "--split", "test", "--milestone", "M4")
    assert first.exit_code == 0, first.output
    assert fake_models == []  # a lexical-only decision loads no embedding model
    target = settings.paths.results_dir / "retrieval_test_M4.json"
    parsed = TestResults.model_validate_json(target.read_text(encoding="utf-8"))
    assert parsed.milestone == "M4"
    ledger = read_ledger(settings.paths.results_dir / "test_ledger.json")
    assert [e.milestone for e in ledger.entries] == ["M4"]
    assert ledger.entries[0].results_sha256 == file_sha256(target)
    assert ledger.entries[0].git_sha == GIT.sha
    before = target.read_bytes()
    second = _eval(config, "--split", "test", "--milestone", "M4")
    assert second.exit_code == 1
    assert "already run for milestone M4" in second.stderr
    assert target.read_bytes() == before  # a refused second run leaves the results untouched


def test_eval_test_loads_the_decided_embedding_model_only(
    config: Path,
    notebooks: list[tuple[str, list[Cell]]],
    fake_models: list[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _prepare(config, notebooks)
    assert _eval(config).exit_code == 0
    dev_path = settings.paths.results_dir / "retrieval_ablation_dev.json"
    dev = DevResults.model_validate_json(dev_path.read_text(encoding="utf-8"))
    forced = dev.model_copy(
        update={"decided": DecidedConfig(chunk_cap=64, pipeline=PipelineSpec(dense_model="base"))}
    )
    dev_path.write_text(forced.model_dump_json(), encoding="utf-8")
    _clean_git(monkeypatch)
    _with_decided(config, 'dense_model = "base"\n')
    fake_models.clear()
    assert _eval(config, "--split", "test", "--milestone", "M4").exit_code == 0
    assert fake_models == ["base"]


def test_eval_dev_without_a_synced_database_fails_clearly(
    config: Path, fake_models: list[str]
) -> None:
    ref = CellRef(notebook_path="n.ipynb", raw_hash="h")
    queries = tuple(
        GoldQuery(id=f"q{n}", text="t", type=QueryType.CONCEPT, relevant=(ref,)) for n in range(6)
    )
    save_gold(freeze_gold(GoldSet(queries=queries)), load_settings(config).paths.gold_path)
    result = _eval(config)
    assert result.exit_code == 1
    assert "the database has no notebooks" in result.stderr
