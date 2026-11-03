"""Enriched indexes, resources with variants, the M5 runners, the steps and the CLI (offline)."""

import json
import re
from pathlib import Path

import pytest

from helpers_retrieval import CONTEXT, MARKER, make_gold, make_resources
from noterecall.chunking.tokens import Tokenizer
from noterecall.config import DecidedConfig, Settings, load_settings
from noterecall.enrich import steps
from noterecall.enrich.contextual import MissingEnrichmentError
from noterecall.enrich.headers import chunk_header
from noterecall.enrich.lookup import CachedBlurbs
from noterecall.enrich.run import run_groups
from noterecall.enrich.steps import (
    EnrichError,
    blurb_setup,
    cleaned_notebooks,
    describe_or_run,
    plan_blurbs,
    plan_clean,
)
from noterecall.eval.ablation import RunKey
from noterecall.eval.gold import Split
from noterecall.eval.run_enrichment import (
    Extras,
    LlmSummary,
    llm_summary,
    reference_variants,
    run_enrichment_dev,
    run_enrichment_test,
)
from noterecall.eval.run_retrieval import (
    Enrichment,
    Resources,
    RunError,
    build_chunks,
    select_queries,
)
from noterecall.index.code_tokenizer import tokenize_plain
from noterecall.index.embeddings import EmbeddingService, EmbeddingSpec, HashEmbedder
from noterecall.index.lexical import LexicalError, LexicalIndex
from noterecall.llm.cache import ResponseCache, Stored, cache_key
from noterecall.llm.client import BudgetExceededError, Prompt, Usage
from noterecall.llm.cost import CostMeter
from noterecall.llm.fake import FakeTransport
from noterecall.llm.transport import CachingClient
from noterecall.models import Cell, Chunk
from noterecall.retrieval.dense import DenseIndex
from noterecall.retrieval.pipeline import PipelineSpec, Variant

SPEC = PipelineSpec(lexical="code", dense_model="small", rrf_k=60)
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


# ------------------------------------------------------------------------------ enriched indexes


def test_a_lexical_index_searches_the_supplied_text_not_the_chunk_text() -> None:
    chunks = [chunk("a", "alpha beta"), chunk("b", "gamma delta")]
    plain = LexicalIndex.build(chunks, tokenize_plain)
    assert plain.search("zebra", 5) == []
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


# ---------------------------------------------------------------------------------- generation


@pytest.fixture(autouse=True)
def _tiny_model_tokenizer(monkeypatch: pytest.MonkeyPatch, tiny_tokenizer: Tokenizer) -> None:
    monkeypatch.setattr(steps, "model_tokenizer", lambda _model: tiny_tokenizer)


def respond(prompt: Prompt) -> str:
    """The fake model: a blurb names the chunk's first word; a clean appends a short suffix."""
    if prompt.name == "contextual_blurb":
        first = prompt.variable.split("\n")[1].split()[0] if "\n" in prompt.variable else "x"
        return json.dumps({"context": f"Notes about {first}."})
    cell = re.search(r"<cell>\n(.*)\n</cell>", prompt.variable, re.DOTALL)
    assert cell is not None
    return json.dumps({"text": cell.group(1) + " (ok)"})


def client_for(settings: Settings, cache: ResponseCache) -> tuple[CachingClient, FakeTransport]:
    transport = FakeTransport(respond)
    meter = CostMeter(settings.llm.prices, settings.llm.max_usd)
    return CachingClient(transport, cache, meter, settings.llm.reasoning_effort), transport


def generate(settings: Settings, corpus: Notebooks, tok: Tokenizer, cache: ResponseCache) -> None:
    """Fill the cache the way `noterecall enrich clean` and `enrich blurbs [--cleaned]` do."""
    settings = with_decided(settings)
    client, _transport = client_for(settings, cache)
    clean = plan_clean(settings, corpus, cache)
    run_groups(clean.groups, client, clean.schema, 2)
    cleaned, _report = cleaned_notebooks(settings, corpus, cache)
    for notebooks in (corpus, cleaned):
        blurbs = plan_blurbs(settings, notebooks, cache, tok)
        run_groups(blurbs.groups, client, blurbs.schema, 2)


def with_decided(settings: Settings, variant: Variant | None = None) -> Settings:
    decided = DecidedConfig(chunk_cap=64, pipeline=SPEC, variant=variant or Variant())
    return settings.model_copy(
        update={"retrieval": settings.retrieval.model_copy(update={"decided": decided})}
    )


@pytest.fixture
def cache(config: Path) -> ResponseCache:
    return ResponseCache(load_settings(config).llm.cache_path)


def enriched_resources(
    settings: Settings, notebooks: Notebooks, tok: Tokenizer, cache: ResponseCache
) -> Resources:
    cleaned, _report = cleaned_notebooks(settings, notebooks, cache)
    blurbs = CachedBlurbs(cache, blurb_setup(settings), dict(notebooks), dict(cleaned))
    enrichment = Enrichment(cleaned=cleaned, blurbs=blurbs)
    base = make_resources(settings, notebooks, tok)
    return Resources(
        notebooks, tok, base.services, base.reranker, base._depths, enrichment=enrichment
    )


def test_generated_blurbs_and_cleaned_cells_are_read_back_from_the_cache(
    config: Path, notebooks: Notebooks, tiny_tokenizer: Tokenizer, cache: ResponseCache
) -> None:
    settings = with_decided(load_settings(config))
    plan = plan_blurbs(settings, notebooks, cache, tiny_tokenizer)
    chunks = build_chunks(notebooks, 64, "structural", tiny_tokenizer)
    assert plan.estimate.jobs_total == len(chunks) == plan.estimate.jobs_to_run
    assert [len(g) for g in plan.groups] == [
        sum(c.notebook_path == p for c in chunks) for p, _ in notebooks
    ]
    generate(settings, notebooks, tiny_tokenizer, cache)
    again = plan_blurbs(settings, notebooks, cache, tiny_tokenizer)
    assert again.estimate.jobs_to_run == 0  # everything is in the cache now
    assert again.estimate.jobs_already_cached == again.estimate.jobs_total
    cleaned, report = cleaned_notebooks(settings, notebooks, cache)
    markdown = sum(
        1
        for _p, cells in notebooks
        for c in cells
        if c.cell_type == "markdown" and c.source.strip()
    )
    assert report.cells == markdown
    assert report.changed > 0
    assert report.rejected_cells > 0  # long cells pass, one-line ones fail the length check
    assert report.changed + report.unchanged + report.rejected_cells == report.cells
    assert set(report.rejected_by_reason) <= {"length_ratio", "missing_token", "code_fence_changed"}
    original = {(p, c.index): c for p, cells in notebooks for c in cells}
    for path, cells in cleaned:
        for c in cells:
            assert c.raw_hash == original[(path, c.index)].raw_hash  # gold labels stay valid


def test_a_chunk_set_without_generated_blurbs_is_a_loud_error(
    config: Path, notebooks: Notebooks, tiny_tokenizer: Tokenizer, cache: ResponseCache
) -> None:
    settings = load_settings(config)
    generate(settings, notebooks, tiny_tokenizer, cache)
    source = CachedBlurbs(cache, blurb_setup(settings), dict(notebooks), None)
    other_cap = build_chunks(notebooks, 128, "structural", tiny_tokenizer)
    with pytest.raises(MissingEnrichmentError, match="blurbs are not in the cache"):
        source(False, other_cap)
    with pytest.raises(MissingEnrichmentError, match="cleaned notebooks were not supplied"):
        source(True, other_cap)


# ----------------------------------------------------------------------------------- resources


def test_resources_build_the_indexed_text_of_each_variant(
    config: Path, notebooks: Notebooks, tiny_tokenizer: Tokenizer, cache: ResponseCache
) -> None:
    settings = load_settings(config)
    generate(settings, notebooks, tiny_tokenizer, cache)
    resources = enriched_resources(settings, notebooks, tiny_tokenizer, cache)

    def key(variant: Variant) -> RunKey:
        return RunKey(cap=64, chunker="structural", spec=SPEC, variant=variant)

    assert resources.indexed_texts(key(Variant())) is None
    header = resources.indexed_texts(key(Variant(header=True)))
    chunks = resources.chunks(64, "structural")
    assert header is not None
    assert all(header[c.id] == f"{chunk_header(c)}\n\n{c.text}" for c in chunks)
    full = resources.indexed_texts(key(Variant(header=True, blurb=True)))
    assert full is not None
    for c in chunks:
        head, blurb, body = full[c.id].split("\n\n", 2)
        assert head == chunk_header(c)
        assert blurb.startswith("Notes about")
        assert body == c.text
    cleaned_chunks = resources.chunks(64, "structural", True)
    assert [c.text for c in cleaned_chunks] != [c.text for c in chunks]  # the clean changed text


def test_a_variant_without_its_inputs_is_a_loud_error(
    config: Path, notebooks: Notebooks, tiny_tokenizer: Tokenizer
) -> None:
    resources = make_resources(load_settings(config), notebooks, tiny_tokenizer)

    def key(variant: Variant) -> RunKey:
        return RunKey(cap=64, chunker="structural", spec=SPEC, variant=variant)

    with pytest.raises(RunError, match="no blurb source"):
        resources.indexed_texts(key(Variant(blurb=True)))
    with pytest.raises(RunError, match="cleaned notebooks were not supplied"):
        resources.chunks(64, "structural", True)
    with pytest.raises(RunError, match="cleaned notebooks were not supplied"):
        resources.coverage(True)


# ------------------------------------------------------------------------------------- runners


def test_the_dev_runner_records_stages_usage_and_never_query_text(
    config: Path, notebooks: Notebooks, tiny_tokenizer: Tokenizer, cache: ResponseCache
) -> None:
    settings = with_decided(load_settings(config))
    generate(settings, notebooks, tiny_tokenizer, cache)
    resources = enriched_resources(settings, notebooks, tiny_tokenizer, cache)
    _cleaned, report = cleaned_notebooks(settings, notebooks, cache)
    extras = Extras(
        llm=llm_summary(cache, settings.llm.prices, settings.llm.max_usd),
        clean=report,
        prompt_versions={"contextual_blurb": "v1"},
    )
    gold = make_gold(notebooks)
    results = run_enrichment_dev(settings, gold, resources, CONTEXT, extras)
    dev_ids = {q.id for q in select_queries(gold, Split.DEV)}
    assert [s.stage for s in results.ablation.stages] == ["1 header", "2 blurb", "3 clean"]
    assert results.decided.variant == results.ablation.variant
    assert (results.decided.chunk_cap, results.decided.pipeline) == (64, SPEC)
    assert results.meta.prompt_versions == {"contextual_blurb": "v1"}
    assert {c.key.variant for c in results.ablation.configs} >= {
        Variant(),
        Variant(header=True),
        Variant(header=True, blurb=True),
    }
    for config_result in results.ablation.configs:
        assert set(config_result.per_query) == dev_ids
    assert len(results.truncation) == len(results.ablation.configs)
    assert all(t.truncated == 0 for t in results.truncation)  # the hash embedder never cuts
    assert {r.prompt for r in results.extras.llm.rows} == {"contextual_blurb", "clean_markdown"}
    assert results.extras.llm.total_cost_usd > 0
    assert MARKER not in results.model_dump_json()
    again = run_enrichment_dev(settings, gold, resources, CONTEXT, extras)
    assert again.decided == results.decided
    assert [c.ci for s in again.ablation.stages for c in s.comparisons] == [
        c.ci for s in results.ablation.stages for c in s.comparisons
    ]


def test_the_runner_needs_the_m4_decisions(
    config: Path, notebooks: Notebooks, tiny_tokenizer: Tokenizer
) -> None:
    settings = load_settings(config)
    resources = make_resources(settings, notebooks, tiny_tokenizer)
    extras = Extras(
        llm=LlmSummary(rows=(), total_cost_usd=0.0, max_usd=1.0), clean=None, prompt_versions={}
    )
    with pytest.raises(RunError, match=r"no \[retrieval.decided\]"):
        run_enrichment_dev(settings, make_gold(notebooks), resources, CONTEXT, extras)
    with pytest.raises(RunError, match=r"no \[retrieval.decided\]"):
        run_enrichment_test(settings, make_gold(notebooks), resources, CONTEXT)


def test_the_test_runner_shows_the_decided_variant_next_to_the_m4_pipeline(
    config: Path, notebooks: Notebooks, tiny_tokenizer: Tokenizer, cache: ResponseCache
) -> None:
    base = load_settings(config)
    generate(base, notebooks, tiny_tokenizer, cache)
    resources = enriched_resources(base, notebooks, tiny_tokenizer, cache)
    settings = with_decided(base, Variant(header=True, blurb=True))
    gold = make_gold(notebooks)
    results = run_enrichment_test(settings, gold, resources, CONTEXT)
    test_ids = {q.id for q in select_queries(gold, Split.TEST)}
    assert results.milestone == "M5"
    assert results.meta.split is Split.TEST
    assert set(results.chosen.per_query) == test_ids
    assert results.chosen.key.variant == Variant(header=True, blurb=True)
    assert [r.key.variant for r in results.references] == [Variant(), Variant(header=True)]
    assert all(c.report_only and c.candidate == results.chosen.name for c in results.comparisons)
    assert MARKER not in results.model_dump_json()


def test_reference_variants() -> None:
    assert reference_variants(Variant()) == [Variant()]
    assert reference_variants(Variant(header=True)) == [Variant()]
    assert reference_variants(Variant(header=True, blurb=True)) == [Variant(), Variant(header=True)]
    assert reference_variants(Variant(header=True, clean=True)) == [Variant(), Variant(header=True)]


def test_llm_summary_is_hand_computed_with_reasoning_and_cache_shown_separately(
    config: Path,
) -> None:
    settings = load_settings(config)
    cache = ResponseCache(settings.llm.cache_path)
    prompt = Prompt(
        name="contextual_blurb", version="v", model="test-model", instructions="i",
        shared="s", variable="v", max_output_tokens=10,
    )  # fmt: skip
    usage = Usage(input_tokens=1000, cached_tokens=400, output_tokens=200, reasoning_tokens=50)
    cache.put("k1", prompt, Stored("{}", usage, 0.1))
    cache.put("k2", prompt, Stored("{}", usage, 0.1))
    summary = llm_summary(cache, settings.llm.prices, 1.0)
    (row,) = summary.rows
    # per call: 600 fresh * 1.0 + 400 cached * 0.1 + 200 output * 4.0 = 1440 per million
    assert (row.calls, row.input_tokens, row.cached_tokens, row.reasoning_tokens) == (
        2,
        2000,
        800,
        100,
    )
    assert row.cost_usd == pytest.approx(2 * 0.00144)
    assert summary.total_cost_usd == pytest.approx(0.00288)


# ----------------------------------------------------------------------------- steps and the cap


def refuse_the_api(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(*_args: object) -> object:
        raise AssertionError("the API transport must not be created")

    monkeypatch.setattr(steps, "create_transport", boom)


def test_a_dry_run_prints_the_estimate_and_never_creates_a_transport(
    config: Path, notebooks: Notebooks, tiny_tokenizer: Tokenizer, cache: ResponseCache,
    monkeypatch: pytest.MonkeyPatch,
) -> None:  # fmt: skip
    refuse_the_api(monkeypatch)
    settings = with_decided(load_settings(config))
    plan = plan_blurbs(settings, notebooks, cache, tiny_tokenizer)
    text = describe_or_run(plan, settings, cache, dry_run=True, confirmed=True)
    assert f"{plan.estimate.jobs_to_run} to run" in text
    assert "estimated cost, prefix caching works" in text
    assert "assumption: 80 output tokens per call" in text
    assert "of the $1.00 cap" in text


def test_a_real_run_is_refused_without_confirmation(
    config: Path, notebooks: Notebooks, tiny_tokenizer: Tokenizer, cache: ResponseCache,
    monkeypatch: pytest.MonkeyPatch,
) -> None:  # fmt: skip
    refuse_the_api(monkeypatch)
    settings = with_decided(load_settings(config))
    plan = plan_blurbs(settings, notebooks, cache, tiny_tokenizer)
    with pytest.raises(EnrichError, match="re-run with --yes"):
        describe_or_run(plan, settings, cache, dry_run=False, confirmed=False)


def test_a_confirmed_run_calls_the_transport_and_reports_the_spend(
    config: Path, notebooks: Notebooks, tiny_tokenizer: Tokenizer, cache: ResponseCache,
    monkeypatch: pytest.MonkeyPatch,
) -> None:  # fmt: skip
    transport = FakeTransport(respond)
    monkeypatch.setattr(steps, "create_transport", lambda *_a: transport)
    settings = with_decided(load_settings(config))
    plan = plan_blurbs(settings, notebooks, cache, tiny_tokenizer)
    text = describe_or_run(plan, settings, cache, dry_run=False, confirmed=True)
    jobs = [job for group in plan.groups for job in group]
    distinct = len({cache_key(j.prompt, settings.llm.reasoning_effort) for j in jobs})
    assert distinct < len(jobs)  # the fixtures repeat a chunk inside one notebook: asked once
    assert len(transport.prompts) == distinct
    assert (
        f"ran {len(jobs)} jobs: {distinct} answered by the API, {len(jobs) - distinct} from" in text
    )
    assert "unusable answers: none" in text
    again = plan_blurbs(settings, notebooks, cache, tiny_tokenizer)
    second = describe_or_run(again, settings, cache, dry_run=False, confirmed=True)
    assert f"ran {len(jobs)} jobs: 0 answered by the API, {len(jobs)} from the cache" in second
    assert len(transport.prompts) == distinct  # nothing was paid for twice


def test_the_money_cap_stops_a_real_run(
    config: Path, notebooks: Notebooks, tiny_tokenizer: Tokenizer, cache: ResponseCache,
    monkeypatch: pytest.MonkeyPatch,
) -> None:  # fmt: skip
    transport = FakeTransport(respond)
    monkeypatch.setattr(steps, "create_transport", lambda *_a: transport)
    base = with_decided(load_settings(config))
    tiny_cap = base.llm.model_copy(update={"max_usd": 1e-9})
    settings = base.model_copy(update={"llm": tiny_cap})
    plan = plan_blurbs(settings, notebooks, cache, tiny_tokenizer)
    with pytest.raises(BudgetExceededError):
        describe_or_run(plan, settings, cache, dry_run=False, confirmed=True)
    assert len(transport.prompts) == 1  # the first call alone crossed the cap, so none followed


def test_the_clean_estimate_assumes_the_output_is_as_long_as_the_input(
    config: Path, notebooks: Notebooks, cache: ResponseCache
) -> None:
    settings = load_settings(config)
    plan = plan_clean(settings, notebooks, cache)
    markdown = sum(
        1
        for _p, cells in notebooks
        for c in cells
        if c.cell_type == "markdown" and c.source.strip()
    )
    assert plan.estimate.jobs_total == markdown
    assert "the average cell" in plan.assumption
    assert plan.estimate.reusable_prefix_tokens == 0  # a clean prompt has no shared notebook part
