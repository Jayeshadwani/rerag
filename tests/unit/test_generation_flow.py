"""Signals, calibration, asking, the generation run, the estimate and the smoke test (offline)."""

import json
import re
from pathlib import Path

import pytest

from helpers_retrieval import CONTEXT, MARKER, decided_settings, make_gold, make_resources
from make_fixtures import FAKE_ANTHROPIC_KEY, FAKE_AWS_ACCESS_KEY
from noterecall.chunking.tokens import Tokenizer
from noterecall.config import Settings, load_settings
from noterecall.eval.ablation import RunKey
from noterecall.eval.gold import GoldSet, QueryType, Split, freeze_gold
from noterecall.eval.run_generation import (
    decided_pipeline,
    gold_subset,
    run_calibration,
    run_generation,
    select_all,
    set_subset,
)
from noterecall.eval.run_retrieval import QueryRef, RunError, select_queries
from noterecall.generate.abstain import Thresholds, refusal_metrics
from noterecall.generate.answer import CitedClaim, Generator, Outcome, Response
from noterecall.generate.ask import Asker, AskResult, Models
from noterecall.generate.citations import Answer
from noterecall.generate.commands import (
    GenerationPlan,
    format_answer,
    format_context,
    format_plan,
    plan_generation,
    plan_stage3,
)
from noterecall.generate.estimate import estimate_generation
from noterecall.generate.runtime import CacheOnly, NoCalls
from noterecall.generate.smoke import (
    choose_step_b,
    format_rows,
    run_step_a,
    run_step_c,
    smoke_row,
    synthetic_chunks,
)
from noterecall.generate.verify import Verifier
from noterecall.llm.cache import ResponseCache, Stored, cache_key
from noterecall.llm.client import LLMCallError, LLMOutputError, Prompt, Usage
from noterecall.llm.cost import CostMeter
from noterecall.llm.fake import FakeTransport, Responder
from noterecall.llm.prompts import load_prompt
from noterecall.llm.transport import CachingClient
from noterecall.models import Cell, Chunk, RetrievalHit
from noterecall.retrieval.pipeline import AbstentionSignals, PipelineError, PipelineSpec
from render_results import main as render_main

Notebooks = list[tuple[str, list[Cell]]]
NEVER_REFUSE = (-1.0, -1.0)  # stage 1 refuses when both signals are below these: never
ALWAYS_REFUSE = (2.0, 1e9)  # ... and when both are below these: always


def first_note_id(prompt: Prompt) -> str:
    match = re.search(r"\[(C\d+)\]", prompt.variable)
    assert match is not None
    return match.group(1)


def answer_with_first_note(prompt: Prompt) -> str:
    claim = {"text": "an answer", "citations": [first_note_id(prompt)]}
    return json.dumps({"insufficient": False, "claims": [claim]})


def make_generator(
    settings: Settings, tmp_path: Path, responder: Responder = answer_with_first_note
) -> tuple[Generator, FakeTransport]:
    transport = FakeTransport(responder)
    meter = CostMeter(settings.llm.prices, settings.llm.max_usd)
    cache = ResponseCache(tmp_path / "gen.sqlite")
    client = CachingClient(transport, cache, meter, settings.llm.reasoning_effort)
    return Generator(client, load_prompt("answer"), settings.llm.model, 100), transport


# ------------------------------------------------------------------------------------ signals


def test_the_signals_are_the_top_cosine_and_the_top_bm25_per_query_token(
    config: Path, notebooks: Notebooks, tiny_tokenizer: Tokenizer
) -> None:
    settings = decided_settings(load_settings(config))
    resources = make_resources(settings, notebooks, tiny_tokenizer)
    pipeline, cap = decided_pipeline(settings, resources)
    hits, signals = pipeline.retrieve_with_signals("groupby mean of each key")
    assert hits
    assert signals.top_cosine > 0
    assert signals.bm25_norm > 0
    _hits, none = pipeline.retrieve_with_signals("zzzqqq xxxwww")
    assert none.bm25_norm == 0.0  # no chunk matches any query token
    # the raw top score divided by the number of query tokens, computed independently
    index = pipeline._lexical
    assert index is not None
    query = "groupby mean of each key"
    raw = index.search(query, 1)[0].score
    assert signals.bm25_norm == pytest.approx(raw / index.query_tokens(query))
    assert cap == 64


def test_signals_need_both_retrievers(
    config: Path, notebooks: Notebooks, tiny_tokenizer: Tokenizer
) -> None:
    settings = load_settings(config)
    resources = make_resources(settings, notebooks, tiny_tokenizer)
    only_bm25 = resources.pipeline(
        RunKey(cap=64, chunker="structural", spec=PipelineSpec(lexical="code"))
    )
    with pytest.raises(PipelineError, match="need both"):
        only_bm25.retrieve_with_signals("anything")


# ---------------------------------------------------------------------------------- calibration


def test_the_calibration_reads_only_dev_queries_and_applies_the_preregistered_rule(
    config: Path, notebooks: Notebooks, tiny_tokenizer: Tokenizer
) -> None:
    settings = decided_settings(load_settings(config))
    gold = make_gold(notebooks)
    resources = make_resources(settings, notebooks, tiny_tokenizer)
    results = run_calibration(settings, gold, resources, CONTEXT)
    dev = {q.id for q in gold.queries if q.split is Split.DEV}
    test = {q.id for q in gold.queries if q.split is Split.TEST}
    assert {p.query_id for p in results.points} == dev  # unanswerable ones included
    assert not {p.query_id for p in results.points} & test
    assert any(p.unanswerable for p in results.points)
    cal = results.calibration
    assert cal.max_false_refusal_pct == 5
    answerable = sum(1 for p in results.points if not p.unanswerable)
    for choice in (cal.cosine_only, cal.cosine_bm25):
        assert choice.metrics.false_refusals * 100 <= 5 * answerable  # the bound holds
    assert cal.cosine_bm25.metrics.caught >= cal.cosine_only.metrics.caught
    assert MARKER not in results.model_dump_json()  # ids and numbers only


def test_select_all_keeps_unanswerable_queries_and_refuses_an_unfrozen_gold(
    notebooks: Notebooks,
) -> None:
    gold = make_gold(notebooks)
    refs = select_all(gold, Split.DEV)
    assert any(r.type is QueryType.UNANSWERABLE for r in refs)
    assert len(refs) > len(select_queries(gold, Split.DEV))
    with pytest.raises(RunError, match="not frozen"):
        select_all(make_gold(notebooks, frozen=False), Split.DEV)


def test_the_calibration_needs_the_decided_pipeline(
    config: Path, notebooks: Notebooks, tiny_tokenizer: Tokenizer
) -> None:
    settings = load_settings(config)
    resources = make_resources(settings, notebooks, tiny_tokenizer)
    with pytest.raises(RunError, match=r"no \[retrieval.decided\]"):
        decided_pipeline(settings, resources)


# ----------------------------------------------------------------------------------- the asker


def make_asker(
    settings: Settings,
    resources_args: tuple[Notebooks, Tokenizer],
    tmp_path: Path,
    thresholds: tuple[float, float],
    responder: Responder = answer_with_first_note,
) -> tuple[Asker, FakeTransport, dict[str, Chunk]]:
    resources = make_resources(settings, *resources_args)
    pipeline, cap = decided_pipeline(settings, resources)
    chunks = {c.id: c for c in resources.chunks(cap, "structural")}
    generator, transport = make_generator(settings, tmp_path, responder)
    asker = Asker(pipeline, chunks, Thresholds(cosine=thresholds[0], bm25=thresholds[1]),
                  Models(generator), settings.generation.max_chunks)  # fmt: skip
    return asker, transport, chunks


def test_a_stage_one_refusal_makes_no_model_call(
    config: Path, notebooks: Notebooks, tiny_tokenizer: Tokenizer, tmp_path: Path
) -> None:
    settings = decided_settings(load_settings(config), ALWAYS_REFUSE)
    asker, transport, chunks = make_asker(
        settings, (notebooks, tiny_tokenizer), tmp_path, ALWAYS_REFUSE
    )
    result = asker.ask("groupby mean of each key")
    assert result.response.outcome is Outcome.REFUSED_STAGE1
    assert transport.prompts == []  # nothing was sent anywhere
    assert "Not in your notes" in format_answer(result, chunks)
    assert "No model call was made" in format_answer(result, chunks)


def test_an_answered_question_cites_chunks_that_were_in_the_context(
    config: Path, notebooks: Notebooks, tiny_tokenizer: Tokenizer, tmp_path: Path
) -> None:
    settings = decided_settings(load_settings(config), NEVER_REFUSE)
    asker, transport, chunks = make_asker(
        settings, (notebooks, tiny_tokenizer), tmp_path, NEVER_REFUSE
    )
    result = asker.ask("groupby mean of each key")
    assert result.response.outcome is Outcome.ANSWERED
    assert len(result.context_ids) <= settings.generation.max_chunks
    cited = {cid for claim in result.response.claims for cid in claim.chunk_ids}
    assert cited <= set(result.context_ids)
    assert len(transport.prompts) == 1
    text = format_answer(result, chunks)
    assert "Sources:" in text
    assert "[1]" in text
    assert "top cosine" in format_context(result, chunks)


def test_the_question_never_reaches_the_model_with_a_secret_in_it(
    config: Path, notebooks: Notebooks, tiny_tokenizer: Tokenizer, tmp_path: Path
) -> None:
    settings = decided_settings(load_settings(config), NEVER_REFUSE)
    asker, transport, _ = make_asker(settings, (notebooks, tiny_tokenizer), tmp_path, NEVER_REFUSE)
    asker.ask(f"groupby {FAKE_AWS_ACCESS_KEY} {FAKE_ANTHROPIC_KEY}")
    for p in transport.prompts:
        sent = p.instructions + p.shared + p.variable
        assert FAKE_AWS_ACCESS_KEY not in sent
        assert FAKE_ANTHROPIC_KEY not in sent


def test_no_note_in_the_context_contains_a_secret_either(
    config: Path, notebooks: Notebooks, tiny_tokenizer: Tokenizer, tmp_path: Path
) -> None:
    settings = decided_settings(load_settings(config), NEVER_REFUSE)
    asker, transport, _ = make_asker(settings, (notebooks, tiny_tokenizer), tmp_path, NEVER_REFUSE)
    asker.ask("aws access key id secret")  # the secret-bearing fixture notebook matches this
    sent = "".join(p.variable for p in transport.prompts)
    assert "[REDACTED" in sent  # the notes were scrubbed at ingest, and it shows
    for secret in (FAKE_AWS_ACCESS_KEY, FAKE_ANTHROPIC_KEY):
        assert secret not in sent


# ------------------------------------------------------------------------------- the generation run


def test_the_generation_run_reports_three_rows_and_validity_without_any_text(
    config: Path, notebooks: Notebooks, tiny_tokenizer: Tokenizer, tmp_path: Path
) -> None:
    settings = decided_settings(load_settings(config), NEVER_REFUSE)
    resources = make_resources(settings, notebooks, tiny_tokenizer)
    generator, transport = make_generator(settings, tmp_path)
    gold = make_gold(notebooks)
    results = run_generation(
        settings, resources, CONTEXT, Models(generator), gold_subset(gold, Split.TEST)
    )
    queries = select_all(gold, Split.TEST)
    assert [o.query_id for o in results.queries] == [q.id for q in queries]
    assert [r.name for r in results.rows] == [
        "cosine only", "cosine + BM25 (stage 1)", "stages 1+2 (current pipeline)"
    ]  # fmt: skip
    assert all(r.metrics.false_refusals == 0 for r in results.rows)  # nothing refused at all
    v = results.validity
    assert v.queries_to_model == len(queries)
    # the fixture gold repeats a question, and an identical prompt is answered from the cache:
    # one API call fewer than queries, never more
    assert len(transport.prompts) == len({p.variable for p in transport.prompts}) < len(queries)
    assert (v.first_attempt_valid, v.retried, v.refused_invalid) == (len(queries), 0, 0)
    assert v.answered == len(queries)
    assert v.uncited_claims_first_attempt == 0
    assert v.answered_answerable == sum(1 for q in queries if q.type is not QueryType.UNANSWERABLE)
    assert results.total_cost_usd > 0
    assert results.total_usage.input_tokens > 0
    assert results.meta.prompt_versions == {"answer": generator.prompt_version}
    assert MARKER not in results.model_dump_json()


def test_with_everything_refused_the_run_calls_no_model_and_counts_the_refusals(
    config: Path, notebooks: Notebooks, tiny_tokenizer: Tokenizer, tmp_path: Path
) -> None:
    settings = decided_settings(load_settings(config), ALWAYS_REFUSE)
    resources = make_resources(settings, notebooks, tiny_tokenizer)
    generator, transport = make_generator(settings, tmp_path)
    gold = make_gold(notebooks)
    results = run_generation(
        settings, resources, CONTEXT, Models(generator), gold_subset(gold, Split.DEV)
    )
    assert transport.prompts == []
    stage1 = results.rows[1].metrics
    assert stage1.false_refusals == stage1.answerable  # every answerable query was refused
    assert stage1.caught == stage1.unanswerable
    assert results.rows[2].metrics == stage1  # no model stage, so the same
    assert results.total_cost_usd == 0
    assert results.validity.queries_to_model == 0


def test_the_model_flag_and_the_validator_are_counted_in_the_two_stage_row(
    config: Path, notebooks: Notebooks, tiny_tokenizer: Tokenizer, tmp_path: Path
) -> None:
    settings = decided_settings(load_settings(config), NEVER_REFUSE)
    resources = make_resources(settings, notebooks, tiny_tokenizer)
    gold = make_gold(notebooks)

    def responder(p: Prompt) -> str:
        if MARKER in p.variable and "nothing" in p.variable:  # the unanswerable queries
            return json.dumps({"insufficient": True, "claims": []})
        return answer_with_first_note(p)

    generator, _ = make_generator(settings, tmp_path, responder)
    results = run_generation(
        settings, resources, CONTEXT, Models(generator), gold_subset(gold, Split.DEV)
    )
    two = results.rows[2].metrics
    assert two.caught == two.unanswerable > 0  # the model caught them
    assert two.false_refusals == 0
    assert results.rows[1].metrics.caught == 0  # stage 1 alone refused nothing
    assert results.validity.refused_by_model == two.unanswerable


def test_the_run_needs_the_calibrated_thresholds_in_config(
    config: Path, notebooks: Notebooks, tiny_tokenizer: Tokenizer, tmp_path: Path
) -> None:
    settings = decided_settings(load_settings(config))  # no [generation.decided]
    resources = make_resources(settings, notebooks, tiny_tokenizer)
    generator, _ = make_generator(settings, tmp_path)
    with pytest.raises(RunError, match=r"no \[generation.decided\]"):
        run_generation(
            settings,
            resources,
            CONTEXT,
            Models(generator),
            gold_subset(make_gold(notebooks), Split.DEV),
        )


# ------------------------------------------------------------------ estimate and the smoke test


def test_the_estimate_is_hand_computed(tmp_path: Path) -> None:
    settings = load_settings(Path("config/default.toml"))
    price = settings.llm.prices[settings.llm.model]
    prompts = [
        Prompt(name="p", version="v", model="m", instructions="a b", shared="",
               variable="c d e", max_output_tokens=1000)
        for _ in range(3)
    ]  # fmt: skip
    est = estimate_generation(prompts, lambda t: len(t.split()), price, assumed_output_tokens=100)
    # each prompt: "a b" + "\n" + "" + "\n" + "c d e" = 5 words, so 15 input tokens in all
    # expected: (15 * 0.25 + 3 * 100 * 2.0) / 1e6 ; bound: 2 * (15 * 0.25 + 3 * 1000 * 2.0) / 1e6
    assert est.input_tokens == 15
    assert est.calls == 3
    assert est.cost_expected == pytest.approx((15 * 0.25 + 600) / 1e6)
    assert est.cost_upper_bound == pytest.approx(2 * (15 * 0.25 + 6000) / 1e6)
    assert estimate_generation([], len, price, 100).cost_upper_bound == 0.0


def test_plan_generation_counts_stage_one_refusals_cached_answers_and_prices_the_rest(
    config: Path, notebooks: Notebooks, tiny_tokenizer: Tokenizer, tmp_path: Path
) -> None:
    settings = decided_settings(load_settings(config), NEVER_REFUSE)
    asker, transport, _ = make_asker(settings, (notebooks, tiny_tokenizer), tmp_path, NEVER_REFUSE)
    queries = select_all(make_gold(notebooks), Split.DEV)
    plan = plan_generation(asker, queries, settings, tiny_tokenizer.count, lambda _p: False)
    assert plan.refused_stage1 == 0
    assert plan.answer.calls == len(queries)
    assert plan.answer_cached == 0
    assert transport.prompts == []
    text = format_plan(plan, 0.0, 2.0)
    assert f"{len(queries)} queries, 0 refused by stage 1" in text
    assert "of the $2.00 cap" in text
    all_cached = plan_generation(asker, queries, settings, len, lambda _p: True)
    assert (all_cached.answer.calls, all_cached.answer_cached) == (0, len(queries))
    always, _t, _c = make_asker(settings, (notebooks, tiny_tokenizer), tmp_path, ALWAYS_REFUSE)
    refused = plan_generation(always, queries, settings, len, lambda _p: False)
    assert refused.refused_stage1 == len(queries)
    assert refused.answer.calls == 0


def test_step_a_uses_only_invented_chunks_and_reports_usage_and_cost(tmp_path: Path) -> None:
    settings = load_settings(Path("config/default.toml"))
    generator, transport = make_generator(settings, tmp_path)
    price = settings.llm.prices[settings.llm.model]
    row = run_step_a(generator, price)
    assert row.outcome == "answered"
    assert row.attempts == 1
    assert all(c.id.startswith("synthetic-") for c in synthetic_chunks())
    sent = transport.prompts[0].variable
    assert "demo/pandas.ipynb" in sent
    assert "ZEBRA" not in sent
    table = format_rows([row])
    assert "step A (invented chunks)" in table
    assert "total cost: $" in table
    assert row.cost_usd > 0


def test_step_b_picks_two_answerable_then_an_unanswerable_that_stage_one_lets_through() -> None:
    def q(i: str, kind: QueryType) -> QueryRef:
        return QueryRef(i, "text", kind, frozenset())

    a, c, d = QueryType.CONCEPT, QueryType.CODE_IDENTIFIER, QueryType.UNANSWERABLE
    items = [(q("q5", c), False), (q("q1", a), False), (q("q3", d), True), (q("q2", a), True),
             (q("q4", d), False), (q("q6", a), False)]  # fmt: skip
    assert [x.id for x in choose_step_b(items)] == ["q1", "q2", "q4"]  # q3 is refused by stage 1
    no_open = [(q("q1", a), False), (q("q2", a), False), (q("q3", d), True), (q("q4", a), False)]
    assert [x.id for x in choose_step_b(no_open)] == ["q1", "q2", "q4"]  # falls back to answerable


def test_smoke_row_prices_the_usage() -> None:
    settings = load_settings(Path("config/default.toml"))
    usage = Usage(input_tokens=1000, cached_tokens=400, output_tokens=200, reasoning_tokens=50)
    row = smoke_row("x", Response(outcome=Outcome.ANSWERED, attempts=1, usage=usage),
                    settings.llm.prices["gpt-5-mini"])  # fmt: skip
    # 600 fresh * 0.25 + 400 cached * 0.025 + 200 output * 2.0 = 150 + 10 + 400 = 560 per million
    assert row.cost_usd == pytest.approx(0.00056)


def test_a_split_without_unanswerable_queries_cannot_be_calibrated_or_reported(
    notebooks: Notebooks,
) -> None:
    gold = make_gold(notebooks)
    answerable_only = freeze_gold(
        GoldSet(queries=tuple(q for q in gold.queries if q.type is not QueryType.UNANSWERABLE))
    )
    with pytest.raises(RunError, match="needs both answerable and unanswerable"):
        select_all(answerable_only, Split.DEV)


def test_a_dense_index_with_no_hits_is_an_error_not_a_made_up_signal(
    config: Path, notebooks: Notebooks, tiny_tokenizer: Tokenizer, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = decided_settings(load_settings(config))
    pipeline, _cap = decided_pipeline(settings, make_resources(settings, notebooks, tiny_tokenizer))
    assert pipeline._dense is not None
    monkeypatch.setattr(pipeline._dense, "search", lambda _query, _k: [])
    with pytest.raises(PipelineError, match="returned no hits"):
        pipeline.retrieve_with_signals("anything")


def test_two_claims_citing_the_same_chunk_share_one_source_number() -> None:
    chunks = {"a": synthetic_chunks()[0], "b": synthetic_chunks()[1]}
    response = Response(
        outcome=Outcome.ANSWERED,
        attempts=1,
        claims=(
            CitedClaim(text="one", chunk_ids=("a", "b")),
            CitedClaim(text="two", chunk_ids=("a",)),
        ),
    )
    result = AskResult(
        response=response,
        drafted=response,
        signals=AbstentionSignals(top_cosine=0.5, bm25_norm=0.1),
        context_ids=("a", "b"),
    )
    text = format_answer(result, chunks)
    assert "- one [1][2]" in text
    assert "- two [1]" in text
    assert text.count("demo/") == 2  # two sources listed once each
    assert "Sources:" in text


def test_the_dry_run_client_refuses_every_call() -> None:
    prompt = Prompt(name="p", version="v", model="m", instructions="i", shared="", variable="v",
                    max_output_tokens=5)  # fmt: skip
    with pytest.raises(LLMCallError, match="never calls the API"):
        NoCalls().complete_json(prompt, Answer)


def test_the_signals_take_the_best_dense_hit_and_the_best_bm25_hit_over_the_query_tokens(
    config: Path, notebooks: Notebooks, tiny_tokenizer: Tokenizer, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = decided_settings(load_settings(config))
    pipeline, _cap = decided_pipeline(settings, make_resources(settings, notebooks, tiny_tokenizer))
    assert pipeline._dense is not None
    assert pipeline._lexical is not None
    dense = [
        RetrievalHit(chunk_id="a", score=0.9, rank=1),
        RetrievalHit(chunk_id="b", score=0.2, rank=2),
    ]
    lexical = [
        RetrievalHit(chunk_id="b", score=6.0, rank=1),
        RetrievalHit(chunk_id="a", score=2.0, rank=2),
    ]
    monkeypatch.setattr(pipeline._dense, "search", lambda _q, _k: dense)
    monkeypatch.setattr(pipeline._lexical, "search", lambda _q, _k: lexical)
    _hits, signals = pipeline.retrieve_with_signals("alpha beta gamma")
    # three query tokens: 6.0 / 3 = 2.0 (the best BM25 hit, not the last one); cosine 0.9 (the best)
    assert pipeline._lexical.query_tokens("alpha beta gamma") == 3
    assert signals == AbstentionSignals(top_cosine=0.9, bm25_norm=2.0)
    monkeypatch.setattr(pipeline._lexical, "search", lambda _q, _k: [])
    assert pipeline.retrieve_with_signals("alpha beta gamma")[1].bm25_norm == 0.0


def test_the_cosine_only_row_applies_the_cosine_only_threshold_to_the_cosine(
    config: Path, notebooks: Notebooks, tiny_tokenizer: Tokenizer, tmp_path: Path
) -> None:
    base = decided_settings(load_settings(config), NEVER_REFUSE)
    generation = base.generation.model_copy(update={"cosine_only": 0.6})
    settings = base.model_copy(update={"generation": generation})
    resources = make_resources(settings, notebooks, tiny_tokenizer)
    generator, _ = make_generator(settings, tmp_path)
    results = run_generation(
        settings,
        resources,
        CONTEXT,
        Models(generator),
        gold_subset(make_gold(notebooks), Split.DEV),
    )
    refused = [(o.unanswerable, o.top_cosine < 0.6) for o in results.queries]
    expected = refusal_metrics(refused)
    assert results.rows[0].metrics == expected
    assert 0 < expected.false_refusals < expected.answerable  # a real, partial refusal
    assert results.rows[1].metrics.false_refusals == 0  # the shipped thresholds refuse nothing
    assert results.cosine_only_threshold == 0.6


def test_the_generation_results_render_with_counts_per_kind_of_query(
    config: Path, notebooks: Notebooks, tiny_tokenizer: Tokenizer, tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:  # fmt: skip
    settings = decided_settings(load_settings(config), NEVER_REFUSE)
    resources = make_resources(settings, notebooks, tiny_tokenizer)

    def responder(p: Prompt) -> str:
        if "nothing" in p.variable:  # the synthetic unanswerable queries
            return json.dumps({"insufficient": True, "claims": []})
        return answer_with_first_note(p)

    generator, _ = make_generator(settings, tmp_path, responder)
    results = run_generation(
        settings,
        resources,
        CONTEXT,
        Models(generator),
        gold_subset(make_gold(notebooks), Split.DEV),
    )
    path = tmp_path / "generation_dev.json"
    path.write_text(results.model_dump_json(), encoding="utf-8")
    assert render_main([str(path)]) == 0
    out = capsys.readouterr().out
    unanswerable = results.rows[2].metrics.unanswerable
    answerable = results.rows[2].metrics.answerable
    assert f"| answerable | {answerable} | {answerable} | 0/{answerable} | 0 | 0 | 0 |" in out
    row = f"| unanswerable | {unanswerable} | 0 | {unanswerable}/{unanswerable} | 0 | "
    assert row + f"{unanswerable} | 0 |" in out
    assert "| stages 1+2 (current pipeline) | 0/" in out
    assert "Total cost $" in out
    assert "answered answerable queries citing a gold-relevant chunk" in out


def test_two_generation_runs_render_side_by_side_and_other_pairs_are_refused(
    config: Path, notebooks: Notebooks, tiny_tokenizer: Tokenizer, tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:  # fmt: skip
    settings = decided_settings(load_settings(config), NEVER_REFUSE)
    resources = make_resources(settings, notebooks, tiny_tokenizer)
    gold = make_gold(notebooks)
    paths = []
    for name, refuse_unanswerable in (("a.json", False), ("b.json", True)):

        def responder(p: Prompt, flag: bool = refuse_unanswerable) -> str:
            if flag and "nothing" in p.variable:
                return json.dumps({"insufficient": True, "claims": []})
            return answer_with_first_note(p)

        (tmp_path / name[0]).mkdir()
        generator, _ = make_generator(settings, tmp_path / name[0], responder)
        results = run_generation(
            settings, resources, CONTEXT, Models(generator), gold_subset(gold, Split.DEV)
        )
        (tmp_path / name).write_text(results.model_dump_json(), encoding="utf-8")
        paths.append(str(tmp_path / name))
    assert render_main(paths) == 0
    out = capsys.readouterr().out
    unanswerable = select_all(gold, Split.DEV)
    n = sum(1 for q in unanswerable if q.type is QueryType.UNANSWERABLE)
    assert f"| unanswerable queries refused | 0/{n} | {n}/{n} |" in out
    assert "| cost USD |" in out
    assert "gold-hit" in out
    assert render_main([paths[0], "results/corpus_stats.json"]) == 1
    assert "both are generation results" in capsys.readouterr().err


# --------------------------------------------------------------------------------- stage 3


def verdict_json(prompt: Prompt, answers: bool = True, supported: bool = True) -> str:
    claims = re.findall(r"^(\d+)\. ", prompt.variable, flags=re.MULTILINE)
    entries = [{"claim": int(n), "supported": supported} for n in claims]
    return json.dumps({"answers_question": answers, "claims": entries})


def staged_responder(refuse_unanswerable: bool = False, drop_all: bool = False) -> Responder:
    """Answer with the first note; the verifier refuses the synthetic unanswerable queries
    ("nothing" in the question) when asked to, and marks every claim unsupported if `drop_all`."""

    def respond(prompt: Prompt) -> str:
        if prompt.name == "verify":
            nothing = "nothing" in prompt.variable
            return verdict_json(prompt, answers=not (refuse_unanswerable and nothing),
                                supported=not drop_all)  # fmt: skip
        return answer_with_first_note(prompt)

    return respond


def make_models(
    settings: Settings, tmp_path: Path, responder: Responder
) -> tuple[Models, FakeTransport]:
    transport = FakeTransport(responder)
    meter = CostMeter(settings.llm.prices, settings.llm.max_usd)
    client = CachingClient(
        transport, ResponseCache(tmp_path / "m.sqlite"), meter, settings.llm.reasoning_effort
    )
    generator = Generator(client, load_prompt("answer"), settings.llm.model, 100)
    verifier = Verifier(client, load_prompt("verify"), settings.llm.model, ("medium", 400))
    return Models(generator, verifier), transport


def test_the_run_compares_the_current_pipeline_with_stage_three_on_the_same_answers(
    config: Path, notebooks: Notebooks, tiny_tokenizer: Tokenizer, tmp_path: Path
) -> None:
    settings = decided_settings(load_settings(config), NEVER_REFUSE)
    resources = make_resources(settings, notebooks, tiny_tokenizer)
    models, transport = make_models(settings, tmp_path, staged_responder(refuse_unanswerable=True))
    gold = make_gold(notebooks)
    results = run_generation(settings, resources, CONTEXT, models, gold_subset(gold, Split.DEV))
    assert [r.name for r in results.rows] == [
        "cosine only", "cosine + BM25 (stage 1)", "stages 1+2 (current pipeline)", "stages 1+2+3"
    ]  # fmt: skip
    current, with3 = results.rows[2].metrics, results.rows[3].metrics
    assert current.caught == 0  # the answer stage refused nothing
    assert with3.caught == with3.unanswerable > 0  # stage 3 refused every unanswerable query
    assert with3.false_refusals == 0
    s3 = results.stage3
    assert s3 is not None
    assert (s3.refused_by_verifier, s3.refused_unsupported, s3.refused_verifier_invalid) == (
        with3.unanswerable, 0, 0
    )  # fmt: skip
    assert s3.calls_queries == len(
        results.queries
    )  # every query was answered, so all reached stage 3
    assert s3.claims_dropped == sum(o.stage3.claims_dropped for o in results.queries if o.stage3)
    assert results.verify_prompt_version == models.verifier.prompt_version  # type: ignore[union-attr]
    assert results.meta.prompt_versions["verify"] == results.verify_prompt_version
    # the fields next to `outcome` still describe the pipeline without stage 3
    assert all(o.outcome is Outcome.ANSWERED for o in results.queries)
    assert results.total_cost_usd == pytest.approx(
        sum(o.cost_usd for o in results.queries) + s3.cost_usd
    )
    assert s3.cost_usd > 0
    assert sum(1 for p in transport.prompts if p.name == "verify") <= len(results.queries)
    assert all(p.reasoning_effort == "medium" for p in transport.prompts if p.name == "verify")
    assert all(p.reasoning_effort is None for p in transport.prompts if p.name == "answer")


def test_stage_three_dropping_every_claim_refuses_answerable_queries_and_it_is_counted(
    config: Path, notebooks: Notebooks, tiny_tokenizer: Tokenizer, tmp_path: Path
) -> None:
    settings = decided_settings(load_settings(config), NEVER_REFUSE)
    resources = make_resources(settings, notebooks, tiny_tokenizer)
    models, _ = make_models(settings, tmp_path, staged_responder(drop_all=True))
    results = run_generation(
        settings, resources, CONTEXT, models, gold_subset(make_gold(notebooks), Split.DEV)
    )
    with3 = results.rows[3].metrics
    assert with3.false_refusals == with3.answerable  # everything answerable was refused
    s3 = results.stage3
    assert s3 is not None
    assert s3.refused_unsupported == len(results.queries)
    assert s3.claims_dropped == s3.claims_in > 0
    assert s3.answered_answerable == 0


def test_without_a_verifier_there_is_no_stage_three_in_the_results(
    config: Path, notebooks: Notebooks, tiny_tokenizer: Tokenizer, tmp_path: Path
) -> None:
    settings = decided_settings(load_settings(config), NEVER_REFUSE)
    resources = make_resources(settings, notebooks, tiny_tokenizer)
    generator, _ = make_generator(settings, tmp_path)
    results = run_generation(
        settings,
        resources,
        CONTEXT,
        Models(generator),
        gold_subset(make_gold(notebooks), Split.DEV),
    )
    assert results.stage3 is None
    assert results.verify_prompt_version is None
    assert [r.name for r in results.rows][-1] == "stages 1+2 (current pipeline)"
    assert all(o.stage3 is None for o in results.queries)


def test_a_set_without_a_split_reports_its_own_label(
    config: Path, notebooks: Notebooks, tiny_tokenizer: Tokenizer, tmp_path: Path
) -> None:
    settings = decided_settings(load_settings(config), NEVER_REFUSE)
    resources = make_resources(settings, notebooks, tiny_tokenizer)
    models, _ = make_models(settings, tmp_path, staged_responder())
    gold = make_gold(notebooks)
    unfrozen = GoldSet(queries=tuple(q.model_copy(update={"split": None}) for q in gold.queries))
    results = run_generation(
        settings, resources, CONTEXT, models, set_subset(unfrozen, "abstention")
    )
    assert results.query_set == "abstention"
    assert results.meta.split is None
    assert len(results.queries) == len(gold.queries)
    with pytest.raises(RunError, match="needs both answerable and unanswerable"):
        set_subset(GoldSet(queries=unfrozen.queries[:1]), "abstention")


def test_the_dry_run_prices_stage_three_from_cached_answers_and_assumes_the_rest(
    config: Path, notebooks: Notebooks, tiny_tokenizer: Tokenizer, tmp_path: Path
) -> None:
    settings = decided_settings(load_settings(config), NEVER_REFUSE)
    resources = make_resources(settings, notebooks, tiny_tokenizer)
    pipeline, cap = decided_pipeline(settings, resources)
    chunks = {c.id: c for c in resources.chunks(cap, "structural")}
    models, transport = make_models(settings, tmp_path, staged_responder())
    asker = Asker(pipeline, chunks, Thresholds(cosine=-1.0, bm25=-1.0), models, 2)
    queries = select_all(make_gold(notebooks), Split.DEV)
    cache = ResponseCache(tmp_path / "m.sqlite")
    effort = settings.llm.reasoning_effort

    def is_cached(p: Prompt) -> bool:
        return cache.get(cache_key(p, p.reasoning_effort or effort)) is not None

    def offline() -> Generator:
        return Generator(CacheOnly(cache, effort), load_prompt("answer"), settings.llm.model, 100)

    def plan_with_stage3() -> GenerationPlan:
        base = plan_generation(asker, queries, settings, tiny_tokenizer.count, is_cached)
        return plan_stage3(base, asker, offline(), models.verifier, (queries, chunks),  # type: ignore[arg-type]
                           settings, (tiny_tokenizer.count, is_cached))  # fmt: skip

    before = plan_with_stage3()
    assert before.stage3_assumed == len(queries)  # no answer is cached: all priced as worst case
    assert before.stage3 is not None
    assert before.stage3.calls == len(queries)
    for q in queries:  # now really answer and verify everything (with the fake)
        asker.ask(q.text)
    calls = len(transport.prompts)
    after = plan_with_stage3()
    assert len(transport.prompts) == calls  # planning made no model call
    assert after.stage3_assumed == 0
    assert after.answer.calls == 0
    assert after.stage3 is not None
    assert after.stage3.calls == 0
    assert after.stage3_cached == len(queries)
    text = format_plan(after, 0.0, 2.0)
    assert "stage 3 calls:" in text
    assert "total estimated cost" in text


def test_cache_only_answers_hits_and_refuses_misses(tmp_path: Path) -> None:
    cache = ResponseCache(tmp_path / "c.sqlite")
    prompt = Prompt(name="p", version="v", model="m", instructions="i", shared="", variable="v",
                    max_output_tokens=5)  # fmt: skip
    reader = CacheOnly(cache, "minimal")
    with pytest.raises(LLMCallError, match="not in the response cache"):
        reader.complete_json(prompt, Answer)
    body = json.dumps({"insufficient": True, "claims": []})
    cache.put(cache_key(prompt, "minimal"), prompt, Stored(body, Usage(input_tokens=3), 1.25))
    hit = reader.complete_json(prompt, Answer)
    assert hit.parsed.insufficient
    assert hit.from_cache
    assert hit.api_latency_s == 1.25
    medium = prompt.model_copy(update={"reasoning_effort": "medium"})
    with pytest.raises(LLMCallError):
        reader.complete_json(medium, Answer)  # another effort is another key
    cache.put(cache_key(medium, "medium"), medium, Stored("{}", Usage(), 0.0))
    with pytest.raises(LLMOutputError, match="no longer parses"):
        reader.complete_json(medium, Answer)


def test_step_c_runs_the_verifier_on_invented_chunks_and_reports_what_it_did(
    tmp_path: Path,
) -> None:
    settings = load_settings(Path("config/default.toml"))
    models, transport = make_models(settings, tmp_path, staged_responder())
    assert models.verifier is not None
    row, detail = run_step_c(models.verifier, settings.llm.prices[settings.llm.model])
    assert row.label.startswith("step C")
    assert row.outcome == "answered"
    assert "answers_question=True; kept 2 of 2 claims" in detail
    sent = transport.prompts[0].variable
    assert "plt.hist draws a pie chart." in sent
    assert "demo/plots.ipynb" in sent
    assert "train_test_split" not in sent  # the uncited chunk was not shown
    assert row.cost_usd > 0


def test_the_stage_three_report_shows_both_pipelines_with_intervals_cost_and_latency(
    config: Path, notebooks: Notebooks, tiny_tokenizer: Tokenizer, tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:  # fmt: skip
    settings = decided_settings(load_settings(config), NEVER_REFUSE)
    resources = make_resources(settings, notebooks, tiny_tokenizer)
    models, _ = make_models(settings, tmp_path, staged_responder(refuse_unanswerable=True))
    results = run_generation(
        settings, resources, CONTEXT, models, gold_subset(make_gold(notebooks), Split.DEV)
    )
    path = tmp_path / "generation_dev_stage3.json"
    path.write_text(results.model_dump_json(), encoding="utf-8")
    assert render_main([str(path)]) == 0
    out = capsys.readouterr().out
    current, with3 = results.rows[2].metrics, results.rows[3].metrics
    assert "| Measure | stages 1+2 (current) | stages 1+2+3 |" in out
    n_u = with3.unanswerable
    assert f"| unanswerable queries refused (95% Wilson) | 0/{n_u} [0.00," in out
    assert f"| {n_u}/{n_u} [" in out
    assert current.false_refusals == 0
    assert with3.false_refusals == 0
    assert "| model cost USD per query |" in out
    assert "| answer-path model latency p50 / p95, s |" in out
    assert "| refused: cited notes do not answer the question |" in out
    assert "prompt `verify` version" in out


def test_stage_three_latency_adds_the_verifier_call_to_the_answer_path(
    config: Path, notebooks: Notebooks, tiny_tokenizer: Tokenizer, tmp_path: Path
) -> None:
    settings = decided_settings(load_settings(config), NEVER_REFUSE)
    resources = make_resources(settings, notebooks, tiny_tokenizer)
    transport = FakeTransport(
        staged_responder(), latency=lambda p: 3.0 if p.name == "verify" else 1.0
    )
    meter = CostMeter(settings.llm.prices, settings.llm.max_usd)
    client = CachingClient(transport, ResponseCache(tmp_path / "l.sqlite"), meter, "minimal")
    models = Models(
        Generator(client, load_prompt("answer"), settings.llm.model, 100),
        Verifier(client, load_prompt("verify"), settings.llm.model, ("medium", 400)),
    )
    results = run_generation(
        settings, resources, CONTEXT, models, gold_subset(make_gold(notebooks), Split.DEV)
    )
    s3 = results.stage3
    assert s3 is not None
    without, with3 = s3.answer_path_latency_without, s3.answer_path_latency_with
    assert without is not None
    assert with3 is not None
    assert (without.p50, without.p95) == (1.0, 1.0)  # the answer call alone: 1.0 s
    assert (with3.p50, with3.p95) == (4.0, 4.0)  # plus the verifier call: 1.0 + 3.0 s
    assert all(o.api_latency_s == 1.0 for o in results.queries)


def test_stage_three_gold_hit_is_the_gold_hit_of_the_claims_it_kept(
    config: Path, notebooks: Notebooks, tiny_tokenizer: Tokenizer, tmp_path: Path
) -> None:
    settings = decided_settings(load_settings(config), NEVER_REFUSE)
    resources = make_resources(settings, notebooks, tiny_tokenizer)
    models, _ = make_models(settings, tmp_path, staged_responder())
    results = run_generation(
        settings, resources, CONTEXT, models, gold_subset(make_gold(notebooks), Split.DEV)
    )
    answerable = [o for o in results.queries if not o.unanswerable]
    hits = [o.cites_relevant_chunk for o in answerable]
    assert True in hits
    assert False in hits
    for o in answerable:
        assert o.stage3 is not None
        assert o.stage3.cites_relevant_chunk == o.cites_relevant_chunk  # all claims were kept
    s3 = results.stage3
    assert s3 is not None
    assert s3.answered_citing_a_relevant_chunk == sum(1 for h in hits if h)
    assert results.validity.answered_citing_a_relevant_chunk == sum(1 for h in hits if h)


def test_the_stage_three_plan_skips_drafts_that_were_already_refused(
    config: Path, notebooks: Notebooks, tiny_tokenizer: Tokenizer, tmp_path: Path
) -> None:
    settings = decided_settings(load_settings(config), NEVER_REFUSE)
    resources = make_resources(settings, notebooks, tiny_tokenizer)
    pipeline, cap = decided_pipeline(settings, resources)
    chunks = {c.id: c for c in resources.chunks(cap, "structural")}

    def respond(prompt: Prompt) -> str:
        if prompt.name == "answer" and "nothing" in prompt.variable:
            return json.dumps({"insufficient": True, "claims": []})
        return staged_responder()(prompt)

    models, _ = make_models(settings, tmp_path, respond)
    asker = Asker(pipeline, chunks, Thresholds(cosine=-1.0, bm25=-1.0), models, 2)
    queries = select_all(make_gold(notebooks), Split.DEV)
    for q in queries:
        asker.ask(q.text)  # fills the cache with the answers (and verifications)
    cache = ResponseCache(tmp_path / "m.sqlite")
    effort = settings.llm.reasoning_effort
    offline = Generator(CacheOnly(cache, effort), load_prompt("answer"), settings.llm.model, 100)
    never_cached = lambda _p: False  # noqa: E731 - every verifier call would still have to be made
    base = plan_generation(asker, queries, settings, len, lambda _p: True)
    assert models.verifier is not None
    plan = plan_stage3(base, asker, offline, models.verifier, (queries, chunks), settings,
                       (len, never_cached))  # fmt: skip
    refused = sum(1 for q in queries if q.type is QueryType.UNANSWERABLE and "nothing" in q.text)
    assert refused > 0
    assert plan.stage3 is not None
    assert plan.stage3.calls == len(queries) - refused
