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
    run_calibration,
    run_generation,
    select_all,
)
from noterecall.eval.run_retrieval import QueryRef, RunError, select_queries
from noterecall.generate.abstain import Thresholds
from noterecall.generate.answer import CitedClaim, Generator, Outcome, Response
from noterecall.generate.ask import Asker, AskResult
from noterecall.generate.citations import Answer
from noterecall.generate.commands import format_answer, format_context, format_plan, plan_generation
from noterecall.generate.estimate import estimate_generation
from noterecall.generate.runtime import NoCalls
from noterecall.generate.smoke import (
    choose_step_b,
    format_rows,
    run_step_a,
    smoke_row,
    synthetic_chunks,
)
from noterecall.llm.cache import ResponseCache
from noterecall.llm.client import LLMCallError, Prompt, Usage
from noterecall.llm.cost import CostMeter
from noterecall.llm.fake import FakeTransport, Responder
from noterecall.llm.prompts import load_prompt
from noterecall.llm.transport import CachingClient
from noterecall.models import Cell, Chunk
from noterecall.retrieval.pipeline import AbstentionSignals, PipelineError, PipelineSpec

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
                  generator, settings.generation.max_chunks)  # fmt: skip
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
    results = run_generation(settings, gold, resources, CONTEXT, generator, Split.TEST)
    queries = select_all(gold, Split.TEST)
    assert [o.query_id for o in results.queries] == [q.id for q in queries]
    assert [r.name for r in results.rows] == [
        "cosine only", "cosine + BM25 (stage 1)", "two-stage (ships)"
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
    results = run_generation(settings, gold, resources, CONTEXT, generator, Split.DEV)
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
    results = run_generation(settings, gold, resources, CONTEXT, generator, Split.DEV)
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
        run_generation(settings, make_gold(notebooks), resources, CONTEXT, generator, Split.DEV)


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


def test_plan_generation_counts_stage_one_refusals_and_prices_the_rest(
    config: Path, notebooks: Notebooks, tiny_tokenizer: Tokenizer, tmp_path: Path
) -> None:
    settings = decided_settings(load_settings(config), NEVER_REFUSE)
    asker, transport, _ = make_asker(settings, (notebooks, tiny_tokenizer), tmp_path, NEVER_REFUSE)
    queries = select_all(make_gold(notebooks), Split.DEV)
    refused, est = plan_generation(asker, asker.generator, queries, settings, tiny_tokenizer.count)
    assert refused == 0
    assert est.calls == len(queries)
    assert transport.prompts == []
    text = format_plan(refused, len(queries), est, 0.0, 2.0)
    assert f"{len(queries)} queries, 0 refused by stage 1" in text
    assert "of the $2.00 cap" in text
    always, _t, _c = make_asker(settings, (notebooks, tiny_tokenizer), tmp_path, ALWAYS_REFUSE)
    refused_all, est_all = plan_generation(always, always.generator, queries, settings, len)
    assert refused_all == len(queries)
    assert est_all.calls == 0


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
