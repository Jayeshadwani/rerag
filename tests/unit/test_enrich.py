"""Headers, blurb prompts, the clean verifier, the job runner and the estimate (all offline)."""

from pathlib import Path

import pytest
from hypothesis import given
from hypothesis import strategies as st
from pydantic import ValidationError

from make_fixtures import (
    FAKE_ANTHROPIC_KEY,
    FAKE_AWS_ACCESS_KEY,
    FAKE_AWS_SECRET_KEY,
    FIXTURE_DIR,
)
from noterecall.chunking.tokens import Tokenizer
from noterecall.config import PriceConfig
from noterecall.enrich.contextual import (
    Blurb,
    MissingEnrichmentError,
    blurb_prompt,
    notebook_text,
    read_blurb,
)
from noterecall.enrich.estimate import estimate_step
from noterecall.enrich.headers import chunk_header, clean_heading, index_text, notebook_title
from noterecall.enrich.run import Job, blurb_groups, clean_groups, read_blurbs, run_groups
from noterecall.ingest.clean import clean_notebook, load_rules
from noterecall.ingest.ipynb import IpynbLoader
from noterecall.ingest.llm_clean import (
    CleanedCell,
    apply_clean,
    fenced_blocks,
    protected_tokens,
    verify_clean,
)
from noterecall.llm.cache import ResponseCache, cache_key
from noterecall.llm.client import BudgetExceededError, LLMCallError, Prompt
from noterecall.llm.cost import CostMeter
from noterecall.llm.fake import FakeTransport, Responder
from noterecall.llm.prompts import load_prompt
from noterecall.llm.transport import CachingClient
from noterecall.models import Cell, CellType, Chunk

PRICE = PriceConfig(input=1.0, cached_input=0.1, output=2.0)
FENCE = "```"


def chunk(
    path: str = "ml/01_intro.ipynb", headings: tuple[str, ...] = (), text: str = "t"
) -> Chunk:
    return Chunk(
        id=f"{path}#{text}",
        notebook_path=path,
        heading_path=headings,
        pieces=(),
        text=text,
        tokens=1,
        cell_hashes=(),
    )


def cell(index: int, source: str, kind: CellType = CellType.MARKDOWN) -> Cell:
    return Cell(index=index, cell_type=kind, raw_hash="r", clean_hash="c", source=source)


# ----------------------------------------------------------------------------------- headers


def test_clean_heading_strips_tags_marks_and_spaces_but_keeps_underscores() -> None:
    assert clean_heading("**<b>Hello</b>**   `train_test_split`\n  now") == (
        "Hello train_test_split now"
    )
    assert clean_heading("<br>") == ""


def test_notebook_title_is_the_file_stem() -> None:
    assert notebook_title("ml/week 1/01_intro.v2.ipynb") == "01_intro.v2"


def test_chunk_header_joins_title_and_every_heading_level_and_skips_empty_ones() -> None:
    c = chunk(headings=("Intro", "<span>Loss</span>", "", "`mse`"))
    assert chunk_header(c) == "01_intro > Intro > Loss > mse"
    assert chunk_header(chunk(headings=())) == "01_intro"


def test_index_text_orders_header_blurb_text_and_leaves_the_chunk_alone() -> None:
    c = chunk(headings=("A",), text="body")
    assert index_text(c, False, None) == "body"
    assert index_text(c, True, None) == "01_intro > A\n\nbody"
    assert index_text(c, True, "about it") == "01_intro > A\n\nabout it\n\nbody"
    assert index_text(c, False, "about it") == "about it\n\nbody"
    assert index_text(c, False, "") == "body"  # an empty blurb adds nothing
    assert c.text == "body"


# ------------------------------------------------------------------------------- blurb prompts


def test_blurb_prompts_share_one_identical_prefix_per_notebook_and_end_with_the_chunk() -> None:
    template = load_prompt("contextual_blurb")
    cells = [cell(0, "# Title\n"), cell(1, "x = 1", CellType.CODE), cell(2, "   "), cell(3, "end")]
    text = notebook_text(cells)
    assert text == "# Title\n\nx = 1\n\nend"  # the blank cell is skipped, outputs are never used
    a = blurb_prompt(template, "m", 80, text, chunk(text="first chunk"))
    b = blurb_prompt(template, "m", 80, text, chunk(text="second chunk"))
    assert a.shared == b.shared
    assert a.instructions == b.instructions
    assert text in a.shared
    assert "first chunk" in a.variable
    assert "first chunk" not in a.shared
    assert a.variable != b.variable
    other = blurb_prompt(template, "m", 80, "different notebook", chunk(text="first chunk"))
    assert other.shared != a.shared
    assert cache_key(a, "minimal") != cache_key(b, "minimal")


def test_blurb_schema_rejects_empty_extra_and_overlong_answers() -> None:
    assert Blurb.model_validate_json('{"context": "ok"}').context == "ok"
    for bad in ('{"context": ""}', '{"context": "ok", "x": 1}', '{"context": "' + "a" * 601 + '"}'):
        with pytest.raises(ValidationError):
            Blurb.model_validate_json(bad)


def test_read_blurb_finds_stored_answers_and_reports_missing_ones(tmp_path: Path) -> None:
    template = load_prompt("contextual_blurb")
    p = blurb_prompt(template, "m", 80, "nb", chunk(text="c"))
    client = CachingClient(
        FakeTransport(lambda _p: '{"context": "situated"}'),
        ResponseCache(tmp_path / "c.sqlite"),
        CostMeter({"m": PRICE}, 1.0),
        "minimal",
    )
    cache = ResponseCache(tmp_path / "c.sqlite")
    assert read_blurb(cache, "minimal", p) is None
    client.complete_json(p, Blurb)
    assert read_blurb(cache, "minimal", p) == "situated"
    assert read_blurb(cache, "low", p) is None  # another effort is another answer


def test_read_blurbs_names_how_many_are_missing(tmp_path: Path) -> None:
    template = load_prompt("contextual_blurb")
    jobs = [
        Job(f"k{i}", blurb_prompt(template, "m", 80, "nb", chunk(text=f"c{i}"))) for i in range(3)
    ]
    cache = ResponseCache(tmp_path / "c.sqlite")
    client = CachingClient(
        FakeTransport(lambda _p: '{"context": "s"}'), cache, CostMeter({"m": PRICE}, 1.0), "minimal"
    )
    client.complete_json(jobs[0].prompt, Blurb)
    with pytest.raises(MissingEnrichmentError, match="2 of 3 blurbs are not in the cache"):
        read_blurbs(jobs, cache, "minimal")
    for job in jobs[1:]:
        client.complete_json(job.prompt, Blurb)
    assert read_blurbs(jobs, cache, "minimal") == {"k0": "s", "k1": "s", "k2": "s"}


# --------------------------------------------------------------------------------- the verifier


def test_protected_tokens_hand_computed() -> None:
    text = "Use `df.head()` and train_test_split with camelCase, x2 and 3.5 but plain words."
    # inline span: df.head(); then outside it: train_test_split (underscore), camelCase (a lower
    # letter then an upper one), x2 (letters and digits). "3.5" has no letter, "plain", "words",
    # "Use" and "with" are ordinary words.
    assert protected_tokens(text) == {"df.head()", "train_test_split", "camelCase", "x2"}
    assert protected_tokens("os.path.join is a dotted path") == {"os.path.join"}
    assert protected_tokens("Nothing special here.") == set()


def test_fenced_blocks_are_found_in_order() -> None:
    text = f"a\n\n{FENCE}python\nx = 1\n{FENCE}\n\nb\n\n{FENCE}\ny\n{FENCE}\n"
    blocks = fenced_blocks(text)
    assert len(blocks) == 2
    assert "x = 1" in blocks[0]
    assert "y" in blocks[1]
    assert fenced_blocks("no code here") == []


def test_a_clean_spelling_fix_is_accepted() -> None:
    original = "Teh `np.mean` function computes the avrage of the values."
    cleaned = "The `np.mean` function computes the average of the values."
    assert verify_clean(original, cleaned).ok


def test_dropping_a_bare_identifier_is_rejected() -> None:
    original = "Call train_test_split before fitting the model on the data."
    cleaned = "Call the splitter before fitting the model on the data."
    verdict = verify_clean(original, cleaned)
    assert not verdict.ok
    assert verdict.reasons == ("missing_token",)


def test_changing_an_inline_code_span_is_rejected() -> None:
    original = "Set `max_depth` to a small number to avoid overfitting the tree."
    cleaned = "Set `maxDepth` to a small number to avoid overfitting the tree."
    assert verify_clean(original, cleaned).reasons == ("missing_token",)


def test_length_ratio_bounds_are_inclusive_and_exact() -> None:
    original = "a" * 10
    assert verify_clean(original, "a" * 7).ok  # 7 / 10 = 0.7 exactly
    assert verify_clean(original, "a" * 13).ok  # 13 / 10 = 1.3 exactly
    assert verify_clean(original, "a" * 6).reasons == ("length_ratio",)  # 0.6
    assert verify_clean(original, "a" * 14).reasons == ("length_ratio",)  # 1.4


def test_a_blank_original_is_never_replaced() -> None:
    assert verify_clean("  \n", "something").reasons == ("length_ratio",)


@pytest.mark.parametrize(
    "cleaned",
    [
        f"intro\n\n{FENCE}python\nx = 2\n{FENCE}\n\nouter text",  # edited inside
        "intro\n\nouter text",  # removed
        f"intro\n\n{FENCE}python\nx = 1\n{FENCE}\n\n{FENCE}\nz\n{FENCE}\n\nouter text",  # added
    ],
)
def test_any_change_to_the_fenced_blocks_is_rejected(cleaned: str) -> None:
    original = f"intro\n\n{FENCE}python\nx = 1\n{FENCE}\n\nouter text"
    verdict = verify_clean(original, cleaned)
    assert "code_fence_changed" in verdict.reasons
    assert not verdict.ok


def test_reordered_fenced_blocks_are_rejected() -> None:
    one, two = f"{FENCE}\nfirst\n{FENCE}", f"{FENCE}\nsecond\n{FENCE}"
    assert verify_clean(f"{one}\n\n{two}", f"{two}\n\n{one}").reasons == ("code_fence_changed",)
    assert verify_clean(f"{one}\n\n{two}", f"{one}\n\n{two}").ok


def test_every_failing_reason_is_reported() -> None:
    original = f"Use `keep_me` here.\n\n{FENCE}\ncode line\n{FENCE}\n"
    verdict = verify_clean(original, "x")
    assert set(verdict.reasons) == {"length_ratio", "missing_token", "code_fence_changed"}


@given(st.text(min_size=1).filter(lambda t: t.strip() != ""))
def test_a_cell_returned_unchanged_always_passes(text: str) -> None:
    assert verify_clean(text, text).ok


@given(st.text(min_size=1).filter(lambda t: t.strip() != ""), st.integers(min_value=1, max_value=5))
def test_a_cell_blown_up_in_length_always_fails_the_ratio(text: str, extra: int) -> None:
    padded = text + "z" * (len(text) * (extra + 1))
    assert "length_ratio" in verify_clean(text, padded).reasons


def test_apply_clean_substitutes_verified_answers_and_counts_the_rest() -> None:
    cells = [
        cell(0, "Teh `np.mean` function computes the avrage."),  # fixed
        cell(1, "Already fine, nothing to do here."),  # returned unchanged
        cell(2, "Call train_test_split before fitting the model."),  # drops an identifier
        cell(3, "short text here ok"),  # answer far too long
        cell(4, "x = 1", CellType.CODE),  # code cells are never touched
        cell(5, "No answer for this one."),  # no answer
    ]
    answers = {
        0: "The `np.mean` function computes the average.",
        1: "Already fine, nothing to do here.",
        2: "Call the splitter before fitting the model.",
        3: "short text here ok " * 5,
        4: "y = 2",
    }
    out, report = apply_clean(cells, answers)
    assert [c.source for c in out] == [
        "The `np.mean` function computes the average.",
        "Already fine, nothing to do here.",
        "Call train_test_split before fitting the model.",
        "short text here ok",
        "x = 1",
        "No answer for this one.",
    ]
    assert out[0].raw_hash == cells[0].raw_hash  # gold-label hashes are untouched
    assert (report.cells, report.changed, report.unchanged, report.rejected_cells) == (5, 1, 1, 2)
    assert report.rejected_by_reason == {"missing_token": 1, "length_ratio": 1}


def test_cleaned_cell_schema_rejects_empty_text() -> None:
    with pytest.raises(ValidationError):
        CleanedCell.model_validate_json('{"text": ""}')


# ------------------------------------------------------------------------------- the job runner


def make_client(
    tmp_path: Path,
    responder: Responder | None = None,
    cap: float = 1.0,
    fail_after: int | None = None,
) -> tuple[CachingClient, FakeTransport]:
    def default(p: Prompt) -> str:
        return '{"context": "about ' + p.variable.split()[-1] + '"}'

    transport = FakeTransport(responder or default, fail_after)
    client = CachingClient(
        transport, ResponseCache(tmp_path / "c.sqlite"), CostMeter({"m": PRICE}, cap), "minimal"
    )
    return client, transport


def two_notebooks() -> list[tuple[str, str, list[Chunk]]]:
    return [
        ("a.ipynb", "notebook A text", [chunk("a.ipynb", text=f"a{i}") for i in range(3)]),
        ("b.ipynb", "notebook B text", [chunk("b.ipynb", text=f"b{i}") for i in range(2)]),
    ]


def test_the_first_job_of_every_notebook_runs_alone_before_the_rest(tmp_path: Path) -> None:
    template = load_prompt("contextual_blurb")
    groups = blurb_groups(two_notebooks(), template, "m", 80)
    client, transport = make_client(tmp_path)
    answers, summary = run_groups(groups, client, Blurb, concurrency=4)
    assert len(answers) == 5
    assert summary.jobs == 5
    assert (summary.from_api, summary.from_cache, summary.failures) == (5, 0, {})
    order = [p.shared for p in transport.prompts]
    assert order[0] == groups[0][0].prompt.shared  # notebook A's first job is the first call
    assert set(order[:3]) == {groups[0][0].prompt.shared}  # A finishes before B starts
    assert order[3] == groups[1][0].prompt.shared
    assert transport.prompts[0].variable == groups[0][0].prompt.variable


def test_a_second_run_is_answered_entirely_from_the_cache(tmp_path: Path) -> None:
    groups = blurb_groups(two_notebooks(), load_prompt("contextual_blurb"), "m", 80)
    client, transport = make_client(tmp_path)
    run_groups(groups, client, Blurb, concurrency=2)
    _answers, summary = run_groups(groups, client, Blurb, concurrency=2)
    assert (summary.from_api, summary.from_cache) == (0, 5)
    assert len(transport.prompts) == 5


def test_unusable_answers_are_counted_by_reason_and_the_rest_continue(tmp_path: Path) -> None:
    groups = blurb_groups(two_notebooks(), load_prompt("contextual_blurb"), "m", 80)

    def responder(p: Prompt) -> str:
        return "not json at all" if "a1" in p.variable else '{"context": "fine"}'

    client, _t = make_client(tmp_path, responder)
    answers, summary = run_groups(groups, client, Blurb, concurrency=2)
    assert len(answers) == 4
    assert summary.jobs == 5
    assert sum(summary.failures.values()) == 1
    assert "a1" not in {k.split("#")[-1] for k in answers}


def test_a_failed_call_stops_the_run_loudly(tmp_path: Path) -> None:
    groups = blurb_groups(two_notebooks(), load_prompt("contextual_blurb"), "m", 80)
    client, _t = make_client(tmp_path, fail_after=2)
    with pytest.raises(LLMCallError, match="told to fail"):
        run_groups(groups, client, Blurb, concurrency=1)


def test_the_spend_cap_stops_the_run(tmp_path: Path) -> None:
    groups = blurb_groups(two_notebooks(), load_prompt("contextual_blurb"), "m", 80)
    client, transport = make_client(tmp_path, cap=0.0)
    with pytest.raises(BudgetExceededError):
        run_groups(groups, client, Blurb, concurrency=1)
    assert transport.prompts == []


def test_clean_groups_make_one_job_per_nonblank_markdown_cell() -> None:
    cells = [cell(0, "# T"), cell(1, "x", CellType.CODE), cell(2, " \n"), cell(3, "words\n")]
    groups = clean_groups(
        [("p.ipynb", cells), ("q.ipynb", [])], load_prompt("clean_markdown"), "m", 500
    )
    assert [[j.key for j in g] for g in groups] == [[("p.ipynb", 0), ("p.ipynb", 3)], []]
    assert groups[0][1].prompt.variable.count("words") == 1
    assert groups[0][0].prompt.shared == ""


# ---------------------------------------------------------------------------------- the estimate


def words(prompt_words: tuple[int, int, int]) -> Prompt:
    a, s, v = prompt_words
    return Prompt(
        name="p",
        version="v",
        model="m",
        instructions=" ".join(["i"] * a),
        shared=" ".join(["s"] * s),
        variable=" ".join(["v"] * v),
        max_output_tokens=50,
    )


def count_words(text: str) -> int:
    return len(text.split())


def test_estimate_is_hand_computed_with_and_without_prefix_caching() -> None:
    group = [Job(i, words((2, 3, 1))) for i in range(3)]
    est = estimate_step([group], lambda _j: False, count_words, PRICE, assumed_output_tokens=4)
    # each prompt: 2 + 3 + 1 = 6 tokens, so 18 in all; two later jobs reuse 2 + 3 = 5 each = 10
    # cached case: (18 - 10) * 1.0 + 10 * 0.1 = 9 input; output 3 * 4 * 2.0 = 24; (9 + 24) / 1e6
    # no caching: 18 * 1.0 + 24 = 42; upper bound: 18 + 3 * 50 * 2.0 = 318
    assert (est.jobs_total, est.jobs_to_run, est.jobs_already_cached) == (3, 3, 0)
    assert (est.input_tokens, est.reusable_prefix_tokens, est.output_cap_per_call) == (18, 10, 50)
    assert est.cost_with_prefix_caching == pytest.approx(33e-6)
    assert est.cost_without_caching == pytest.approx(42e-6)
    assert est.cost_upper_bound == pytest.approx(318e-6)


def test_estimate_leaves_out_jobs_already_in_the_cache() -> None:
    group = [Job(i, words((2, 3, 1))) for i in range(3)]
    est = estimate_step([group], lambda j: j.key == 0, count_words, PRICE, assumed_output_tokens=4)
    # jobs 1 and 2 remain: input 12, the second reuses 5; cached case (12 - 5) + 0.5 = 7.5,
    # output 2 * 4 * 2.0 = 16 -> 23.5 / 1e6
    assert (est.jobs_to_run, est.jobs_already_cached) == (2, 1)
    assert (est.input_tokens, est.reusable_prefix_tokens) == (12, 5)
    assert est.cost_with_prefix_caching == pytest.approx(23.5e-6)


def test_a_job_without_a_shared_part_reuses_nothing() -> None:
    group = [Job(i, words((2, 0, 1))) for i in range(2)]
    est = estimate_step([group], lambda _j: False, count_words, PRICE, assumed_output_tokens=0)
    assert est.reusable_prefix_tokens == 0
    assert est.cost_with_prefix_caching == est.cost_without_caching


def test_an_empty_estimate_costs_nothing() -> None:
    est = estimate_step([], lambda _j: False, count_words, PRICE, assumed_output_tokens=4)
    assert est.jobs_total == 0
    assert est.cost_upper_bound == 0.0


# ------------------------------------------------------------------------------ secrets stay out


def test_a_planted_secret_never_reaches_what_would_be_sent_to_the_model(
    tiny_tokenizer: Tokenizer, tmp_path: Path
) -> None:
    rules = load_rules(Path(__file__).resolve().parents[2] / "config" / "cleaning_rules.yaml")
    raw = IpynbLoader().load(FIXTURE_DIR / "05_fake_secret.ipynb", FIXTURE_DIR, "h")
    nb = clean_notebook(raw, rules, 20, tiny_tokenizer, "p")
    text = notebook_text(nb.cells)
    chunks = [chunk(nb.path, text=c.source) for c in nb.cells if c.source.strip()]
    blurbs = blurb_groups([(nb.path, text, chunks)], load_prompt("contextual_blurb"), "m", 80)
    cleans = clean_groups([(nb.path, list(nb.cells))], load_prompt("clean_markdown"), "m", 500)
    client, transport = make_client(tmp_path)
    run_groups(blurbs, client, Blurb, concurrency=2)
    run_groups(cleans, client, CleanedCell, concurrency=2)
    assert transport.prompts, "the test must actually send something"
    assert {p.name for p in transport.prompts} == {"contextual_blurb", "clean_markdown"}
    for p in transport.prompts:
        sent = p.instructions + p.shared + p.variable
        for secret in (FAKE_ANTHROPIC_KEY, FAKE_AWS_ACCESS_KEY, FAKE_AWS_SECRET_KEY):
            assert secret not in sent
    assert "[REDACTED" in "".join(p.shared for p in transport.prompts)
