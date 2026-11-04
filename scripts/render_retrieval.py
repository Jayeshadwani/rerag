"""Markdown tables for the M4 retrieval results (dev ablation and the single test run).

Every number is read from the results file; nothing is typed by hand. Used by
`scripts/render_results.py`.
"""

from collections.abc import Callable, Sequence

from noterecall.eval.ablation import Comparison, ConfigResult, LatencyStat
from noterecall.eval.run_generation import GenerationResults, Row
from noterecall.eval.run_header import HeaderDevResults
from noterecall.eval.run_retrieval import DevResults, TestResults
from noterecall.eval.stats import wilson_interval


def _cell(text: str) -> str:
    """A table cell: configuration names contain `|`, which would split the column."""
    return text.replace("|", "\\|")


def _grid(header: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    lines.extend("| " + " | ".join(_cell(c) for c in row) + " |" for row in rows)
    return "\n".join(lines)


def _f(value: float) -> str:
    return f"{value:.3f}"


def _ms(value: float) -> str:
    return f"{value:,.1f}"


def _coverage_metric(budget_metric: str) -> str:
    return budget_metric.replace("recall@", "tokcov@")


def config_rows(configs: Sequence[ConfigResult], budget_metric: str) -> list[list[str]]:
    """One row per configuration with the headline metrics and total latency."""
    coverage = _coverage_metric(budget_metric)
    return [
        [
            c.name,
            _f(c.means["recall@5"]),
            _f(c.means["recall@10"]),
            _f(c.means["mrr@10"]),
            _f(c.means["ndcg@10"]),
            _f(c.means[budget_metric]),
            _f(c.means[coverage]),
            f"{_ms(c.latency.total.p50)} / {_ms(c.latency.total.p95)}",
        ]
        for c in configs
    ]


def config_table(configs: Sequence[ConfigResult], budget_metric: str) -> str:
    """Metrics per configuration (means over the queries)."""
    header = (
        "Configuration",
        "Recall@5",
        "Recall@10",
        "MRR@10",
        "nDCG@10",
        budget_metric.replace("recall@", "Recall@"),
        _coverage_metric(budget_metric).replace("tokcov@", "Token coverage@"),
        "Total ms p50 / p95",
    )
    return _grid(header, config_rows(configs, budget_metric))


def comparison_rows(comparisons: Sequence[Comparison]) -> list[list[str]]:
    """One row per ADR-010 comparison."""
    rows = []
    for c in comparisons:
        label = "ADOPT" if c.decision.adopt else "reject"
        decision = (
            f"report only: {c.decision.reason}"
            if c.report_only
            else f"{label}: {c.decision.reason}"
        )
        if c.gate:
            decision += f" ({c.gate})"
        rows.append(
            [
                c.stage,
                f"{c.candidate} vs {c.baseline}",
                c.metric,
                f"{c.ci.mean:+.3f}",
                f"[{c.ci.low:+.3f}, {c.ci.high:+.3f}]",
                "yes" if c.free else "no",
                decision,
            ]
        )
    return rows


def decision_table(comparisons: Sequence[Comparison]) -> str:
    """Paired bootstrap interval and the ADR-010 decision for every comparison."""
    header = ("Stage", "Comparison", "Metric", "Mean delta", "95% CI", "Free", "Decision")
    return _grid(header, comparison_rows(comparisons))


def by_type_table(result: DevResults, names: Sequence[str]) -> str:
    """Mean Recall@10 per query type for the named configurations (descriptive, no decision)."""
    types = sorted(set(result.meta.query_types.values()))
    counts = {t: sum(1 for v in result.meta.query_types.values() if v == t) for t in types}
    by_name = {c.name: c for c in result.ablation.configs}
    rows = []
    for name in names:
        config = by_name[name]
        cells = []
        for qtype in types:
            ids = [q for q, t in result.meta.query_types.items() if t == qtype]
            cells.append(_f(sum(config.per_query[q]["recall@10"] for q in ids) / len(ids)))
        rows.append([name, *cells])
    header = ["Recall@10 by query type", *(f"{t} (n={counts[t]})" for t in types)]
    return _grid(header, rows)


def _type_table_names(result: DevResults) -> list[str]:
    """The decided cap's BM25 baseline, single retrievers, hybrid and reranked configurations."""
    cap = result.decided.chunk_cap
    wanted = [
        c.name
        for c in result.ablation.configs
        if c.key.cap == cap and c.key.chunker == "structural"
    ]
    return [n for n in wanted if "bm25-code" in n or "dense" in n]


def render_dev(result: DevResults) -> str:
    """The full dev report: decisions, every configuration, truncation, sensitivity."""
    comparisons = [c for s in result.ablation.stages for c in s.comparisons]
    budget = result.params.budget_metric
    notes = [f"{s.stage}: {s.note}" for s in result.ablation.stages if s.note]
    truncation = _grid(
        ("Embedding model", "Chunk cap", "Chunks", "Cut at the input limit"),
        [[t.model, str(t.cap), str(t.chunks), str(t.truncated)] for t in result.truncation],
    )
    pipeline = result.decided.pipeline
    decided = (
        f"Decided: chunk cap {result.decided.chunk_cap}, lexical {pipeline.lexical}, "
        f"dense {pipeline.dense_model}, RRF k {pipeline.rrf_k}, rerank {pipeline.rerank}."
    )
    meta = result.meta
    footer = (
        f"{meta.queries} dev queries, seed {meta.seed}, {meta.bootstrap_resamples:,} bootstrap "
        f"resamples, commit `{meta.git.sha[:12]}`{' (dirty tree)' if meta.git.dirty else ''}."
    )
    limitations = [f"{note}" for note in result.notes]
    return (
        "\n\n".join(
            [
                decision_table(comparisons),
                *notes,
                "Token coverage, structural vs fixed (secondary metric, report only):",
                decision_table(result.ablation.secondary),
                decided,
                config_table(result.ablation.configs, budget),
                "Recall@10 by query type (report only, no decision):",
                by_type_table(result, _type_table_names(result)),
                "Decided pipeline at each chunk cap (report only, no decision):",
                config_table(result.sensitivity, budget),
                truncation,
                *limitations,
                footer,
            ]
        )
        + "\n"
    )


def render_test(result: TestResults) -> str:
    """The single test-split run: the decided pipeline and its comparison rows."""
    budget = f"recall@tokens={result.meta.config['retrieval']['budget_tokens']}"  # type: ignore[index]
    meta = result.meta
    footer = (
        f"{meta.queries} test queries, milestone {result.milestone}, seed {meta.seed}, commit "
        f"`{meta.git.sha[:12]}`{' (dirty tree)' if meta.git.dirty else ''}."
    )
    return (
        "\n\n".join(
            [
                config_table([result.chosen, *result.references], budget),
                decision_table(result.comparisons),
                footer,
            ]
        )
        + "\n"
    )


def render_header(result: HeaderDevResults) -> str:
    """The M5 header row: with and without the header on dev, and what it means."""
    ab = result.ablation
    budget = f"recall@tokens={result.meta.config['retrieval']['budget_tokens']}"  # type: ignore[index]
    truncation = _grid(
        ("Variant", "Chunks", "Cut at the input limit"),
        [
            ["header on" if t.variant.header else "header off", str(t.chunks), str(t.truncated)]
            for t in result.truncation
        ],
    )
    meta = result.meta
    footer = (
        f"{meta.queries} dev queries, seed {meta.seed}, {meta.bootstrap_resamples:,} bootstrap "
        f"resamples, commit `{meta.git.sha[:12]}`{' (dirty tree)' if meta.git.dirty else ''}. "
        "The LLM blurb and the LLM clean were not tested (proposal 005, owner decision)."
    )
    return (
        "\n\n".join(
            [
                decision_table([ab.comparison]),
                ab.note,
                config_table([ab.without_header, ab.with_header], budget),
                truncation,
                footer,
            ]
        )
        + "\n"
    )


def _count(outcomes: Sequence[str], name: str) -> int:
    return sum(1 for o in outcomes if o == name)


def _verify_version(result: GenerationResults) -> str:
    version = result.verify_prompt_version
    return f", prompt `verify` version {version}" if version else ""


def _with_interval(successes: int, n: int) -> str:
    """`k/n` and its 95% Wilson interval (n = 0 has no interval)."""
    if n == 0:
        return f"{successes}/{n}"
    low, high = wilson_interval(successes, n)
    return f"{successes}/{n} [{low:.2f}, {high:.2f}]"


def _row(result: GenerationResults, name: str) -> Row:
    return next(r for r in result.rows if r.name == name)


def render_stage3(result: GenerationResults) -> list[str]:
    """The current pipeline against stage 3 added, on the same answers (proposal 007)."""
    s3 = result.stage3
    if s3 is None:
        return []
    current = _row(result, "stages 1+2 (current pipeline)").metrics
    with3 = _row(result, "stages 1+2+3").metrics
    v = result.validity
    shown = sum(o.claims for o in result.queries if o.outcome.value == "answered")
    answered = sum(1 for o in result.queries if o.outcome.value == "answered")
    n = len(result.queries)
    cost_now = sum(o.cost_usd for o in result.queries)
    cost_s3 = cost_now + s3.cost_usd
    without, with_s3 = s3.answer_path_latency_without, s3.answer_path_latency_with

    def latency(stat: LatencyStat | None) -> str:
        return "n/a" if stat is None else f"{stat.p50:.2f} / {stat.p95:.2f}"

    side = _grid(
        ("Measure", "stages 1+2 (current)", "stages 1+2+3"),
        [
            [
                "answerable queries refused (95% Wilson)",
                _with_interval(current.false_refusals, current.answerable),
                _with_interval(with3.false_refusals, with3.answerable),
            ],
            [
                "unanswerable queries refused (95% Wilson)",
                _with_interval(current.caught, current.unanswerable),
                _with_interval(with3.caught, with3.unanswerable),
            ],
            [
                "answered answerable queries citing a gold chunk",
                f"{v.answered_citing_a_relevant_chunk}/{v.answered_answerable}",
                f"{s3.answered_citing_a_relevant_chunk}/{s3.answered_answerable}",
            ],
            ["claims shown", str(shown), str(shown - s3.claims_dropped)],
            ["model cost USD (all queries)", f"{cost_now:.4f}", f"{cost_s3:.4f}"],
            ["model cost USD per query", f"{cost_now / n:.5f}", f"{cost_s3 / n:.5f}"],
            [
                "answer-path model latency p50 / p95, s",
                latency(without),
                latency(with_s3),
            ],
        ],
    )
    u = s3.usage
    tokens = f"{u.input_tokens} / {u.cached_tokens} / {u.output_tokens} / {u.reasoning_tokens}"
    detail = _grid(
        ("Stage 3", "Count"),
        [
            [
                "queries that reached stage 3 (answered by stages 1+2)",
                f"{s3.calls_queries}/{answered}",
            ],
            ["refused: cited notes do not answer the question", str(s3.refused_by_verifier)],
            ["refused: no supported claim left", str(s3.refused_unsupported)],
            ["refused: verifier output invalid after the retry", str(s3.refused_verifier_invalid)],
            ["verifier retried", str(s3.retried)],
            ["claims dropped / claims checked", f"{s3.claims_dropped}/{s3.claims_in}"],
            ["tokens: input / cached / output / reasoning", tokens],
            ["stage 3 cost USD", f"{s3.cost_usd:.4f}"],
        ],
    )
    return [side, detail]


def render_generation(result: GenerationResults) -> str:
    """The M6 report: outcomes per kind of query, citation validity, the three rows and the cost."""
    split = result.query_set
    outcome_rows = []
    for label, flag in (("answerable", False), ("unanswerable", True)):
        outs = [q.outcome.value for q in result.queries if q.unanswerable is flag]
        refused = len(outs) - _count(outs, "answered")
        outcome_rows.append(
            [
                label,
                str(len(outs)),
                str(_count(outs, "answered")),
                f"{refused}/{len(outs)}",
                str(_count(outs, "refused_stage1")),
                str(_count(outs, "refused_model")),
                str(_count(outs, "refused_invalid")),
            ]
        )
    v = result.validity
    validity = _grid(
        ("Measure", "Count"),
        [
            ["queries sent to the model", str(v.queries_to_model)],
            ["first attempt valid", f"{v.first_attempt_valid}/{v.queries_to_model}"],
            ["first attempt unparsable", str(v.first_attempt_unparsable)],
            ["retried", f"{v.retried}/{v.queries_to_model}"],
            [
                "uncited claims, first attempt",
                f"{v.uncited_claims_first_attempt}/{v.claims_first_attempt}",
            ],
            ["unknown citation ids, first attempt", str(v.unknown_citations_first_attempt)],
            [
                "answered answerable queries citing a gold-relevant chunk",
                f"{v.answered_citing_a_relevant_chunk}/{v.answered_answerable}",
            ],
        ],
    )
    rows = _grid(
        (
            "Row",
            "False refusals (answerable)",
            "Unanswerable refused",
            "Precision",
            "F1 (reference)",
        ),
        [
            [
                r.name,
                f"{r.metrics.false_refusals}/{r.metrics.answerable}",
                f"{r.metrics.caught}/{r.metrics.unanswerable}",
                "n/a" if r.metrics.precision is None else _f(r.metrics.precision),
                "n/a" if r.metrics.f1 is None else _f(r.metrics.f1),
            ]
            for r in result.rows
        ],
    )
    u = result.total_usage
    cost = (
        f"Tokens: input {u.input_tokens} (cached {u.cached_tokens}), output {u.output_tokens} "
        f"(reasoning {u.reasoning_tokens}). Total cost ${result.total_cost_usd:.4f}."
    )
    meta = result.meta
    footer = (
        f"{meta.queries} {split} queries, prompt `answer` version "
        f"{result.meta.prompt_versions['answer']}"
        f"{_verify_version(result)}, commit `{meta.git.sha[:12]}`"
        f"{' (dirty tree)' if meta.git.dirty else ''}. Stage-1 thresholds: cosine "
        f"{result.thresholds.cosine:.4f}, BM25 {result.thresholds.bm25:.4f}."
    )
    outcomes = _grid(
        ("Queries", "n", "Answered", "Refused", "by stage 1", "by the model", "invalid twice"),
        outcome_rows,
    )
    parts = [outcomes, validity, rows, cost]
    if result.stage3 is not None:
        parts.extend(render_stage3(result))
    return "\n\n".join([*parts, footer]) + "\n"


def _split_counts(result: GenerationResults) -> tuple[str, str]:
    answerable = [q for q in result.queries if not q.unanswerable]
    unanswerable = [q for q in result.queries if q.unanswerable]
    refused_a = sum(1 for q in answerable if q.outcome.value != "answered")
    refused_u = sum(1 for q in unanswerable if q.outcome.value != "answered")
    return f"{refused_a}/{len(answerable)}", f"{refused_u}/{len(unanswerable)}"


def render_generation_pair(first: GenerationResults, second: GenerationResults) -> str:
    """Two runs of the same split side by side (for example two answer prompts)."""
    columns = [f"prompt {r.meta.prompt_versions['answer']}" for r in (first, second)]

    def valid(r: GenerationResults) -> str:
        return f"{r.validity.first_attempt_valid}/{r.validity.queries_to_model}"

    def retried(r: GenerationResults) -> str:
        return f"{r.validity.retried}/{r.validity.queries_to_model}"

    def uncited(r: GenerationResults) -> str:
        return f"{r.validity.uncited_claims_first_attempt}/{r.validity.claims_first_attempt}"

    def gold_hit(r: GenerationResults) -> str:
        return f"{r.validity.answered_citing_a_relevant_chunk}/{r.validity.answered_answerable}"

    def tokens(r: GenerationResults) -> str:
        return f"{r.total_usage.input_tokens} / {r.total_usage.output_tokens}"

    measures: list[tuple[str, Callable[[GenerationResults], str]]] = [
        ("answerable queries refused", lambda r: _split_counts(r)[0]),
        ("unanswerable queries refused", lambda r: _split_counts(r)[1]),
        ("first attempt valid", valid),
        ("retried", retried),
        ("uncited claims, first attempt", uncited),
        ("gold-hit (answered answerable citing a gold chunk)", gold_hit),
        ("claims in all answers, first attempt", lambda r: str(r.validity.claims_first_attempt)),
        ("input / output tokens", tokens),
        ("cost USD", lambda r: f"{r.total_cost_usd:.4f}"),
    ]
    rows = [[label, *(fn(r) for r in (first, second))] for label, fn in measures]
    meta = first.meta
    footer = (
        f"{meta.queries} {first.query_set} queries, same retrieval and thresholds; commits "
        f"`{first.meta.git.sha[:12]}` and `{second.meta.git.sha[:12]}`."
    )
    return _grid(("Measure", *columns), rows) + "\n\n" + footer + "\n"
