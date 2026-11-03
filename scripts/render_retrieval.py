"""Markdown tables for the M4 retrieval results (dev ablation and the single test run).

Every number is read from the results file; nothing is typed by hand. Used by
`scripts/render_results.py`.
"""

from collections.abc import Sequence

from noterecall.eval.ablation import Comparison, ConfigResult
from noterecall.eval.run_header import HeaderDevResults
from noterecall.eval.run_retrieval import DevResults, TestResults


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
