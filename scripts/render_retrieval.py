"""Markdown tables for the M4 retrieval results (dev ablation and the single test run).

Every number is read from the results file; nothing is typed by hand. Used by
`scripts/render_results.py`.
"""

from collections.abc import Sequence

from noterecall.eval.ablation import Comparison, ConfigResult
from noterecall.eval.run_retrieval import DevResults, TestResults


def _grid(header: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def _f(value: float) -> str:
    return f"{value:.3f}"


def _ms(value: float) -> str:
    return f"{value:,.1f}"


def config_rows(configs: Sequence[ConfigResult], budget_metric: str) -> list[list[str]]:
    """One row per configuration with the headline metrics and total latency."""
    return [
        [
            c.name,
            _f(c.means["recall@5"]),
            _f(c.means["recall@10"]),
            _f(c.means["mrr@10"]),
            _f(c.means["ndcg@10"]),
            _f(c.means[budget_metric]),
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
        "Total ms p50 / p95",
    )
    return _grid(header, config_rows(configs, budget_metric))


def comparison_rows(comparisons: Sequence[Comparison]) -> list[list[str]]:
    """One row per ADR-010 comparison."""
    rows = []
    for c in comparisons:
        decision = ("ADOPT" if c.decision.adopt else "reject") + f": {c.decision.reason}"
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
    return (
        "\n\n".join(
            [
                decision_table(comparisons),
                *notes,
                decided,
                config_table(result.ablation.configs, budget),
                "Decided pipeline at each chunk cap (report only, no decision):",
                config_table(result.sensitivity, budget),
                truncation,
                footer,
            ]
        )
        + "\n"
    )


def render_test(result: TestResults) -> str:
    """The single test-split run: decided pipeline vs the BM25 baseline."""
    budget = f"recall@tokens={result.meta.config['retrieval']['budget_tokens']}"  # type: ignore[index]
    meta = result.meta
    footer = (
        f"{meta.queries} test queries, milestone {result.milestone}, seed {meta.seed}, commit "
        f"`{meta.git.sha[:12]}`{' (dirty tree)' if meta.git.dirty else ''}."
    )
    return (
        "\n\n".join(
            [
                config_table([result.chosen, result.reference], budget),
                decision_table([result.comparison]),
                footer,
            ]
        )
        + "\n"
    )
