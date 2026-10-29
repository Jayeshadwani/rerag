"""Render a results JSON file as Markdown tables (CLAUDE.md: numbers come from results/).

Renders `corpus_stats.json`, `chunk_stats.json` and `evidence_rule_eval.json`; later milestones
add their result files.
Every number printed is read from the file (or a ratio of two numbers in it); none is typed
by hand. Usage: `python scripts/render_results.py results/corpus_stats.json`.
"""

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from pydantic import BaseModel, ValidationError

from noterecall.chunking.report import ChunkStats
from noterecall.ingest.stats import CorpusStats, Distribution, GitState


class Interval(BaseModel):
    """A confidence interval, as stored in the evidence-rule record."""

    low: float
    high: float


class EvidenceEval(BaseModel):
    """The summary of the evidence-rule evaluation (a closed negative result; code removed)."""

    seed: int
    rule_fingerprint: str
    population_kept: int
    population_dropped: int
    sampled_kept: int
    sampled_dropped: int
    labelled_evidence_in_kept: int
    labelled_evidence_in_dropped: int
    precision: float | None
    precision_wilson95: Interval | None
    recall_weighted: float | None
    evidence_prevalence_weighted: float | None


class EvidenceReport(BaseModel):
    """`results/evidence_rule_eval.json`: only the parts the table needs are validated here."""

    git: GitState
    summary: EvidenceEval


def _table(header: tuple[str, str], rows: list[tuple[str, str]]) -> str:
    lines = [f"| {header[0]} | {header[1]} |", "|---|---:|"]
    lines.extend(f"| {name} | {value} |" for name, value in rows)
    return "\n".join(lines)


def _n(value: int) -> str:
    return f"{value:,}"


def render_corpus_stats(stats: CorpusStats) -> str:
    """Markdown for one corpus stats report."""
    tokens = stats.tokens
    removed_pct = 100 * (tokens.raw - tokens.clean) / tokens.raw if tokens.raw else 0.0
    corpus = [
        ("Notebooks", _n(stats.notebooks)),
        ("Cells before cleaning", _n(stats.cells_raw)),
        ("Cells kept: markdown", _n(stats.cells_kept["markdown"])),
        ("Cells kept: code", _n(stats.cells_kept["code"])),
        (f"Tokens before cleaning ({tokens.tokenizer})", _n(tokens.raw)),
        (f"Tokens after cleaning ({tokens.tokenizer})", _n(tokens.clean)),
        ("Tokens removed", f"{removed_pct:.1f}%"),
    ]
    dist = [
        ("Cells per notebook (min / median / max)", _dist(stats.cells_per_notebook)),
        ("Tokens per notebook (min / median / max)", _dist(stats.clean_tokens_per_notebook)),
    ]
    dups = [(k.replace("_", " "), _n(v)) for k, v in stats.duplicate_extra_copies.items()]
    removed = [(k, _n(v)) for k, v in stats.removed_by_rule.items()]
    secrets = [(k, _n(v)) for k, v in stats.secrets_scrubbed_by_type.items()]
    sections = [
        _table(("Corpus", "Value"), corpus),
        _table(("Per notebook", "Value"), dist),
        _table(("Removed by rule", "Count"), removed),
        _table(("Duplicate extra copies", "Count"), dups),
        _table(("Secrets scrubbed by type", "Count"), secrets or [("(none found)", "0")]),
        f"Commit `{stats.git.sha[:12]}`{' (dirty tree)' if stats.git.dirty else ''}, "
        f"pipeline `{stats.pipeline_hash[:12]}`, rules `{stats.rules_sha256[:12]}`.",
    ]
    return "\n\n".join(sections) + "\n"


def _dist(d: Distribution) -> str:
    low, median, high = d.min, d.median, d.max
    return f"{_n(low)} / {median:,.1f} / {_n(high)}"


def _grid(header: tuple[str, ...], rows: Sequence[tuple[str, ...]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{100 * value:.1f}%"


def render_chunk_stats(stats: ChunkStats) -> str:
    """Markdown for one chunk stats report."""
    rows: list[tuple[str, ...]] = []
    for r in stats.rows:
        flagged = ", ".join(f"{k} {_n(v)}" for k, v in r.flagged_by_reason.items() if v) or "none"
        spread = (
            f"{_n(r.tokens.min)} / {r.tokens.median:,.1f} / {_n(r.tokens.p90)} / {_n(r.tokens.max)}"
        )
        rows.append(
            (
                r.chunker,
                _n(r.cap),
                _n(r.chunks),
                spread,
                _n(r.under_quarter_cap),
                flagged,
                _n(r.cells_split),
                _n(r.unparsed_code_cells),
            )
        )
    header = (
        "Chunker",
        "Cap",
        "Chunks",
        "Tokens min / median / p90 / max",
        "Under cap/4",
        "Flagged oversize (by reason)",
        "Cells split",
        "Unparsed code cells",
    )
    return (
        "\n\n".join(
            [
                _grid(header, rows),
                f"{_n(stats.notebooks)} notebooks, {_n(stats.cells)} cells. "
                f"Commit `{stats.git.sha[:12]}`{' (dirty tree)' if stats.git.dirty else ''}.",
            ]
        )
        + "\n"
    )


def render_evidence_eval(result: EvidenceEval) -> str:
    """Markdown for one evidence-rule evaluation."""
    interval = result.precision_wilson95
    precision = _pct(result.precision) + (
        f" (95% Wilson {_pct(interval.low)} to {_pct(interval.high)})" if interval else ""
    )
    rows = [
        (
            "Outputs the rule keeps / drops (population)",
            f"{_n(result.population_kept)} / {_n(result.population_dropped)}",
        ),
        ("Sampled from each", f"{_n(result.sampled_kept)} / {_n(result.sampled_dropped)}"),
        ("Labelled evidence among sampled kept", _n(result.labelled_evidence_in_kept)),
        ("Labelled evidence among sampled dropped", _n(result.labelled_evidence_in_dropped)),
        ("Precision (kept that are evidence)", precision),
        (
            "Recall (evidence that is kept, weighted to the population)",
            _pct(result.recall_weighted),
        ),
        ("Evidence prevalence (weighted)", _pct(result.evidence_prevalence_weighted)),
    ]
    note = f"Sampling seed {result.seed}, rule `{result.rule_fingerprint[:12]}`."
    return _table(("Evidence rule vs owner labels", "Value"), rows) + "\n\n" + note + "\n"


def load_results(path: Path) -> CorpusStats | ChunkStats | EvidenceReport:
    """Read a results file and validate it as the report type its keys identify."""
    text = path.read_text(encoding="utf-8")
    keys = json.loads(text)
    if "rows" in keys:
        return ChunkStats.model_validate_json(text)
    if "summary" in keys:
        return EvidenceReport.model_validate_json(text)
    return CorpusStats.model_validate_json(text)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point: prints Markdown to stdout; exit 1 on a missing or invalid file."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n", maxsplit=1)[0])
    parser.add_argument("path", type=Path)
    args = parser.parse_args(argv)
    try:
        report = load_results(args.path)
    except (OSError, ValidationError, json.JSONDecodeError) as exc:
        print(f"error: cannot render {args.path}: {exc}", file=sys.stderr)
        return 1
    if isinstance(report, ChunkStats):
        sys.stdout.write(render_chunk_stats(report))
    elif isinstance(report, EvidenceReport):
        sys.stdout.write(render_evidence_eval(report.summary))
    else:
        sys.stdout.write(render_corpus_stats(report))
    return 0


if __name__ == "__main__":
    sys.exit(main())
