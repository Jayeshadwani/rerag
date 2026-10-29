"""Render a results JSON file as Markdown tables (CLAUDE.md: numbers come from results/).

Currently renders `results/corpus_stats.json` only; later milestones add their result files.
Every number printed is read from the file (or a ratio of two numbers in it); none is typed
by hand. Usage: `python scripts/render_results.py results/corpus_stats.json`.
"""

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from pydantic import ValidationError

from noterecall.ingest.stats import CorpusStats, Distribution


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


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point: prints Markdown to stdout; exit 1 on a missing or invalid file."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n", maxsplit=1)[0])
    parser.add_argument("path", type=Path)
    args = parser.parse_args(argv)
    try:
        stats = CorpusStats.model_validate_json(args.path.read_text(encoding="utf-8"))
    except (OSError, ValidationError) as exc:
        print(f"error: cannot render {args.path}: {exc}", file=sys.stderr)
        return 1
    sys.stdout.write(render_corpus_stats(stats))
    return 0


if __name__ == "__main__":
    sys.exit(main())
