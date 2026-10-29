"""Command line interface. Infrastructure; no ADR.

`noterecall sync` brings the database in line with the raw notebook directory (ADR-003,
ADR-015). `noterecall stats` writes the aggregate corpus report to
`results/corpus_stats.json`. Known failures print one line to stderr and exit with code 1;
unexpected errors are not caught.
"""

import sqlite3
import sys
from pathlib import Path
from typing import Annotated

import structlog
import tiktoken
import typer

from noterecall.chunking.evidence import judge_output, rule_fingerprint
from noterecall.chunking.evidence_check import (
    EvidenceCheckError,
    EvidenceReport,
    OutputRef,
    draw_sample,
    evaluate,
    read_labels,
    read_manifest,
    render_sample,
    score_items,
    write_manifest,
)
from noterecall.chunking.report import compute_chunk_stats
from noterecall.chunking.tokens import TokenizerError, load_tokenizer
from noterecall.config import ConfigError, Settings, load_settings
from noterecall.index.store import StoreError, connect, notebook_cells
from noterecall.ingest.clean import RulesError, load_rules
from noterecall.ingest.ipynb import IpynbLoader, NotebookParseError
from noterecall.ingest.stats import (
    Provenance,
    StatsError,
    compute_stats,
    git_state,
    write_stats,
)
from noterecall.ingest.sync import (
    Pipeline,
    SyncError,
    compute_pipeline_hash,
    file_sha256,
)
from noterecall.ingest.sync import (
    sync as run_sync,
)

KNOWN_ERRORS = (
    ConfigError,
    EvidenceCheckError,
    NotebookParseError,
    RulesError,
    StatsError,
    StoreError,
    SyncError,
    TokenizerError,
)

app = typer.Typer(no_args_is_help=True, add_completion=False, help="NoteRecall command line.")

ConfigOption = Annotated[
    Path | None,
    typer.Option("--config", help="Config TOML to use instead of config/default.toml."),
]


def configure_logging(level: str) -> None:
    """Key-value logs to stderr. No exception renderer that prints local variables."""
    structlog.configure(
        processors=[
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.KeyValueRenderer(key_order=["timestamp", "level", "event"]),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        logger_factory=structlog.PrintLoggerFactory(file=sys.stderr),
        cache_logger_on_first_use=False,
    )


def _settings(config: Path | None) -> Settings:
    settings = load_settings(config)
    configure_logging(settings.logging.level)
    return settings


def _fail(exc: Exception) -> typer.Exit:
    typer.echo(f"error: {exc}", err=True)
    return typer.Exit(code=1)


def _pipeline_hash(settings: Settings) -> str:
    return compute_pipeline_hash(
        settings.cleaning.rules_path,
        settings.cleaning.max_output_lines,
        settings.tokenizer.encoding,
    )


def _sync(settings: Settings) -> str:
    tokenizer = load_tokenizer(settings.tokenizer.encoding)
    pipeline = Pipeline(
        rules=load_rules(settings.cleaning.rules_path),
        max_output_lines=settings.cleaning.max_output_lines,
        tokenizer=tokenizer,
        pipeline_hash=_pipeline_hash(settings),
    )
    conn: sqlite3.Connection = connect(settings.paths.db_path)
    try:
        report = run_sync(settings.paths.raw_dir, conn, IpynbLoader(), pipeline)
    finally:
        conn.close()
    return (
        f"sync: {report.added} added, {report.updated} updated, "
        f"{report.unchanged} unchanged, {report.removed} removed"
    )


@app.command()
def sync(config: ConfigOption = None) -> None:
    """Sync the raw notebook directory into the database (incremental)."""
    try:
        typer.echo(_sync(_settings(config)))
    except KNOWN_ERRORS as exc:
        raise _fail(exc) from exc


def _stats(settings: Settings, out: Path | None) -> str:
    provenance = Provenance(
        pipeline_hash=_pipeline_hash(settings),
        rules_sha256=file_sha256(settings.cleaning.rules_path),
        tokenizer_name=settings.tokenizer.encoding,
        git=git_state(),
    )
    target = out if out is not None else settings.paths.results_dir / "corpus_stats.json"
    conn = connect(settings.paths.db_path)
    try:
        write_stats(compute_stats(conn, settings, provenance), target)
    finally:
        conn.close()
    return f"stats: wrote {target}"


@app.command()
def stats(
    config: ConfigOption = None,
    out: Annotated[Path | None, typer.Option("--out", help="Output JSON path.")] = None,
) -> None:
    """Write the aggregate corpus report (no note text, no file names)."""
    try:
        typer.echo(_stats(_settings(config), out))
    except KNOWN_ERRORS as exc:
        raise _fail(exc) from exc


def _provenance(settings: Settings) -> Provenance:
    return Provenance(
        pipeline_hash=_pipeline_hash(settings),
        rules_sha256=file_sha256(settings.cleaning.rules_path),
        tokenizer_name=settings.tokenizer.encoding,
        git=git_state(),
    )


def _chunk_stats(settings: Settings, out: Path | None) -> str:
    tokenizer = load_tokenizer(settings.tokenizer.encoding)
    target = out if out is not None else settings.paths.results_dir / "chunk_stats.json"
    conn = connect(settings.paths.db_path)
    try:
        stats = compute_chunk_stats(conn, settings, _provenance(settings), tokenizer)
    finally:
        conn.close()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(stats.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return f"chunk-stats: wrote {target}"


@app.command("chunk-stats")
def chunk_stats(
    config: ConfigOption = None,
    out: Annotated[Path | None, typer.Option("--out", help="Output JSON path.")] = None,
) -> None:
    """Write aggregate chunk statistics for every candidate cap (no note text, no file names)."""
    try:
        typer.echo(_chunk_stats(_settings(config), out))
    except KNOWN_ERRORS as exc:
        raise _fail(exc) from exc


def _evidence_dir(settings: Settings) -> Path:
    """Private working files live beside the database, under the git-ignored data directory."""
    return settings.paths.db_path.parent / "evidence"


def _evidence_sample(settings: Settings, n: int, seed: int) -> str:
    tokenizer = load_tokenizer(settings.tokenizer.encoding)
    conn = connect(settings.paths.db_path)
    try:
        notebooks = list(notebook_cells(conn))
    finally:
        conn.close()
    texts = {
        (path, cell.index): cell.output_text
        for path, cells in notebooks
        for cell in cells
        if cell.output_text.strip()
    }
    refs = [
        OutputRef(notebook_path=p, cell_index=i, verdict=judge_output(t, tokenizer))
        for (p, i), t in texts.items()
    ]
    if not refs:
        raise EvidenceCheckError(
            "the database has no code-cell outputs to sample; run `sync` first"
        )
    manifest = draw_sample(refs, n // 2, seed, rule_fingerprint())
    write_manifest(manifest, _evidence_dir(settings) / "sample.json")
    typer.echo(
        f"sample of {len(manifest.items)} drawn blind from {len(refs)} outputs "
        f"(seed {seed}); labels go in {_evidence_dir(settings) / 'labels.json'}",
        err=True,
    )
    return render_sample(manifest, texts)


@app.command("evidence-sample")
def evidence_sample(
    config: ConfigOption = None,
    n: Annotated[int, typer.Option("--n", min=2, help="Sample size, split in two strata.")] = 50,
    seed: Annotated[int, typer.Option("--seed", help="Sampling seed.")] = 20261029,
) -> None:
    """Draw the blind validation sample of outputs and print it (for the owner to label)."""
    try:
        typer.echo(_evidence_sample(_settings(config), n, seed), nl=False)
    except KNOWN_ERRORS as exc:
        raise _fail(exc) from exc


def _evidence_eval(settings: Settings, labels: Path | None, out: Path | None) -> str:
    directory = _evidence_dir(settings)
    manifest = read_manifest(directory / "sample.json")
    if manifest.rule_fingerprint != rule_fingerprint():
        raise EvidenceCheckError(
            "the evidence rule changed after this sample was drawn; its labels no longer apply"
        )
    scored = read_labels(labels or directory / "labels.json", manifest)
    tokenizer = load_tokenizer(settings.tokenizer.encoding)
    conn = connect(settings.paths.db_path)
    try:
        texts = {
            (path, cell.index): cell.output_text
            for path, cells in notebook_cells(conn)
            for cell in cells
        }
    finally:
        conn.close()
    verdicts = {
        i.sample_id: judge_output(texts[(i.notebook_path, i.cell_index)], tokenizer)
        for i in manifest.items
    }
    report = EvidenceReport(
        schema_version=2,
        git=git_state(),
        config=settings.model_dump(mode="json"),
        tokenizer=settings.tokenizer.encoding,
        tiktoken_version=tiktoken.__version__,
        summary=evaluate(manifest, scored),
        items=score_items(manifest, scored, verdicts),
    )
    target = out if out is not None else settings.paths.results_dir / "evidence_rule_eval.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(report.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return f"evidence-eval: wrote {target}"


@app.command("evidence-eval")
def evidence_eval(
    config: ConfigOption = None,
    labels: Annotated[Path | None, typer.Option("--labels", help="Labels JSON.")] = None,
    out: Annotated[Path | None, typer.Option("--out", help="Output JSON path.")] = None,
) -> None:
    """Report the evidence rule's precision and recall against the owner's labels."""
    try:
        typer.echo(_evidence_eval(_settings(config), labels, out))
    except KNOWN_ERRORS as exc:
        raise _fail(exc) from exc
