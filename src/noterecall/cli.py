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
import typer

from noterecall.chunking.report import compute_chunk_stats
from noterecall.chunking.tokens import TokenizerError, load_tokenizer
from noterecall.config import ConfigError, Settings, load_settings
from noterecall.eval.gold import (
    CellRef,
    GoldError,
    GoldQuery,
    GoldSet,
    Issue,
    QueryType,
    freeze_gold,
    load_gold,
    next_query_id,
    relevant_keys,
    save_gold,
    summarize,
    validate_gold,
)
from noterecall.eval.labeling import Entry, QuerySession, build_entries
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
    GoldError,
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


gold_app = typer.Typer(no_args_is_help=True, help="Inspect, validate and freeze the gold set.")
app.add_typer(gold_app, name="gold")

_EMPTY_DB = "the database has no notebooks; run `noterecall sync` first"

GoldPathOption = Annotated[
    Path | None,
    typer.Option("--gold-path", help="Gold file to use instead of paths.gold_path."),
]


def _gold_path(settings: Settings, override: Path | None) -> Path:
    return override if override is not None else settings.paths.gold_path


def _notebook_hashes(settings: Settings) -> dict[str, frozenset[str]]:
    conn = connect(settings.paths.db_path)
    try:
        hashes = {p: frozenset(c.raw_hash for c in cells) for p, cells in notebook_cells(conn)}
    finally:
        conn.close()
    if not hashes:
        raise GoldError(_EMPTY_DB)
    return hashes


def _issue_line(issue: Issue) -> str:
    where = f" {issue.query_id}:" if issue.query_id else ""
    return f"{issue.level:7} [{issue.code}]{where} {issue.message}"


def _report(issues: list[Issue]) -> int:
    """Print the findings and a summary; returns the number of errors."""
    for issue in issues:
        typer.echo(_issue_line(issue))
    errors = sum(1 for i in issues if i.level == "error")
    typer.echo(f"{errors} error(s), {len(issues) - errors} warning(s)")
    return errors


def _gold_validate(settings: Settings, path: Path, ready: bool) -> int:
    gold = load_gold(path)
    return _report(validate_gold(gold, _notebook_hashes(settings), require_complete=ready))


@gold_app.command("validate")
def gold_validate(
    config: ConfigOption = None,
    gold_path: GoldPathOption = None,
    ready: Annotated[bool, typer.Option("--ready", help="Also require the ADR-009 size.")] = False,
) -> None:
    """Report labels whose cells no longer exist, and other problems (ids and counts only)."""
    try:
        settings = _settings(config)
        errors = _gold_validate(settings, _gold_path(settings, gold_path), ready)
    except KNOWN_ERRORS as exc:
        raise _fail(exc) from exc
    if errors:
        raise typer.Exit(code=1)


def _gold_freeze(settings: Settings, path: Path) -> str:
    gold = load_gold(path)
    issues = validate_gold(gold, _notebook_hashes(settings), require_complete=True)
    if any(i.level == "error" for i in issues):
        _report(issues)
        raise GoldError("the gold set is not ready to freeze; fix the errors above")
    frozen = freeze_gold(gold)
    save_gold(frozen, path)
    split = summarize(frozen).by_split
    return f"frozen: {split.get('dev', 0)} dev, {split.get('test', 0)} test"


@gold_app.command("freeze")
def gold_freeze(config: ConfigOption = None, gold_path: GoldPathOption = None) -> None:
    """Assign the 70/30 dev/test split, once."""
    try:
        settings = _settings(config)
        typer.echo(_gold_freeze(settings, _gold_path(settings, gold_path)))
    except KNOWN_ERRORS as exc:
        raise _fail(exc) from exc


def _gold_stats(path: Path) -> str:
    summary = summarize(load_gold(path))
    lines = [f"queries: {summary.queries} ({'frozen' if summary.frozen else 'not frozen'})"]
    for qtype, row in sorted(summary.by_type_and_split.items()):
        lines.append(
            f"  {qtype}: {sum(row.values())} "
            + ", ".join(f"{k} {v}" for k, v in sorted(row.items()))
        )
    if summary.labels_per_answerable_query:
        low, median, high = summary.labels_per_answerable_query
        lines.append(f"labels per answerable query (min/median/max): {low} / {median:.1f} / {high}")
    return "\n".join(lines)


@gold_app.command("stats")
def gold_stats(config: ConfigOption = None, gold_path: GoldPathOption = None) -> None:
    """Counts by type and split (no query text)."""
    try:
        settings = _settings(config)
        typer.echo(_gold_stats(_gold_path(settings, gold_path)))
    except KNOWN_ERRORS as exc:
        raise _fail(exc) from exc


def _ask_type() -> QueryType:
    types = list(QueryType)
    for number, qtype in enumerate(types, start=1):
        typer.echo(f"  {number}. {qtype.value}")
    while True:
        answer = typer.prompt("type (number)").strip()
        if answer.isdigit() and 1 <= int(answer) <= len(types):
            return types[int(answer) - 1]
        typer.echo(f"enter a number from 1 to {len(types)}")


def _run_session(session: QuerySession) -> bool:
    """Drive one dialogue to the end; True if the owner saved it."""
    typer.echo(session.handle("/help"))
    while session.status == "open":
        typer.echo(session.handle(typer.prompt("label", prompt_suffix=" > ")))
    return session.status == "done"


def _labelled(base: GoldQuery, session: QuerySession) -> GoldQuery:
    refs = tuple(CellRef(notebook_path=p, raw_hash=h) for p, h in sorted(session.marked))
    return base.model_copy(update={"relevant": refs})


def _label_one(gold: GoldSet, entries: list[Entry], path: Path) -> GoldSet:
    text = typer.prompt("query text").strip()
    qtype = _ask_type()
    session = QuerySession(entries, text, qtype is QueryType.UNANSWERABLE)
    if not _run_session(session):
        return gold
    new = _labelled(GoldQuery(id=next_query_id(gold), text=text, type=qtype), session)
    updated = gold.model_copy(update={"queries": (*gold.queries, new)})
    save_gold(updated, path)
    typer.echo(f"saved {new.id}")
    return updated


def _edit_one(gold: GoldSet, query_id: str, entries: list[Entry], path: Path) -> None:
    """Redo the labels of an existing query (allowed after freezing; text, type, split stay)."""
    matches = [q for q in gold.queries if q.id == query_id]
    if not matches:
        raise GoldError(f"no query with id {query_id} in {path}")
    query = matches[0]
    session = QuerySession(
        entries, query.text, query.type is QueryType.UNANSWERABLE, sorted(relevant_keys(query))
    )
    if not _run_session(session):
        return
    queries = tuple(_labelled(q, session) if q.id == query_id else q for q in gold.queries)
    save_gold(gold.model_copy(update={"queries": queries}), path)
    typer.echo(f"saved {query_id}")


def _label(settings: Settings, path: Path, edit: str | None) -> None:
    conn = connect(settings.paths.db_path)
    try:
        entries = build_entries(list(notebook_cells(conn)))
    finally:
        conn.close()
    if not entries:
        raise GoldError(_EMPTY_DB)
    gold = load_gold(path) if path.exists() else GoldSet()
    if edit is not None:
        _edit_one(gold, edit, entries, path)
        return
    if gold.frozen:
        raise GoldError("the gold set is frozen: queries cannot be added; use --edit to fix labels")
    while True:
        gold = _label_one(gold, entries, path)
        if not typer.confirm("another query?", default=True):
            return


@app.command()
def label(
    config: ConfigOption = None,
    gold_path: GoldPathOption = None,
    edit: Annotated[
        str | None, typer.Option("--edit", help="Redo the labels of this query id.")
    ] = None,
) -> None:
    """Write gold queries and mark their relevant cells (you decide relevance, not the tool)."""
    try:
        settings = _settings(config)
        _label(settings, _gold_path(settings, gold_path), edit)
    except KNOWN_ERRORS as exc:
        raise _fail(exc) from exc
