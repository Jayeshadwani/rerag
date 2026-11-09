"""Command line interface. Infrastructure; no ADR.

`noterecall sync` brings the database in line with the raw notebook directory (ADR-003,
ADR-015). `noterecall stats` writes the aggregate corpus report to
`results/corpus_stats.json`. Known failures print one line to stderr and exit with code 1;
unexpected errors are not caught.
"""

import importlib.metadata
import sqlite3
import sys
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Annotated

import structlog
import typer

from noterecall.chunking.report import compute_chunk_stats
from noterecall.chunking.tokens import TokenizerError, load_tokenizer
from noterecall.config import ConfigError, DecidedConfig, PriceConfig, Settings, load_settings
from noterecall.eval.gold import (
    CellRef,
    GoldError,
    GoldQuery,
    GoldSet,
    Issue,
    QueryType,
    Split,
    freeze_gold,
    load_gold,
    next_query_id,
    relevant_keys,
    save_gold,
    summarize,
    validate_abstention,
    validate_gold,
)
from noterecall.eval.labeling import Entry, QuerySession, build_entries
from noterecall.eval.ledger import (
    LedgerEntry,
    LedgerError,
    check_test_allowed,
    read_ledger,
    record_run,
)
from noterecall.eval.onnx_parity import run_parity
from noterecall.eval.run_generation import (
    GenerationResults,
    Subset,
    decided_pipeline,
    gold_subset,
    run_calibration,
    run_generation,
    select_all,
    set_subset,
)
from noterecall.eval.run_header import HeaderDevResults, run_header_dev
from noterecall.eval.run_retrieval import (
    DevResults,
    Resources,
    RunContext,
    RunError,
    run_dev,
    run_test,
)
from noterecall.generate.abstain import Choice, Thresholds
from noterecall.generate.ask import Asker
from noterecall.generate.commands import (
    format_answer,
    format_context,
    format_plan,
    plan_generation,
)
from noterecall.generate.runtime import (
    NoCalls,
    build_client,
    build_generator,
    model_tokenizer,
    spent_so_far,
)
from noterecall.generate.smoke import (
    SMOKE_CAP_USD,
    choose_step_b,
    format_rows,
    run_step_a,
    smoke_row,
)
from noterecall.index.embeddings import EmbeddingService, EmbeddingSpec
from noterecall.index.onnx_embedder import fetch_onnx_files, load_onnx_embedder
from noterecall.index.quantize import QuantizeError, quantize_int8
from noterecall.index.st_models import load_cross_encoder, load_embedder
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
from noterecall.llm.cache import ResponseCache, cache_key
from noterecall.llm.client import LLMError, Prompt
from noterecall.retrieval.pipeline import Depths
from noterecall.serve.artifacts import runtime_label
from noterecall.serve.build import build_index

KNOWN_ERRORS = (
    QuantizeError,
    LLMError,
    ConfigError,
    GoldError,
    LedgerError,
    RunError,
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


def _gold_validate(settings: Settings, path: Path, ready: bool, kind: str = "gold") -> int:
    gold = load_gold(path)
    if kind == "abstention":
        reference_path = settings.paths.gold_path
        reference = load_gold(reference_path) if reference_path.exists() else None
        return _report(validate_abstention(gold, _notebook_hashes(settings), reference))
    return _report(validate_gold(gold, _notebook_hashes(settings), require_complete=ready))


@gold_app.command("validate")
def gold_validate(
    config: ConfigOption = None,
    gold_path: GoldPathOption = None,
    ready: Annotated[bool, typer.Option("--ready", help="Also require the ADR-009 size.")] = False,
    kind: Annotated[
        str, typer.Option("--kind", help="gold (default) or abstention (the report-only set).")
    ] = "gold",
) -> None:
    """Report labels whose cells no longer exist, and other problems (ids and counts only)."""
    try:
        settings = _settings(config)
        if kind not in ("gold", "abstention"):
            raise GoldError("--kind must be gold or abstention")
        default = settings.paths.abstention_path if kind == "abstention" else None
        path = _gold_path(settings, gold_path if gold_path is not None else default)
        errors = _gold_validate(settings, path, ready, kind)
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


def _label_one(gold: GoldSet, entries: list[Entry], path: Path, prefix: str = "q") -> GoldSet:
    text = typer.prompt("query text").strip()
    qtype = _ask_type()
    session = QuerySession(entries, text, qtype is QueryType.UNANSWERABLE)
    if not _run_session(session):
        return gold
    new = _labelled(GoldQuery(id=next_query_id(gold, prefix), text=text, type=qtype), session)
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


def _label(settings: Settings, path: Path, edit: str | None, prefix: str = "q") -> None:
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
        gold = _label_one(gold, entries, path, prefix)
        if not typer.confirm("another query?", default=True):
            return


@app.command()
def label(
    config: ConfigOption = None,
    gold_path: GoldPathOption = None,
    edit: Annotated[
        str | None, typer.Option("--edit", help="Redo the labels of this query id.")
    ] = None,
    id_prefix: Annotated[
        str, typer.Option("--id-prefix", help="Prefix of new ids (a for the abstention set).")
    ] = "q",
) -> None:
    """Write gold queries and mark their relevant cells (you decide relevance, not the tool)."""
    try:
        settings = _settings(config)
        _label(settings, _gold_path(settings, gold_path), edit, id_prefix)
    except KNOWN_ERRORS as exc:
        raise _fail(exc) from exc


LIBRARIES = ("numpy", "bm25s", "tiktoken", "torch", "sentence-transformers", "transformers")


def _versions() -> dict[str, str]:
    return {name: importlib.metadata.version(name) for name in LIBRARIES}


def _resources(
    settings: Settings,
    conn: sqlite3.Connection,
    models: set[str] | None = None,
) -> Resources:
    """Corpus, embedding services (cached in the database) and the reranker, all pinned."""
    notebooks = list(notebook_cells(conn))
    if not notebooks:
        raise RunError("the database has no notebooks; run `noterecall sync` first")
    services = {
        spec.id: EmbeddingService(load_embedder(spec), spec, conn)
        for spec in settings.embeddings.candidates
        if models is None or spec.id in models
    }
    depths = Depths(
        candidate=settings.retrieval.candidate_depth, fused=settings.retrieval.fused_depth
    )
    reranker = load_cross_encoder(settings.rerank.model)
    return Resources(
        notebooks, load_tokenizer(settings.tokenizer.encoding), services, reranker, depths
    )


def _write_json(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text + "\n", encoding="utf-8")


def _eval_dev(settings: Settings, out: Path | None) -> str:
    gold = load_gold(settings.paths.gold_path)
    conn = connect(settings.paths.db_path)
    try:
        results = run_dev(
            settings,
            gold,
            _resources(settings, conn),
            RunContext(git=git_state(), versions=_versions()),
        )
    finally:
        conn.close()
    target = out if out is not None else settings.paths.results_dir / "retrieval_ablation_dev.json"
    _write_json(target, results.model_dump_json(indent=2))
    return f"eval: wrote {target}"


def _check_matches_dev(settings: Settings) -> None:
    """The committed decisions must be the ones the dev results file reached."""
    decided = settings.retrieval.decided
    if decided is None:
        raise RunError("config has no [retrieval.decided]; commit the dev decisions first")
    path = settings.paths.results_dir / "retrieval_ablation_dev.json"
    try:
        dev = DevResults.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise RunError(f"cannot read the dev results at {path}: {exc}") from exc
    m4 = decided.model_copy(update={"variant": dev.decided.variant})
    if dev.decided != m4:
        raise RunError("[retrieval.decided] differs from the decisions in the dev results file")
    if decided.variant.header:
        _check_matches_header_dev(settings, decided)


def _check_matches_header_dev(settings: Settings, decided: DecidedConfig) -> None:
    """With the header on, the header dev results must exist and name the same decisions."""
    path = settings.paths.results_dir / "header_ablation_dev.json"
    try:
        dev = HeaderDevResults.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise RunError(f"cannot read the header dev results at {path}: {exc}") from exc
    if dev.decided != decided:
        raise RunError("[retrieval.decided] differs from the decisions in the header dev results")


def _eval_test(settings: Settings, milestone: str, out: Path | None) -> str:
    gold = load_gold(settings.paths.gold_path)
    git = git_state()
    ledger_path = settings.paths.results_dir / "test_ledger.json"
    check_test_allowed(read_ledger(ledger_path), milestone, git.dirty, gold.frozen)
    _check_matches_dev(settings)
    decided = settings.retrieval.decided
    models = {decided.pipeline.dense_model} if decided and decided.pipeline.dense_model else set()
    conn = connect(settings.paths.db_path)
    try:
        resources = _resources(settings, conn, models)
        results = run_test(
            settings, gold, resources, RunContext(git=git, versions=_versions()), milestone
        )
    finally:
        conn.close()
    target = (
        out if out is not None else settings.paths.results_dir / f"retrieval_test_{milestone}.json"
    )
    _write_json(target, results.model_dump_json(indent=2))
    record_run(
        ledger_path,
        LedgerEntry(
            milestone=milestone,
            git_sha=git.sha,
            results_file=target.name,
            results_sha256=file_sha256(target),
        ),
    )
    return f"eval: wrote {target} and recorded the test run for {milestone}"


class Suite(StrEnum):
    """Which evaluation `noterecall eval` runs."""

    RETRIEVAL = "retrieval"
    HEADER = "header"
    ABSTENTION = "abstention"
    GENERATION = "generation"
    ONNX_PARITY = "onnx-parity"


class QuerySet(StrEnum):
    """Which queries `--suite generation` asks: the frozen gold set or the abstention set."""

    GOLD = "gold"
    ABSTENTION = "abstention"


def _eval_header_dev(settings: Settings, out: Path | None) -> str:
    decided = settings.retrieval.decided
    if decided is None:
        raise RunError("config has no [retrieval.decided]; the M4 decisions are the base of M5")
    gold = load_gold(settings.paths.gold_path)
    conn = connect(settings.paths.db_path)
    try:
        models = {decided.pipeline.dense_model} if decided.pipeline.dense_model else set()
        results = run_header_dev(
            settings,
            gold,
            _resources(settings, conn, {m for m in models if m}),
            RunContext(git=git_state(), versions=_versions()),
        )
    finally:
        conn.close()
    target = out if out is not None else settings.paths.results_dir / "header_ablation_dev.json"
    _write_json(target, results.model_dump_json(indent=2))
    return f"eval: wrote {target}"


def _eval_onnx_parity(settings: Settings, out: Path | None) -> str:
    """Torch against ONNX on the dev split (no LLM call): writes `results/onnx_parity_dev.json`."""
    gold = load_gold(settings.paths.gold_path)
    model = _dense_spec(settings)
    conn = connect(settings.paths.db_path)
    try:
        resources = _resources(settings, conn, {model.id})
        directory = _served_onnx_dir(settings, model)
        onnx_service = EmbeddingService(
            load_onnx_embedder(directory), model, conn, namespace=settings.api.onnx_subdir
        )
        results = run_parity(
            settings, gold, resources, onnx_service,
            RunContext(git=git_state(), versions=_versions()),
        )  # fmt: skip
    finally:
        conn.close()
    name = settings.api.onnx_subdir.replace("-", "_")
    target = out if out is not None else settings.paths.results_dir / f"{name}_parity_dev.json"
    results = results.model_copy(update={"variant": settings.api.onnx_subdir})
    _write_json(target, results.model_dump_json(indent=2))
    t, o = results.torch.means, results.onnx.means
    lines = [f"eval: wrote {target}", f"  decision: {results.decision}: {results.reason}"]
    lines += [
        f"  {m}: torch {t[m]:.4f}, onnx {o[m]:.4f}" for m in ("recall@10", "mrr@10", "ndcg@10")
    ]
    lines.append(f"  max |cosine difference|: {results.cosine.max_abs_cosine_difference:.6f}")
    return "\n".join(lines)


def _dense_spec(settings: Settings) -> EmbeddingSpec:
    decided = settings.retrieval.decided
    if decided is None or decided.pipeline.dense_model is None:
        raise RunError("config has no [retrieval.decided] with a dense model")
    return next(c for c in settings.embeddings.candidates if c.id == decided.pipeline.dense_model)


def _dense_models(settings: Settings) -> set[str]:
    decided = settings.retrieval.decided
    if decided is None:
        raise RunError("config has no [retrieval.decided]; M6 builds on the M4/M5 decisions")
    return {decided.pipeline.dense_model} if decided.pipeline.dense_model else set()


def _eval_abstention_dev(settings: Settings, out: Path | None) -> str:
    gold = load_gold(settings.paths.gold_path)
    conn = connect(settings.paths.db_path)
    try:
        results = run_calibration(
            settings,
            gold,
            _resources(settings, conn, _dense_models(settings)),
            RunContext(git=git_state(), versions=_versions()),
        )
    finally:
        conn.close()
    target = (
        out if out is not None else settings.paths.results_dir / "abstention_calibration_dev.json"
    )
    _write_json(target, results.model_dump_json(indent=2))
    cal = results.calibration
    lines = [f"eval: wrote {target}"]
    lines.append(_describe_choice("cosine + BM25", cal.cosine_bm25))
    lines.append(_describe_choice("cosine only", cal.cosine_only))
    return "\n".join(lines)


def _describe_choice(label: str, choice: Choice) -> str:
    t, m = choice.thresholds, choice.metrics
    return (
        f"  {label}: tau_cos {t.cosine:.4f}, tau_bm25 {t.bm25:.4f}: caught "
        f"{m.caught}/{m.unanswerable} unanswerable, "
        f"{m.false_refusals}/{m.answerable} false refusals"
    )


def _asker(settings: Settings, resources: Resources, cache: ResponseCache, real: bool) -> Asker:
    """An asker on the decided pipeline; `real` is False for estimates (no API client)."""
    thresholds = settings.generation.decided
    if thresholds is None:
        raise RunError("config has no [generation.decided]; run and commit the dev calibration")
    pipeline, cap = decided_pipeline(settings, resources)
    chunks = {c.id: c for c in resources.chunks(cap, "structural")}
    client = build_client(settings, cache) if real else NoCalls()
    return Asker(
        pipeline,
        chunks,
        Thresholds(cosine=thresholds.cosine, bm25=thresholds.bm25),
        build_generator(settings, client),
        settings.generation.max_chunks,
    )


@dataclass(frozen=True)
class GenerationRun:
    """What `eval --suite generation` was asked to do."""

    split: Split
    query_set: QuerySet
    milestone: str | None
    out: Path | None
    dry_run: bool
    confirmed: bool


def _generation_plan(
    settings: Settings, resources: Resources, cache: ResponseCache, run: GenerationRun
) -> str:
    """The dry-run text (no API client is built)."""
    subset = _subset(settings, run)
    asker = _asker(settings, resources, cache, real=False)
    counter = model_tokenizer(settings.llm.model).count
    effort = settings.llm.reasoning_effort

    def is_cached(prompt: Prompt) -> bool:
        return cache.get(cache_key(prompt, prompt.reasoning_effort or effort)) is not None

    plan = plan_generation(asker, subset.queries, settings, counter, is_cached)
    return format_plan(plan, spent_so_far(settings, cache), settings.llm.max_usd)


def _subset(settings: Settings, run: GenerationRun) -> Subset:
    if run.query_set is QuerySet.ABSTENTION:
        path = settings.paths.abstention_path
        if not path.exists():
            raise RunError(f"the abstention set was not found at {path}")
        return set_subset(load_gold(path), "abstention")
    return gold_subset(load_gold(settings.paths.gold_path), run.split)


def _save_generation(settings: Settings, results: GenerationResults, run: GenerationRun) -> str:
    stem = "generation_" + (
        "abstention" if run.query_set is QuerySet.ABSTENTION else results.query_set
    )
    if run.split is Split.TEST and run.query_set is QuerySet.GOLD:
        stem = f"generation_test_{run.milestone}"
    target = run.out if run.out is not None else settings.paths.results_dir / f"{stem}.json"
    _write_json(target, results.model_dump_json(indent=2))
    if run.split is Split.TEST and run.query_set is QuerySet.GOLD and run.milestone is not None:
        git = results.meta.git
        entry = LedgerEntry(
            milestone=run.milestone,
            git_sha=git.sha,
            results_file=target.name,
            results_sha256=file_sha256(target),
        )
        record_run(settings.paths.results_dir / "test_ledger.json", entry)
    return f"eval: wrote {target} (cost ${results.total_cost_usd:.4f})"


def _check_test_run(settings: Settings, run: GenerationRun) -> None:
    """The test split runs once per milestone from a clean tree on a frozen gold set."""
    if run.split is not Split.TEST or run.query_set is not QuerySet.GOLD or run.dry_run:
        return
    if run.milestone is None:
        raise RunError("--split test needs --milestone (the test split runs once per milestone)")
    gold = load_gold(settings.paths.gold_path)
    ledger = read_ledger(settings.paths.results_dir / "test_ledger.json")
    check_test_allowed(ledger, run.milestone, git_state().dirty, gold.frozen)


def _eval_generation(settings: Settings, run: GenerationRun) -> str:
    _check_test_run(settings, run)
    conn = connect(settings.paths.db_path)
    cache = ResponseCache(settings.llm.cache_path)
    try:
        resources = _resources(settings, conn, _dense_models(settings))
        if run.dry_run or not run.confirmed:
            plan = _generation_plan(settings, resources, cache, run)
            if run.dry_run:
                return plan
            raise RunError(f"{plan}\n\nno call was made: re-run with --yes to run this")
        generator = _asker(settings, resources, cache, real=True).generator
        context = RunContext(git=git_state(), versions=_versions())
        results = run_generation(settings, resources, context, generator, _subset(settings, run))
    finally:
        cache.close()
        conn.close()
    return _save_generation(settings, results, run)


def _served_onnx_dir(settings: Settings, model: EmbeddingSpec) -> Path:
    """The directory of the model the service runs: the fetched fp32 files (`onnx`) or the
    quantized model made from them (anything else must already exist, see `quantize-onnx`)."""
    base = settings.api.artifacts_dir
    fp32 = fetch_onnx_files(model.id, model.revision, base / "onnx")
    served = base / settings.api.onnx_subdir
    if served != fp32 and not (served / "model.onnx").is_file():
        raise RunError(f"missing {served / 'model.onnx'}; run `noterecall quantize-onnx` first")
    return served


@app.command("quantize-onnx")
def quantize_onnx_command(config: ConfigOption = None) -> None:
    """Quantize the fetched fp32 ONNX model to int8 into the served directory."""
    try:
        settings = _settings(config)
        model = _dense_spec(settings)
        base = settings.api.artifacts_dir
        fp32 = fetch_onnx_files(model.id, model.revision, base / "onnx")
        record = quantize_int8(fp32, base / settings.api.onnx_subdir)
        typer.echo(
            f"quantize-onnx: {record.source_bytes} -> {record.output_bytes} bytes "
            f"({record.method}, {record.weight_type})"
        )
    except KNOWN_ERRORS as exc:
        raise _fail(exc) from exc


def _build_index(settings: Settings, out: Path | None) -> str:
    """Chunk, embed with ONNX and write the artifacts the service loads."""
    decided = settings.retrieval.decided
    if decided is None:
        raise RunError("config has no [retrieval.decided]; nothing to build")
    model = _dense_spec(settings)
    directory = _served_onnx_dir(settings, model)
    service = EmbeddingService(
        load_onnx_embedder(directory), model, None, namespace=settings.api.onnx_subdir
    )
    conn = connect(settings.paths.db_path)
    try:
        notebooks = list(notebook_cells(conn))
    finally:
        conn.close()
    if not notebooks:
        raise RunError("the database has no notebooks; run `noterecall sync` first")
    target = out if out is not None else settings.api.artifacts_dir
    tokenizer = load_tokenizer(settings.tokenizer.encoding)
    manifest = build_index(
        notebooks, decided, tokenizer, service, target, runtime_label(settings.api.onnx_subdir)
    )
    return f"build-index: wrote {manifest.chunks} chunks ({manifest.dim} dims) to {target}"


@app.command("build-index")
def build_index_command(
    config: ConfigOption = None,
    out: Annotated[Path | None, typer.Option("--out", help="Directory for the artifacts.")] = None,
) -> None:
    """Build the served index (chunks, ONNX vectors, manifest) from the synced notebooks."""
    try:
        typer.echo(_build_index(_settings(config), out))
    except KNOWN_ERRORS as exc:
        raise _fail(exc) from exc


@app.command("eval")
def eval_command(  # noqa: PLR0913, PLR0917 - the options of one CLI command
    config: ConfigOption = None,
    split: Annotated[
        Split, typer.Option("--split", help="dev runs the ablation; test runs once.")
    ] = Split.DEV,
    milestone: Annotated[
        str | None, typer.Option("--milestone", help="Required for --split test.")
    ] = None,
    out: Annotated[Path | None, typer.Option("--out", help="Output JSON path.")] = None,
    suite: Annotated[
        Suite,
        typer.Option("--suite", help="retrieval (M4), header (M5), abstention or generation (M6)."),
    ] = Suite.RETRIEVAL,
    dry_run: Annotated[
        bool, typer.Option("--dry-run", help="generation: print the estimate, no API call.")
    ] = False,
    yes: Annotated[
        bool, typer.Option("--yes", help="generation: accept the estimate and call the API.")
    ] = False,
    query_set: Annotated[
        QuerySet, typer.Option("--set", help="generation: the gold split or the abstention set.")
    ] = QuerySet.GOLD,
) -> None:
    """Run an evaluation: the dev ablation, or the decided configuration once on test."""
    try:
        settings = _settings(config)
        run = GenerationRun(split, query_set, milestone, out, dry_run, yes)
        typer.echo(_dispatch_eval(settings, suite, run))
    except KNOWN_ERRORS as exc:
        raise _fail(exc) from exc


def _dispatch_eval(settings: Settings, suite: Suite, run: GenerationRun) -> str:
    split, milestone, out = run.split, run.milestone, run.out
    if suite is Suite.GENERATION:
        return _eval_generation(settings, run)
    if suite in (Suite.HEADER, Suite.ABSTENTION, Suite.ONNX_PARITY):
        if split is Split.TEST:
            raise RunError(f"--suite {suite.value} is a dev report; there is no test run")
        runners = {
            Suite.HEADER: _eval_header_dev,
            Suite.ABSTENTION: _eval_abstention_dev,
            Suite.ONNX_PARITY: _eval_onnx_parity,
        }
        return runners[suite](settings, out)
    if split is Split.DEV:
        return _eval_dev(settings, out)
    if milestone is None:
        raise RunError("--split test needs --milestone (the test split runs once per milestone)")
    return _eval_test(settings, milestone, out)


@app.command()
def ask(
    question: Annotated[str, typer.Argument(help="Your question about your notes.")],
    config: ConfigOption = None,
    show_context: Annotated[
        bool, typer.Option("--show-context", help="Also print the signals and the context chunks.")
    ] = False,
) -> None:
    """Answer a question from your notes, with citations, or say it is not in them."""
    try:
        settings = _settings(config)
        typer.echo(_ask(settings, question, show_context))
    except KNOWN_ERRORS as exc:
        raise _fail(exc) from exc


def _ask(settings: Settings, question: str, show_context: bool) -> str:
    conn = connect(settings.paths.db_path)
    cache = ResponseCache(settings.llm.cache_path)
    try:
        resources = _resources(settings, conn, _dense_models(settings))
        asker = _asker(settings, resources, cache, real=True)
        result = asker.ask(question)
        decided = settings.retrieval.decided
        chunks = (
            {c.id: c for c in resources.chunks(decided.chunk_cap, "structural")} if decided else {}
        )
    finally:
        cache.close()
        conn.close()
    text = format_answer(result, chunks)
    return f"{format_context(result, chunks)}\n\n{text}" if show_context else text


@app.command("smoke-answer")
def smoke_answer(
    step: Annotated[
        str,
        typer.Option("--step", help="a: invented chunks; b: 3 dev queries."),
    ],
    config: ConfigOption = None,
) -> None:
    """Real-API smoke test with a $0.05 cap: shows token usage and cost (owner's pre-flight)."""
    try:
        typer.echo(_smoke(_settings(config), step))
    except KNOWN_ERRORS as exc:
        raise _fail(exc) from exc


def _smoke(settings: Settings, step: str) -> str:
    if step not in ("a", "b"):
        raise RunError("--step must be a or b")
    price = settings.llm.prices[settings.llm.model]
    conn = connect(settings.paths.db_path)
    cache = ResponseCache(settings.llm.cache_path)
    capped_llm = settings.llm.model_copy(
        update={"max_usd": spent_so_far(settings, cache) + SMOKE_CAP_USD}
    )
    capped = settings.model_copy(update={"llm": capped_llm})
    try:
        if step == "a":
            row = run_step_a(build_generator(capped, build_client(capped, cache)), price)
            return format_rows([row])
        return _smoke_step_b(capped, conn, cache, price)
    finally:
        cache.close()
        conn.close()


def _smoke_step_b(
    settings: Settings, conn: sqlite3.Connection, cache: ResponseCache, price: PriceConfig
) -> str:
    gold = load_gold(settings.paths.gold_path)
    resources = _resources(settings, conn, _dense_models(settings))
    asker = _asker(settings, resources, cache, real=True)
    queries = select_all(gold, Split.DEV)
    refusals = []
    for q in queries:
        _chunks, signals = asker.context_for(q.text)
        refusals.append((q, asker.stage1_refuses(signals)))
    rows = [smoke_row(q.id, asker.ask(q.text).response, price) for q in choose_step_b(refusals)]
    return format_rows(rows)


if __name__ == "__main__":
    app()
