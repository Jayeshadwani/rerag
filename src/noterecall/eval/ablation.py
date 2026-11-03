"""The staged ablation and its pre-registered decisions. Implements ADR-010 (and ADR-004/005/006).

Every decision is the ADR-010 rule applied to per-query deltas on the dev split: a paired
bootstrap (10,000 resamples) 95% interval on the mean delta, then `stats.adopt` with a `free`
flag. The stages run in a fixed order and each uses the winner of the one before:

0. chunking: structural vs fixed baseline at each cap, then each cap vs the middle cap, all on
   Recall@token-budget (chunk size changes what "top 10" means), probed with BM25 (code tokenizer);
1. BM25 tokenizer: code-aware vs plain (free);
2. embedding model: each larger candidate vs the smallest (not free);
3. hybrid RRF (default k) vs the better single retriever (free, a post-hoc owner decision: it
   was pre-registered as not free; see proposal 003);
4. RRF k: each challenger vs the default k (free);
5. rerank vs no rerank (not free, and p95 latency must stay inside the budget).

Everything here is pure: the I/O (building indexes, running queries) is behind the `Evaluator`
callable, so the decision logic is tested with hand-made scores.
"""

import math
from collections.abc import Callable, Sequence
from typing import Literal

from pydantic import BaseModel, ConfigDict

from noterecall.eval.stats import BootstrapCI, Decision, adopt, paired_bootstrap, paired_deltas
from noterecall.retrieval.pipeline import PipelineSpec, StageTimes

Chunker = Literal["structural", "fixed"]
PRIMARY = "recall@10"


class RunKey(BaseModel):
    """What was run: the chunker and cap, and the retrieval pipeline."""

    model_config = ConfigDict(frozen=True)

    cap: int
    chunker: Chunker
    spec: PipelineSpec


class LatencyStat(BaseModel):
    """Median and 95th percentile (nearest rank) in milliseconds."""

    model_config = ConfigDict(frozen=True)

    p50: float
    p95: float


class Latency(BaseModel):
    """Per-stage and total latency over the dev queries (warm caches, one query at a time)."""

    model_config = ConfigDict(frozen=True)

    lexical: LatencyStat
    dense: LatencyStat
    fusion: LatencyStat
    rerank: LatencyStat
    total: LatencyStat


class ConfigResult(BaseModel):
    """One configuration's scores. Per-query scores are keyed by query id only."""

    model_config = ConfigDict(frozen=True)

    name: str
    key: RunKey
    means: dict[str, float]
    per_query: dict[str, dict[str, float]]
    latency: Latency


class Comparison(BaseModel):
    """One ADR-010 comparison, with its interval and the decision."""

    model_config = ConfigDict(frozen=True)

    stage: str
    candidate: str
    baseline: str
    metric: str
    free: bool
    ci: BootstrapCI
    decision: Decision
    gate: str | None = None
    report_only: bool = False


class StageOutcome(BaseModel):
    """A stage's comparisons and which configuration it carried forward."""

    model_config = ConfigDict(frozen=True)

    stage: str
    comparisons: tuple[Comparison, ...]
    chosen: str
    note: str = ""


class Ablation(BaseModel):
    """All stages, every configuration that was run, and the decided pipeline."""

    model_config = ConfigDict(frozen=True)

    stages: tuple[StageOutcome, ...]
    configs: tuple[ConfigResult, ...]
    chunk_cap: int
    pipeline: PipelineSpec
    reference: ConfigResult
    secondary: tuple[Comparison, ...] = ()


class AblationError(Exception):
    """Raised when the stage logic reaches an impossible state (a bug, not bad data)."""


Evaluator = Callable[[RunKey], ConfigResult]


class AblationParams(BaseModel):
    """The pre-registered inputs of the ablation (from config)."""

    model_config = ConfigDict(frozen=True)

    caps: tuple[int, ...]
    baseline_cap: int
    budget_metric: str
    secondary_metric: str
    embedding_models: tuple[str, ...]
    rrf_default: int
    rrf_challengers: tuple[int, ...]
    latency_budget_ms: float
    seed: int


def percentile_stat(values: Sequence[float]) -> LatencyStat:
    """Median and nearest-rank 95th percentile of a non-empty list."""
    ordered = sorted(values)
    middle = len(ordered) // 2
    median = ordered[middle] if len(ordered) % 2 else (ordered[middle - 1] + ordered[middle]) / 2
    return LatencyStat(p50=median, p95=ordered[max(1, math.ceil(0.95 * len(ordered))) - 1])


def summarize_latency(times: Sequence[StageTimes]) -> Latency:
    """Per-stage latency statistics over the queries."""
    return Latency(
        lexical=percentile_stat([t.lexical_ms for t in times]),
        dense=percentile_stat([t.dense_ms for t in times]),
        fusion=percentile_stat([t.fusion_ms for t in times]),
        rerank=percentile_stat([t.rerank_ms for t in times]),
        total=percentile_stat([t.total_ms for t in times]),
    )


def describe(key: RunKey) -> str:
    """A short readable name for a configuration."""
    spec = key.spec
    parts = []
    if spec.lexical:
        parts.append(f"bm25-{spec.lexical}")
    if spec.dense_model:
        parts.append(f"dense:{spec.dense_model}")
    if spec.rrf_k is not None:
        parts.append(f"rrf{spec.rrf_k}")
    if spec.rerank:
        parts.append("rerank")
    return f"{key.chunker}-{key.cap} | {' + '.join(parts)}"


class Rule(BaseModel):
    """What a comparison is about: its stage, the metric, and whether the component is free."""

    model_config = ConfigDict(frozen=True)

    stage: str
    metric: str
    free: bool


def compare(rule: Rule, candidate: ConfigResult, baseline: ConfigResult, seed: int) -> Comparison:
    """Paired bootstrap of candidate minus baseline on one metric, and the ADR-010 decision."""
    deltas = paired_deltas(
        {q: s[rule.metric] for q, s in candidate.per_query.items()},
        {q: s[rule.metric] for q, s in baseline.per_query.items()},
    )
    ci = paired_bootstrap(deltas, seed)
    return Comparison(
        stage=rule.stage,
        candidate=candidate.name,
        baseline=baseline.name,
        metric=rule.metric,
        free=rule.free,
        ci=ci,
        decision=adopt(ci, rule.free),
    )


def choose(
    baseline: ConfigResult, challengers: Sequence[tuple[ConfigResult, Comparison]]
) -> ConfigResult:
    """The adopted challenger with the best mean delta (ties: earlier), else the baseline."""
    adopted = [(r, c) for r, c in challengers if c.decision.adopt]
    if not adopted:
        return baseline
    best = max(range(len(adopted)), key=lambda i: (adopted[i][1].ci.mean, -i))
    return adopted[best][0]


class _Runner:
    """Runs configurations once each (cached) and applies the stage rules in order."""

    def __init__(self, evaluate: Evaluator, params: AblationParams) -> None:
        self._evaluate = evaluate
        self._params = params
        self._results: dict[RunKey, ConfigResult] = {}

    @property
    def configs(self) -> tuple[ConfigResult, ...]:
        return tuple(self._results.values())

    def run(self, cap: int, spec: PipelineSpec, chunker: Chunker = "structural") -> ConfigResult:
        key = RunKey(cap=cap, chunker=chunker, spec=spec)
        if key not in self._results:
            self._results[key] = self._evaluate(key)
        return self._results[key]

    def _compare(
        self, stage: str, cand: ConfigResult, base: ConfigResult, metric: str, free: bool
    ) -> Comparison:
        return compare(Rule(stage=stage, metric=metric, free=free), cand, base, self._params.seed)

    def chunking(self) -> tuple[int, StageOutcome, tuple[Comparison, ...]]:
        probe, metric = PipelineSpec(lexical="code"), self._params.budget_metric
        comparisons = [
            self._compare(
                "0a structural vs fixed",
                self.run(cap, probe),
                self.run(cap, probe, "fixed"),
                metric,
                True,
            )
            for cap in self._params.caps
        ]
        secondary = tuple(
            self._compare(
                "0a token coverage (report only)",
                self.run(cap, probe),
                self.run(cap, probe, "fixed"),
                self._params.secondary_metric,
                True,
            ).model_copy(update={"report_only": True})
            for cap in self._params.caps
        )
        base = self.run(self._params.baseline_cap, probe)
        others = [c for c in self._params.caps if c != self._params.baseline_cap]
        challengers = [
            (self.run(c, probe), self._compare("0b cap", self.run(c, probe), base, metric, True))
            for c in others
        ]
        chosen = choose(base, challengers)
        comparisons.extend(c for _, c in challengers)
        outcome = StageOutcome(
            stage="0 chunking", comparisons=tuple(comparisons), chosen=chosen.name
        )
        return chosen.key.cap, outcome, secondary

    def tokenizer(self, cap: int) -> tuple[Literal["plain", "code"], StageOutcome]:
        plain = self.run(cap, PipelineSpec(lexical="plain"))
        code = self.run(cap, PipelineSpec(lexical="code"))
        comparison = self._compare("1 tokenizer", code, plain, PRIMARY, True)
        chosen = choose(plain, [(code, comparison)])
        lexical: Literal["plain", "code"] = "code" if chosen is code else "plain"
        return lexical, StageOutcome(
            stage="1 tokenizer", comparisons=(comparison,), chosen=chosen.name
        )

    def embedding(self, cap: int) -> tuple[str, StageOutcome]:
        models = self._params.embedding_models
        base = self.run(cap, PipelineSpec(dense_model=models[0]))
        challengers = [
            (r, self._compare("2 embedding", r, base, PRIMARY, False))
            for r in (self.run(cap, PipelineSpec(dense_model=m)) for m in models[1:])
        ]
        chosen = choose(base, challengers)
        model = chosen.key.spec.dense_model
        if model is None:
            raise AblationError(
                "the embedding stage produced a configuration without a dense model"
            )
        outcome = StageOutcome(
            stage="2 embedding", comparisons=tuple(c for _, c in challengers), chosen=chosen.name
        )
        return model, outcome

    def hybrid(
        self, cap: int, lexical: Literal["plain", "code"], model: str
    ) -> tuple[PipelineSpec, StageOutcome]:
        """Hybrid vs the better single retriever.

        Post-hoc: this was pre-registered as "not free" (it needs both retrievers). The owner
        reclassified it as free after the dev run because the measured added latency is a few
        milliseconds (proposal 003). The comparison records that and the measured numbers.
        """
        bm25 = self.run(cap, PipelineSpec(lexical=lexical))
        dense = self.run(cap, PipelineSpec(dense_model=model))
        single = dense if dense.means[PRIMARY] > bm25.means[PRIMARY] else bm25
        hybrid = self.run(
            cap, PipelineSpec(lexical=lexical, dense_model=model, rrf_k=self._params.rrf_default)
        )
        comparison = self._compare("3 hybrid", hybrid, single, PRIMARY, True)
        added = (
            f"{hybrid.latency.total.p50 - single.latency.total.p50:+.1f} ms p50 / "
            f"{hybrid.latency.total.p95 - single.latency.total.p95:+.1f} ms p95"
        )
        gate = (
            f"post-hoc: classified free by the owner (proposal 003); measured added latency {added}"
        )
        comparison = comparison.model_copy(update={"gate": gate})
        chosen = choose(single, [(hybrid, comparison)])
        return chosen.key.spec, StageOutcome(
            stage="3 hybrid", comparisons=(comparison,), chosen=chosen.name
        )

    def rrf_k(self, cap: int, spec: PipelineSpec) -> tuple[PipelineSpec, StageOutcome]:
        base = self.run(cap, spec)
        challengers = [
            (r, self._compare("4 rrf k", r, base, PRIMARY, True))
            for r in (
                self.run(cap, spec.model_copy(update={"rrf_k": k}))
                for k in self._params.rrf_challengers
            )
        ]
        chosen = choose(base, challengers)
        outcome = StageOutcome(
            stage="4 rrf k", comparisons=tuple(c for _, c in challengers), chosen=chosen.name
        )
        return chosen.key.spec, outcome

    def rerank(self, cap: int, spec: PipelineSpec) -> tuple[PipelineSpec, StageOutcome]:
        base = self.run(cap, spec)
        reranked = self.run(cap, spec.model_copy(update={"rerank": True}))
        comparison = self._compare("5 rerank", reranked, base, PRIMARY, False)
        p95, budget = reranked.latency.total.p95, self._params.latency_budget_ms
        gate = f"p95 total latency {p95:.0f} ms vs budget {budget:.0f} ms"
        if comparison.decision.adopt and p95 >= budget:
            reason = f"{comparison.decision.reason}, but {gate}: over budget"
            comparison = comparison.model_copy(
                update={"decision": Decision(adopt=False, reason=reason)}
            )
        comparison = comparison.model_copy(update={"gate": gate})
        chosen = choose(base, [(reranked, comparison)])
        return chosen.key.spec, StageOutcome(
            stage="5 rerank", comparisons=(comparison,), chosen=chosen.name
        )


def run_ablation(evaluate: Evaluator, params: AblationParams) -> Ablation:
    """Run all stages in order and return every outcome and the decided configuration."""
    runner = _Runner(evaluate, params)
    cap, chunking, secondary = runner.chunking()
    lexical, tokenizer = runner.tokenizer(cap)
    model, embedding = runner.embedding(cap)
    spec, hybrid = runner.hybrid(cap, lexical, model)
    stages = [chunking, tokenizer, embedding, hybrid]
    if spec.rrf_k is not None:
        spec, rrf_stage = runner.rrf_k(cap, spec)
        stages.append(rrf_stage)
    else:
        note = "skipped: the hybrid was not adopted, so there is no fusion to tune"
        stages.append(
            StageOutcome(stage="4 rrf k", comparisons=(), chosen=hybrid.chosen, note=note)
        )
    spec, rerank_stage = runner.rerank(cap, spec)
    stages.append(rerank_stage)
    reference = runner.run(cap, PipelineSpec(lexical="code"))
    return Ablation(
        stages=tuple(stages),
        configs=runner.configs,
        chunk_cap=cap,
        pipeline=spec,
        reference=reference,
        secondary=secondary,
    )
