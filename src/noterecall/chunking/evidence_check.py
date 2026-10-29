"""Validation of the evidence-output rule against the owner's labels (proposal 002).

Draws a blind, shuffled sample in two random strata (outputs the rule keeps, outputs it drops) so
that both precision and recall can be estimated; recall is weighted back to the whole population.
Sample membership is stored only in the git-ignored data directory. The results file holds counts
and rates only: no output text, no notebook paths. This module is separate from `evidence.py` so
that the frozen rule's fingerprint does not change when tooling changes.
"""

import json
import math
import random
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, ValidationError

from noterecall.chunking.evidence import EvidenceVerdict
from noterecall.ingest.stats import GitState


class EvidenceCheckError(Exception):
    """Raised when a sample or labels file is missing, malformed or inconsistent."""


class OutputRef(BaseModel):
    """One stored code-cell output and the rule's verdict on it."""

    model_config = ConfigDict(frozen=True)

    notebook_path: str
    cell_index: int
    verdict: EvidenceVerdict

    @property
    def keeps(self) -> bool:
        """Whether the rule keeps this output in `evidence` mode."""
        return self.verdict is EvidenceVerdict.EVIDENCE


class SampleItem(BaseModel):
    """One sampled output. `sample_id` is all the labeller sees; the rest stays private."""

    model_config = ConfigDict(frozen=True)

    sample_id: int
    notebook_path: str
    cell_index: int
    rule_keeps: bool


class SampleManifest(BaseModel):
    """The drawn sample plus what is needed to weight it back to the population."""

    model_config = ConfigDict(frozen=True)

    seed: int
    rule_fingerprint: str
    population_kept: int
    population_dropped: int
    items: tuple[SampleItem, ...]


class Interval(BaseModel):
    """A confidence interval."""

    model_config = ConfigDict(frozen=True)

    low: float
    high: float


class EvidenceEval(BaseModel):
    """Aggregate results of comparing the rule with the owner's labels."""

    model_config = ConfigDict(frozen=True)

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


class ItemScore(BaseModel):
    """One sampled output's score: ids and verdicts only, never text or paths."""

    model_config = ConfigDict(frozen=True)

    sample_id: int
    stratum: Literal["kept", "dropped"]
    rule_verdict: EvidenceVerdict
    labelled_evidence: bool


class EvidenceReport(BaseModel):
    """The contents of `results/evidence_rule_eval.json` (CLAUDE.md section 4 metadata)."""

    model_config = ConfigDict(frozen=True)

    schema_version: Literal[2]
    git: GitState
    config: dict[str, Any]
    tokenizer: str
    tiktoken_version: str
    summary: EvidenceEval
    items: tuple[ItemScore, ...]


def score_items(
    manifest: SampleManifest, labels: dict[int, bool], verdicts: dict[int, EvidenceVerdict]
) -> tuple[ItemScore, ...]:
    """Per-item scores in sample order. A verdict must exist for every sampled item."""
    missing = sorted({i.sample_id for i in manifest.items} - set(verdicts))
    if missing:
        raise EvidenceCheckError(f"no rule verdict for sample ids {missing}")
    return tuple(
        ItemScore(
            sample_id=item.sample_id,
            stratum="kept" if item.rule_keeps else "dropped",
            rule_verdict=verdicts[item.sample_id],
            labelled_evidence=labels[item.sample_id],
        )
        for item in manifest.items
    )


def wilson_interval(successes: int, n: int, z: float = 1.96) -> Interval:
    """Wilson score interval for a binomial proportion."""
    if n <= 0:
        raise ValueError("n must be positive")
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return Interval(low=max(0.0, centre - half), high=min(1.0, centre + half))


def draw_sample(
    refs: Sequence[OutputRef], per_stratum: int, seed: int, fingerprint: str
) -> SampleManifest:
    """Random sample of up to `per_stratum` kept and `per_stratum` dropped outputs, shuffled."""
    ordered = sorted(refs, key=lambda r: (r.notebook_path, r.cell_index))
    kept = [r for r in ordered if r.keeps]
    dropped = [r for r in ordered if not r.keeps]
    rng = random.Random(seed)  # noqa: S311 - reproducible sampling, not cryptography
    chosen = rng.sample(kept, min(per_stratum, len(kept))) + rng.sample(
        dropped, min(per_stratum, len(dropped))
    )
    rng.shuffle(chosen)
    items = tuple(
        SampleItem(
            sample_id=i,
            notebook_path=r.notebook_path,
            cell_index=r.cell_index,
            rule_keeps=r.keeps,
        )
        for i, r in enumerate(chosen, start=1)
    )
    return SampleManifest(
        seed=seed,
        rule_fingerprint=fingerprint,
        population_kept=len(kept),
        population_dropped=len(dropped),
        items=items,
    )


def write_manifest(manifest: SampleManifest, path: Path) -> None:
    """Write the sample (private; keep it under the git-ignored data directory)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(manifest.model_dump_json(indent=2) + "\n", encoding="utf-8")


def read_manifest(path: Path) -> SampleManifest:
    """Read a sample written by `write_manifest`."""
    try:
        return SampleManifest.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValidationError) as exc:
        raise EvidenceCheckError(f"cannot read the sample at {path}: {exc}") from exc


class _LabelsFile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    labels: dict[int, Literal["E", "N"]]


def read_labels(path: Path, manifest: SampleManifest) -> dict[int, bool]:
    """Labels as {sample_id: is_evidence}. Every sampled id must be labelled, and only those."""
    try:
        parsed = _LabelsFile.model_validate(json.loads(path.read_text(encoding="utf-8")))
    except (OSError, json.JSONDecodeError, ValidationError) as exc:
        raise EvidenceCheckError(f"cannot read the labels at {path}: {exc}") from exc
    expected = {item.sample_id for item in manifest.items}
    if set(parsed.labels) != expected:
        missing, extra = (
            sorted(expected - set(parsed.labels)),
            sorted(set(parsed.labels) - expected),
        )
        raise EvidenceCheckError(
            f"labels do not match the sample: missing {missing}, unexpected {extra}"
        )
    return {sample_id: label == "E" for sample_id, label in parsed.labels.items()}


def evaluate(manifest: SampleManifest, labels: dict[int, bool]) -> EvidenceEval:
    """Precision from the kept stratum; recall weighted by stratum size (inverse sampling rate)."""
    kept = [i for i in manifest.items if i.rule_keeps]
    dropped = [i for i in manifest.items if not i.rule_keeps]
    tp = sum(labels[i.sample_id] for i in kept)
    fn = sum(labels[i.sample_id] for i in dropped)
    precision = tp / len(kept) if kept else None
    interval = wilson_interval(tp, len(kept)) if kept else None
    est_tp = tp * manifest.population_kept / len(kept) if kept else 0.0
    est_fn = fn * manifest.population_dropped / len(dropped) if dropped else 0.0
    positives = est_tp + est_fn
    population = manifest.population_kept + manifest.population_dropped
    return EvidenceEval(
        seed=manifest.seed,
        rule_fingerprint=manifest.rule_fingerprint,
        population_kept=manifest.population_kept,
        population_dropped=manifest.population_dropped,
        sampled_kept=len(kept),
        sampled_dropped=len(dropped),
        labelled_evidence_in_kept=tp,
        labelled_evidence_in_dropped=fn,
        precision=precision,
        precision_wilson95=interval,
        recall_weighted=est_tp / positives if positives > 0 else None,
        evidence_prevalence_weighted=positives / population if population else None,
    )


_SHOWN_CHARS = 600


def render_sample(manifest: SampleManifest, texts: dict[tuple[str, int], str]) -> str:
    """The sample as plain text for the labeller. Shows no path and no verdict, only an id."""
    blocks: list[str] = []
    for item in manifest.items:
        text = texts[(item.notebook_path, item.cell_index)].strip("\n")
        shown = text[:_SHOWN_CHARS]
        more = len(text) - len(shown)
        tail = f"\n[... {more} more characters]" if more > 0 else ""
        header = f"#{item.sample_id}  ({text.count(chr(10)) + 1} lines, {len(text)} characters)"
        blocks.append(f"{header}\n{shown}{tail}")
    return "\n\n----\n\n".join(blocks) + "\n"
