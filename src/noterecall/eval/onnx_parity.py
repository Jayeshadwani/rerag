"""Torch against ONNX for the served dense model. Implements ADR-010 reporting (proposal 009).

The M5 dev retrieval evaluation (the decided hybrid pipeline with the context header) is run twice
on the dev split, once with the torch embedder and once with the ONNX embedder, and compared:
Recall@5/10, MRR@10, nDCG@10, a paired bootstrap interval on the Recall@10 difference, and how far
apart the two runtimes' cosine similarities are (every dev query against every chunk). No LLM call
is made. The rule of proposal 009: if the ONNX Recall@10 is lower than the torch one by more than
0.01, the result is "stop" and the ONNX embedder is not deployed.
"""

import numpy as np
from pydantic import BaseModel, ConfigDict

from noterecall.config import Settings
from noterecall.eval.ablation import Comparison, ConfigResult, Rule, RunKey, compare
from noterecall.eval.gold import GoldSet, Split
from noterecall.eval.run_generation import select_all
from noterecall.eval.run_retrieval import (
    SEED,
    Resources,
    RunContext,
    RunError,
    RunMeta,
    make_evaluator,
    make_meta,
    score_params,
    select_queries,
)
from noterecall.index.embeddings import EmbeddingService

TOLERANCE = 0.01
HEADLINE = ("recall@5", "recall@10", "mrr@10", "ndcg@10")


class CosineDifference(BaseModel):
    """How far apart the two runtimes are, over every dev query against every chunk."""

    model_config = ConfigDict(frozen=True)

    pairs: int
    max_abs_cosine_difference: float
    mean_abs_cosine_difference: float
    min_self_cosine_chunks: float  # the same chunk text embedded by both: 1.0 is identical
    min_self_cosine_queries: float


class ParityResults(BaseModel):
    """The contents of `results/onnx_parity_dev.json`."""

    model_config = ConfigDict(frozen=True)

    meta: RunMeta
    variant: str = "onnx"  # the model directory that was compared with torch
    torch: ConfigResult
    onnx: ConfigResult
    recall10_comparison: Comparison
    cosine: CosineDifference
    tolerance: float
    decision: str  # "pass" or "stop"
    reason: str


def cosine_difference(
    queries: tuple[np.ndarray, np.ndarray], chunks: tuple[np.ndarray, np.ndarray]
) -> CosineDifference:
    """Compare the (torch, onnx) vectors of the queries and of the chunks. All are unit length."""
    q_t, q_o = queries
    c_t, c_o = chunks
    diff = np.abs(q_t @ c_t.T - q_o @ c_o.T)
    return CosineDifference(
        pairs=int(diff.size),
        max_abs_cosine_difference=float(diff.max()),
        mean_abs_cosine_difference=float(diff.mean()),
        min_self_cosine_chunks=float((c_t * c_o).sum(axis=1).min()),
        min_self_cosine_queries=float((q_t * q_o).sum(axis=1).min()),
    )


def parity_decision(torch_recall: float, onnx_recall: float) -> tuple[str, str]:
    """`stop` if ONNX loses more than the tolerance in Recall@10, else `pass`."""
    drop = round(torch_recall - onnx_recall, 9)  # a drop of exactly the tolerance passes
    if drop > TOLERANCE:
        return (
            "stop",
            f"ONNX Recall@10 is lower by {drop:.4f} (more than {TOLERANCE}): do not deploy",
        )
    return "pass", f"ONNX Recall@10 changed by {-drop:+.4f}; a drop of up to {TOLERANCE} is allowed"


def run_parity(
    settings: Settings,
    gold: GoldSet,
    torch_resources: Resources,
    onnx_service: EmbeddingService,
    context: RunContext,
) -> ParityResults:
    """Evaluate both runtimes on dev and compare them."""
    decided = settings.retrieval.decided
    if decided is None or decided.pipeline.dense_model is None:
        raise RunError("config has no [retrieval.decided] with a dense model to compare")
    model = decided.pipeline.dense_model
    onnx_resources = torch_resources.with_services({model: onnx_service})
    queries = select_queries(gold, Split.DEV)
    params = score_params(settings)
    key = RunKey(cap=decided.chunk_cap, chunker="structural", spec=decided.pipeline,
                 variant=decided.variant)  # fmt: skip
    torch_result = make_evaluator(torch_resources, queries, params)(key)
    onnx_result = make_evaluator(onnx_resources, queries, params)(key)
    rule = Rule(stage="onnx vs torch (report only)", metric="recall@10", free=True)
    comparison = compare(rule, onnx_result, torch_result, SEED).model_copy(
        update={"report_only": True}
    )
    cosine = _cosines(settings, gold, torch_resources, onnx_resources, key)
    decision, reason = parity_decision(
        torch_result.means["recall@10"], onnx_result.means["recall@10"]
    )
    return ParityResults(
        meta=make_meta(settings, Split.DEV, queries, context),
        torch=torch_result,
        onnx=onnx_result,
        recall10_comparison=comparison,
        cosine=cosine,
        tolerance=TOLERANCE,
        decision=decision,
        reason=reason,
    )


def _cosines(
    settings: Settings, gold: GoldSet, torch_res: Resources, onnx_res: Resources, key: RunKey
) -> CosineDifference:
    model = key.spec.dense_model
    if model is None:
        raise RunError("the decided pipeline has no dense model")
    texts = torch_res.indexed_texts(key)
    chunks = torch_res.chunks(key.cap, key.chunker)
    passages = [texts[c.id] if texts is not None else c.text for c in chunks]
    texts_q = [q.text for q in select_all(gold, Split.DEV)]
    vectors = []
    for resources in (torch_res, onnx_res):
        service = resources.services[model]
        vectors.append(
            (np.stack([service.embed_query(t) for t in texts_q]), service.embed_passages(passages))
        )
    (q_t, c_t), (q_o, c_o) = vectors
    return cosine_difference((q_t, q_o), (c_t, c_o))
