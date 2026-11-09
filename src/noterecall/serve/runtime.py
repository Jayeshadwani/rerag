"""Loading everything the service needs, once, at startup. Implements proposal 009.

Reads the config, the built index and the ONNX model, builds the BM25 index in memory, and wires
the answer and rewrite steps to the OpenAI client (`OPENAI_API_KEY` from the environment only).
Any missing piece raises at startup; there is no degraded mode. `UI_API_SECRET` is optional:
without it the API never trusts a forwarded client IP."""

import os

import structlog
from fastapi import FastAPI

from noterecall.api import create_app
from noterecall.config import Settings, load_settings
from noterecall.enrich.headers import index_text
from noterecall.generate.abstain import Thresholds
from noterecall.generate.answer import Generator
from noterecall.generate.ask import Asker
from noterecall.generate.rewrite import Rewriter
from noterecall.generate.runtime import build_client
from noterecall.index.code_tokenizer import tokenize_code
from noterecall.index.embeddings import EmbeddingService, EmbeddingSpec
from noterecall.index.lexical import LexicalIndex
from noterecall.index.onnx_embedder import load_onnx_embedder
from noterecall.llm.cache import ResponseCache
from noterecall.llm.client import LLMClient
from noterecall.llm.prompts import load_prompt
from noterecall.retrieval.dense import DenseIndex
from noterecall.retrieval.pipeline import Components, Depths, RetrievalPipeline
from noterecall.serve.artifacts import Artifacts, load_artifacts, runtime_label
from noterecall.serve.ratelimit import RateLimiter
from noterecall.serve.service import QAService
from noterecall.srs.clock import SystemClock

log = structlog.get_logger()


class StartupError(Exception):
    """Raised when the service cannot start with the config and files it was given."""


def dense_spec(settings: Settings) -> EmbeddingSpec:
    """The embedding spec of the decided dense model."""
    decided = settings.retrieval.decided
    if decided is None or decided.pipeline.dense_model is None:
        raise StartupError("config has no [retrieval.decided] with a dense model")
    return next(c for c in settings.embeddings.candidates if c.id == decided.pipeline.dense_model)


def build_pipeline(
    settings: Settings, artifacts: Artifacts, service: EmbeddingService
) -> RetrievalPipeline:
    """The decided hybrid pipeline over the built index (BM25 in memory, vectors from disk)."""
    decided = settings.retrieval.decided
    if decided is None or decided.pipeline.lexical is None:
        raise StartupError("config has no [retrieval.decided] with a lexical retriever")
    manifest = artifacts.manifest
    spec = service.spec
    if (manifest.model, manifest.revision) != (spec.id, spec.revision):
        raise StartupError("the built index was made by another model or revision than the config")
    if manifest.chunk_cap != decided.chunk_cap or manifest.header != decided.variant.header:
        raise StartupError("the built index does not match the decided chunk cap and header")
    if manifest.runtime != runtime_label(settings.api.onnx_subdir):
        raise StartupError(
            f"the built index was made with {manifest.runtime}, "
            f"the config serves {runtime_label(settings.api.onnx_subdir)}"
        )
    chunks = artifacts.chunks
    texts = {c.id: index_text(c, decided.variant.header) for c in chunks}
    lexical = LexicalIndex.build(chunks, tokenize_code, texts)
    dense = DenseIndex([c.id for c in chunks], artifacts.matrix, service)
    depths = Depths(
        candidate=settings.retrieval.candidate_depth, fused=settings.retrieval.fused_depth
    )
    return RetrievalPipeline(
        decided.pipeline,
        {c.id: c for c in chunks},
        depths,
        Components(lexical=lexical, dense=dense),
    )


def build_qa(settings: Settings, artifacts: Artifacts, client: LLMClient) -> QAService:
    """The question-answering service over a client (the real one, or a fake in tests)."""
    spec = dense_spec(settings)
    onnx_dir = settings.api.artifacts_dir / settings.api.onnx_subdir
    embedder = load_onnx_embedder(onnx_dir)
    service = EmbeddingService(embedder, spec, None, namespace=settings.api.onnx_subdir)
    pipeline = build_pipeline(settings, artifacts, service)
    thresholds = settings.generation.decided
    if thresholds is None:
        raise StartupError("config has no [generation.decided]")
    generator = Generator(
        client, load_prompt("answer"), settings.llm.model, settings.generation.max_output_tokens
    )
    chunks = {c.id: c for c in artifacts.chunks}
    asker = Asker(
        pipeline,
        chunks,
        Thresholds(cosine=thresholds.cosine, bm25=thresholds.bm25),
        generator,
        settings.generation.max_chunks,
    )
    api = settings.api
    rewriter = Rewriter(
        client,
        load_prompt("rewrite_question"),
        settings.llm.model,
        (api.rewrite_max_output_tokens, api.max_question_chars, api.max_turn_chars),
    )
    return QAService(asker, rewriter, chunks, api)


def build_service() -> FastAPI:
    """Everything, from the environment and the files: the app the server runs."""
    settings = load_settings(None)
    artifacts = load_artifacts(settings.api.artifacts_dir)
    cache = ResponseCache(settings.llm.cache_path)
    client = build_client(settings, cache)  # raises if OPENAI_API_KEY is not set
    qa = build_qa(settings, artifacts, client)
    secret = os.environ.get("UI_API_SECRET") or None
    if secret is None:
        log.warning("startup.no_ui_secret", note="forwarded client IPs will never be trusted")
    limiter = RateLimiter(settings.api.daily_limit, SystemClock())
    facts: dict[str, object] = {
        "chunks": len(artifacts.chunks),
        "model": artifacts.manifest.model,
        "revision": artifacts.manifest.revision,
        "runtime": artifacts.manifest.runtime,
    }
    log.info("startup.ready", **facts)
    return create_app(qa, limiter, secret, settings.api, facts)
