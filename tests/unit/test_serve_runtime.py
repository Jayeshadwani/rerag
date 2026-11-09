"""Startup of the service: files and config in, a working app out (offline, fake model)."""

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from noterecall.chunking.tokens import load_tokenizer
from noterecall.config import DecidedConfig, load_settings
from noterecall.index.embeddings import EmbeddingService, HashEmbedder
from noterecall.llm.client import Prompt
from noterecall.llm.fake import FakeTransport
from noterecall.models import Cell, CellType
from noterecall.serve import runtime
from noterecall.serve.artifacts import ArtifactError, load_artifacts, write_artifacts
from noterecall.serve.build import build_index

DIM = 16
NOT_IN_NOTES = json.dumps({"insufficient": True, "claims": []})


def make_index(directory: Path) -> EmbeddingService:
    settings = load_settings()
    decided = settings.retrieval.decided
    assert isinstance(decided, DecidedConfig)
    spec = runtime.dense_spec(settings)
    service = EmbeddingService(HashEmbedder(DIM), spec, None, namespace="onnx")
    cells = [
        Cell(
            index=0, cell_type=CellType.MARKDOWN, raw_hash="h0", clean_hash="c0",
            source="# Linear regression\n\nFit a line with least squares.",
        )
    ]  # fmt: skip
    build_index(
        [("nb/a.ipynb", cells)], decided, load_tokenizer(settings.tokenizer.encoding), service,
        directory,
        "onnxruntime:onnx-int8",
    )  # fmt: skip
    return service


@pytest.fixture
def served(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[TestClient, FakeTransport]:
    make_index(tmp_path / "idx")
    transport = FakeTransport(lambda _p: NOT_IN_NOTES)
    monkeypatch.setenv("NOTERECALL_API__ARTIFACTS_DIR", str(tmp_path / "idx"))
    monkeypatch.setenv("NOTERECALL_LLM__CACHE_PATH", str(tmp_path / "cache.sqlite"))
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-sentinel-key")
    monkeypatch.setenv("UI_API_SECRET", "ui-secret")
    monkeypatch.setattr(runtime, "load_onnx_embedder", lambda _d: HashEmbedder(DIM))
    monkeypatch.setattr("noterecall.generate.runtime.create_transport", lambda *_a, **_k: transport)
    return TestClient(runtime.build_service()), transport


def test_the_service_starts_and_answers_with_a_refusal(
    served: tuple[TestClient, FakeTransport], capsys: pytest.CaptureFixture[str]
) -> None:
    client, transport = served
    health = client.get("/health").json()
    assert health["status"] == "ok"
    assert health["chunks"] == 1
    reply = client.post("/ask", json={"question": "what is linear regression?"})
    assert reply.status_code == 200
    assert reply.json()["status"] == "refused"
    assert reply.json()["refusal_reason"] == "not_in_notes"
    assert len(transport.prompts) == 1  # the answer call only: no history, so no rewrite
    seen = capsys.readouterr()
    everything = seen.out + seen.err + json.dumps(health) + reply.text
    assert "sk-test-sentinel-key" not in everything


def test_a_follow_up_makes_a_rewrite_call_first(
    served: tuple[TestClient, FakeTransport],
) -> None:
    client, transport = served

    def responder(prompt: Prompt) -> str:
        return (
            json.dumps({"question": "What is least squares?"})
            if "<follow_up>" in (prompt.variable)
            else NOT_IN_NOTES
        )

    transport._responder = responder
    history = [{"question": "what is linear regression?", "answer": "A line fit."}]
    reply = client.post("/ask", json={"question": "and how?", "history": history})
    assert reply.json()["standalone_question"] == "What is least squares?"
    assert len(transport.prompts) == 2


def test_startup_fails_loudly_on_a_mismatched_index(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    make_index(tmp_path / "idx")
    artifacts = load_artifacts(tmp_path / "idx")
    settings = load_settings()
    spec = runtime.dense_spec(settings)
    service = EmbeddingService(HashEmbedder(DIM), spec, None, namespace="onnx")
    other = artifacts.manifest.model_copy(update={"revision": "someone-else"})
    write_artifacts(tmp_path / "bad", artifacts.chunks, artifacts.matrix, other)
    with pytest.raises(runtime.StartupError, match="another model or revision"):
        runtime.build_pipeline(settings, load_artifacts(tmp_path / "bad"), service)
    capped = artifacts.manifest.model_copy(update={"chunk_cap": 999})
    write_artifacts(tmp_path / "bad2", artifacts.chunks, artifacts.matrix, capped)
    with pytest.raises(runtime.StartupError, match="decided chunk cap"):
        runtime.build_pipeline(settings, load_artifacts(tmp_path / "bad2"), service)


def test_startup_fails_without_the_built_index(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("NOTERECALL_API__ARTIFACTS_DIR", str(tmp_path / "missing"))
    with pytest.raises(ArtifactError, match="build-index"):
        runtime.build_service()


def test_startup_fails_without_decided_config() -> None:
    settings = load_settings()
    undecided = settings.model_copy(
        update={"retrieval": settings.retrieval.model_copy(update={"decided": None})}
    )
    with pytest.raises(runtime.StartupError, match=r"retrieval\.decided"):
        runtime.dense_spec(undecided)


def test_startup_refuses_an_index_made_with_another_onnx_variant(tmp_path: Path) -> None:
    make_index(tmp_path / "idx")
    artifacts = load_artifacts(tmp_path / "idx")
    settings = load_settings()
    service = EmbeddingService(HashEmbedder(DIM), runtime.dense_spec(settings), None)
    other = artifacts.manifest.model_copy(update={"runtime": "onnxruntime:onnx"})
    write_artifacts(tmp_path / "fp32", artifacts.chunks, artifacts.matrix, other)
    with pytest.raises(runtime.StartupError, match="onnxruntime:onnx-int8"):
        runtime.build_pipeline(settings, load_artifacts(tmp_path / "fp32"), service)
