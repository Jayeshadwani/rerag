"""The build step of the served index. Implements ADR-008 (storage) and ADR-007 (headers).

`build_index` chunks the synced notebooks with the decided structural chunker and cap, embeds the
indexed text of every chunk (header on or off as decided) with the ONNX model, and writes the
three artifact files that the service loads. It runs where the notes are (the API image build,
or the owner's machine), never in the running service."""

from collections.abc import Sequence
from pathlib import Path

from noterecall.chunking.tokens import Tokenizer
from noterecall.config import DecidedConfig
from noterecall.enrich.headers import index_text
from noterecall.eval.run_retrieval import build_chunks
from noterecall.index.embeddings import EmbeddingService
from noterecall.models import Cell
from noterecall.serve.artifacts import Manifest, notebook_titles, write_artifacts

SCHEMA_VERSION = 1


def build_index(  # noqa: PLR0913, PLR0917 - the inputs of one build step
    notebooks: Sequence[tuple[str, list[Cell]]],
    decided: DecidedConfig,
    tokenizer: Tokenizer,
    service: EmbeddingService,
    directory: Path,
    runtime: str,
) -> Manifest:
    """Write the artifacts for `notebooks` into `directory` and return the manifest."""
    chunks = build_chunks(list(notebooks), decided.chunk_cap, "structural", tokenizer)
    texts = [index_text(c, decided.variant.header) for c in chunks]
    matrix = service.embed_passages(texts)
    manifest = Manifest(
        schema_version=SCHEMA_VERSION,
        model=service.spec.id,
        revision=service.spec.revision,
        runtime=runtime,
        chunk_cap=decided.chunk_cap,
        header=decided.variant.header,
        chunks=len(chunks),
        dim=int(matrix.shape[1]) if len(chunks) else 0,
        notebooks=notebook_titles(path for path, _cells in notebooks),
    )
    write_artifacts(directory, chunks, matrix, manifest)
    return manifest
