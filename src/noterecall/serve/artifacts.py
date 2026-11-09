"""The built index that the service loads at startup. Implements ADR-008 (storage) for the MVP.

A build step (`noterecall build-index`, which needs the ingestion libraries) writes, into one
directory: `chunks.jsonl` (one `Chunk` per line, the exact chunks of the decided pipeline),
`embeddings.npy` (the unit-length dense vectors of their indexed text, float32, one row per chunk
in the same order) and `manifest.json` (which model and revision made the vectors, the chunk cap,
the header setting). The service reads these files and builds the BM25 index in memory; it needs
neither the notebooks nor the database. `load_artifacts` checks that the files agree with each
other before anything is served."""

import json
from pathlib import Path

import numpy as np
from pydantic import BaseModel, ConfigDict

from noterecall.models import Chunk

CHUNKS_FILE = "chunks.jsonl"
EMBEDDINGS_FILE = "embeddings.npy"
MANIFEST_FILE = "manifest.json"


def runtime_label(onnx_subdir: str) -> str:
    """What the manifest records as the runtime of the vectors (and the service checks)."""
    return f"onnxruntime:{onnx_subdir}"


class ArtifactError(Exception):
    """Raised when the built index is missing, inconsistent or not what the config expects."""


class Manifest(BaseModel):
    """What the vectors and chunks were built from."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: int
    model: str
    revision: str
    runtime: str  # "onnxruntime"
    chunk_cap: int
    header: bool
    chunks: int
    dim: int


class Artifacts(BaseModel):
    """The loaded index files."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    manifest: Manifest
    chunks: list[Chunk]
    matrix: np.ndarray


def write_artifacts(
    directory: Path, chunks: list[Chunk], matrix: np.ndarray, manifest: Manifest
) -> None:
    """Write the three files (the directory is created)."""
    if matrix.shape != (len(chunks), manifest.dim) or manifest.chunks != len(chunks):
        raise ArtifactError("the chunks, the vectors and the manifest disagree on their sizes")
    directory.mkdir(parents=True, exist_ok=True)
    lines = [c.model_dump_json() for c in chunks]
    (directory / CHUNKS_FILE).write_text("\n".join(lines) + "\n", encoding="utf-8")
    np.save(directory / EMBEDDINGS_FILE, matrix.astype(np.float32, copy=False))
    (directory / MANIFEST_FILE).write_text(
        manifest.model_dump_json(indent=2) + "\n", encoding="utf-8"
    )


def load_artifacts(directory: Path) -> Artifacts:
    """Read and cross-check the three files."""
    paths = [directory / n for n in (CHUNKS_FILE, EMBEDDINGS_FILE, MANIFEST_FILE)]
    for path in paths:
        if not path.is_file():
            raise ArtifactError(f"missing {path}; run `noterecall build-index` first")
    manifest = Manifest.model_validate_json(paths[2].read_text(encoding="utf-8"))
    rows = [ln for ln in paths[0].read_text(encoding="utf-8").splitlines() if ln.strip()]
    chunks = [Chunk.model_validate(json.loads(line)) for line in rows]
    matrix = np.load(paths[1])
    if len({c.id for c in chunks}) != len(chunks):
        raise ArtifactError("two chunks share an id")
    if matrix.shape != (len(chunks), manifest.dim) or manifest.chunks != len(chunks):
        raise ArtifactError(
            f"the files disagree: {len(chunks)} chunks, vectors {matrix.shape}, manifest "
            f"{manifest.chunks} x {manifest.dim}"
        )
    if matrix.dtype != np.float32 or not np.isfinite(matrix).all():
        raise ArtifactError("the vectors must be finite float32")
    if not np.allclose(np.linalg.norm(matrix, axis=1), 1.0, atol=1e-3):
        raise ArtifactError("the vectors are not unit length")
    return Artifacts(manifest=manifest, chunks=chunks, matrix=matrix)
