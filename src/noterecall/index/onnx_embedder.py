"""The ONNX embedder, the served form of the dense model. Implements ADR-005/ADR-008 (proposal 009).

The same model at the same revision as the torch embedder (`BAAI/bge-base-en-v1.5`, whose repository
ships `onnx/model.onnx` at that revision) runs on `onnxruntime` (CPU only, fp32, no
quantization), so the API image needs no torch. Text is tokenized with the repository's
`tokenizer.json` (lower-casing and accent rules included), truncated at 512 tokens and padded to
the longest input in a batch. The embedding is the `[CLS]` vector of the last hidden state (the
model's pooling config), scaled to unit length. Query instructions and caching stay in
`EmbeddingService`, unchanged.
"""

from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

import numpy as np
from tokenizers import Tokenizer

MAX_LENGTH = 512
BATCH = 16
PAD_TOKEN = "[PAD]"  # noqa: S105 - a token string, not a secret
INPUT_NAMES = ("input_ids", "attention_mask", "token_type_ids")


class OnnxEmbedderError(Exception):
    """Raised when the model or tokenizer files are missing or do not look as expected."""


class Session(Protocol):
    """What the embedder needs of an `onnxruntime.InferenceSession` (tests pass a stand-in)."""

    def run(
        self, output_names: None, input_feed: dict[str, np.ndarray], /
    ) -> Sequence[np.ndarray]: ...


class OnnxEmbedder:
    """`Embedder` over an ONNX model; `load_onnx_embedder` builds the real one."""

    def __init__(
        self, session: Session, tokenizer: Tokenizer, max_length: int = MAX_LENGTH
    ) -> None:
        pad_id = tokenizer.token_to_id(PAD_TOKEN)
        if pad_id is None:
            raise OnnxEmbedderError(f"the tokenizer has no {PAD_TOKEN} token to pad with")
        self._session = session
        self._max_length = max_length
        self._tokenizer = tokenizer
        self._tokenizer.enable_truncation(max_length=max_length)
        self._tokenizer.enable_padding(pad_id=pad_id, pad_token=PAD_TOKEN)

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        """Unit-length float32 vectors, shape (len(texts), dim)."""
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)
        rows = [self._encode_batch(texts[i : i + BATCH]) for i in range(0, len(texts), BATCH)]
        return np.concatenate(rows).astype(np.float32, copy=False)

    def _encode_batch(self, texts: Sequence[str]) -> np.ndarray:
        encodings = self._tokenizer.encode_batch(list(texts))
        ids = np.array([e.ids for e in encodings], dtype=np.int64)
        mask = np.array([e.attention_mask for e in encodings], dtype=np.int64)
        feed = {
            "input_ids": ids,
            "attention_mask": mask,
            "token_type_ids": np.zeros_like(ids),
        }
        hidden = self._session.run(None, feed)[0]
        cls = np.asarray(hidden)[:, 0, :]
        norms = np.linalg.norm(cls, axis=1, keepdims=True)
        return np.asarray(cls / np.maximum(norms, 1e-12))

    def count_truncated(self, texts: Sequence[str]) -> int:
        """How many strings are longer than the input limit (special tokens included)."""
        count = 0
        for start in range(0, len(texts), BATCH):
            batch = self._tokenizer.encode_batch(list(texts[start : start + BATCH]))
            count += sum(1 for e in batch if any(e.overflowing))
        return count


def load_onnx_embedder(directory: Path, threads: int | None = None) -> OnnxEmbedder:
    """The real embedder from `directory/model.onnx` and `directory/tokenizer.json`."""
    import onnxruntime as ort  # noqa: PLC0415 - only the code paths that serve need it

    model, tokenizer = directory / "model.onnx", directory / "tokenizer.json"
    for path in (model, tokenizer):
        if not path.is_file():
            raise OnnxEmbedderError(f"missing {path}; fetch the ONNX files first")
    options = ort.SessionOptions()
    if threads is not None:
        options.intra_op_num_threads = threads
    session = ort.InferenceSession(
        str(model), sess_options=options, providers=["CPUExecutionProvider"]
    )
    names = {i.name for i in session.get_inputs()}
    if not set(INPUT_NAMES) <= names:
        raise OnnxEmbedderError(f"the model takes {sorted(names)}, expected {INPUT_NAMES}")
    return OnnxEmbedder(session, Tokenizer.from_file(str(tokenizer)))


def fetch_onnx_files(repo: str, revision: str, directory: Path) -> Path:
    """Download `onnx/model.onnx` and `tokenizer.json` at the pinned revision into `directory`.

    Needs `huggingface_hub` (the `build` dependency group): used when an image is built and by the
    evaluation, never by the running service."""
    from huggingface_hub import hf_hub_download  # noqa: PLC0415 - build-time dependency only

    directory.mkdir(parents=True, exist_ok=True)
    for remote, local in (("onnx/model.onnx", "model.onnx"), ("tokenizer.json", "tokenizer.json")):
        cached = Path(hf_hub_download(repo, remote, revision=revision))
        target = directory / local
        if not target.exists():
            target.write_bytes(cached.read_bytes())
    return directory
