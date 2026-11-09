"""Int8 dynamic quantization of the served ONNX model. Implements ADR-005 (proposal 009, add. 1).

`quantize_int8` turns the fp32 `model.onnx` into an int8 model with onnxruntime's dynamic
quantization (weights stored as signed 8-bit integers, activations quantized at run time, no
calibration data), copies the tokenizer next to it and writes `quantization.json`: the method, its
parameters, the versions of the tools and the SHA-256 and size of both files, so that the served
model can be traced to the pinned fp32 revision. It needs the `build` dependency group (the `onnx`
package) and runs when an image is built or on the owner's machine, never in the running service.
Whether the quantized model is good enough is decided by the dev parity check, not here.
"""

import hashlib
import shutil
from pathlib import Path

from pydantic import BaseModel, ConfigDict

METHOD = "onnxruntime.quantization.quantize_dynamic"
WEIGHT_TYPE = "QInt8"
RECORD_FILE = "quantization.json"


class QuantizeError(Exception):
    """Raised when the source model is missing or the quantized model was not produced."""


class QuantizationRecord(BaseModel):
    """How the served model was made."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    method: str
    weight_type: str
    per_channel: bool
    reduce_range: bool
    onnx_version: str
    onnxruntime_version: str
    source_sha256: str
    source_bytes: int
    output_sha256: str
    output_bytes: int


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def quantize_int8(source_dir: Path, target_dir: Path) -> QuantizationRecord:
    """Quantize `source_dir/model.onnx` into `target_dir` (model, tokenizer, record)."""
    import onnx  # noqa: PLC0415 - build-time dependency only
    import onnxruntime  # noqa: PLC0415
    from onnxruntime.quantization import QuantType, quantize_dynamic  # noqa: PLC0415

    source, tokenizer = source_dir / "model.onnx", source_dir / "tokenizer.json"
    for path in (source, tokenizer):
        if not path.is_file():
            raise QuantizeError(f"missing {path}; fetch the fp32 ONNX files first")
    target_dir.mkdir(parents=True, exist_ok=True)
    output = target_dir / "model.onnx"
    quantize_dynamic(
        source, output, weight_type=QuantType.QInt8, per_channel=False, reduce_range=False
    )
    if not output.is_file():
        raise QuantizeError("the quantizer produced no model file")
    shutil.copyfile(tokenizer, target_dir / "tokenizer.json")
    record = QuantizationRecord(
        method=METHOD,
        weight_type=WEIGHT_TYPE,
        per_channel=False,
        reduce_range=False,
        onnx_version=onnx.__version__,
        onnxruntime_version=onnxruntime.__version__,
        source_sha256=sha256_of(source),
        source_bytes=source.stat().st_size,
        output_sha256=sha256_of(output),
        output_bytes=output.stat().st_size,
    )
    (target_dir / RECORD_FILE).write_text(record.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return record
