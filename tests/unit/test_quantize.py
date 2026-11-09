"""Int8 dynamic quantization on a tiny model (the real model is checked by the parity run)."""

import json
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import pytest
from onnx import TensorProto, helper, numpy_helper

from noterecall.index.quantize import (
    METHOD,
    RECORD_FILE,
    QuantizeError,
    quantize_int8,
    sha256_of,
)


def tiny_model(path: Path) -> np.ndarray:
    """x (1x64) @ W (64x64) -> y; returns W."""
    rng = np.random.default_rng(0)
    weights = rng.normal(size=(64, 64)).astype(np.float32)
    graph = helper.make_graph(
        [helper.make_node("MatMul", ["x", "w"], ["y"])],
        "tiny",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 64])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 64])],
        [numpy_helper.from_array(weights, "w")],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 9
    onnx.save(model, path)
    return weights


def test_quantize_writes_an_int8_model_a_tokenizer_and_a_record(tmp_path: Path) -> None:
    source, target = tmp_path / "fp32", tmp_path / "int8"
    source.mkdir()
    weights = tiny_model(source / "model.onnx")
    (source / "tokenizer.json").write_text("{}")
    record = quantize_int8(source, target)
    assert record.method == METHOD
    assert record.weight_type == "QInt8"
    assert record.output_bytes < record.source_bytes / 2  # weights went from 4 bytes to 1
    assert record.source_sha256 == sha256_of(source / "model.onnx")
    assert record.output_sha256 == sha256_of(target / "model.onnx")
    assert record.onnx_version == onnx.__version__
    assert record.onnxruntime_version == ort.__version__
    assert (target / "tokenizer.json").read_text() == "{}"
    assert json.loads((target / RECORD_FILE).read_text())["output_sha256"] == record.output_sha256
    x = np.random.default_rng(1).normal(size=(1, 64)).astype(np.float32)
    session = ort.InferenceSession(str(target / "model.onnx"), providers=["CPUExecutionProvider"])
    got = session.run(None, {"x": x})[0]
    expected = x @ weights
    assert not np.array_equal(got, expected)  # it really is quantized
    cosine = float(np.vdot(got, expected) / (np.linalg.norm(got) * np.linalg.norm(expected)))
    assert cosine > 0.99  # and still close


def test_quantize_needs_the_fp32_files(tmp_path: Path) -> None:
    with pytest.raises(QuantizeError, match="missing"):
        quantize_int8(tmp_path, tmp_path / "out")
