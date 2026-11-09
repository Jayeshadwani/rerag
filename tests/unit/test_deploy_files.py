"""Guards for what the images may contain and where the Render settings live (proposal 009)."""

import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
FORBIDDEN = ("torch", "sentence_transformers", "tiktoken", "transformers")


def test_the_api_modules_do_not_import_the_evaluation_stack() -> None:
    code = (
        "import sys, noterecall.api, noterecall.serve.runtime\n"
        f"bad = [m for m in {FORBIDDEN!r} if m in sys.modules]\n"
        "print(','.join(bad))\n"
    )
    out = subprocess.run(  # noqa: S603 - fixed interpreter and code
        [sys.executable, "-I", "-c", code],
        capture_output=True, text=True, check=True, env={"PYTHONPATH": str(ROOT / "src")},
    )  # fmt: skip
    assert out.stdout.strip() == ""


def test_render_yaml_has_two_free_docker_services_and_no_secret_values() -> None:
    spec = yaml.safe_load((ROOT / "render.yaml").read_text())
    services = {s["name"]: s for s in spec["services"]}
    assert set(services) == {"noterecall-api", "noterecall-ui"}
    for service in services.values():
        assert service["plan"] == "free"
        assert service["runtime"] == "docker"
        assert all(v.get("sync") is False and "value" not in v for v in service["envVars"])
    keys = {n: {v["key"] for v in s["envVars"]} for n, s in services.items()}
    # the data token is a Secret File, not an environment variable
    assert keys["noterecall-api"] == {"OPENAI_API_KEY", "UI_API_SECRET"}
    assert keys["noterecall-ui"] == {"API_URL", "UI_API_SECRET"}


def test_the_ui_image_installs_only_the_ui_group_and_copies_no_package() -> None:
    text = (ROOT / "Dockerfile.ui").read_text()
    assert "--only-group ui-image" in text
    assert "COPY src" not in text
    assert "data" not in text.lower().replace("metadata", "")


def test_the_data_token_is_a_build_secret_and_stays_in_the_index_stage() -> None:
    text = (ROOT / "Dockerfile.api").read_text()
    assert "--mount=type=secret,id=DATA_REPO_TOKEN,required=true" in text
    assert text.startswith("# syntax=docker/dockerfile:")
    final = text.split("\nFROM ${PY}\n")[1]
    assert "DATA_REPO_TOKEN" not in final
    assert "rerag-data" not in final
    assert "COPY --from=index /src/data/artifacts/onnx-int8 " in final
    assert "artifacts/onnx " not in final  # the fp32 model is not in the final image
    assert "quantize-onnx" in text.split("\nFROM ${PY}\n")[0]


def test_the_image_build_can_call_the_cli_as_a_module() -> None:
    out = subprocess.run(
        [sys.executable, "-m", "noterecall.cli", "--help"],
        capture_output=True, text=True, check=True, env={"PYTHONPATH": str(ROOT / "src")},
    )  # fmt: skip
    for command in ("sync", "quantize-onnx", "build-index"):
        assert command in out.stdout
