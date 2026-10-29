"""scripts/validate_bootstrap.py writes a valid results file (tiny settings for speed)."""

from pathlib import Path

import pytest

import validate_bootstrap
from noterecall.eval.stats import BootstrapValidation
from noterecall.ingest.stats import GitState


def test_script_writes_one_simulation_per_effect(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(validate_bootstrap, "TRIALS", 5)
    monkeypatch.setattr(validate_bootstrap, "RESAMPLES", 50)
    target = tmp_path / "out" / "v.json"
    assert validate_bootstrap.main([str(target)]) == 0
    assert capsys.readouterr().out.strip() == f"wrote {target}"
    report = BootstrapValidation.model_validate_json(target.read_text(encoding="utf-8"))
    assert [s.spec.effect for s in report.simulations] == [0.0, 0.02, 0.05]
    assert all(s.spec.n == 60 and s.spec.trials == 5 for s in report.simulations)
    assert report.seed == validate_bootstrap.SEED


def test_script_defaults_to_the_results_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(validate_bootstrap, "TRIALS", 1)
    monkeypatch.setattr(validate_bootstrap, "RESAMPLES", 10)
    monkeypatch.setattr(
        validate_bootstrap, "git_state", lambda: GitState(sha="a" * 40, dirty=False)
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("sys.argv", ["validate_bootstrap"])
    assert validate_bootstrap.main() == 0
    assert (tmp_path / "results" / "bootstrap_validation.json").is_file()
