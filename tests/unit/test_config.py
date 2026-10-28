"""Config loading: TOML defaults, env override, strict validation, typed errors."""

from pathlib import Path

import pytest

from noterecall.config import DEFAULT_CONFIG_PATH, ConfigError, load_settings

VALID_TOML = """
[paths]
raw_dir = "raw"
db_path = "db.sqlite"
results_dir = "out"

[logging]
level = "WARNING"
"""


def _write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "config.toml"
    path.write_text(text, encoding="utf-8")
    return path


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("NOTERECALL_LOGGING__LEVEL", "NOTERECALL_PATHS__DB_PATH", "NOTERECALL_BOGUS"):
        monkeypatch.delenv(name, raising=False)


def test_default_config_file_loads() -> None:
    settings = load_settings()
    assert DEFAULT_CONFIG_PATH.is_file()
    assert settings.paths.raw_dir == Path("data/rerag-data/notebooks")
    assert settings.paths.db_path == Path("data/noterecall.sqlite")
    assert settings.paths.results_dir == Path("results")
    assert settings.logging.level == "INFO"


def test_default_db_is_outside_the_data_repo() -> None:
    paths = load_settings().paths
    data_repo = Path("data/rerag-data")
    assert data_repo in paths.raw_dir.parents
    assert data_repo not in paths.db_path.parents


def test_values_come_from_given_file(tmp_path: Path) -> None:
    settings = load_settings(_write(tmp_path, VALID_TOML))
    assert settings.paths.raw_dir == Path("raw")
    assert settings.logging.level == "WARNING"


def test_env_overrides_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NOTERECALL_LOGGING__LEVEL", "DEBUG")
    monkeypatch.setenv("NOTERECALL_PATHS__DB_PATH", "elsewhere.sqlite")
    settings = load_settings(_write(tmp_path, VALID_TOML))
    assert settings.logging.level == "DEBUG"
    assert settings.paths.db_path == Path("elsewhere.sqlite")
    # Untouched keys in the same section keep the file's value.
    assert settings.paths.raw_dir == Path("raw")


def test_settings_are_frozen(tmp_path: Path) -> None:
    settings = load_settings(_write(tmp_path, VALID_TOML))
    with pytest.raises(ValueError, match="frozen"):
        settings.logging = settings.logging.model_copy()  # type: ignore[misc]


def test_unknown_key_in_section_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="bogus"):
        load_settings(_write(tmp_path, VALID_TOML + "bogus = 1\n"))


def test_unknown_section_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="extra"):
        load_settings(_write(tmp_path, VALID_TOML + "\n[retrieval]\nk = 60\n"))


def test_invalid_log_level_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="level"):
        load_settings(_write(tmp_path, VALID_TOML.replace("WARNING", "LOUD")))


def test_invalid_env_value_is_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NOTERECALL_LOGGING__LEVEL", "LOUD")
    with pytest.raises(ConfigError, match="level"):
        load_settings(_write(tmp_path, VALID_TOML))


def test_missing_field_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="results_dir"):
        load_settings(_write(tmp_path, VALID_TOML.replace('results_dir = "out"\n', "")))


def test_missing_file_is_a_config_error(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="not found"):
        load_settings(tmp_path / "nope.toml")


def test_malformed_toml_is_a_config_error(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="not valid TOML"):
        load_settings(_write(tmp_path, "[paths\nraw_dir = "))
