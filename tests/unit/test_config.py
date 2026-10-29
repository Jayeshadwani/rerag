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

[cleaning]
rules_path = "rules.yaml"
max_output_lines = 5

[tokenizer]
encoding = "tiny"

[chunking]
candidate_caps = [8, 16]
"""


def _write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "config.toml"
    path.write_text(text, encoding="utf-8")
    return path


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "NOTERECALL_LOGGING__LEVEL",
        "NOTERECALL_PATHS__DB_PATH",
        "NOTERECALL_CLEANING__MAX_OUTPUT_LINES",
        "NOTERECALL_BOGUS",
    ):
        monkeypatch.delenv(name, raising=False)


def test_default_config_file_loads() -> None:
    settings = load_settings()
    assert DEFAULT_CONFIG_PATH.is_file()
    assert settings.paths.raw_dir == Path("data/rerag-data/notebooks")
    assert settings.paths.db_path == Path("data/noterecall.sqlite")
    assert settings.paths.results_dir == Path("results")
    assert settings.logging.level == "INFO"
    assert settings.cleaning.rules_path == Path("config/cleaning_rules.yaml")
    assert settings.cleaning.max_output_lines == 20
    assert settings.tokenizer.encoding == "cl100k_base"
    assert settings.chunking.candidate_caps == (256, 512, 1024)


def test_default_db_is_outside_the_data_repo() -> None:
    paths = load_settings().paths
    data_repo = Path("data/rerag-data")
    assert data_repo in paths.raw_dir.parents
    assert data_repo not in paths.db_path.parents


def test_values_come_from_given_file(tmp_path: Path) -> None:
    settings = load_settings(_write(tmp_path, VALID_TOML))
    assert settings.paths.raw_dir == Path("raw")
    assert settings.logging.level == "WARNING"
    assert settings.cleaning.max_output_lines == 5


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


def test_max_output_lines_must_be_positive(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="max_output_lines"):
        load_settings(
            _write(tmp_path, VALID_TOML.replace("max_output_lines = 5", "max_output_lines = 0"))
        )


def test_env_overrides_nested_cleaning_value(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("NOTERECALL_CLEANING__MAX_OUTPUT_LINES", "7")
    assert load_settings(_write(tmp_path, VALID_TOML)).cleaning.max_output_lines == 7


@pytest.mark.parametrize("value", ["[]", "[0]", "[256, -1]"])
def test_candidate_caps_must_be_positive_and_non_empty(tmp_path: Path, value: str) -> None:
    text = VALID_TOML.replace("candidate_caps = [8, 16]", f"candidate_caps = {value}")
    with pytest.raises(ConfigError, match="candidate_caps"):
        load_settings(_write(tmp_path, text))
