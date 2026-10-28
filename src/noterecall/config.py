"""Configuration loading. Infrastructure; no ADR.

Settings come from a TOML file (`config/default.toml` unless another path is
given) with environment variables layered on top (`NOTERECALL_<SECTION>__<KEY>`,
e.g. `NOTERECALL_LOGGING__LEVEL=DEBUG`). Every field is required in the file, so
the TOML file is the single source of defaults. Unknown keys and wrong types
raise `ConfigError`; nothing falls back silently.
"""

import tomllib
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, ValidationError
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
)

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "default.toml"


class ConfigError(Exception):
    """Raised when configuration cannot be read or does not validate."""


class PathsConfig(BaseModel):
    """Filesystem locations. Relative paths are resolved against the working directory."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    raw_dir: Path
    db_path: Path
    results_dir: Path


class LoggingConfig(BaseModel):
    """Logging settings."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    level: Literal["DEBUG", "INFO", "WARNING", "ERROR"]


class Settings(BaseSettings):
    """Validated application settings."""

    model_config = SettingsConfigDict(
        env_prefix="NOTERECALL_",
        env_nested_delimiter="__",
        extra="forbid",
        frozen=True,
    )

    paths: PathsConfig
    logging: LoggingConfig

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        """Environment overrides the file; the file arrives as init kwargs."""
        return (env_settings, init_settings)


def _read_toml(path: Path) -> dict[str, Any]:
    try:
        with path.open("rb") as handle:
            return tomllib.load(handle)
    except FileNotFoundError as exc:
        raise ConfigError(f"config file not found: {path}") from exc
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"config file {path} is not valid TOML: {exc}") from exc


def load_settings(config_path: Path | None = None) -> Settings:
    """Load settings from `config_path` (default: `config/default.toml`) plus environment."""
    path = DEFAULT_CONFIG_PATH if config_path is None else config_path
    data = _read_toml(path)
    try:
        return Settings(**data)
    except ValidationError as exc:
        raise ConfigError(f"invalid configuration from {path} and environment: {exc}") from exc

# TODO: planted to prove the banned-pattern check fails
