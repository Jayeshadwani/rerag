"""Configuration loading. Infrastructure; no ADR.

Settings come from a TOML file (`config/default.toml` unless another path is
given) with environment variables layered on top (`NOTERECALL_<SECTION>__<KEY>`,
e.g. `NOTERECALL_LOGGING__LEVEL=DEBUG`). Every field is required in the file, so
the TOML file is the single source of defaults. Unknown keys and wrong types
raise `ConfigError`; nothing falls back silently.
"""

import tomllib
from pathlib import Path
from typing import Annotated, Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
)

from noterecall.index.embeddings import EmbeddingSpec
from noterecall.retrieval.pipeline import PipelineSpec, Variant
from noterecall.retrieval.rerank import RerankSpec

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "default.toml"


class ConfigError(Exception):
    """Raised when configuration cannot be read or does not validate."""


class PathsConfig(BaseModel):
    """Filesystem locations. Relative paths are resolved against the working directory."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    raw_dir: Path
    db_path: Path
    results_dir: Path
    gold_path: Path
    abstention_path: Path = Path("data/rerag-data/gold/abstention.json")  # report-only set


class LoggingConfig(BaseModel):
    """Logging settings."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    level: Literal["DEBUG", "INFO", "WARNING", "ERROR"]


class CleaningConfig(BaseModel):
    """Deterministic cleaning parameters (ADR-003)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    rules_path: Path
    max_output_lines: int = Field(ge=1)


class TokenizerConfig(BaseModel):
    """The single tokenizer used everywhere (PLAN section 5)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    encoding: str


class ChunkingConfig(BaseModel):
    """Chunking parameters. The cap is data-decided in M4, so only the candidates live here."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    candidate_caps: tuple[Annotated[int, Field(ge=1)], ...] = Field(min_length=1)


class DecidedConfig(BaseModel):
    """Data-decided M4 values. Written only after the experiment produced its results file."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    chunk_cap: int = Field(ge=1)
    pipeline: PipelineSpec
    variant: Variant = Variant()


class RetrievalConfig(BaseModel):
    """Retrieval parameters. The depths and budgets are pre-registered design parameters."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    candidate_depth: int = Field(ge=1)
    fused_depth: int = Field(ge=1)
    final_k: int = Field(ge=1)
    budget_tokens: int = Field(ge=1)
    sensitivity_budgets: tuple[Annotated[int, Field(ge=1)], ...]
    rrf_default: int = Field(ge=1)
    rrf_challengers: tuple[Annotated[int, Field(ge=1)], ...]
    latency_budget_ms: float = Field(gt=0)
    decided: DecidedConfig | None = None


class DecidedThresholds(BaseModel):
    """Data-decided M6 values (proposal 006): written only after the calibration results file."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    cosine: float
    bm25: float


class GenerationConfig(BaseModel):
    """Grounded Q&A parameters (ADR-011, proposal 006). All but `decided` are pre-registered."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    max_chunks: int = Field(ge=1)
    max_output_tokens: int = Field(ge=1)
    max_false_refusal_pct: int = Field(ge=0, le=100)
    decided: DecidedThresholds | None = None
    cosine_only: float | None = None  # tau_cos of the "cosine only" reference row


class EmbeddingsConfig(BaseModel):
    """Candidate embedding models, smallest first (the first is the ADR-010 baseline)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    candidates: tuple[EmbeddingSpec, ...] = Field(min_length=1)


class RerankConfig(BaseModel):
    """The single cross-encoder (ADR-006)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    model: RerankSpec


class PriceConfig(BaseModel):
    """USD per one million tokens for one model (supplied by the owner, never guessed)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    input: float = Field(ge=0)
    cached_input: float = Field(ge=0)
    output: float = Field(ge=0)


class LlmConfig(BaseModel):
    """LLM provider settings (proposal 004). The API key comes from the environment, not here."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    provider: Literal["openai"]
    model: str
    reasoning_effort: Literal["none", "minimal", "low", "medium", "high", "xhigh", "max"]
    timeout_s: float = Field(gt=0)
    max_retries: int = Field(ge=0)
    concurrency: int = Field(ge=1)
    max_usd: float = Field(gt=0)
    cache_path: Path
    prices: dict[str, PriceConfig]

    @model_validator(mode="after")
    def _every_model_has_a_price(self) -> Self:
        if self.model not in self.prices:
            raise ValueError(f"no [llm.prices] entry for model {self.model!r}")
        return self


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
    cleaning: CleaningConfig
    tokenizer: TokenizerConfig
    chunking: ChunkingConfig
    llm: LlmConfig
    retrieval: RetrievalConfig
    generation: GenerationConfig
    embeddings: EmbeddingsConfig
    rerank: RerankConfig

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
