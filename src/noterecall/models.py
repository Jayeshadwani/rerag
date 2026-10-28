"""Domain models for ingest and storage. Implements ADR-003 (cleaning) and ADR-009 (cell hashes).

`Raw*` models are what the parser reads from a notebook before any cleaning. `Cell` and
`Notebook` are the cleaned form that is stored. `raw_hash` is the sha256 of the cell's
newline-normalised source *before* scrubbing and cleaning, so gold-set labels (ADR-009)
stay valid when cleaning rules change.
"""

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class CellType(StrEnum):
    """The only cell types the pipeline accepts."""

    MARKDOWN = "markdown"
    CODE = "code"


class OutputKind(StrEnum):
    """How a notebook output is treated by the cleaner."""

    TEXT = "text"
    IMAGE = "image"
    HTML = "html"
    OTHER = "other"
    ERROR = "error"


class RawOutput(BaseModel):
    """One notebook output. `text` is empty for image-only and html-only outputs."""

    model_config = ConfigDict(frozen=True)

    kind: OutputKind
    text: str = ""
    has_html: bool = False


class RawCell(BaseModel):
    """A cell exactly as read from the notebook (secrets still present)."""

    model_config = ConfigDict(frozen=True)

    index: int = Field(ge=0)
    cell_type: CellType
    source: str
    outputs: tuple[RawOutput, ...] = ()


class RawNotebook(BaseModel):
    """A parsed notebook before cleaning. `path` is relative to the raw directory."""

    model_config = ConfigDict(frozen=True)

    path: str
    file_hash: str
    cells: tuple[RawCell, ...]


class Cell(BaseModel):
    """A cleaned cell as stored. `index` is the position in the original notebook."""

    model_config = ConfigDict(frozen=True)

    index: int = Field(ge=0)
    cell_type: CellType
    raw_hash: str
    clean_hash: str
    source: str
    output_text: str = ""


class Notebook(BaseModel):
    """A cleaned notebook with its token counts and per-rule counts."""

    model_config = ConfigDict(frozen=True)

    path: str
    file_hash: str
    pipeline_hash: str
    raw_tokens: int = Field(ge=0)
    clean_tokens: int = Field(ge=0)
    counts: dict[str, int]
    cells: tuple[Cell, ...]
