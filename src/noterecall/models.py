"""Domain models for ingest, storage and chunking. Implements ADR-003, ADR-004 and ADR-009.

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


class PieceKind(StrEnum):
    """Which part of a cell a piece comes from."""

    MARKDOWN = "markdown"
    CODE = "code"
    OUTPUT = "output"


class PieceUnit(StrEnum):
    """The unit a piece was cut at. Statements, sentences, fences and unparsed code are atoms."""

    CELL = "cell"
    PARAGRAPH = "paragraph"
    STATEMENT = "statement"
    SENTENCE = "sentence"
    FENCED_BLOCK = "fenced_block"
    UNPARSED_CODE = "unparsed_code"
    WINDOW = "window"


class OversizeReason(StrEnum):
    """Why a chunk is allowed to exceed the cap: it is one indivisible piece."""

    STATEMENT = "statement"
    SENTENCE = "sentence"
    FENCED_BLOCK = "fenced_block"
    UNPARSED_CODE = "unparsed_code"


class OutputMode(StrEnum):
    """Which cell outputs go into chunk text (proposal 002, M4 ablation arms A, B, C)."""

    NONE = "none"
    ALL = "all"
    EVIDENCE = "evidence"


class Piece(BaseModel):
    """A contiguous slice of one cell. A cell's pieces concatenate to its exact text."""

    model_config = ConfigDict(frozen=True)

    cell_index: int = Field(ge=0)
    kind: PieceKind
    part: int = Field(ge=0)
    unit: PieceUnit
    text: str


class Chunk(BaseModel):
    """A retrievable unit. `cell_hashes` are the `raw_hash` of every cell it touches."""

    model_config = ConfigDict(frozen=True)

    id: str
    notebook_path: str
    heading_path: tuple[str, ...]
    pieces: tuple[Piece, ...]
    text: str
    tokens: int = Field(ge=0)
    flags: tuple[OversizeReason, ...] = ()
    cell_hashes: tuple[str, ...]
