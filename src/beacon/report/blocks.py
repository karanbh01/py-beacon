# src/beacon/report/blocks.py
"""
The block model: what a report is made of, as plain data.

A report template is a page setup and an ordered list of blocks. Nothing here
knows how to draw anything — these are dataclasses that round-trip through JSON
so a template can be stored, edited by a client, and rendered later by a
process that never saw the one that created it. The drawing lives in
`beacon.report.pdf` behind the `pdf` extra, which means a client can build and
persist templates with no PDF library installed at all.

Blocks carry a ``kind`` discriminator so a stored template can be read back
into the right class. That is the one piece of ceremony in here, and it is what
lets `DocumentStore` hold a heterogeneous list without the reader having to
guess.

## Why the chart block holds a path rather than a figure

A chart is rendered by the plotting layer (BN-78) into an image, and this block
points at it. Holding a live matplotlib figure would drag an optional
dependency into the data model and make a template unserialisable — the exact
coupling the split above exists to avoid. Until those images exist, a chart
block with no image renders as a labelled placeholder, so a template can be
designed and reviewed before the charts it will hold are built.
"""
from dataclasses import asdict, dataclass, field
from typing import Any

from ..exceptions import ReportingError

# Page sizes in points (1/72 inch), the unit PDF works in natively.
PAGE_SIZES = {
    "A4": (595.276, 841.890),
    "LETTER": (612.0, 792.0),
    "A5": (419.528, 595.276),
}

PORTRAIT = "portrait"
LANDSCAPE = "landscape"
ORIENTATIONS = (PORTRAIT, LANDSCAPE)

DEFAULT_MARGIN = 48.0


@dataclass(frozen=True)
class PageSetup:
    """The sheet a report is drawn on.

    Attributes:
        size: A key of PAGE_SIZES.
        orientation: PORTRAIT or LANDSCAPE.
        margin: Blank border in points, applied on all four sides.
    """
    size: str = "A4"
    orientation: str = PORTRAIT
    margin: float = DEFAULT_MARGIN

    def __post_init__(self) -> None:
        if self.size not in PAGE_SIZES:
            raise ReportingError(
                f"unknown page size '{self.size}'. Available: "
                f"{', '.join(sorted(PAGE_SIZES))}.")

        if self.orientation not in ORIENTATIONS:
            raise ReportingError(
                f"unknown orientation '{self.orientation}'. Available: "
                f"{', '.join(ORIENTATIONS)}.")

        if self.margin < 0.0:
            raise ReportingError(f"margin must be non-negative, got {self.margin}.")

    @property
    def dimensions(self) -> tuple[float, float]:
        """(width, height) in points, after orientation."""
        width, height = PAGE_SIZES[self.size]

        return (height, width) if self.orientation == LANDSCAPE else (width, height)

    @property
    def content_width(self) -> float:
        """Drawable width between the margins."""
        return self.dimensions[0] - 2 * self.margin

    @property
    def content_height(self) -> float:
        """Drawable height between the margins."""
        return self.dimensions[1] - 2 * self.margin

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready form."""
        return asdict(self)

    @classmethod
    def from_dict(cls,
                  data: dict[str, Any]) -> "PageSetup":
        """Rebuild from stored form."""
        return cls(**data)


@dataclass(frozen=True)
class Block:
    """Base for everything that can appear in a report.

    Subclasses set ``kind`` and are registered in BLOCK_TYPES so a stored
    template can be read back.
    """
    kind: str = field(default="", init=False)

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready form, carrying the discriminator."""
        return {"kind": self.kind, **asdict(self)}


@dataclass(frozen=True)
class Header(Block):
    """The title band at the top of a report.

    Attributes:
        title: Main line.
        subtitle: Optional second line.
        as_of: Optional date string. Kept as text rather than a date because a
            report's as-of label is presentation — "31 Dec 2024" and
            "2024-12-31" are the same date and a different report.
    """
    title: str
    subtitle: str = ""
    as_of: str = ""
    kind: str = field(default="header", init=False)


@dataclass(frozen=True)
class Text(Block):
    """A paragraph.

    Attributes:
        body: The text. Wrapped to the content width by the renderer.
        size: Font size in points.
        muted: Whether to draw in the muted ink rather than the primary.
    """
    body: str
    size: float = 9.0
    muted: bool = False
    kind: str = field(default="text", init=False)


@dataclass(frozen=True)
class Stat:
    """One labelled number in a StatGrid.

    Attributes:
        label: What it is.
        value: Preformatted for display. The block model does not format
            numbers — a percentage, a currency amount and a ratio all need
            different treatment, and the caller knows which this is.
        change: Optional secondary line, e.g. a period change.
    """
    label: str
    value: str
    change: str = ""


@dataclass(frozen=True)
class StatGrid(Block):
    """A row of headline figures.

    Attributes:
        stats: The figures, laid out left to right.
        columns: How many per row. Extra stats wrap onto further rows.
    """
    stats: list[Stat]
    columns: int = 4
    kind: str = field(default="stat_grid", init=False)

    def __post_init__(self) -> None:
        if self.columns < 1:
            raise ReportingError(f"columns must be at least 1, got {self.columns}.")

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready form."""
        return {"kind": self.kind,
                "stats": [asdict(stat) for stat in self.stats],
                "columns": self.columns}

    @classmethod
    def from_dict(cls,
                  data: dict[str, Any]) -> "StatGrid":
        """Rebuild from stored form."""
        return cls(stats=[Stat(**stat) for stat in data["stats"]],
                   columns=data.get("columns", 4))


@dataclass(frozen=True)
class Table(Block):
    """A grid of values.

    Attributes:
        columns: Header labels.
        rows: Row values, already formatted for display.
        title: Optional caption above the table.
        align_right: Column indices to right-align. Numbers read far better
            right-aligned, and the block model cannot tell which columns hold
            them because every cell is already a string.
    """
    columns: list[str]
    rows: list[list[str]]
    title: str = ""
    align_right: list[int] = field(default_factory=list)
    kind: str = field(default="table", init=False)

    def __post_init__(self) -> None:
        width = len(self.columns)
        for position, row in enumerate(self.rows):
            if len(row) != width:
                raise ReportingError(
                    f"table row {position} has {len(row)} cells but there are "
                    f"{width} columns.")

    @classmethod
    def from_dict(cls,
                  data: dict[str, Any]) -> "Table":
        """Rebuild from stored form."""
        return cls(columns=list(data["columns"]),
                   rows=[list(row) for row in data["rows"]],
                   title=data.get("title", ""),
                   align_right=list(data.get("align_right", [])))


@dataclass(frozen=True)
class BarChart(Block):
    """A simple horizontal bar chart, drawn natively rather than as an image.

    Kept separate from :class:`Chart` because a handful of labelled bars — top
    holdings, sector weights, per-factor contributions — is most of what a
    factsheet actually shows, and routing that through an image pipeline would
    mean a report could not be produced without the plotting extra.

    Attributes:
        categories: Bar labels, top to bottom.
        values: One value per category. Negative values are drawn to the left
            of the axis, so a contribution chart reads correctly.
        title: Optional caption.
        height: Drawing height in points.
    """
    categories: list[str]
    values: list[float]
    title: str = ""
    height: float = 140.0
    kind: str = field(default="bar_chart", init=False)

    def __post_init__(self) -> None:
        if len(self.categories) != len(self.values):
            raise ReportingError(
                f"{len(self.categories)} categories but {len(self.values)} values.")

        if not self.categories:
            raise ReportingError("a bar chart needs at least one category.")

    @classmethod
    def from_dict(cls,
                  data: dict[str, Any]) -> "BarChart":
        """Rebuild from stored form."""
        return cls(categories=list(data["categories"]),
                   values=[float(value) for value in data["values"]],
                   title=data.get("title", ""),
                   height=float(data.get("height", 140.0)))


@dataclass(frozen=True)
class Chart(Block):
    """A rendered chart image, or a placeholder for one.

    Attributes:
        image_path: Path to a rendered image. None, or a path that does not
            exist, draws a labelled placeholder instead of failing — a template
            is designed before the charts it will hold are built, and a missing
            image should not stop a layout being reviewed.
        title: Optional caption.
        height: Drawing height in points.
    """
    image_path: str | None = None
    title: str = ""
    height: float = 200.0
    kind: str = field(default="chart", init=False)

    @classmethod
    def from_dict(cls,
                  data: dict[str, Any]) -> "Chart":
        """Rebuild from stored form."""
        return cls(image_path=data.get("image_path"),
                   title=data.get("title", ""),
                   height=float(data.get("height", 200.0)))


# Discriminator to class, for reading a stored template back.
BLOCK_TYPES: dict[str, type[Block]] = {
    "header": Header,
    "text": Text,
    "stat_grid": StatGrid,
    "table": Table,
    "bar_chart": BarChart,
    "chart": Chart,
}


def block_from_dict(data: dict[str, Any]) -> Block:
    """Rebuild one block from its stored form.

    Args:
        data: A block's dict, carrying its ``kind``.

    Returns:
        Block: The reconstructed block.

    Raises:
        ReportingError: If the kind is missing or unknown. Skipping an
            unreadable block would silently drop content from a report, which
            is worse than refusing to render it.
    """
    kind = data.get("kind")
    if kind not in BLOCK_TYPES:
        raise ReportingError(
            f"unknown block kind {kind!r}. Available: "
            f"{', '.join(sorted(BLOCK_TYPES))}.")

    block_type = BLOCK_TYPES[kind]
    payload = {key: value for key, value in data.items() if key != "kind"}

    builder = getattr(block_type, "from_dict", None)
    if builder is not None:
        rebuilt: Block = builder(data)
        return rebuilt

    return block_type(**payload)


@dataclass(frozen=True)
class ReportTemplate:
    """A page setup and the blocks to draw on it.

    Attributes:
        template_id: Stable identifier, used as the DocumentStore key.
        name: Human-readable name.
        page: Sheet setup.
        blocks: Content, drawn top to bottom in order.
    """
    template_id: str
    name: str
    page: PageSetup = field(default_factory=PageSetup)
    blocks: list[Block] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready form, suitable for DocumentStore."""
        return {"template_id": self.template_id,
                "name": self.name,
                "page": self.page.to_dict(),
                "blocks": [block.to_dict() for block in self.blocks]}

    @classmethod
    def from_dict(cls,
                  data: dict[str, Any]) -> "ReportTemplate":
        """Rebuild a stored template.

        Args:
            data: The stored form.

        Returns:
            ReportTemplate: The template.

        Raises:
            ReportingError: If a block cannot be read.
        """
        return cls(template_id=data["template_id"],
                   name=data.get("name", ""),
                   page=PageSetup.from_dict(data.get("page", {})),
                   blocks=[block_from_dict(block)
                           for block in data.get("blocks", [])])
