# src/beacon/report/pdf.py
"""
Rendering a report template to PDF.

Separate from the block model so a template can be built and stored without
reportlab installed; this module is the only thing behind the `pdf` extra.

## Deterministic by construction

Two renders of the same template produce byte-identical files. That is not
free — a PDF normally carries a creation timestamp and a random document id, so
the default behaviour is for every render to differ. reportlab's ``invariant``
mode pins both, and it is switched on here rather than offered as an option:

* a report is a record, and two people running the same template on the same
  data should be able to confirm they got the same thing by comparing hashes;
* the alternative is a diffing story where every rerender looks like a change,
  which makes review worthless.

## Colours come from the design tokens

Specifically from the `raw.paper-*` values, which are mode-independent. A
factsheet is a print artefact: it must look the same whichever theme the
surrounding application is wearing, and identical on screen to the PDF it
becomes. Using the themed tokens here would produce a dark-mode PDF, which is
not a thing anyone wants.

## One page

The renderer lays blocks out top to bottom and refuses when they do not fit,
naming the block that overflowed. Pagination is real work — headers repeating,
tables splitting, blocks that must not break across a boundary — and guessing
at it would produce documents that look almost right. Refusing is the honest
foundation; BN-75 can build on it.
"""
import logging
import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from .._optional import require
from ..exceptions import ReportingError
from ..tokens import raw_colours
from .blocks import (
    BarChart,
    Block,
    Chart,
    Header,
    PageSetup,
    ReportTemplate,
    StatGrid,
    Table,
    Text,
)

require("reportlab", "PDF reporting")

from reportlab.lib.colors import HexColor  # noqa: E402
from reportlab.lib.utils import ImageReader  # noqa: E402
from reportlab.pdfgen import canvas as pdf_canvas  # noqa: E402

logger = logging.getLogger(__name__)

# Base-14 fonts, so nothing has to be embedded and the output stays small and
# reproducible on machines with different fonts installed.
BODY_FONT = "Helvetica"
BOLD_FONT = "Helvetica-Bold"

TITLE_SIZE = 18.0
SUBTITLE_SIZE = 10.0
LABEL_SIZE = 7.5
VALUE_SIZE = 13.0
TABLE_SIZE = 8.0

# Vertical breathing room after each block, in points.
BLOCK_GAP = 16.0
LINE_GAP = 1.35

# Bars lighter than this fraction of the row height look like rules rather than
# bars, so a near-zero value still draws something visible.
MINIMUM_BAR_WIDTH = 1.0


@dataclass
class _Ink:
    """The colours a page is drawn in, read once from the design tokens."""
    page: str
    text: str
    muted: str
    rule: str
    accent: str

    @classmethod
    def from_tokens(cls) -> "_Ink":
        """Read the mode-independent paper colours."""
        raw = raw_colours()

        return cls(page=raw["paper-page"],
                   text=raw["paper-ink"],
                   muted=raw["paper-ink-muted"],
                   rule=raw["paper-rule"],
                   accent=raw["paper-accent"])


class _Cursor:
    """Where the next block goes, and how much room is left.

    PDF coordinates start at the bottom-left and grow upward, which is the
    opposite of how a report is read and written. This tracks the top edge of
    the next block in reading order and converts on the way out, so the drawing
    code below never has to think in flipped coordinates.
    """

    def __init__(self,
                 page: PageSetup):
        self.page = page
        self.top = page.dimensions[1] - page.margin
        self.floor = page.margin

    @property
    def remaining(self) -> float:
        """Points of vertical room left."""
        return self.top - self.floor

    def take(self,
             height: float,
             block: Block) -> float:
        """Reserve *height* and return the y coordinate of the block's baseline.

        Raises:
            ReportingError: If the block does not fit on the page.
        """
        if height > self.remaining + 1e-9:
            raise ReportingError(
                f"the {block.kind} block needs {height:.0f}pt but only "
                f"{self.remaining:.0f}pt of the page is left. This renderer "
                f"produces a single page: shorten the content, raise the page "
                f"size, or drop a block.")

        self.top -= height

        return self.top


def render(template: ReportTemplate,
           output_path: str | Path) -> Path:
    """Draw a template to a PDF file.

    Args:
        template: What to draw.
        output_path: Where to write it. Parent directories are created.

    Returns:
        Path: The written file.

    Raises:
        ReportingError: If the content does not fit on one page.
        MissingDependencyError: If reportlab is not installed.
    """
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    ink = _Ink.from_tokens()
    page = template.page

    # invariant=1 pins the creation timestamp and document id, which is what
    # makes two renders of one template byte-identical.
    canvas = pdf_canvas.Canvas(str(destination),
                               pagesize=page.dimensions,
                               invariant=1)
    canvas.setTitle(template.name)
    canvas.setAuthor("Beacon")
    canvas.setSubject(template.template_id)

    _fill_page(canvas, page, ink)

    cursor = _Cursor(page)
    for block in template.blocks:
        _draw(canvas, block, cursor, page, ink)

    canvas.showPage()
    canvas.save()

    logger.info(
        f"Rendered report '{template.template_id}' with {len(template.blocks)} "
        f"block(s) to {destination}.")

    return destination


def _fill_page(canvas: "pdf_canvas.Canvas",
               page: PageSetup,
               ink: _Ink) -> None:
    """Lay down the page colour behind everything else."""
    canvas.setFillColor(HexColor(ink.page))
    canvas.rect(0, 0, *page.dimensions, stroke=0, fill=1)


def _draw(canvas: "pdf_canvas.Canvas",
          block: Block,
          cursor: _Cursor,
          page: PageSetup,
          ink: _Ink) -> None:
    """Dispatch one block to its drawing routine."""
    drawers: dict[type[Block], Callable[..., None]] = {
        Header: _draw_header,
        Text: _draw_text,
        StatGrid: _draw_stat_grid,
        Table: _draw_table,
        BarChart: _draw_bar_chart,
        Chart: _draw_chart,
    }

    drawer = drawers.get(type(block))
    if drawer is None:
        raise ReportingError(f"no renderer for a {block.kind} block.")

    drawer(canvas, block, cursor, page, ink)
    cursor.top -= BLOCK_GAP


def _draw_header(canvas: "pdf_canvas.Canvas",
                 block: Header,
                 cursor: _Cursor,
                 page: PageSetup,
                 ink: _Ink) -> None:
    """Title, optional subtitle and as-of, with a rule beneath."""
    height = TITLE_SIZE * LINE_GAP + 10.0
    if block.subtitle or block.as_of:
        height += SUBTITLE_SIZE * LINE_GAP

    baseline = cursor.take(height, block)
    left = page.margin

    canvas.setFillColor(HexColor(ink.text))
    canvas.setFont(BOLD_FONT, TITLE_SIZE)
    canvas.drawString(left, baseline + height - TITLE_SIZE, block.title)

    if block.subtitle or block.as_of:
        canvas.setFont(BODY_FONT, SUBTITLE_SIZE)
        canvas.setFillColor(HexColor(ink.muted))
        canvas.drawString(left, baseline + 8.0, block.subtitle)
        if block.as_of:
            canvas.drawRightString(page.dimensions[0] - page.margin,
                                   baseline + 8.0, block.as_of)

    canvas.setStrokeColor(HexColor(ink.rule))
    canvas.setLineWidth(0.75)
    canvas.line(left, baseline + 2.0, page.dimensions[0] - page.margin, baseline + 2.0)


def _draw_text(canvas: "pdf_canvas.Canvas",
               block: Text,
               cursor: _Cursor,
               page: PageSetup,
               ink: _Ink) -> None:
    """A wrapped paragraph."""
    lines = _wrap(canvas, block.body, block.size, page.content_width)
    height = len(lines) * block.size * LINE_GAP

    baseline = cursor.take(height, block)

    canvas.setFont(BODY_FONT, block.size)
    canvas.setFillColor(HexColor(ink.muted if block.muted else ink.text))

    for index, line in enumerate(lines):
        offset = height - (index + 1) * block.size * LINE_GAP
        canvas.drawString(page.margin, baseline + offset, line)


def _wrap(canvas: "pdf_canvas.Canvas",
          body: str,
          size: float,
          width: float) -> list[str]:
    """Break text into lines that fit *width*, measured in the real font.

    Measured rather than estimated by character count, because Helvetica is
    proportional and a count-based guess overflows on capitals and wastes half
    the line on digits.
    """
    lines: list[str] = []

    for paragraph in body.split("\n"):
        current = ""

        for word in paragraph.split():
            candidate = f"{current} {word}".strip()
            if canvas.stringWidth(candidate, BODY_FONT, size) <= width:
                current = candidate
                continue

            if current:
                lines.append(current)
            current = word

        lines.append(current)

    return lines


def _draw_stat_grid(canvas: "pdf_canvas.Canvas",
                    block: StatGrid,
                    cursor: _Cursor,
                    page: PageSetup,
                    ink: _Ink) -> None:
    """Headline figures in a grid, label above value."""
    rows = -(-len(block.stats) // block.columns)
    row_height = LABEL_SIZE * LINE_GAP + VALUE_SIZE * LINE_GAP + 12.0
    height = rows * row_height

    baseline = cursor.take(height, block)
    column_width = page.content_width / block.columns

    for index, stat in enumerate(block.stats):
        row, column = divmod(index, block.columns)
        left = page.margin + column * column_width
        top = baseline + height - row * row_height

        canvas.setFont(BODY_FONT, LABEL_SIZE)
        canvas.setFillColor(HexColor(ink.muted))
        canvas.drawString(left, top - LABEL_SIZE, stat.label.upper())

        canvas.setFont(BOLD_FONT, VALUE_SIZE)
        canvas.setFillColor(HexColor(ink.text))
        canvas.drawString(left, top - LABEL_SIZE - VALUE_SIZE - 2.0, stat.value)

        if stat.change:
            canvas.setFont(BODY_FONT, LABEL_SIZE)
            canvas.setFillColor(HexColor(ink.muted))
            canvas.drawString(left, top - LABEL_SIZE - VALUE_SIZE - 12.0,
                              stat.change)


def _draw_table(canvas: "pdf_canvas.Canvas",
                block: Table,
                cursor: _Cursor,
                page: PageSetup,
                ink: _Ink) -> None:
    """A header row, a rule, and the data rows."""
    row_height = TABLE_SIZE * LINE_GAP + 4.0
    title_height = (TABLE_SIZE + 6.0) if block.title else 0.0
    height = title_height + row_height * (len(block.rows) + 1) + 4.0

    baseline = cursor.take(height, block)
    top = baseline + height
    column_width = page.content_width / len(block.columns)

    if block.title:
        canvas.setFont(BOLD_FONT, TABLE_SIZE + 1.0)
        canvas.setFillColor(HexColor(ink.text))
        canvas.drawString(page.margin, top - TABLE_SIZE, block.title)
        top -= title_height

    canvas.setFont(BOLD_FONT, TABLE_SIZE)
    canvas.setFillColor(HexColor(ink.muted))
    for index, label in enumerate(block.columns):
        _cell(canvas, label, index, column_width, top - TABLE_SIZE, page, block)

    rule_y = top - TABLE_SIZE - 3.0
    canvas.setStrokeColor(HexColor(ink.rule))
    canvas.setLineWidth(0.5)
    canvas.line(page.margin, rule_y, page.dimensions[0] - page.margin, rule_y)

    canvas.setFont(BODY_FONT, TABLE_SIZE)
    canvas.setFillColor(HexColor(ink.text))
    for row_index, row in enumerate(block.rows):
        text_y = rule_y - (row_index + 1) * row_height + 2.0
        for column_index, value in enumerate(row):
            _cell(canvas, value, column_index, column_width, text_y, page, block)


def _cell(canvas: "pdf_canvas.Canvas",
          value: str,
          index: int,
          column_width: float,
          text_y: float,
          page: PageSetup,
          block: Table) -> None:
    """Draw one cell, honouring the block's right-alignment list."""
    left = page.margin + index * column_width

    if index in block.align_right:
        canvas.drawRightString(left + column_width - 6.0, text_y, value)
    else:
        canvas.drawString(left, text_y, value)


def _draw_bar_chart(canvas: "pdf_canvas.Canvas",
                    block: BarChart,
                    cursor: _Cursor,
                    page: PageSetup,
                    ink: _Ink) -> None:
    """Horizontal bars, drawn natively so no plotting extra is needed."""
    title_height = (TABLE_SIZE + 6.0) if block.title else 0.0
    height = block.height + title_height

    baseline = cursor.take(height, block)
    top = baseline + height

    if block.title:
        canvas.setFont(BOLD_FONT, TABLE_SIZE + 1.0)
        canvas.setFillColor(HexColor(ink.text))
        canvas.drawString(page.margin, top - TABLE_SIZE, block.title)
        top -= title_height

    _draw_bars(canvas, block, top, page, ink)


def _draw_bars(canvas: "pdf_canvas.Canvas",
               block: BarChart,
               top: float,
               page: PageSetup,
               ink: _Ink) -> None:
    """The bars themselves, with a zero axis when any value is negative."""
    label_width = page.content_width * 0.32
    plot_left = page.margin + label_width
    plot_width = page.content_width - label_width
    row_height = block.height / len(block.categories)

    largest = max(abs(value) for value in block.values) or 1.0
    has_negative = any(value < 0 for value in block.values)

    # With negatives present the axis sits in the middle so bars grow both
    # ways; otherwise it sits at the left and the full width is available.
    axis = plot_left + (plot_width / 2 if has_negative else 0.0)
    span = (plot_width / 2) if has_negative else plot_width

    for index, (label, value) in enumerate(zip(block.categories, block.values,
                                               strict=True)):
        centre = top - (index + 0.5) * row_height

        canvas.setFont(BODY_FONT, LABEL_SIZE)
        canvas.setFillColor(HexColor(ink.muted))
        canvas.drawString(page.margin, centre - LABEL_SIZE / 2, label)

        length = max(abs(value) / largest * span, MINIMUM_BAR_WIDTH)
        left = axis - length if value < 0 else axis

        canvas.setFillColor(HexColor(ink.accent))
        canvas.rect(left, centre - row_height * 0.3, length, row_height * 0.6,
                    stroke=0, fill=1)

    if has_negative:
        canvas.setStrokeColor(HexColor(ink.rule))
        canvas.setLineWidth(0.5)
        canvas.line(axis, top - block.height, axis, top)


def _draw_chart(canvas: "pdf_canvas.Canvas",
                block: Chart,
                cursor: _Cursor,
                page: PageSetup,
                ink: _Ink) -> None:
    """A rendered image, or a placeholder when there is not one yet."""
    title_height = (TABLE_SIZE + 6.0) if block.title else 0.0
    height = block.height + title_height

    baseline = cursor.take(height, block)
    top = baseline + height

    if block.title:
        canvas.setFont(BOLD_FONT, TABLE_SIZE + 1.0)
        canvas.setFillColor(HexColor(ink.text))
        canvas.drawString(page.margin, top - TABLE_SIZE, block.title)
        top -= title_height

    if block.image_path and os.path.exists(block.image_path):
        canvas.drawImage(ImageReader(block.image_path),
                         page.margin, top - block.height,
                         width=page.content_width, height=block.height,
                         preserveAspectRatio=True, anchor="c", mask="auto")
        return

    _draw_placeholder(canvas, block, top, page, ink)


def _draw_placeholder(canvas: "pdf_canvas.Canvas",
                      block: Chart,
                      top: float,
                      page: PageSetup,
                      ink: _Ink) -> None:
    """An outlined box saying what is missing.

    A template gets designed before the charts it will hold exist, so a missing
    image draws a placeholder rather than failing. It says so on the page — an
    empty box with no explanation reads as a rendering bug.
    """
    canvas.setStrokeColor(HexColor(ink.rule))
    canvas.setLineWidth(0.75)
    canvas.setDash(3, 3)
    canvas.rect(page.margin, top - block.height, page.content_width, block.height,
                stroke=1, fill=0)
    canvas.setDash()

    canvas.setFont(BODY_FONT, LABEL_SIZE)
    canvas.setFillColor(HexColor(ink.muted))
    canvas.drawCentredString(page.margin + page.content_width / 2,
                             top - block.height / 2,
                             block.title or "chart")
