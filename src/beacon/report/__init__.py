# src/beacon/report/__init__.py
"""
Paginated report documents.

Two halves, deliberately separable. `blocks` is the data model — what a report
is made of — and needs nothing beyond the standard library, so a client can
build templates and persist them through `DocumentStore` with no PDF library
installed. `pdf` renders one, and is the only part behind the `pdf` extra:

    pip install "py-beacon[pdf]"

Complementary to the Excel reporting in `beacon.portfolio.reporting` rather
than a replacement for it: a spreadsheet is for numbers someone will pick up
and work with, a PDF is a paginated document that looks the same everywhere and
is meant to be read as it was laid out.

Importing `beacon.report.pdf` without reportlab raises MissingDependencyError
naming the extra; importing `beacon.report.blocks` always works.
"""
from .blocks import (
    BLOCK_TYPES,
    PAGE_SIZES,
    BarChart,
    Block,
    Chart,
    Header,
    PageSetup,
    ReportTemplate,
    Stat,
    StatGrid,
    Table,
    Text,
    block_from_dict,
)

__all__ = [
    "BLOCK_TYPES",
    "PAGE_SIZES",
    "BarChart",
    "Block",
    "Chart",
    "Header",
    "PageSetup",
    "ReportTemplate",
    "Stat",
    "StatGrid",
    "Table",
    "Text",
    "block_from_dict",
]
