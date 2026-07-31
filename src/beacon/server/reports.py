# src/beacon/server/reports.py
"""
Rendering reports: the built-in factsheet, and rendering a stored template.

Two kinds of template, and the distinction matters:

* A **stored template** is a block list a user built. It is rendered exactly as
  saved, because that is what "I designed this page" means. Nothing is
  substituted into it.
* A **built-in template** is generated from a completed run. `FACTSHEET-A4`
  is the first: it reads an index's latest backtest and lays out the headline
  figures, the constituents and the contribution chart.

The alternative was a templating language — placeholders in a stored document,
filled at render time. That is a real feature and a large one, and inventing a
half-version of it (string substitution into text blocks) would produce
something that looks like a template engine and breaks like a string replace.
Built-in templates are the honest smaller thing: they are code, they are
testable, and a user who needs a different factsheet gets a new built-in rather
than a syntax to learn.

## Rendering is a job

The PDF itself takes milliseconds. Building the factsheet does not: it reads a
stored run and derives contributions from it. More importantly the result is
*bytes*, and bytes do not belong in a JSON job payload — so the render writes a
file under the storage root and the job result carries its id, which
`GET /reports/renders/{render_id}` then streams.
"""
import logging
import uuid
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

import pandas as pd

from ..exceptions import DataNotFoundError, ReportingError
from ..report.blocks import (
    BarChart,
    Chart,
    Header,
    PageSetup,
    ReportTemplate,
    Stat,
    StatGrid,
    Table,
    Text,
)
from ..report.pdf import render
from .jobs import ProgressReporter
from .schemas import RenderRequest, RenderResult

logger = logging.getLogger(__name__)

# The built-in template ids a caller can render without storing anything.
FACTSHEET = "FACTSHEET-A4"
BUILT_IN = (FACTSHEET,)

# How many constituents the factsheet's table and chart show. Enough to fill
# the page without pushing the chart off it.
TOP_HOLDINGS = 8


def is_built_in(template_id: str) -> bool:
    """Whether a template is generated rather than stored."""
    return template_id in BUILT_IN


def build_factsheet(index_name: str,
                    run: dict[str, Any]) -> ReportTemplate:
    """Lay out a one-page factsheet from a completed backtest.

    Args:
        index_name: Display name for the header.
        run: A stored backtest result.

    Returns:
        ReportTemplate: The blocks, in reading order.

    Raises:
        DataNotFoundError: If the run carries no composition — a result stored
            before BN-71 has a level but no constituents, and a factsheet
            without holdings is not a factsheet.
    """
    snapshots = run.get("rebalances") or []
    if not snapshots:
        raise DataNotFoundError(
            "rebalance snapshots on this run",
            source="the run predates composition being stored; re-run the backtest")

    latest = snapshots[-1]
    metrics = run["metrics"]
    level = run["level"]
    as_of = str(level["index"][-1])[:10] if level["index"] else latest["date"]

    weights = sorted(latest["weights"].items(), key=lambda item: item[1],
                     reverse=True)
    shown = weights[:TOP_HOLDINGS]

    return ReportTemplate(
        template_id=FACTSHEET,
        name=f"{index_name} factsheet",
        page=PageSetup(size="A4"),
        blocks=[
            Header(title=index_name,
                   subtitle="Index factsheet",
                   as_of=as_of),
            StatGrid(stats=_headline(metrics, len(latest["weights"]))),
            Text(body=_summary(run, latest), muted=True),
            BarChart(categories=[name for name, _ in shown],
                     values=[value for _, value in shown],
                     title=f"Top {len(shown)} constituents by weight",
                     height=110.0),
            Table(columns=["Constituent", "Weight"],
                  rows=[[name, f"{value:.2%}"] for name, value in shown],
                  title="Holdings",
                  align_right=[1]),
            Chart(title="Index level", height=105.0),
        ])


def _headline(metrics: dict[str, Any],
              constituents: int) -> list[Stat]:
    """The four figures a factsheet leads with.

    Formatted here rather than in the block model: a percentage, a ratio and a
    count all need different treatment, and only the caller knows which is
    which.
    """
    return [
        Stat(label="Total return", value=_percent(metrics.get("total_return"))),
        Stat(label="Annualised", value=_percent(metrics.get("annualised_return"))),
        Stat(label="Volatility", value=_percent(metrics.get("volatility"))),
        Stat(label="Constituents", value=str(constituents)),
    ]


def _percent(value: Any) -> str:
    """A fraction as a percentage, or an em dash when it is missing.

    A dash rather than 0.00%: a metric that could not be computed and one that
    came out at zero are different statements, and a factsheet is read by
    people who will not check which.
    """
    if value is None:
        return "—"

    return f"{float(value):.2%}"


def _summary(run: dict[str, Any],
             latest: dict[str, Any]) -> str:
    """A sentence describing how the index is built."""
    cap = latest.get("cap")
    capping = (f" No constituent may exceed {float(cap):.1%} of the index."
               if cap is not None else "")

    tracking = run["metrics"].get("tracking_error")
    replication = (f" The simulated portfolio tracked the index to within "
                   f"{float(tracking):.2%}." if tracking is not None else "")

    return (f"Rebalanced across {len(run.get('rebalances', []))} dates over the "
            f"period shown.{capping}{replication}")


def render_path(directory: Path,
                render_id: str) -> Path:
    """Where a rendered PDF lives."""
    return directory / f"{render_id}.pdf"


def build_render_job(render_id: str,
                     request: RenderRequest,
                     template: ReportTemplate,
                     directory: Path
                     ) -> Callable[[ProgressReporter], Awaitable[dict[str, Any]]]:
    """Build the coroutine that renders a template to a PDF.

    Returns:
        A coroutine function suitable for JobRegistry.submit.
    """
    async def run(report: ProgressReporter) -> dict[str, Any]:
        await report(0.2, f"Rendering '{template.template_id}'.")

        destination = render_path(directory, render_id)
        render(template, destination)

        await report(1.0, "Complete.")

        result = RenderResult(render_id=render_id,
                              template_id=template.template_id,
                              index_id=request.index_id,
                              name=template.name,
                              blocks=len(template.blocks),
                              bytes=destination.stat().st_size,
                              rendered_at=pd.Timestamp.now(tz="UTC").isoformat())

        logger.info(
            f"Rendered report '{render_id}' from template "
            f"'{template.template_id}': {result.bytes} bytes.")

        return result.model_dump()

    return run


def new_render_id() -> str:
    """An identifier for one rendered document."""
    return str(uuid.uuid4())


def ensure_renderable(template: ReportTemplate) -> None:
    """Raise if a template obviously cannot produce a page.

    Only the checks that are cheap and certain. Whether the blocks *fit* is the
    renderer's answer, and it gives a better one — naming the block that
    overflowed — so it is left to say it.
    """
    if not template.blocks:
        raise ReportingError(
            f"template '{template.template_id}' has no blocks, so it would "
            f"render an empty page.")
