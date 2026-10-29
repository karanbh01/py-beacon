# src/beacon/analysis/request.py
"""
What a run's analysis should give: the reports wanted, each at a level, with
statistics added to or taken from them, the time view each is shown in, and
every choice the analysis makes.

    Backtest(...).run(definition, start, end,
                      analysis=AnalysisRequest(reports={"factsheet": "core"},
                                               add={"factsheet": ["sortino_ratio"]}))

The run then carries the analytics block (`analyse`): each statistic once,
however many reports show it, and the reports as lists of statistic ids. Only
what the statistics asked for need is worked out, so a factsheet's core never
builds the risk model.
"""
# BN-398. Replaces BN-397's portfolio/risk/attribution switches: the reports
# decide what is worked out (Karan, 2026-10-29).
import dataclasses
import logging
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from ..strategy.factors import Factor
from .performance.settings import AnalysisSettings
from .reports.options import AnalysisOptions

logger = logging.getLogger(__name__)

PERFORMANCE = "performance"
PORTFOLIO = "portfolio"
RISK = "risk"
ATTRIBUTION = "attribution"
PARTS = (PERFORMANCE, PORTFOLIO, RISK, ATTRIBUTION)

# Building the run's tables, which every part but performance reads.
TABLES = "tables"

# What each part says as it starts.
MESSAGES = {PERFORMANCE: "Computing performance", PORTFOLIO: "Computing the portfolio figures",
            RISK: "Building the risk model", ATTRIBUTION: "Working out the attribution"}

# The code of a statistic that failed to be worked out for another reason.
FAILED = "failed"


@dataclass(frozen=True)
class AnalysisRequest:
    """What a run's analysis should give.

    Attributes:
        reports: Each report wanted, by id, with its level: ``core`` or
            ``extended``.
        add: By report, optional statistics to show.
        remove: By report, included statistics not to show.
        time_views: By report, statistic ids with the time view to show
            them in: ``point``, ``rolling`` or ``both``.
        statistics: Statistics wanted outside any report, over the window.
        options: Every choice the analysis makes; the defaults when left out.
        factors: The factors risk and attribution use; the defaults (size,
            value, momentum) when None.

    Raises:
        ValueError: If a report, level or statistic is unknown, a statistic
            added is not optional at the report's level or one taken out is
            not in it, or a time view is unknown or rolls a statistic that
            does not roll.
    """
    reports: Mapping[str, str] = field(default_factory=lambda: {"factsheet": "core"})
    add: Mapping[str, Sequence[str]] = field(default_factory=dict)
    remove: Mapping[str, Sequence[str]] = field(default_factory=dict)
    time_views: Mapping[str, Mapping[str, str]] = field(default_factory=dict)
    statistics: Sequence[str] = ()
    options: AnalysisOptions = field(default_factory=AnalysisOptions)
    factors: Sequence[Factor] | None = None

    def __post_init__(self) -> None:
        self.selection()

    @property
    def settings(self) -> AnalysisSettings:
        """The window the figures are measured over and against."""
        return self.options.window

    def selection(self) -> dict[str, list[tuple[str, str]]]:
        """Each report's statistics to show, in order, with their views:
        its level's included ones, with those added and without those taken
        out."""
        from .reports.catalogue import CATALOGUE  # noqa: PLC0415
        from .reports.definitions import REPORTS, VIEWS  # noqa: PLC0415

        _known(list(self.statistics), CATALOGUE.statistics, "statistic")
        _known([*self.add, *self.remove, *self.time_views], self.reports, "report asked for")
        chosen = {}

        for report_id, level_name in self.reports.items():
            _known([report_id], REPORTS, "report")
            level = REPORTS[report_id].level(level_name)
            added, removed = list(self.add.get(report_id, ())), list(self.remove.get(report_id,
                                                                                       ()))
            _check(added, level.optional, f"optional in the {report_id} {level_name}")
            _check(removed, level.included, f"included in the {report_id} {level_name}")
            views = dict(self.time_views.get(report_id, {}))

            for statistic_id, view in views.items():
                if view not in VIEWS:
                    raise ValueError(f"A time view is {', '.join(VIEWS)}, got {view!r}.")

                if view != "point" and CATALOGUE[statistic_id].rolling is None:
                    raise ValueError(f"{CATALOGUE[statistic_id].label} does not roll.")

            chosen[report_id] = [(entry.id, views.get(entry.id, entry.view))
                                 for entry in level.entries
                                 if (entry.state == "included" and entry.id not in removed)
                                 or entry.id in added]

        return chosen

    def wanted(self) -> dict[str, set[str]]:
        """Each statistic to work out, with the views it is wanted in."""
        found: dict[str, set[str]] = {statistic_id: {"point"} for statistic_id in self.statistics}

        for entries in self.selection().values():
            for statistic_id, view in entries:
                found.setdefault(statistic_id, set()).update(
                    {"point", "rolling"} if view == "both" else {view})

        return found

    def parts(self) -> list[str]:
        """The parts the statistics wanted are worked out by, in order."""
        from .reports.catalogue import CATALOGUE  # noqa: PLC0415

        needed = {CATALOGUE[statistic_id].part for statistic_id in self.wanted()}

        return [part for part in PARTS if part in needed]

    def as_data(self) -> dict[str, Any]:
        """The request as plain data, every default filled in."""
        return {"reports": dict(self.reports),
                "add": {name: list(ids) for name, ids in self.add.items()},
                "remove": {name: list(ids) for name, ids in self.remove.items()},
                "time_views": {name: dict(views) for name, views in self.time_views.items()},
                "statistics": list(self.statistics),
                "options": _plain(dataclasses.asdict(self.options)),
                "factors": None if self.factors is None else [factor.name
                                                              for factor in self.factors]}


def analyse(result: Any,
            request: AnalysisRequest,
            progress: Callable[[str], None] | None = None) -> dict[str, Any]:
    """The analytics block *request* asks for, of the finished run *result*.

    Each statistic is worked out once, part by part, with a message as each
    part starts; one that cannot be worked out for the run is named under
    ``unavailable`` with a code and a reason.

    Returns:
        dict: ``request``, ``reports`` (each report's level and statistics,
        by id, with their views), ``statistics`` (each one's kind, its
        ``point`` value and, where wanted, its ``rolling`` table, and a
        table's ``summary``), ``unavailable``, ``active_against`` and
        ``timings`` (seconds, the run's tables first).
    """
    from .reports.context import Context  # noqa: PLC0415

    run = result.run_data
    context = Context(run, request.options, getattr(result, "_data_fetcher", None),
                      request.factors)

    return analyse_run(context, request, progress)


def analyse_run(context: Any,
                request: AnalysisRequest,
                progress: Callable[[str], None] | None = None) -> dict[str, Any]:
    """`analyse` on a run's `Context`, for a stored run as for a finished one."""
    # The panels import the backtest package, which imports this module.
    from .data import EXPECTED  # noqa: PLC0415
    from .reports.catalogue import CATALOGUE  # noqa: PLC0415

    say = progress or (lambda message: None)
    wanted = request.wanted()
    start = time.perf_counter()

    for name in EXPECTED:
        if name in context.run.tables:
            context.run.tables.get(name)

    timings = {TABLES: time.perf_counter() - start}
    statistics: dict[str, Any] = {}
    unavailable: dict[str, Any] = {}

    for part in request.parts():
        say(MESSAGES[part])
        start = time.perf_counter()

        chosen = {statistic_id: views for statistic_id, views in wanted.items()
                  if CATALOGUE[statistic_id].part == part}

        for statistic_id in chosen:
            _work_out(CATALOGUE[statistic_id], context, statistics, unavailable)

        _roll([CATALOGUE[statistic_id] for statistic_id, views in chosen.items()
               if "rolling" in views and statistic_id in statistics], context, statistics,
              unavailable)

        timings[part] = time.perf_counter() - start

    reports = {report_id: {"level": request.reports[report_id],
                           "statistics": [{"id": statistic_id, "view": view}
                                          for statistic_id, view in entries]}
               for report_id, entries in request.selection().items()}

    return {"request": request.as_data(), "reports": reports, "statistics": statistics,
            "unavailable": unavailable, "active_against": context.tracked, "timings": timings}


def _work_out(statistic: Any,
              context: Any,
              statistics: dict[str, Any],
              unavailable: dict[str, Any]) -> None:
    """One statistic's point value, or why it has none."""
    from .reports.statistics import Unavailable  # noqa: PLC0415
    from .reports.wire import payload  # noqa: PLC0415

    try:
        entry = {"kind": statistic.kind, **payload(statistic.kind, statistic.compute(context))}
    except Unavailable as reason:
        unavailable[statistic.id] = {"code": reason.code, "reason": reason.reason}
        return
    except Exception as error:  # A failure shows in place, and does not stop the rest.
        logger.warning(f"{statistic.label} could not be worked out: {error}")
        unavailable[statistic.id] = {"code": FAILED,
                                     "reason": f"It could not be worked out: {error}"}
        return

    statistics[statistic.id] = entry


def _roll(chosen: list[Any],
          context: Any,
          statistics: dict[str, Any],
          unavailable: dict[str, Any]) -> None:
    """The rolling tables of *chosen*, worked out together; one that fails
    alone is named as failing, and the rest keep theirs."""
    from .reports.rolling import rolling_all, step_of  # noqa: PLC0415
    from .reports.wire import table_payload  # noqa: PLC0415

    if not chosen:
        return

    try:
        tables = rolling_all(chosen, context)
    except Exception:  # Worked out one at a time, to find the one that fails.
        tables = {}

        for statistic in chosen:
            try:
                tables.update(rolling_all([statistic], context))
            except Exception as error:
                logger.warning(f"{statistic.label} could not be rolled: {error}")
                statistics.pop(statistic.id, None)
                unavailable[statistic.id] = {"code": FAILED,
                                             "reason": f"It could not be rolled: {error}"}

    for statistic in chosen:
        if statistic.id in tables:
            statistics[statistic.id]["rolling"] = {
                "window": context.options.rolling.window, "step": step_of(statistic, context),
                "table": table_payload(tables[statistic.id])}


def _known(names: list[str],
           known: Mapping[str, Any],
           what: str) -> None:
    unknown = [name for name in names if name not in known]

    if unknown:
        raise ValueError(f"Unknown {what}: {', '.join(unknown)}. Known: {', '.join(known)}.")


def _check(names: list[str],
           allowed: list[str],
           what: str) -> None:
    wrong = [name for name in names if name not in allowed]

    if wrong:
        raise ValueError(f"Not {what}: {', '.join(wrong)}.")


def _plain(value: Any) -> Any:
    """Options as plain data: tuples as lists, dates as text."""
    if isinstance(value, dict):
        return {key: _plain(item) for key, item in value.items()}

    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]

    if isinstance(value, pd.Timestamp):
        return value.strftime("%Y-%m-%d")

    return value
