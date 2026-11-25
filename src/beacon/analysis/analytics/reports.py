# src/beacon/analysis/analytics/reports.py
"""
A run's reports, the second way into its analytics: a report's statistics
worked out together.

    report = result.analytics.reports.factsheet(level="extended", add=["calmar_ratio"])
    report.statistics            # the ids, in the report's order
    report["sharpe_ratio"]       # what portfolio.performance.sharpe_ratio() gives
    report.to_frame()            # its single figures side by side
    result.analytics.reports.asked     # the reports the run's request named

Each value is what the statistic's analytic gives called bare, at the run's
settings or the ``options`` given. A statistic folded into another's form or
held by the result (`beacon.analysis.placement`) keeps its id here, for the
reports that show it. The reports and their levels are
`beacon.analysis.reports.definitions`'.
"""
# BN-456 (Karan, 2026-11-24).
from collections.abc import Sequence
from functools import cached_property
from typing import Any

import pandas as pd

from ..reports.catalogue import CATALOGUE
from ..reports.definitions import CORE, REPORTS
from ..reports.options import AnalysisOptions
from ..reports.statistics import Unavailable
from ..request import FAILED, AnalysisRequest, analyse_run
from .tables import single_figure


class Report:
    """One report of a run, its statistics worked out.

    Attributes:
        name: The report's id, such as ``factsheet``.
        label: How it reads.
        level: ``core`` or ``extended``.
        statistics: The ids of its statistics, in the report's order.
        views: Each statistic's time view: ``point``, ``rolling`` or
            ``both``.
        unavailable: Why each statistic the run cannot give has no value, by
            id.
        optional: The statistics ``add=`` could bring in at its level.
    """

    def __init__(self,
                 analytics: Any,
                 request: AnalysisRequest,
                 name: str):
        definition = REPORTS[name]
        level = request.reports[name]
        self.name = name
        self.label = definition.label
        self.level = level
        self._analytics = analytics
        self._request = request
        entries = request.selection()[name]
        self.views = dict(entries)
        self.statistics = [statistic_id for statistic_id, _ in entries]
        self.optional = [statistic_id for statistic_id in definition.level(level).optional
                         if statistic_id not in self.views]
        self._values: dict[str, Any] = {}
        self.unavailable: dict[str, str] = {}

        for statistic_id in self.statistics:
            self._work_out(statistic_id)

    def __getitem__(self,
                    statistic: str) -> Any:
        """The value of *statistic*, by id or `Statistic` member.

        Raises:
            KeyError: If the report does not hold it, saying whether the
                run could not give it or ``add=`` could bring it in.
        """
        statistic_id = str(statistic)

        if statistic_id in self._values:
            return self._values[statistic_id]

        raise KeyError(self._missing(statistic_id))

    def __contains__(self,
                     statistic: object) -> bool:
        return str(statistic) in self._values

    def values(self) -> dict[str, Any]:
        """Each statistic's value, by id, in the report's order."""
        return dict(self._values)

    def to_frame(self) -> pd.DataFrame:
        """Its single-figure statistics side by side: a row a statistic, a
        column a portfolio, as `analytics.table(..., of="all")` gives them."""
        ids = [statistic_id for statistic_id in self._values
               if single_figure(CATALOGUE[statistic_id])]

        return self._analytics.table(ids, of="all") if ids else pd.DataFrame()

    def to_dict(self) -> dict[str, Any]:
        """The report as the API gives it: the analytics block of a request
        for this report alone (`beacon.analysis.request.analyse`)."""
        return analyse_run(self._analytics._context, self._request)

    def _work_out(self,
                  statistic_id: str) -> None:
        try:
            self._values[statistic_id] = self._analytics._value(statistic_id, {})
        except Unavailable as reason:
            self.unavailable[statistic_id] = reason.reason
        except Exception as error:  # A failure is named, and does not stop the rest.
            self.unavailable[statistic_id] = f"{FAILED}: {error}"

    def _missing(self,
                 statistic_id: str) -> str:
        """Why the report holds no *statistic_id*."""
        if statistic_id not in CATALOGUE:
            return f"Unknown statistic {statistic_id!r}."

        if statistic_id in self.unavailable:
            return f"{statistic_id} could not be given: {self.unavailable[statistic_id]}"

        if statistic_id in self.optional:
            return (f"{statistic_id} is optional in the {self.name} {self.level}; add it with "
                    f"add=[{statistic_id!r}].")

        return f"The {self.name} report does not hold {statistic_id} at either level."

    def __repr__(self) -> str:
        return (f"Report({self.name}, {self.level}: {len(self._values)} statistics, "
                f"{len(self.unavailable)} unavailable)")


class Reports:
    """A run's reports: a method a report, and those its request asked
    for (`asked`)."""

    def __init__(self,
                 analytics: Any):
        self._analytics = analytics

    @cached_property
    def asked(self) -> dict[str, Report]:
        """The reports the run's request named, at its levels and options."""
        request = self._analytics.request or AnalysisRequest()

        return {name: Report(self._analytics, request, name) for name in request.reports}

    def factsheet(self,
                  *,
                  level: str = CORE,
                  add: Sequence[str] = (),
                  remove: Sequence[str] = (),
                  options: AnalysisOptions | None = None) -> Report:
        """The factsheet: the headline figures across every category."""
        return self._report("factsheet", level, add, remove, options)

    def performance(self,
                    *,
                    level: str = CORE,
                    add: Sequence[str] = (),
                    remove: Sequence[str] = (),
                    options: AnalysisOptions | None = None) -> Report:
        """The performance report: returns, drawdowns and risk-adjusted
        ratios."""
        return self._report("performance", level, add, remove, options)

    def risk(self,
             *,
             level: str = CORE,
             add: Sequence[str] = (),
             remove: Sequence[str] = (),
             options: AnalysisOptions | None = None) -> Report:
        """The risk report: volatility, the distribution of returns, value
        at risk and the risk model."""
        return self._report("risk", level, add, remove, options)

    def tracking_error(self,
                       *,
                       level: str = CORE,
                       add: Sequence[str] = (),
                       remove: Sequence[str] = (),
                       options: AnalysisOptions | None = None) -> Report:
        """The tracking error report: the fund against its index."""
        return self._report("tracking_error", level, add, remove, options)

    def factor(self,
               *,
               level: str = CORE,
               add: Sequence[str] = (),
               remove: Sequence[str] = (),
               options: AnalysisOptions | None = None) -> Report:
        """The factor report: exposures, returns and betas."""
        return self._report("factor", level, add, remove, options)

    def portfolio(self,
                  *,
                  level: str = CORE,
                  add: Sequence[str] = (),
                  remove: Sequence[str] = (),
                  options: AnalysisOptions | None = None) -> Report:
        """The portfolio report: holdings, concentration and liquidity."""
        return self._report("portfolio", level, add, remove, options)

    def attribution(self,
                    *,
                    level: str = CORE,
                    add: Sequence[str] = (),
                    remove: Sequence[str] = (),
                    options: AnalysisOptions | None = None) -> Report:
        """The attribution report: where the return came from."""
        return self._report("attribution", level, add, remove, options)

    def implementation(self,
                       *,
                       level: str = CORE,
                       add: Sequence[str] = (),
                       remove: Sequence[str] = (),
                       options: AnalysisOptions | None = None) -> Report:
        """The implementation report: trading, costs, cash and the
        vehicle."""
        return self._report("implementation", level, add, remove, options)

    def _report(self,
                name: str,
                level: str,
                add: Sequence[str],
                remove: Sequence[str],
                options: AnalysisOptions | None) -> Report:
        """Report *name* at *level*, worked out at the run's settings or
        *options*.

        Raises:
            ValueError: If the level is unknown, a statistic added is not
                optional at it, or one removed is not in it.
        """
        asked = self._analytics.request or AnalysisRequest()
        chosen = options or asked.options
        request = AnalysisRequest(reports={name: level},
                                  add={name: [str(id_) for id_ in add]},
                                  remove={name: [str(id_) for id_ in remove]},
                                  statistic_tables={}, options=chosen, factors=asked.factors)
        analytics = self._analytics if chosen == asked.options else self._analytics._with(request)

        return Report(analytics, request, name)

    def __repr__(self) -> str:
        return f"Reports({', '.join(REPORTS)})"


__all__ = ["Report", "Reports"]
