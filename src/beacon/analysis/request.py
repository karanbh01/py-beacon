# src/beacon/analysis/request.py
"""
What a backtest should analyse up front: its performance always, and its
portfolio, risk and attribution when asked, with the settings they use.

    Backtest(...).run(definition, start, end,
                      analysis=AnalysisRequest(risk=True, attribution=True))

The result keeps the request and builds each part on first read either way
(`result.analytics`, `result.portfolio_analytics`, `result.risk`,
`result.attribution`); asking for a part works out its standard figures
during the run and records how long each part took.
"""
# BN-397. Which figures make each part's standard set is provisional until
# the review of standard and optional figures at the end of the analytics
# work (Karan, 2026-10-27).
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from ..strategy.factors import Factor
from .performance.settings import AnalysisSettings

PERFORMANCE = "performance"
PORTFOLIO = "portfolio"
RISK = "risk"
ATTRIBUTION = "attribution"
PARTS = (PERFORMANCE, PORTFOLIO, RISK, ATTRIBUTION)

# Building the run's tables, which every part but performance reads.
TABLES = "tables"

# The grouping field attribution and group figures use by default.
DEFAULT_GROUPING = "SECTOR"


@dataclass(frozen=True)
class AnalysisRequest:
    """What to analyse with a run, and how.

    Attributes:
        portfolio: Work out the portfolio figures (holdings, turnover, costs,
            characteristics, income and flows).
        risk: Work out the risk figures (factors, value at risk, risk over
            time, concentration, liquidity).
        attribution: Work out the attribution (contribution, Brinson-Fachler
            and the implementation breakdown).
        settings: The window and conventions performance figures use;
            the defaults when None.
        factors: The factors risk and attribution use; the defaults (size,
            value, momentum) when None.
        grouping: The grouping field attribution and group figures use.
    """
    portfolio: bool = False
    risk: bool = False
    attribution: bool = False
    settings: AnalysisSettings | None = None
    factors: Sequence[Factor] | None = None
    grouping: str = DEFAULT_GROUPING

    def parts(self) -> list[str]:
        """The parts asked for, performance first."""
        asked = {PORTFOLIO: self.portfolio, RISK: self.risk, ATTRIBUTION: self.attribution}

        return [PERFORMANCE, *(part for part, wanted in asked.items() if wanted)]


def standard_figures(result: Any,
                     request: AnalysisRequest) -> dict[str, Callable[[], Any]]:
    """Each part's standard figures, as calls on *result*'s analytics."""
    grouping = request.grouping

    def performance() -> Any:
        analytics = result.analytics
        tracked = "index" in analytics.run.comparators

        return (analytics.summary(), analytics.comparison("index") if tracked else None,
                analytics.periods("monthly"), analytics.periods("annual"))

    def portfolio() -> Any:
        found = result.portfolio_analytics

        return (found.turnover("year"), found.costs(), found.costs("year"),
                found.characteristics(), found.income())

    def risk() -> Any:
        found = result.risk
        factors = found.factors

        return ([factors.exposures(day) for day in factors.dates], factors.regression(),
                found.loss.table(), found.over_time.summary(), found.concentration.by_weight(),
                found.liquidity.profile())

    def attribution() -> Any:
        # The panels import the backtest package, which imports this module.
        from .panels import REFERENCE  # noqa: PLC0415

        found = result.attribution
        tracked = "index" in result.run_data.comparators
        # Group figures only for a field the run kept for its names.
        grouped = grouping in result.run_data.tables[REFERENCE].columns

        return (found.contribution(),
                found.contribution(field=grouping) if grouped else None,
                found.brinson(grouping) if tracked and grouped else None,
                found.implementation() if tracked else None)

    return {PERFORMANCE: performance, PORTFOLIO: portfolio, RISK: risk,
            ATTRIBUTION: attribution}


def analyse(result: Any,
            request: AnalysisRequest) -> dict[str, float]:
    """Work out the standard figures of each part *request* asks for on
    *result*, keeping them, and return the seconds each part took, with the
    run's tables built first (``tables``) so each part is timed on its own."""
    # The panels import the backtest package, which imports this module.
    from .data import EXPECTED  # noqa: PLC0415

    calls = standard_figures(result, request)
    start = time.perf_counter()
    tables = result.run_data.tables

    for name in EXPECTED:
        tables.get(name)

    timings = {TABLES: time.perf_counter() - start}

    for part in request.parts():
        start = time.perf_counter()
        calls[part]()
        timings[part] = time.perf_counter() - start

    return timings
