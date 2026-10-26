# src/beacon/server/periods.py
"""
A run's figures by calendar period: `GET /strategies/{id}/periods`.

Read from the saved run, so nothing is recalculated: its daily returns, the
first from the run's starting point, and its comparators' levels. Each period
runs from the previous period's close and the first from the run's start, so
the periods compound to the whole run's figures.
"""
# BN-324.
from typing import Any

import pandas as pd

from ..analysis.data import RunData
from ..analysis.performance.analytics import PerformanceAnalytics
from ..assumptions import DEFAULT_PERIODS_PER_YEAR
from .schemas import PeriodComparison, PeriodFigures, PeriodRow, PeriodsView, SeriesPayload

STRATEGY_FIGURES = ("total_return", "volatility", "sharpe_ratio", "max_drawdown",
                    "positive_days")


def build_periods(strategy_id: str,
                  index_id: str,
                  run: dict[str, Any],
                  frequency: str) -> PeriodsView:
    """A saved run's figures in each period of *frequency*."""
    assumptions = (run.get("settings") or {}).get("modelling_assumptions") or {}
    periods_per_year = int(assumptions.get("periods_per_year") or DEFAULT_PERIODS_PER_YEAR)
    risk_free = float(assumptions.get("risk_free_rate") or 0.0)
    returns = _series(run["returns"])
    comparisons = run.get("comparisons") or {}
    comparators = {name: _series(comparison["level"])
                   for name in ("index", "benchmark")
                   if (comparison := comparisons.get(name)) is not None}
    # BN-372: measured by the run's analysis, from what the run saved.
    analytics = PerformanceAnalytics(RunData.of_returns(returns, periods_per_year, risk_free,
                                                        comparators))
    rows = [PeriodRow(period=period.period,
                      start=str(period.start.date()),
                      end=str(period.end.date()),
                      strategy=PeriodFigures(**{name: period.figures[name]
                                                for name in STRATEGY_FIGURES}),
                      index=_comparison(period.against.get("index")),
                      benchmark=_comparison(period.against.get("benchmark")))
            for period in analytics.periods(frequency, list(comparators))]

    return PeriodsView(strategy_id=strategy_id, index_id=index_id, frequency=frequency,
                       periods=rows)


def _comparison(figures: dict[str, Any] | None) -> PeriodComparison | None:
    """A period's comparison on the wire, or None."""
    return PeriodComparison.model_validate(figures) if figures is not None else None


def _series(payload: dict[str, Any]) -> pd.Series:
    """A stored SeriesPayload as a dated Series."""
    parsed = SeriesPayload.model_validate(payload)

    return pd.Series(parsed.data, index=pd.to_datetime(parsed.index), dtype=float)
