# src/beacon/analysis/reports/catalogue/relative.py
"""
The statistics of the fund against its comparators, the spread of its rolling
returns, and how it held up in crises and regimes.
"""
# BN-398/BN-400.
from dataclasses import asdict
from typing import Any

import pandas as pd

from ...performance.comparisons import relative_figures
from ...performance.relative_analytics import RelativeAnalytics
from ..context import Context
from ..statistics import (
    FIGURES,
    LIGHT,
    PERFORMANCE,
    TABLE,
    Column,
    Statistic,
    Summarised,
)
from .build import memo, per_comparator
from .performance import EPISODE_COLUMNS

RELATIVE, ROLLING, CRISES = "relative", "rolling", "crises_and_regimes"

COMPARED = ("comparator",)


def _basic(name: str) -> Any:
    """One of the comparison figures, worked out once a comparator."""
    def read(relative: RelativeAnalytics,
             context: Context) -> Any:
        return memo(relative, "basic", lambda: relative_figures(
            relative.ours, relative.theirs, relative.periods_per_year))[name]

    return read


def _regression(name: str) -> Any:
    def read(relative: RelativeAnalytics,
             context: Context) -> Any:
        found = memo(relative, "regression", relative.regression)

        return None if found is None else getattr(found, name)

    return read


def _alpha(relative: RelativeAnalytics,
           context: Context) -> dict[str, Any]:
    return {"alpha": _regression("alpha")(relative, context),
            "t": _regression("alpha_t")(relative, context),
            "p": _regression("alpha_p")(relative, context)}


def _by_year(context: Context) -> Summarised:
    years, spread = context.relative().tracking_difference_by_year()

    return Summarised(pd.DataFrame([asdict(year) for year in years],
                                   columns=["year", "fund", "comparator", "difference"])
                      .set_index("year"), {"spread": spread})


def _relative_drawdowns(context: Context) -> Summarised:
    found = context.relative().relative_drawdowns(context.options.returns.worst_drawdowns)
    table = pd.DataFrame([asdict(drawdown) for drawdown in found.worst],
                         columns=[column.name for column in EPISODE_COLUMNS])

    return Summarised(table, {name: value for name, value in asdict(found).items()
                              if name not in ("worst", "current_drawdown")})


def _outcomes(context: Context) -> dict[str, Any]:
    return context.performance.rolling_outcomes(context.options.rolling.window,
                                                context.tracked)


def _crises(context: Context) -> pd.DataFrame:
    rows = []

    for crisis in context.crises.crises():
        row = {name: value for name, value in asdict(crisis).items() if name != "after"}
        row.update({f"after/{name}": value for name, value in (crisis.after or {}).items()})
        rows.append(row)

    return pd.DataFrame(rows)


def _regimes(context: Context) -> dict[str, Any]:
    return {name: regime for name, regime in context.regimes.regimes().items()
            if regime is not None}


def statistics() -> list[Statistic]:
    """The relative, rolling, crisis and regime statistics, in order."""
    def relative(id_: str,
                 label: str,
                 description: str,
                 read: Any,
                 unit: str | None = "fraction",
                 rolling: str | None = LIGHT,
                 names: tuple[str, ...] = COMPARED) -> Statistic:
        return Statistic(id_, label, RELATIVE, FIGURES, PERFORMANCE, description,
                         per_comparator(read), unit=unit, rolling=rolling,
                         figure_names=names)

    return [
        relative("tracking_difference", "Tracking difference", "The fund's return less "
                 "the comparator's over the window.", _basic("relative_return")),
        relative("tracking_error", "Tracking error", "The annualised standard deviation of "
                 "daily active returns.", _basic("tracking_error")),
        relative("information_ratio", "Information ratio", "Mean daily active return over "
                 "its spread, annualised.", _basic("information_ratio"), unit="ratio"),
        relative("information_ratio_t", "Information ratio t-statistic", "How many standard "
                 "errors the mean active return is from zero, and the chance of that.",
                 lambda found, context: found.information_ratio_t(), unit="ratio",
                 names=("comparator", "field")),
        relative("annualised_relative_return", "Annualised relative return", "The yearly "
                 "rate of the fund's growth relative to the comparator's.",
                 _basic("annualised_relative_return")),
        relative("beta", "Beta", "How much the fund moves with the comparator.",
                 _basic("beta"), unit="ratio"),
        relative("correlation", "Correlation", "How closely the fund's daily returns move "
                 "with the comparator's.", _basic("correlation"), unit="ratio"),
        relative("alpha", "Alpha and its t-statistic", "The intercept of the fund's excess "
                 "returns regressed on the comparator's, a year, and its significance.",
                 _alpha, names=("comparator", "field")),
        relative("r_squared", "R-squared", "The share of the fund's return variance the "
                 "comparator explains.", _regression("r_squared")),
        relative("residual_risk", "Residual risk", "The annualised spread of the "
                 "regression's residuals.", _regression("residual_risk")),
        relative("treynor_ratio", "Treynor ratio", "Excess annual return per unit of beta.",
                 lambda found, context: found.treynor(), unit="ratio"),
        relative("capture", "Up and down capture", "The fund's mean return in the "
                 "comparator's up months (and down months) over the comparator's.",
                 lambda found, context: found.capture(), unit="ratio",
                 names=("comparator", "field")),
        relative("batting_average", "Batting average", "The share of months the fund beat "
                 "the comparator.", lambda found, context: found.batting_average()),
        relative("downside_tracking_error", "Downside tracking error", "Tracking error "
                 "counting only days the fund lagged.",
                 lambda found, context: found.downside_tracking_error()),
        Statistic("relative_drawdowns", "Relative drawdowns", RELATIVE, TABLE, PERFORMANCE,
                  "Drawdowns of the fund's value relative to the tracked comparator's.",
                  _relative_drawdowns, rolling=None, settings=("returns.worst_drawdowns",
                                                               "window.tracked"),
                  columns=EPISODE_COLUMNS),
        Statistic("tracking_difference_by_year", "Tracking difference by year", RELATIVE,
                  TABLE, PERFORMANCE, "Each year's tracking difference against the tracked "
                  "comparator, and their spread.", _by_year, settings=("window.tracked",),
                  columns=(Column("fund", "Fund", "fraction"),
                           Column("comparator", "Comparator", "fraction"),
                           Column("difference", "Difference", "fraction"))),
        Statistic("rolling_outcomes", "Rolling outcomes", ROLLING, FIGURES, PERFORMANCE,
                  "The worst, median and best rolling return, the share of windows that "
                  "gained, and the share ahead of the tracked comparator.", _outcomes,
                  unit="fraction", settings=("rolling.window", "window.tracked"),
                  figure_names=("field",)),
        Statistic("crisis_returns", "Crisis returns", CRISES, TABLE, PERFORMANCE,
                  "The fund's and comparator's return, outperformance, drawdowns, beta and "
                  "alpha in each crisis.", _crises,
                  settings=("crises.market", "crises.threshold", "window.tracked"),
                  columns=(Column("name", "Crisis"), Column("start", "Start", "date"),
                           Column("end", "End", "date"), Column("days", "Days", "days"),
                           Column("fund", "Fund", "fraction"),
                           Column("comparator", "Comparator", "fraction"),
                           Column("outperformance", "Outperformance", "fraction"),
                           Column("fund_drawdown", "Fund's drawdown", "fraction"),
                           Column("comparator_drawdown", "Comparator's drawdown", "fraction"),
                           Column("beta", "Beta", "ratio"), Column("alpha", "Alpha", "fraction"),
                           Column("recovery_days", "Days to recover", "days"))),
        Statistic("crisis_summary", "Crisis summary", CRISES, FIGURES, PERFORMANCE,
                  "Across crises: average outperformance, share outperformed, capture, "
                  "pooled alpha.", lambda context: context.crises.summary(), unit="fraction",
                  settings=("crises.market", "crises.threshold"), figure_names=("field",)),
        Statistic("regimes", "Regime figures", CRISES, FIGURES, PERFORMANCE,
                  "Returns and relative figures in up and down months, calm and volatile "
                  "months, bear markets and corrections.", _regimes, unit="fraction",
                  settings=("crises.market", "window.tracked"),
                  figure_names=("regime", "field"), rolling=None),
    ]
