# src/beacon/analysis/performance/analytics.py
"""
A run's performance figures, from its data and the settings it is measured
with.

    analytics = PerformanceAnalytics(RunData.from_record(folder))
    analytics.summary()

Every figure is one of `beacon.analysis.performance.formulas`, so a finished
run and a stored one give the same figures, and a backtest's summary is
these.
"""
# BN-371.
import pandas as pd

from ..data import BENCHMARK, INDEX, OPTIMISED, RunData
from . import formulas
from .comparisons import Comparison, Period, comparison_of, periods_of
from .settings import AnalysisSettings


class PerformanceAnalytics:
    """A run's performance figures.

    Args:
        run: The run's data.
        settings: What the figures are measured over and against; the run's
            own when None.
    """

    def __init__(self,
                 run: RunData,
                 settings: AnalysisSettings | None = None):
        self.run = run
        self.settings = settings or AnalysisSettings()

    @property
    def periods_per_year(self) -> int:
        """How many daily returns make a year."""
        return self.settings.periods_per_year or self.run.periods_per_year

    @property
    def risk_free_rate(self) -> float:
        """The annual rate the Sharpe ratio is in excess of."""
        rate = self.settings.risk_free_rate

        return self.run.risk_free_rate if rate is None else rate

    @property
    def returns(self) -> pd.Series:
        """The fund's daily returns over the days measured."""
        return self._window(self.run.returns)

    @property
    def tracked(self) -> pd.Series | None:
        """The levels of the comparator tracking is measured against, or None
        when the run has none."""
        comparators = self.run.comparators
        name = self.settings.tracked or (OPTIMISED if OPTIMISED in comparators else INDEX)

        return comparators.get(name)

    def headline(self) -> dict[str, float]:
        """The five headline figures: total and annualised return,
        volatility, Sharpe ratio and maximum drawdown.

        Over the whole run, the total return and drawdown run from what the
        fund started with, so the cost of buying in on the first day counts;
        over a window, from the close before its first day.
        """
        returns = self.returns

        if not self._windowed:
            path, start = self._performance()
            total = (0.0 if path.empty or start == 0
                     else formulas.total_return(start, float(path.iloc[-1])))
            # From the starting value, so a fall on the first day counts.
            drawn = (pd.concat([pd.Series([start]), path], ignore_index=True)
                     if not path.empty and start > 0 else path)
        else:
            drawn = formulas.path_of(returns)
            total = float(drawn.iloc[-1] - 1.0) if len(returns) else 0.0

        return formulas.headline_figures(returns, total, drawn, self.periods_per_year,
                                         self.risk_free_rate)

    def hit_rate_monthly(self) -> float | None:
        """The share of calendar months with a positive return."""
        return formulas.hit_rate_monthly(self.returns)

    def tracking_error(self) -> float | None:
        """The annualised spread of the daily returns less the tracked
        comparator's; None without one or with fewer than two days."""
        active = self._active()

        return None if active is None else formulas.tracking_error(active, self.periods_per_year)

    def tracking_difference(self) -> float | None:
        """The compounded return less the tracked comparator's; None without
        one."""
        theirs = self._theirs()

        if theirs is None or self.returns.empty or theirs.empty:
            return None

        return formulas.tracking_difference(self.returns, theirs)

    def information_ratio(self) -> float | None:
        """The mean daily return less the tracked comparator's, over its
        spread, annualised; None without one or without active risk."""
        active = self._active()

        return None if active is None else formulas.information_ratio(active,
                                                                      self.periods_per_year)

    def money_weighted_return(self) -> float | None:
        """The annual internal rate of return of the investors' money; None
        for a run without flows, or when no rate exists."""
        flows = self.run.investor_flows

        if flows is None or flows.empty:
            return None

        return formulas.money_weighted_return([(pd.Timestamp(date), float(amount))
                                               for date, amount in flows.items()
                                               if amount != 0.0])

    def comparison(self,
                   comparator: str | pd.Series) -> Comparison | None:
        """The run against a comparator, given by name or by its levels, over
        the days both have: the comparator's own figures and the run's
        relative to it. None for a comparator the run does not have, or one
        sharing fewer than two days with it."""
        levels = (self.run.comparators.get(comparator) if isinstance(comparator, str)
                  else comparator)

        if levels is None:
            return None

        return comparison_of(self.returns, levels, self.periods_per_year, self.risk_free_rate)

    def periods(self,
                frequency: str,
                comparators: list[str] | None = None) -> list[Period]:
        """The run's figures in each calendar year, quarter or month, and its
        comparators' in each, by name: the index and the benchmark, of those
        the run has, unless *comparators* names others.

        Raises:
            ValueError: If the frequency is not ``annual``, ``quarterly`` or
                ``monthly``.
        """
        names = comparators if comparators is not None else [
            name for name in (INDEX, BENCHMARK) if name in self.run.comparators]
        theirs = {name: formulas.returns_from_start(self.run.comparators[name])
                  for name in names if name in self.run.comparators}

        return periods_of(self.returns, frequency, self.periods_per_year, self.risk_free_rate,
                          theirs)

    def summary(self) -> dict[str, float | None]:
        """The headline figures, the monthly hit rate, and, when the run has
        them, the tracking error and difference against the tracked
        comparator and the money-weighted return."""
        result: dict[str, float | None] = {**self.headline()}
        result["hit_rate_monthly"] = self.hit_rate_monthly()
        tracking_error = self.tracking_error()
        tracking_difference = self.tracking_difference()

        if tracking_error is not None:
            result["tracking_error"] = tracking_error
        if tracking_difference is not None:
            result["tracking_difference"] = tracking_difference

        if self.run.investor_flows is not None:
            result["money_weighted_return"] = self.money_weighted_return()

        return result

    @property
    def _windowed(self) -> bool:
        """Whether the figures cover less than the whole run."""
        return self.settings.start is not None or self.settings.end is not None

    def _window(self,
                series: pd.Series) -> pd.Series:
        """*series* over the days measured."""
        if not self._windowed:
            return series

        start = pd.Timestamp(self.settings.start) if self.settings.start is not None else None
        end = pd.Timestamp(self.settings.end) if self.settings.end is not None else None

        return series.loc[start:end]

    def _performance(self) -> tuple[pd.Series, float]:
        """The level the run's figures are measured along, and its start."""
        if self.run.performance is not None and self.run.performance_start is not None:
            return self.run.performance, self.run.performance_start

        return self.run.nav, self.run.initial_capital

    def _theirs(self) -> pd.Series | None:
        """The tracked comparator's daily returns over the days measured,
        its first day's nothing."""
        levels = self.tracked

        if levels is None:
            return None

        return self._window(formulas.returns_from_start(levels).dropna())

    def _active(self) -> pd.Series | None:
        """The daily returns less the tracked comparator's, on the days both
        have; None without a comparator."""
        theirs = self._theirs()

        if theirs is None:
            return None

        return (self.returns - theirs).dropna()
