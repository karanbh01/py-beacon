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
from typing import Any

import pandas as pd

from ..data import BENCHMARK, INDEX, OPTIMISED, RunData
from . import distribution as shape
from . import formulas, ratios, return_figures, tables
from . import rolling as moving
from .comparisons import Comparison, Period, aligned, comparison_of, periods_of
from .drawdowns import WORST, DrawdownFigures, drawdown_figures, episodes
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
    def before(self) -> pd.Timestamp | None:
        """The run's close before the days measured, when they start after
        the run's first day; None when they start with it."""
        days = self.run.returns.index
        measured = self.returns

        if measured.empty or not len(days) or measured.index[0] == days[0]:
            return None

        position = days.get_loc(measured.index[0])

        return days[position - 1] if isinstance(position, int) and position > 0 else None

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

    def trailing_returns(self) -> dict[str, return_figures.WindowReturn | None]:
        """The returns over the last 1, 3 and 6 months and 1, 3, 5 and 10
        years, for the month, quarter and year to date, and since inception,
        by name (``1M`` ... ``10Y``, ``MTD``, ``QTD``, ``YTD``,
        ``inception``). Each is annualised as well when it is longer than a
        year. None for a window longer than the run.

        Since inception is the headline's total and annualised return, so
        over the whole run it starts from what the fund started with.
        """
        returns = self.returns
        figures = return_figures.trailing_returns(returns, self.periods_per_year)
        figures.update(return_figures.returns_to_date(returns, self.periods_per_year))
        headline = self.headline()
        figures["inception"] = (None if returns.empty else return_figures.WindowReturn(
            start=returns.index[0], end=returns.index[-1], days=len(returns),
            total=headline["total_return"],
            annualised=(headline["annualised_return"]
                        if return_figures.longer_than_a_year(returns.index[0], returns.index[-1])
                        else None)))

        return figures

    def best_and_worst(self) -> dict[str, tuple[return_figures.PeriodReturn,
                                                return_figures.PeriodReturn] | None]:
        """The best and worst day, week, month, quarter and year, by
        frequency (``daily`` ... ``annual``), with their dates and whether
        the run covered all of each."""
        return {frequency: return_figures.best_and_worst(self.returns, frequency)
                for frequency in return_figures.EXTREMES}

    def positive_shares(self) -> dict[str, float | None]:
        """The share of days, months and years with a positive return, by
        frequency."""
        return {frequency: return_figures.positive_share(self.returns, frequency)
                for frequency in (return_figures.DAILY, formulas.MONTHLY, formulas.ANNUAL)}

    def gains_and_losses(self,
                         frequency: str = return_figures.DAILY) -> return_figures.GainsAndLosses:
        """The average gain and loss over days (or months, ...), and their
        ratio."""
        return return_figures.gains_and_losses(self.returns, frequency)

    def longest_streaks(self) -> tuple[return_figures.Streak | None,
                                       return_figures.Streak | None]:
        """The longest run of months that all gained, and of months that all
        lost."""
        return return_figures.longest_streaks(self.returns)

    def means(self) -> dict[str, float | None]:
        """The arithmetic and geometric mean daily return, and each a year."""
        return return_figures.means(self.returns, self.periods_per_year)

    def levels(self) -> pd.Series:
        """The level the figures are measured along, dated, from its start:
        over the whole run what the fund started with, then its NAV or a
        unit's value; over a window, 1 grown by its returns. The start is
        dated the business day before the first return."""
        if not self._windowed:
            path, start = self._performance()
        else:
            path, start = (1.0 + self.returns).cumprod(), 1.0

        if path.empty or not start > 0:
            return path

        eve = pd.Timestamp(path.index[0]) - pd.offsets.BDay(1)

        return pd.concat([pd.Series([start], index=pd.DatetimeIndex([eve])), path])

    def drawdowns(self,
                  worst: int = WORST) -> DrawdownFigures:
        """The run's drawdowns: the deepest *worst* with their peak, trough
        and recovery dates, the current one, and the figures over all of
        them. The deepest is the headline's maximum drawdown."""
        return drawdown_figures(self.levels(), self.headline()["total_return"], worst)

    def drawdown_series(self) -> pd.Series:
        """How far below its peak the level stood each day."""
        return formulas.drawdown(self.levels())

    def ratios(self,
               sharpe_benchmark: float = 0.0,
               trials: int | None = None,
               trial_variance: float | None = None) -> dict[str, float | None]:
        """The risk-adjusted ratios beyond the Sharpe ratio, by name.

        ``sortino``, ``kappa_3`` (daily), ``upside_potential`` and ``omega``
        measure return against the shortfall below the risk-free rate;
        ``calmar``, ``martin``, ``sterling`` and ``burke`` against drawdowns;
        ``gain_to_pain`` against losing months. ``probabilistic_sharpe`` is the
        probability the true Sharpe ratio is above *sharpe_benchmark* (a
        year); ``deflated_sharpe`` the same against the best of *trials*
        strategies whose annual Sharpe ratios vary by *trial_variance*, None
        unless both are given; ``lo_sharpe`` the Sharpe ratio corrected for
        serial correlation.
        """
        returns = self.returns
        excess = ratios.excess_returns(returns, self.periods_per_year, self.risk_free_rate)
        annualised = self.headline()["annualised_return"]
        depths = [drawdown.depth for drawdown in episodes(self.levels())]
        ulcer = drawdown_figures(self.levels(), 0.0).ulcer_index

        return {
            "sortino": ratios.sortino(excess, self.periods_per_year),
            "kappa_3": ratios.kappa_3(excess),
            "upside_potential": ratios.upside_potential(excess),
            "omega": ratios.omega(excess),
            "calmar": ratios.calmar(annualised, min(depths, default=0.0)),
            "martin": ratios.martin(excess, ulcer, self.periods_per_year),
            "sterling": ratios.sterling(annualised, depths),
            "burke": ratios.burke(annualised, depths),
            "gain_to_pain": ratios.gain_to_pain(formulas.period_returns(returns,
                                                                        formulas.MONTHLY)),
            "probabilistic_sharpe": ratios.probabilistic_sharpe(excess, self.periods_per_year,
                                                                sharpe_benchmark),
            "deflated_sharpe": ratios.deflated_sharpe(excess, self.periods_per_year, trials,
                                                      trial_variance),
            "lo_sharpe": ratios.lo_sharpe(excess, self.periods_per_year),
        }

    def distribution(self,
                     bins: int = shape.BINS) -> dict[str, Any]:
        """The shape of the daily returns, by name: ``downside_deviation``
        (below the risk-free rate) and ``semi_variance`` (below the mean),
        both annualised; ``skew`` and ``excess_kurtosis``; ``tail_ratio``;
        ``serial_correlation`` by lag in days; and a ``histogram`` of *bins*
        bins with the normal distribution to compare."""
        returns = self.returns

        return {"downside_deviation": shape.downside_deviation(
                    returns, self.periods_per_year, self.risk_free_rate),
                "semi_variance": shape.semi_variance(returns, self.periods_per_year),
                "skew": shape.skew(returns),
                "excess_kurtosis": shape.excess_kurtosis(returns),
                "tail_ratio": shape.tail_ratio(returns),
                "serial_correlation": shape.serial_correlation(returns),
                "histogram": shape.histogram(returns, bins)}

    def bootstrap(self,
                  samples: int = shape.SAMPLES,
                  confidence: float = shape.CONFIDENCE,
                  block: int = 1,
                  seed: int = shape.SEED) -> dict[str, shape.Interval] | None:
        """How far the annualised return and the Sharpe ratio could have come
        out otherwise, from runs made by drawing the run's own days again;
        see `beacon.analysis.performance.distribution.bootstrap`."""
        return shape.bootstrap(self.returns, self.periods_per_year, self.risk_free_rate,
                                      samples, confidence, block, seed)

    def rolling(self,
                window: int = moving.WINDOWS[1],
                comparator: str | pd.Series | None = None) -> pd.DataFrame:
        """Each day's return, volatility, Sharpe ratio, Sortino ratio and
        maximum drawdown over the *window* days ending that day, blank until
        the first full window; with a comparator (by name or levels), its
        relative return, tracking error, information ratio, beta and
        correlation too, on the days both have."""
        figures = moving.rolling_figures(self.returns, window, self.periods_per_year,
                                          self.risk_free_rate)
        theirs = self._comparator_returns(comparator)

        if theirs is None:
            return figures

        relative = moving.rolling_relative(theirs[0], theirs[1], window, self.periods_per_year)

        return figures.join(relative, how="left")

    def rolling_outcomes(self,
                         window: int = moving.WINDOWS[1],
                         comparator: str | pd.Series | None = None) -> dict[str, float | None]:
        """Over every full *window*: the worst, middle and best return, the
        share that gained, and the share ahead of the comparator when one is
        given."""
        figures = self.rolling(window, comparator)

        return moving.outcomes(figures["return"], figures.get("relative_return"))

    def monthly_grid(self,
                     of: str = tables.FUND) -> pd.DataFrame:
        """The calendar grid of monthly returns, a row a year, with each
        year's: of the fund, of a comparator by name, or of the fund less a
        comparator (``difference index``).

        Raises:
            KeyError: If the run has no such comparator.
        """
        if of == tables.FUND:
            return tables.monthly_grid(self.returns)

        name = of.removeprefix(f"{tables.DIFFERENCE} ")
        ours, theirs = tables.comparator_returns(self.returns,
                                                 {name: self.run.comparators[name]})[name]

        return (tables.difference_grid(ours, theirs) if of != name
                else tables.monthly_grid(theirs))

    def period_table(self,
                     frequency: str = formulas.ANNUAL) -> pd.DataFrame:
        """Each year's (or quarter's, month's) return, volatility, Sharpe
        ratio and maximum drawdown for the fund, with its share of positive
        days, for its index and benchmark, and the fund's less each's."""
        comparators = {name: formulas.returns_from_start(self.run.comparators[name])
                       for name in (INDEX, BENCHMARK) if name in self.run.comparators}

        return tables.period_table(self.returns, frequency, self.periods_per_year,
                                   self.risk_free_rate, comparators)

    def trailing_table(self) -> pd.DataFrame:
        """The trailing returns (`trailing_returns`) of the fund, its index
        and benchmark and the fund's less each's, a row a window."""
        comparators = {}

        for name, (_, theirs) in tables.comparator_returns(self.returns, {
                name: self.run.comparators[name] for name in (INDEX, BENCHMARK)
                if name in self.run.comparators}).items():
            trailing = return_figures.trailing_returns(theirs, self.periods_per_year)
            trailing.update(return_figures.returns_to_date(theirs, self.periods_per_year))
            trailing["inception"] = return_figures.window_return(
                theirs, theirs.index[0], self.periods_per_year,
                annualise=return_figures.longer_than_a_year(theirs.index[0], theirs.index[-1]))
            comparators[name] = trailing

        return tables.trailing_table(self.trailing_returns(), comparators)

    def month_of_year(self) -> pd.Series:
        """The fund's mean return in each calendar month over the years."""
        return tables.month_of_year(self.returns)

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

    def _comparator_returns(self,
                            comparator: str | pd.Series | None) -> tuple[pd.Series,
                                                                         pd.Series] | None:
        """The run's and a comparator's daily returns on the days both have;
        None without a comparator, or one the run does not have."""
        if comparator is None:
            return None

        levels = (self.run.comparators.get(comparator) if isinstance(comparator, str)
                  else comparator)

        return None if levels is None else aligned(self.returns, levels, self.before)

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
