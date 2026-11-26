# src/beacon/analysis/analytics/instrument.py
"""
A run's analytics a name at a time: `result.analytics.instrument`.

    instrument = result.analytics.instrument
    instrument.volatility()                         # a value a name
    instrument.beta(names=["AAPL", "MSFT"], against="benchmark")
    instrument.total_return(by="year")              # a row a year, a column a name
    instrument.tracking_error(rolling="1Y")         # dates by names
    instrument.volatility(frequency="weekly")
    instrument.annualised_return(frequency="weekly")  # as the efficient frontier measures it
    instrument.average_weight(of="all")             # a row a name, a column a portfolio
    instrument.days_to_sell()                       # at the last rebalance or month end

Each figure is measured from the run's daily returns of every name it held
or its index held, in the portfolio's currency, over the analysis window
(`start`, `end`), as `beacon.analysis.performance.name_figures` measures
them. ``frequency`` compounds them into weeks or months first. Every result
says what it was measured on in ``attrs``: its ``basis`` (``daily``,
``weekly`` or ``monthly``) and, against a comparator, ``against``. An
annualised return is blank for a name with returns over less than the
shortest window an annualised rate is given over
(`beacon.analysis.annualising`), as the fund's is.

The weights each portfolio gave each name, the days each holding would take
to sell, and the value each name traded are here too, a row a name; and one
name over the run, its profile:

    instrument.profile("AAPL").daily_weights
"""
# BN-452 (Karan, 2026-11-24). BN-461 (Karan, 2026-11-25): member_risk_return
# is removed; its figures are table(frequency="weekly")'s.
from collections.abc import Sequence
from typing import Any

import pandas as pd

from ...data.base import as_of_position
from ..annualising import days_in, long_enough
from ..panels import NAME_RETURNS, TRADED_VALUES
from ..performance import name_figures
from ..performance.name_figures import DAILY, RELATIVE
from ..reports.context import FUND
from ..reports.rolling import window_ends
from ..reports.statistics import NO_DATA_STORE, RUN_PREDATES, TOO_SHORT, Unavailable
from ..risk.liquidity import BOOKS, LiquidityAnalytics
from .category import Category, Day
from .forms import PERIODS
from .groupings import FREQUENCIES
from .instrument_table import per_name
from .profile import Profile
from .shape import ALL

DAILY_STEP, MONTH_END = "daily", "month_end"

# Names, one or several; every name the run measured when None.
Names = str | Sequence[str] | None


class InstrumentLevel(Category):
    """A run's analytics a name at a time: a figure for each name, and the
    statistics given a row a name."""

    name = "instrument"
    statistics = ("style_positions",)
    extras = (*name_figures.FIGURES, "average_weight", "days_to_sell", "traded_value", "table",
              "profile")

    def style_positions(self) -> Any:
        """Where each name held sits by size and style, from 0 to 3 on each,
        with each portfolio's weight in it."""
        return self._get("style_positions")

    def total_return(self,
                     *,
                     names: Names = None,
                     start: Day = None,
                     end: Day = None,
                     by: str | None = None,
                     rolling: int | str | None = None,
                     step: str | None = None,
                     frequency: str = DAILY) -> Any:
        """Each name's compounded return over the window."""
        return self._figure("total_return", names, start, end, by, rolling, step, frequency)

    def annualised_return(self,
                          *,
                          names: Names = None,
                          start: Day = None,
                          end: Day = None,
                          by: str | None = None,
                          rolling: int | str | None = None,
                          step: str | None = None,
                          frequency: str = DAILY,
                          annualise: bool | None = None) -> Any:
        """Each name's mean return times the periods a year, as the
        efficient frontier measures it: blank for a name with too few
        returns to annualise, unless ``annualise=True``; the mean return a
        period with ``annualise=False``."""
        # BN-454 (Karan, 2026-11-25).
        return self._figure("annualised_return", names, start, end, by, rolling, step,
                            frequency, annualise=annualise)

    def volatility(self,
                   *,
                   names: Names = None,
                   start: Day = None,
                   end: Day = None,
                   by: str | None = None,
                   rolling: int | str | None = None,
                   step: str | None = None,
                   frequency: str = DAILY) -> Any:
        """Each name's annualised standard deviation of returns."""
        return self._figure("volatility", names, start, end, by, rolling, step, frequency)

    def max_drawdown(self,
                     *,
                     names: Names = None,
                     start: Day = None,
                     end: Day = None,
                     by: str | None = None,
                     rolling: int | str | None = None,
                     step: str | None = None,
                     frequency: str = DAILY) -> Any:
        """Each name's deepest fall from a running peak. Rolled, its windows
        end each month end unless ``step`` says otherwise."""
        return self._figure("max_drawdown", names, start, end, by, rolling, step, frequency)

    def excess_return(self,
                      *,
                      names: Names = None,
                      start: Day = None,
                      end: Day = None,
                      by: str | None = None,
                      rolling: int | str | None = None,
                      step: str | None = None,
                      against: str | None = None,
                      frequency: str = DAILY) -> Any:
        """Each name's compounded return less the comparator's, over the
        days the name has a return."""
        # BN-454: Drilldown's, worked out here rather than by the server.
        return self._figure("excess_return", names, start, end, by, rolling, step, frequency,
                            against)

    def beta(self,
             *,
             names: Names = None,
             start: Day = None,
             end: Day = None,
             by: str | None = None,
             rolling: int | str | None = None,
             step: str | None = None,
             against: str | None = None,
             frequency: str = DAILY) -> Any:
        """Each name's covariance with the comparator over its variance."""
        return self._figure("beta", names, start, end, by, rolling, step, frequency, against)

    def correlation(self,
                    *,
                    names: Names = None,
                    start: Day = None,
                    end: Day = None,
                    by: str | None = None,
                    rolling: int | str | None = None,
                    step: str | None = None,
                    against: str | None = None,
                    frequency: str = DAILY) -> Any:
        """Each name's correlation with the comparator."""
        return self._figure("correlation", names, start, end, by, rolling, step, frequency,
                            against)

    def tracking_error(self,
                       *,
                       names: Names = None,
                       start: Day = None,
                       end: Day = None,
                       by: str | None = None,
                       rolling: int | str | None = None,
                       step: str | None = None,
                       against: str | None = None,
                       frequency: str = DAILY) -> Any:
        """Each name's annualised standard deviation of its returns less the
        comparator's."""
        return self._figure("tracking_error", names, start, end, by, rolling, step, frequency,
                            against)

    def residual_risk(self,
                      *,
                      names: Names = None,
                      start: Day = None,
                      end: Day = None,
                      by: str | None = None,
                      rolling: int | str | None = None,
                      step: str | None = None,
                      against: str | None = None,
                      frequency: str = DAILY) -> Any:
        """Each name's annualised spread of the residuals of its returns
        regressed on the comparator's."""
        return self._figure("residual_risk", names, start, end, by, rolling, step, frequency,
                            against)

    def average_weight(self,
                       *,
                       of: str = FUND,
                       names: Names = None,
                       start: Day = None,
                       end: Day = None) -> Any:
        """Each name's mean daily weight in the fund, its index, optimised
        index or benchmark (``of``), or each (``of="all"``, a column a
        portfolio), over the window: 0 on the days it was not held. The
        daily weights themselves are the result's.

        Raises:
            ValueError: If the run kept no weights for that portfolio.
        """
        # BN-453.
        context = self._analytics._context_for(start=start, end=end)
        days = context.performance.returns.index
        books = list(BOOKS) if of == ALL else [of]
        found = {}

        for book in books:
            weights = context.run.tables.get(BOOKS[book]) if book in BOOKS else None

            if weights is None or weights.empty:
                if of == ALL:
                    continue
                raise ValueError(f"The run kept no weights for {book!r}.")

            held = weights.loc[(weights.index >= days[0]) & (weights.index <= days[-1])]
            found[book] = held.fillna(0.0).mean()

        table = pd.DataFrame(found).fillna(0.0)
        table = _picked(table.T, names).T if names is not None else table

        return table[of] if of != ALL else table

    def days_to_sell(self,
                     *,
                     of: str = FUND,
                     names: Names = None,
                     as_of: Day = None) -> Any:
        """Each holding's value, its average daily traded value and the days
        it would take to sell at the participation rate the liquidity
        options set, slowest first, at the last rebalance or month end, or
        the last on or before ``as_of``; infinite for a name with no traded
        value."""
        liquidity = _liquidity(self._analytics._context)
        dates = _liquidity_dates(liquidity)
        at = as_of_position(pd.DatetimeIndex(dates),
                            pd.Timestamp(as_of) if as_of is not None else dates[-1])

        if at is None:
            raise ValueError(f"The run kept no liquidity on or before {as_of}.")

        found = liquidity.days_to_sell(dates[at], of)
        found = _picked(found.T, names).T if names is not None else found
        found.attrs = {"date": dates[at]}

        return found

    def traded_value(self,
                     *,
                     names: Names = None,
                     start: Day = None,
                     end: Day = None) -> Any:
        """Each name's average daily value traded on its market, at each
        rebalance and month end the run kept, dates by names."""
        context = self._analytics._context_for(start=start, end=end)
        liquidity = _liquidity(context)
        _liquidity_dates(liquidity)
        traded = liquidity.traded_values()
        days = context.performance.returns.index
        found = traded.loc[(traded.index >= days[0]) & (traded.index <= days[-1])]

        return _picked(found, names) if names is not None else found

    def table(self,
              *,
              columns: Sequence[str] | None = None,
              names: Names = None,
              start: Day = None,
              end: Day = None,
              against: str | None = None,
              frequency: str = DAILY) -> pd.DataFrame:
        """Every per-name figure in one table, a row a name: its return and
        risk, against the comparator, its weights, its contributions to
        return and risk, and its days to sell
        (`beacon.analysis.analytics.instrument_table`). ``columns`` picks
        some; one the run cannot give is left out, and ``attrs`` say why."""
        # BN-454.
        return per_name(self, columns, names, start, end, against, frequency)

    def profile(self,
                name: str,
                *,
                start: Day = None,
                end: Day = None,
                against: str | None = None,
                by: str = "quarter") -> Profile:
        """One name over the run, a dated table a part
        (`beacon.analysis.analytics.profile`): how each rebalance weighted
        it, how its weight drifted each day and what it contributed each
        month, quarter or year (``by``), against the comparator given as
        ``against`` or the one the analysis tracks.

        Raises:
            ValueError: If no portfolio of the run held *name*, or *by* is
                not ``month``, ``quarter`` or ``year``.
        """
        # BN-468; BN-469 added by.
        return Profile(name, self._analytics._context_for(start=start, end=end), against, by)

    def _observations(self,
                      window: dict[str, Any],
                      frequency: str) -> pd.Series:
        """How many returns each name has over *window* on the basis
        *frequency* names."""
        context = self._analytics._context_for(**window)
        returns, _, _ = name_figures.on_basis(_name_returns(context), None, frequency,
                                              context.performance.periods_per_year)

        return returns.notna().sum().rename("observations")

    def _figure(self,
                figure: str,
                names: Names,
                start: Day,
                end: Day,
                by: str | None,
                rolling: int | str | None,
                step: str | None,
                frequency: str,
                against: str | None = None,
                annualise: bool | None = None) -> Any:
        """Figure *figure* of each name in the form the keywords ask for.

        Raises:
            ValueError: If a name is not one the run measured, a keyword
                is not one it takes, or both ``by`` and ``rolling`` are
                given.
            Unavailable: If the run kept no names' returns, or lacks the
                comparator.
        """
        if by is not None and rolling is not None:
            raise ValueError("Give one of by and rolling at a time, not both.")

        if step is not None and rolling is None:
            raise ValueError("step goes with rolling.")

        context = self._analytics._context_for(start=start, end=end, annualise=annualise)
        returns = _picked(_name_returns(context), names)
        comparator = None if figure not in RELATIVE else context.require(
            against or context.require_tracked())
        theirs = None if comparator is None else context.book(comparator).returns
        returns, theirs, periods = name_figures.on_basis(returns, theirs, frequency,
                                                         context.performance.periods_per_year)
        periods = 1 if annualise is False else periods
        span = context.performance.annualising[0] if figure == "annualised_return" else None

        if by is not None:
            found: Any = _by_period(figure, returns, theirs, by, periods, span)
        elif rolling is not None:
            found = _rolling(figure, returns, theirs, rolling, step, periods, span)
        else:
            found = name_figures.figure(figure, returns, theirs, periods).rename(figure)
            found = found.where(_enough(returns.notna().sum(), span, periods))

        found.attrs = {"basis": frequency, **({"against": comparator} if comparator else {})}

        return found


def _name_returns(context: Any) -> pd.DataFrame:
    """The run's names' daily returns over the analysis window."""
    tables = context.run.tables

    if NAME_RETURNS not in tables:
        raise Unavailable(RUN_PREDATES)

    days = context.performance.returns.index

    if not len(days):
        raise Unavailable(TOO_SHORT)

    found: pd.DataFrame = tables[NAME_RETURNS]

    return found.loc[(found.index >= days[0]) & (found.index <= days[-1])]


def _liquidity(context: Any) -> LiquidityAnalytics:
    """The run's liquidity under the analysis's options: measured from the
    data store when it is loaded, else from the traded values the run
    kept."""
    options = context.options.liquidity

    return LiquidityAnalytics(context.run, context.source, participation=options.participation,
                              lookback_days=options.lookback_days)


def _liquidity_dates(liquidity: LiquidityAnalytics) -> list[pd.Timestamp]:
    """The dates the run kept traded values on, or `Unavailable`."""
    try:
        dates: list[pd.Timestamp] = liquidity.dates()
    except KeyError as error:
        code = RUN_PREDATES if TRADED_VALUES in str(error) else NO_DATA_STORE

        raise Unavailable(code) from error

    if not dates:
        raise Unavailable(TOO_SHORT)

    return dates


def _picked(returns: pd.DataFrame,
            names: Names) -> pd.DataFrame:
    """*returns* of *names*, every name when None."""
    if names is None:
        return returns

    wanted = [names] if isinstance(names, str) else list(names)
    unknown = [name for name in wanted if name not in returns.columns]

    if unknown:
        raise ValueError(f"The run measured no {', '.join(unknown[:5])}"
                         f"{' and others' if len(unknown) > 5 else ''}.")

    return returns[wanted]


def _enough(counts: pd.Series | pd.DataFrame,
            span: str | None,
            periods: int) -> pd.Series | pd.DataFrame:
    """Whether each count of returns is enough to annualise over: always
    when *span* is None, as for a figure not annualised."""
    if span is None:
        return counts >= 0

    return counts.map(lambda count: long_enough(int(count), span, periods))


def _by_period(figure: str,
               returns: pd.DataFrame,
               theirs: pd.Series | None,
               by: str,
               periods: int,
               span: str | None) -> pd.DataFrame:
    """*figure* of each name in each calendar period, a row a period."""
    if by not in PERIODS:
        raise ValueError(f"by is one of {', '.join(PERIODS)}, got {by!r}.")

    labels = pd.DatetimeIndex(returns.index).to_period(FREQUENCIES[by]).astype(str)
    rows = {label: name_figures.figure(figure, returns[labels == label],
                                       None if theirs is None else theirs[labels == label],
                                       periods)
            for label in pd.unique(labels)}
    found = pd.DataFrame.from_dict(rows, orient="index").rename_axis("period")
    counts = returns.notna().groupby(labels).sum().reindex(found.index)

    return found.where(_enough(counts, span, periods))


def _rolling(figure: str,
             returns: pd.DataFrame,
             theirs: pd.Series | None,
             rolling: int | str,
             step: str | None,
             periods: int,
             span: str | None) -> pd.DataFrame:
    """*figure* of each name over each trailing window, dates by names: blank
    when the window is too short to annualise *figure* over."""
    window = days_in(rolling, periods) if isinstance(rolling, str) else rolling

    if isinstance(window, bool) or not isinstance(window, int) or window < 2:
        raise ValueError(f"rolling is a number of periods, at least 2, or a span such as "
                         f"'1Y', got {rolling!r}.")

    step = step or (MONTH_END if figure == "max_drawdown" else DAILY_STEP)

    if step not in (DAILY_STEP, MONTH_END):
        raise ValueError(f"step is {DAILY_STEP!r} or {MONTH_END!r}, got {step!r}.")

    ends = (window_ends(pd.DatetimeIndex(returns.index), window, MONTH_END)
            if step == MONTH_END else None)
    found = name_figures.rolled(figure, returns, theirs, window, periods, ends)

    if span is not None and not long_enough(window, span, periods):
        found = found.iloc[:0]

    return found.dropna(how="all").rename_axis("DATE")


__all__ = ["InstrumentLevel", "Names"]
