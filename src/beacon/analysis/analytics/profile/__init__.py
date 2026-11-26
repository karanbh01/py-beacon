# src/beacon/analysis/analytics/profile/__init__.py
"""
One name over a run: `result.analytics.instrument.profile(name)`.

    profile = result.analytics.instrument.profile("AAPL")
    profile.rebalance_weights          # a row a rebalance
    profile.daily_weights              # a row a day
    profile.attribution                # a row a quarter, or as by= says
    profile.risk                       # a row a rebalance
    profile.liquidity                  # a row a rebalance
    profile.factor_exposures           # a row a rebalance
    profile.factor_betas               # a row a rebalance

Each part is a dated table over the analysis window the profile was asked
for (``start``, ``end``), worked out the first time it is asked for.
``Profile.parts`` names them, in order.

``rebalance_weights`` has a row for each of the run's rebalances: the
strategy's weight in the name before any implementation stage (``target``),
after the screens and redistribution (``screened``) and what the trades aimed
at (``weight``), each 0 when that stage did not hold it; the weight a cap cut
it to (``capped``) and the screen that took it out (``removed``), blank when
none did; and each comparator's weight in it that day (``index``,
``optimised``, ``benchmark``, for those the run kept weights for).

``daily_weights`` has a row for each day: the fund's weight in the name as it
drifted between rebalances (the units held times the day's price, over the
NAV), each comparator's, each 0 on a day that portfolio did not hold it, and
the fund's active weight against the comparator (``active``): the one given
as ``against``, else the one the analysis tracks. Its ``attrs`` say which.

``attribution`` has a row for each month, quarter or year (``by``), dated by
its last day: the name's contribution to the fund's return
(``contribution``), its contribution to the fund's return over the
comparator from its active weight (``active_contribution``), and each's
total from the window's start (``contribution_to_date``,
``active_contribution_to_date``), the active ones when the run kept the
comparator's weights. Each day's contribution is as the run's attribution
gives it; the analysis options' ``attribution.linking`` says how the days
are linked. By default (``window``) each period is its share of the whole
window's contribution, so the periods add up to the per-name table's figure
over the window and the last total is it. With ``period`` each period is
linked on its own, as the per-name table over the window from the last day
before the period to its last day gives it, and each total as the table over
the window to the period's end: then the periods do not add up to the
total. Its ``attrs`` say the comparator, the periods and the linking.

``risk`` has a row for each rebalance the risk model could be fitted for:
the name's contribution to the fund's expected volatility and its share of it
(``volatility_contribution``, ``volatility_share``), and to the fund's
expected tracking error against the comparator, from its active weight
(``tracking_error_contribution``, ``tracking_error_share``), each 0 when no
portfolio held it, from the model fitted that day
(`beacon.analysis.risk.contributions`). Its ``attrs`` say the comparator.

``liquidity`` has a row for each rebalance the run kept traded values on: the
fund's holding of the name (``value``), the name's average daily traded value
(``traded_value``) and the days the holding would take to sell
(``days_to_sell``) at the participation rate the liquidity options set, as
`instrument.days_to_sell()` gives them, 0 days when the fund did not hold it
and infinite when the name has no traded value.

``factor_exposures`` has a row for each rebalance the risk model could be
fitted for, what the name is: the model's exposure of it to each factor that
day, the market's first (``exposure/market``, ``exposure/value``, ...), and in
a global model the country, sector and currency it put the name in
(``country``, ``sector``, ``currency``, the currency blank for a name in the
portfolio's own), all blank where the model left the name out for want of
returns.

``factor_betas`` has a row for each rebalance with the rolling options'
window of days before it, how the name has behaved: its daily returns over
the risk-free rate regressed on the run's daily factor returns over that
window, each factor's beta and t-statistic (``beta/market``,
``t_stat/market``, ...), the regression's R-squared and the days fitted on.
A single name's betas are much noisier than a portfolio's; the t-statistics
say how much. Its ``attrs`` say the window.

A stored run's risk and exposures are read from what its analysis kept for
the same factors and risk model (`beacon.analysis.risk.risk_store`);
otherwise the model is fitted at every rebalance, and the factor betas read
the factor returns, both of which need the data store.
"""
# BN-468 (Karan, 2026-11-25): the first of the enhanced Drilldown's
# sections; BN-469..BN-474 add theirs to the same profile. A package since
# BN-471, a module for each section.
from functools import cached_property
from typing import Any

import pandas as pd

from ....exceptions import CalculationError
from ...panels import NAME_RETURNS, REFERENCE
from ...reports.statistics import TOO_SHORT, Unavailable
from ...risk.liquidity import BOOKS
from ..groupings import FREQUENCIES
from . import contribution, factors, risk, weights
from .columns import ACTIVE, LABELS, columns_of
from .risk import liquidity_dates, liquidity_of


class Profile:
    """One name over a run: a dated table a part, worked out when first
    asked for."""

    parts = ("rebalance_weights", "daily_weights", "attribution", "risk", "liquidity",
             "factor_exposures", "factor_betas")

    def __init__(self,
                 name: str,
                 context: Any,
                 against: str | None = None,
                 by: str = "quarter",
                 whole: Any = None):
        """
        Raises:
            ValueError: If no portfolio of the run held *name*, or *by* is
                not ``month``, ``quarter`` or ``year``.
            Unavailable: If the run lacks the comparator given.

        *whole* is the analysis over the whole run, whose work over every
        rebalance (the risk model's) is done once; *context* when None.
        """
        if by not in FREQUENCIES:
            raise ValueError(f"by is {', '.join(FREQUENCIES)}; not {by!r}.")

        self.name = name
        self.by = by
        self._context = context
        self._whole = whole if whole is not None else context
        self._against = None if against is None else context.require(against)

        if not any(name in held.columns for held in self.portfolios().values()):
            raise ValueError(f"No portfolio of the run held {name}.")

    @cached_property
    def rebalance_weights(self) -> pd.DataFrame:
        """At each rebalance: the strategy's weight in the name at each
        implementation stage, any cap or screen applied to it, and each
        comparator's weight in it that day.

        Raises:
            Unavailable: If the run was kept before its rebalances were.
        """
        return weights.rebalance_weights(self.name, self._context.run.tables, self.days(),
                                         self.portfolios())

    @cached_property
    def daily_weights(self) -> pd.DataFrame:
        """Each day: the fund's and each comparator's weight in the name,
        and the fund's active weight against the comparator."""
        return weights.daily_weights(self.name, self.days(), self.portfolios(),
                                     self.comparator())

    @cached_property
    def attribution(self) -> pd.DataFrame:
        """Each period: the name's contribution to the fund's return and to
        its return over the comparator, and each's total to date.

        Raises:
            Unavailable: If the window attributes no day.
        """
        against = self.comparator()
        held = against if against in self.portfolios() else None

        return contribution.attribution(self.name, self._context, self.by, held)

    @cached_property
    def risk(self) -> pd.DataFrame:
        """At each rebalance: the name's contribution to and share of the
        fund's expected volatility, and of its expected tracking error
        against the comparator.

        Raises:
            Unavailable: Without the data store for a run that kept no risk
                for this model, or when no rebalance has enough returns
                before it for its model.
        """
        return risk.risk(self.name, self._fitted("name_risk"), self.days(), self.comparator())

    @cached_property
    def liquidity(self) -> pd.DataFrame:
        """At each rebalance: the fund's holding of the name, the name's
        average daily traded value and the days to sell the holding.

        Raises:
            Unavailable: If the run kept no traded values, or none in the
                window.
        """
        return risk.liquidity(self.name, liquidity_of(self._whole), self.days(),
                              self.rebalances())

    @cached_property
    def factor_exposures(self) -> pd.DataFrame:
        """At each rebalance: the risk model's exposure of the name to each
        factor, and in a global model its country, sector and currency.

        Raises:
            Unavailable: As `risk` is.
        """
        return factors.factor_exposures(self.name, self._fitted("name_exposures"), self.days())

    @cached_property
    def factor_betas(self) -> pd.DataFrame:
        """At each rebalance: the name's returns' beta to each factor over
        the rolling window before it, with its t-statistic, and the
        regression's R-squared.

        Raises:
            Unavailable: Without the data store, or when no rebalance has the
                window's days before it.
        """
        context = self._whole
        days = self.days()
        returns = context.run.tables[NAME_RETURNS]
        returns = returns.set_axis(returns.columns.astype(str), axis=1)
        mine = returns[self.name] if self.name in returns.columns else pd.Series(dtype=float)
        within = [day for day in self.rebalances() if days[0] <= day <= days[-1]]

        try:
            factor_returns = context.factor_analytics.factor_returns()
        except CalculationError as error:
            raise Unavailable(TOO_SHORT, str(error)) from error

        return factors.factor_betas(self.name, mine, factor_returns, within,
                                    context.options.rolling.window, context.run.risk_free_rate,
                                    context.run.periods_per_year)

    def comparator(self) -> str | None:
        """The comparator active figures are measured against: the one
        given, else the one the analysis tracks."""
        against: str | None = self._against or self._context.tracked

        return against

    def rebalances(self) -> list[pd.Timestamp]:
        """The run's rebalances."""
        reference = self._context.run.tables.get(REFERENCE)

        if reference is None or reference.empty:
            return []

        return sorted(pd.Timestamp(day) for day in reference.index.get_level_values(0).unique())

    def days(self) -> pd.DatetimeIndex:
        """The analysis window's days, or `Unavailable` when it has none."""
        found = pd.DatetimeIndex(self._context.performance.returns.index)

        if not len(found):
            raise Unavailable(TOO_SHORT)

        return found

    def portfolios(self) -> dict[str, pd.DataFrame]:
        """Each portfolio's daily weights the run kept, by portfolio."""
        tables = self._context.run.tables
        found = {book: tables.get(table_name) for book, table_name in BOOKS.items()}

        return {book: held for book, held in found.items()
                if held is not None and not held.empty}

    def _fitted(self,
                table: str) -> pd.DataFrame:
        """A table of the risk model's work at every rebalance
        (`Context.name_risk`, `Context.name_exposures`)."""
        try:
            found: pd.DataFrame = getattr(self._whole, table)
        except CalculationError as error:
            raise Unavailable(TOO_SHORT, str(error)) from error

        return found

    def __repr__(self) -> str:
        return f"Profile({self.name!r}: {', '.join(self.parts)})"


__all__ = ["ACTIVE", "LABELS", "Profile", "columns_of", "liquidity_dates", "liquidity_of"]
