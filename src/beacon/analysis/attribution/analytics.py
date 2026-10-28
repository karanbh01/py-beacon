# src/beacon/analysis/attribution/analytics.py
"""
A run's attribution: what each name, or each group, contributed to a book's
return, or to the fund's return over a comparator.

Each day a name contributes its weight at the close before times its return
that day (the run's own daily total return for it, in the book's currency),
and the days are linked over the window by Carino's factors so the
contributions add up to the compounded return (see
`beacon.analysis.attribution.contribution`). A group contributes what its
names did, each in the group it was in at that close.

**Active** attribution takes the fund's weights less the comparator's, both
as they drift, against the fund's daily return less the comparator's. What the
names do not explain (cash, costs, fees, futures, the timing of dividends) is
the residual, reported with every result and never folded into a name.
"""
# BN-393.
import pandas as pd

from ..data import INDEX, RunData
from ..panels import BENCHMARK_WEIGHTS, CURRENCIES, FUND_WEIGHTS, INDEX_WEIGHTS, NAME_RETURNS
from ..performance.comparisons import aligned
from ..risk.factors import FactorAnalytics
from . import currency, model
from .brinson import BrinsonResult, brinson, by_period
from .contribution import AttributionResult, attribute
from .effects import EffectsResult, linked_effects
from .groups import by_group, memberships

FUND = "fund"
BOOKS = {FUND: FUND_WEIGHTS, "index": INDEX_WEIGHTS, "benchmark": BENCHMARK_WEIGHTS}


class Attribution:
    """What produced a run's returns, by name or group.

    Args:
        run: The run's data, with its weights, name returns and reference.
    """

    def __init__(self,
                 run: RunData):
        self.run = run

    def contribution(self,
                     book: str = FUND,
                     field: str | None = None,
                     start: str | pd.Timestamp | None = None,
                     end: str | pd.Timestamp | None = None) -> AttributionResult:
        """Each name's (or, with *field*, each group's) contribution to a
        book's return (``fund``, ``index`` or ``benchmark``) over the window,
        the whole run by default."""
        returns = self._returns(book)
        weights = self.weights(book)

        return self._attributed(returns, weights, field, start, end)

    def active(self,
               against: str = INDEX,
               field: str | None = None,
               start: str | pd.Timestamp | None = None,
               end: str | pd.Timestamp | None = None) -> AttributionResult:
        """Each name's (or group's) contribution to the fund's daily return
        less *against*'s, from its active weight, over the window."""
        ours, theirs = aligned(self.run.returns, self._levels(against))
        fund, comparator = self.weights(FUND), self.weights(against)
        names = sorted(set(fund.columns) | set(comparator.columns))
        days = fund.index
        active = (fund.reindex(columns=names, fill_value=0.0)
                  - comparator.reindex(index=days, method="ffill")
                  .reindex(columns=names).fillna(0.0))

        return self._attributed(ours - theirs, active, field, start, end)

    def brinson(self,
                field: str,
                against: str = INDEX,
                start: str | pd.Timestamp | None = None,
                end: str | pd.Timestamp | None = None) -> BrinsonResult:
        """The fund's Brinson-Fachler attribution against *against* by the
        grouping field *field*, over the window, the whole run by default:
        each group's allocation, selection and interaction, cash a group of
        its own, linked over the days."""
        return brinson(self.run, self.weights(FUND), self.weights(against),
                       self._levels(against), self._name_returns(),
                       field, self._days(start, end))

    def brinson_by_period(self,
                          field: str,
                          against: str = INDEX,
                          frequency: str = "Y") -> pd.DataFrame:
        """The Brinson-Fachler effects over each year (``Y``) or month
        (``M``)."""
        fund, theirs = self.weights(FUND), self.weights(against)
        levels, returns = self._levels(against), self._name_returns()

        return by_period(self._days(None, None),
                         lambda days: brinson(self.run, fund, theirs, levels, returns, field,
                                              days),
                         frequency)

    def _days(self,
              start: str | pd.Timestamp | None,
              end: str | pd.Timestamp | None) -> pd.DatetimeIndex:
        """The days the fund holds names and the names have returns, in the
        window."""
        weights = self.weights(FUND)
        held = weights.index[weights.sum(axis=1) > 0.0]
        days = held.intersection(self._name_returns().index)

        if not len(days):
            return days

        days = pd.DatetimeIndex(_window(pd.Series(0.0, index=days), start, end).index)

        return days

    def _name_returns(self) -> pd.DataFrame:
        returns = self.run.tables[NAME_RETURNS].sort_index()
        returns.columns = returns.columns.astype(str)

        return returns

    def currency(self,
                 against: str | None = None,
                 by_currency: bool = False,
                 start: str | pd.Timestamp | None = None,
                 end: str | pd.Timestamp | None = None) -> EffectsResult:
        """The fund's return (or, *against* a comparator, its active return)
        split into its names' local returns, their currencies' moves and the
        cross term; with *by_currency*, each by the names' currencies too."""
        days = self._days(start, end)
        weights = self._book_weights(against)
        lagged = weights.reindex(days, method="ffill").shift(1).fillna(0.0)
        parts = currency.currency_parts(lagged, self._name_returns(),
                                        currency.local_returns(self.run))
        daily = pd.DataFrame({name: frame.sum(axis=1) for name, frame in parts.items()})
        table = (currency.by_currency(parts, self.run.tables[CURRENCIES]["CURRENCY"])
                 if by_currency else None)
        ours, theirs = self._paired(against)

        return linked_effects(daily.iloc[1:], ours, theirs, table)

    def factors(self,
                analytics: FactorAnalytics,
                against: str | None = None,
                method: str = model.HOLDINGS,
                start: str | pd.Timestamp | None = None,
                end: str | pd.Timestamp | None = None) -> EffectsResult:
        """The fund's return (or its active return *against* a comparator)
        explained by the run's factors: from its holdings' exposures
        (``holdings``: market, each factor, specific) or from its returns'
        betas (``returns``: risk-free, alpha, each factor, residual).

        Raises:
            ValueError: If the method is neither.
        """
        days = self._days(start, end)
        ours, theirs = self._paired(against)

        if method == model.HOLDINGS:
            lagged = (self._book_weights(against).reindex(days, method="ffill").shift(1)
                      .fillna(0.0))
            daily = model.holdings_parts(analytics, lagged, self._name_returns())
        elif method == model.RETURNS:
            target = ours - theirs if theirs is not None else ours
            rate = (0.0 if theirs is not None
                    else model.daily_rate(self.run.risk_free_rate, self.run.periods_per_year))
            daily = model.returns_parts(target.reindex(days[1:]).dropna(),
                                        analytics.factor_returns(), rate,
                                        self.run.periods_per_year)
        else:
            raise ValueError(f"Unknown method {method!r}. Known: {', '.join(model.METHODS)}.")

        return linked_effects(daily, ours, theirs)

    def _book_weights(self,
                      against: str | None) -> pd.DataFrame:
        """The fund's weights, less *against*'s when one is named."""
        fund = self.weights(FUND)

        if against is None:
            return fund

        theirs = self.weights(against)
        names = sorted(set(fund.columns) | set(theirs.columns))

        return (fund.reindex(columns=names, fill_value=0.0)
                - theirs.reindex(index=fund.index, method="ffill").reindex(columns=names)
                .fillna(0.0))

    def _paired(self,
                against: str | None) -> tuple[pd.Series, pd.Series | None]:
        """The fund's daily returns, and the comparator's on the same days."""
        if against is None:
            return self.run.returns, None

        return aligned(self.run.returns, self._levels(against))

    def weights(self,
                book: str = FUND) -> pd.DataFrame:
        """A book's daily weights, by name."""
        table = self.run.tables.get(BOOKS[book])

        if table is None or table.empty:
            raise KeyError(f"The run kept no weights for its {book}.")

        table = table.sort_index().fillna(0.0)
        table.columns = table.columns.astype(str)

        return table

    def _attributed(self,
                    returns: pd.Series,
                    weights: pd.DataFrame,
                    field: str | None,
                    start: str | pd.Timestamp | None,
                    end: str | pd.Timestamp | None) -> AttributionResult:
        """*returns* attributed to *weights*' names, or to their groups."""
        name_returns = self.run.tables[NAME_RETURNS].sort_index()
        name_returns.columns = name_returns.columns.astype(str)
        # The days the names have returns and the book weights, so by name
        # and by group cover the same days.
        returns = _window(returns.dropna(), start, end)
        returns = returns.loc[returns.index.intersection(name_returns.index)
                              .intersection(weights.index)]

        if field is None:
            return attribute(returns, weights, name_returns)

        lagged = weights.reindex(returns.index, method="ffill").shift(1)
        names = [name for name in lagged.columns if name in name_returns.columns]
        contributions = lagged[names] * name_returns.reindex(returns.index)[names]
        groups = memberships(self.run, field)
        held = by_group(lagged[names].fillna(0.0), groups)
        earned = by_group(contributions.fillna(0.0), groups)
        group_returns = (earned / held.where(held != 0.0)).fillna(0.0)

        # attribute() takes the weights held at the end of each day and lags
        # them itself, so the lagged group weights go in one day earlier.
        return attribute(returns, held.shift(-1), group_returns)

    def _returns(self,
                 book: str) -> pd.Series:
        """A book's daily returns: the fund's, or a comparator's from its
        levels."""
        if book == FUND:
            return self.run.returns

        _, theirs = aligned(self.run.returns, self._levels(book))

        return theirs

    def _levels(self,
                book: str) -> pd.Series:
        if book not in self.run.comparators:
            raise KeyError(f"The run has no {book}: it has "
                           f"{', '.join(self.run.comparators) or 'no comparators'}.")

        return self.run.comparators[book]


def _window(series: pd.Series,
            start: str | pd.Timestamp | None,
            end: str | pd.Timestamp | None) -> pd.Series:
    """*series* from *start* to *end*, both included."""
    if start is not None:
        series = series[series.index >= pd.Timestamp(start)]

    if end is not None:
        series = series[series.index <= pd.Timestamp(end)]

    return series
