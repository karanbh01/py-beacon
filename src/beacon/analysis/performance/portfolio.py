# src/beacon/analysis/performance/portfolio.py
"""
What a run held: how many names, and its weight in each group of a grouping
field (sector, region, currency, ...) each day, against its index's and its
benchmark's.

    portfolio = PortfolioAnalytics(RunData.from_record(folder))
    portfolio.group_weights("SECTOR")

A name's group on a day is the one it had at the latest rebalance on or
before that day, as the run's reference table recorded it. A name with no
group recorded is in ``Unknown``. A book's group weights add up to its
invested weight each day: the fund's cash is in none of them.
"""
# BN-382.
import numpy as np
import pandas as pd

from ...data.base import as_of_position
from ..data import BENCHMARK, INDEX, RunData
from ..panels import (
    BENCHMARK_WEIGHTS,
    CASH,
    CHARACTERISTICS,
    COSTS,
    FUND_WEIGHTS,
    INDEX_WEIGHTS,
    MARKET,
    REFERENCE,
    TRADES,
)
from . import costs, fund_figures, trading

# The books, by name, and the tables their daily weights are in.
FUND = "fund"
BOOKS = {FUND: FUND_WEIGHTS, INDEX: INDEX_WEIGHTS, BENCHMARK: BENCHMARK_WEIGHTS}

# The group of a name with none recorded.
UNKNOWN = "Unknown"


class PortfolioAnalytics:
    """What a run held, and in which groups.

    Args:
        run: The run's data.
    """

    def __init__(self,
                 run: RunData):
        self.run = run
        self._groups: dict[str, dict[pd.Timestamp, pd.Series]] = {}

    def weights(self,
                book: str = FUND) -> pd.DataFrame | None:
        """A book's daily weights, dates by name: ``fund``, ``index`` or
        ``benchmark``; None for a book the run did not keep weights for (a
        benchmark given as a level series holds no names)."""
        table = BOOKS[book]

        return self.run.tables.get(table)

    def holdings_count(self,
                       book: str = FUND) -> pd.Series | None:
        """How many names a book held each day."""
        weights = self.weights(book)

        return None if weights is None else (weights.fillna(0.0) > 0).sum(axis=1)

    def group_weights(self,
                      field: str,
                      book: str = FUND) -> pd.DataFrame | None:
        """A book's weight in each group of *field* each day, dates by
        group, the groups sorted; None for a book without weights.

        Raises:
            KeyError: If the run kept no reference data, or none for *field*.
        """
        weights = self.weights(book)

        if weights is None:
            return None

        memberships = self._memberships(field)
        values = weights.fillna(0.0)
        grid = values.to_numpy(dtype=float)
        names = [str(name) for name in values.columns]
        each = {start: membership.reindex(names).fillna(UNKNOWN).to_numpy(dtype=object)
                for start, membership in memberships.items()}
        groups = sorted(pd.unique(np.concatenate(list(each.values()))))
        found = np.zeros((len(values), len(groups)))
        starts = sorted(memberships)
        # Each day's groups are its latest rebalance's. A day before the first
        # rebalance holds nothing, as a book's first weights are its first
        # rebalance's; its groups are taken as that rebalance's.
        dates = pd.DatetimeIndex(starts)
        which = np.array([as_of_position(dates, day) or 0 for day in values.index])

        for position, start in enumerate(starts):
            rows = which == position

            if not rows.any():
                continue

            # Each name's group as a column of a one-hot matrix, by code.
            codes = pd.Categorical(each[start], categories=groups).codes
            found[rows] = grid[rows] @ np.eye(len(groups))[codes]

        return pd.DataFrame(found, index=values.index, columns=pd.Index(groups, name=field))

    def active_group_weights(self,
                             field: str,
                             against: str = INDEX) -> pd.DataFrame | None:
        """The fund's weight in each group less *against*'s (``index`` or
        ``benchmark``), on the days the fund has; None when the comparator
        holds no names."""
        ours = self.group_weights(field)
        theirs = self.group_weights(field, against)

        if ours is None or theirs is None:
            return None

        groups = sorted(set(ours.columns) | set(theirs.columns))
        theirs = theirs.reindex(index=ours.index, columns=groups).ffill().fillna(0.0)

        return ours.reindex(columns=groups, fill_value=0.0) - theirs

    def turnover(self,
                 by: str = "day") -> pd.DataFrame:
        """The fund's buys, sells, number of trades and their mean value, and
        its one-way turnover (the lesser of buys and sells) and two-way
        turnover (all of it), over its assets: each trading day (``day``,
        with whether it was a rebalance), each rebalance's own trades
        (``rebalance``) or each year (``year``).

        Raises:
            KeyError: If the run kept no trades (one stored before 0.10.0).
            ValueError: If *by* is none of the three.
        """
        trades = self.run.tables[TRADES]

        if by == "day":
            return trading.daily_turnover(trades, self.run.nav, self.run.initial_capital)
        if by == "rebalance":
            return trading.rebalance_turnover(trades, self.run.nav, self.run.initial_capital)
        if by == "year":
            return trading.yearly_turnover(trades, self.run.nav)

        raise ValueError(f"Unknown turnover by {by!r}. Known: day, rebalance, year.")

    def trading_by_reason(self) -> pd.DataFrame:
        """The fund's trading by why it was made, a row a reason."""
        return trading.by_reason(self.run.tables[TRADES])

    def costs(self,
              by: str = "run") -> pd.DataFrame:
        """The fund's costs by kind (fixed, impact, futures rolls), the
        creation fees it received and the net: over the run (``run``), with
        each in basis points of its mean NAV a year, or each year
        (``year``), with the net in basis points of the year's mean NAV.

        Raises:
            ValueError: If *by* is neither.
        """
        table = self.run.tables[COSTS]

        if by == "run":
            return costs.cost_summary(table, self.run.nav, self.run.periods_per_year)
        if by == "year":
            return costs.yearly_costs(table, self.run.nav)

        raise ValueError(f"Unknown costs by {by!r}. Known: run, year.")

    def cost_per_traded(self) -> dict[str, float | None]:
        """The trading costs in basis points of the value traded."""
        return costs.per_traded(self.run.tables[TRADES])

    def costs_by_reason(self) -> pd.DataFrame:
        """The trading costs by why the trade was made."""
        return costs.by_reason(self.run.tables[TRADES])

    def characteristics(self) -> pd.DataFrame:
        """The weighted market cap, P/E, P/B and dividend yield of the fund,
        its index and its benchmark (when it holds names) at each rebalance:
        a row a rebalance, columns by book and figure."""
        books = {name: weights for name in BOOKS
                 if (weights := self.weights(name)) is not None}

        return fund_figures.characteristics_over_time(self.run.tables[CHARACTERISTICS], books)

    def income(self) -> pd.DataFrame:
        """Each day's distribution yield and trailing dividend yield, over
        the year to it."""
        return fund_figures.trailing_yields(self.run.tables[CASH], self.run.nav)

    def dividend_growth(self) -> pd.DataFrame:
        """Each calendar year's dividends and their growth."""
        return fund_figures.dividend_growth(self.run.tables[CASH])

    def aum(self) -> pd.DataFrame:
        """The fund's assets under management and units outstanding each
        day."""
        return pd.DataFrame({"aum": self.run.nav, "units": self.run.units})

    def net_flows(self,
                  frequency: str = "M") -> pd.DataFrame:
        """Each month's (``M``) or year's (``Y``) net flows and their share of
        the AUM it began with."""
        return fund_figures.net_flows(self.run.tables[CASH], self.run.nav, frequency)

    def flow_costs(self) -> pd.DataFrame:
        """The cost of trading each day's flows, less the creation fees
        received."""
        return fund_figures.flow_costs(self.run.tables[TRADES], self.run.tables[CASH],
                                       self.run.tables[COSTS])

    def etf_premium(self) -> dict[str, float | None] | None:
        """An ETF's premium to NAV on the exchange: its mean, volatility and
        widest each way, and the mean spread; None for another vehicle."""
        market = self.run.tables.get(MARKET)

        return None if market is None else fund_figures.premium_figures(market)

    def _memberships(self,
                     field: str) -> dict[pd.Timestamp, pd.Series]:
        """Each rebalance's group of every name, by date."""
        if field in self._groups:
            return self._groups[field]

        reference = self.run.tables[REFERENCE]

        if field not in reference.columns:
            raise KeyError(f"The run kept no {field} for its names: it has "
                           f"{', '.join(c for c in reference.columns if c != 'IDENTIFIER')}.")

        found = {}

        for date, rows in reference.groupby(level=0):
            found[pd.Timestamp(date)] = (rows.set_index("IDENTIFIER")[field].astype(object)
                                         .where(lambda s: s.notna(), UNKNOWN).astype(str))

        self._groups[field] = found

        return found
