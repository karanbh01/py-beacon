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
from collections.abc import Mapping

import numpy as np
import pandas as pd

from ...data.base import as_of_position
from ..class_panels import CLASS_DAYS, CLASS_FLOWS, CLASS_HEDGES
from ..data import BENCHMARK, INDEX, RunData
from ..panels import (
    BENCHMARK_WEIGHTS,
    CASH,
    CASH_BAND_EVENTS,
    CHARACTERISTICS,
    COSTS,
    FUND_WEIGHTS,
    FUTURES_DAYS,
    FUTURES_TRADES,
    INDEX_WEIGHTS,
    MARKET,
    REFERENCE,
    TRADES,
)
from . import cash_management, costs, dealing, fund_figures, futures_figures, trading

# The books, by name, and the tables their daily weights are in.
FUND = "fund"
BOOKS = {FUND: FUND_WEIGHTS, INDEX: INDEX_WEIGHTS, BENCHMARK: BENCHMARK_WEIGHTS}

# The group of a name with none recorded.
UNKNOWN = "Unknown"

# What the holdings range gives, a row each.
LEAST, MOST, LATEST = "least", "most", "latest"

# The fund's weight in no group: its cash, in the latest exposures.
CASH_GROUP = "Cash"

# A cash weight smaller than this is rounding, and left out.
NEGLIGIBLE_CASH = 1e-9


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

    def holdings_range(self) -> pd.DataFrame:
        """The fewest and most names each book held on any day of the run,
        and how many it held on the last: a row a measure (``least``,
        ``most``, ``latest``, in the ``measure`` column) and a column a book;
        blank for a book the run kept no weights for."""
        # BN-423. Every day, not only rebalances: a name that stops trading
        # between them leaves the count until the next.
        found: dict[str, object] = {"measure": [LEAST, MOST, LATEST]}

        for book in BOOKS:
            counts = self.holdings_count(book)
            found[book] = ([None] * 3 if counts is None or counts.empty
                           else [int(counts.min()), int(counts.max()), int(counts.iloc[-1])])

        return pd.DataFrame(found)

    def holdings_distribution(self) -> pd.DataFrame:
        """How many days the fund held each number of names, fewest first:
        columns ``names``, ``days`` and ``share``, the share of the run's
        days, which sum to 1."""
        counts = self.holdings_count()

        if counts is None or counts.empty:
            return pd.DataFrame({"names": pd.Series(dtype=int), "days": pd.Series(dtype=int),
                                 "share": pd.Series(dtype=float)})

        days = counts.value_counts().sort_index()

        return pd.DataFrame({"names": days.index.astype(int), "days": days.to_numpy(dtype=int),
                             "share": days.to_numpy(dtype=float) / float(days.sum())})

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

    def group_exposures(self,
                        fields: Mapping[str, str]) -> pd.DataFrame:
        """Each book's weight in each group on the fund's last day, for each
        grouping of *fields* (a grouping's name, such as ``sector``, and the
        reference field it is read from).

        Columns ``grouping``, ``group``, then a column a book, blank for a
        book the run kept no weights for. A grouping's groups are in the
        order of the index's weight, largest first, then those the index does
        not hold by the fund's, then ``Unknown`` and the fund's ``Cash``: its
        weight in no group, so each book's weights sum to 1 in each grouping.
        A grouping whose field the run did not keep is left out.

        Raises:
            KeyError: If the run kept none of the fields.
        """
        # BN-424, for beacon-ui's Summary (BU-294).
        rows: list[dict[str, object]] = []
        kept = 0

        for grouping, field in fields.items():
            try:
                latest = {book: self._latest_groups(field, book) for book in BOOKS}
            except KeyError:
                continue

            kept += 1
            rows.extend({"grouping": grouping, "group": group,
                         **{book: (None if weights is None else float(weights.get(group, 0.0)))
                            for book, weights in latest.items()}}
                        for group in _exposure_order(latest))

        if not kept:
            raise KeyError(f"No reference data for {', '.join(fields.values())}.")

        return pd.DataFrame(rows, columns=["grouping", "group", *BOOKS])

    def _latest_groups(self,
                       field: str,
                       book: str) -> pd.Series | None:
        """*book*'s weight in each group of *field* on the fund's last day,
        the fund's with its cash; None for a book without weights."""
        held = self.group_weights(field, book)

        if held is None or held.empty:
            return None

        fund = self.weights(FUND)
        last = fund.index[-1] if fund is not None and not fund.empty else held.index[-1]
        position = as_of_position(held.index, last)
        weights = held.iloc[position if position is not None else 0]
        weights = weights[weights.abs() > 0.0]

        if book == FUND and abs(cash := 1.0 - float(weights.sum())) > NEGLIGIBLE_CASH:
            weights = pd.concat([weights, pd.Series({CASH_GROUP: cash})])

        return weights

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
        """The fund's costs by kind (fixed, impact, futures rolls and futures trading), the
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
        """The weighted market cap, P/E, P/B, dividend yield and earnings and
        sales growth of the fund, its index and its benchmark (when it holds
        names) at each rebalance: a row a rebalance, columns by book and
        figure."""
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

    def cash_band_events(self) -> pd.DataFrame:
        """Each time the cash left its band and was raised or spent back to
        its target: the cash before and after, the target and band, the gap
        closed, what was traded and its cost. Empty for a run without a band."""
        return cash_management.band_events(self._band_events())

    def cash_band_summary(self) -> pd.DataFrame:
        """The raises, the spend-downs and both: how many, what they traded
        and cost, and how close they brought the cash to its target."""
        return cash_management.band_summary(self._band_events())

    def cash_band_calendar(self,
                           frequency: str = "M") -> pd.DataFrame:
        """How many raises and spend-downs each month (``M``) or year (``Y``)
        had, and their cost."""
        return cash_management.band_calendar(self._band_events(), frequency)

    def cash_trading(self) -> pd.DataFrame:
        """Every trade made to raise or spend cash, by tag and reason."""
        return cash_management.cash_trading(self.run.tables[TRADES])

    def dealing_costs(self) -> pd.DataFrame:
        """Each day's subscriptions or redemptions (an ETF's creations or
        redemptions): the trading they caused, what the dealing investors
        paid toward it, the net cost to the fund, and for an ETF the cost to
        the AP."""
        return dealing.dealing_costs(self.run.tables[TRADES], self.run.tables[CASH],
                                     self.run.tables[COSTS], self.run.tables.get(MARKET))

    def dealing_summary(self,
                        by: str = "run") -> pd.DataFrame:
        """The flows' costs added up by kind, over the run (``run``) or each
        year (``year``)."""
        return dealing.dealing_summary(self.dealing_costs(), by)

    def class_costs(self,
                    by: str = "run") -> pd.DataFrame:
        """Each share class's fees, hedging costs less the carry earned, the
        trading its flows caused less what its dealing investors paid, and
        its share of the rest of the fund's trading, in the book's currency:
        over the run (``run``) or each year (``year``).

        Raises:
            KeyError: If the run had no share classes.
        """
        tables = {name: self.run.tables[name] for name in (CLASS_DAYS, CLASS_FLOWS, CLASS_HEDGES)}

        return dealing.class_costs(tables, self.run.tables[TRADES], self.run.tables[CASH],
                                   self.run.tables[COSTS], self.run.periods_per_year, by)

    def futures_trading(self) -> pd.DataFrame:
        """The fund's futures trading by why each trade was made, between
        rolls and all of it: trades, contracts, notional, cost per contract
        and in basis points, and turnover.

        Raises:
            KeyError: If the run held no index futures.
        """
        return futures_figures.futures_trading(self.run.tables[FUTURES_TRADES], self.run.nav,
                                               self.run.periods_per_year)

    def futures_basis(self,
                      daily: bool = False) -> pd.DataFrame:
        """The futures' basis to their index, its carry and its mispricing:
        each day (*daily*), or their mean, volatility and extremes in basis
        points."""
        days = self.run.tables[FUTURES_DAYS]

        return futures_figures.daily_basis(days) if daily else futures_figures.basis_summary(days)

    def futures_basis_risk(self) -> pd.DataFrame:
        """The tracking error the futures bring, split into the pricing basis
        (against their own index) and the underlying mismatch (their index
        against the fund's)."""
        return futures_figures.basis_risk(self.run.tables[FUTURES_DAYS],
                                          self.run.periods_per_year)

    def _band_events(self) -> pd.DataFrame:
        """The run's cash band events; none when it kept none."""
        found = self.run.tables.get(CASH_BAND_EVENTS)

        if found is None:
            return pd.DataFrame(columns=list(cash_management.EVENT_COLUMNS),
                                index=pd.DatetimeIndex([], name="date"))

        return found

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


def _exposure_order(latest: Mapping[str, pd.Series | None]) -> list[str]:
    """The groups any book holds, by the index's weight, then the fund's,
    with ``Unknown`` and ``Cash`` last."""
    index = latest.get(INDEX)
    fund = latest.get(FUND)
    groups = {str(group) for weights in latest.values() if weights is not None
              for group in weights.index}
    last = [group for group in (UNKNOWN, CASH_GROUP) if group in groups]
    rest = groups - set(last)

    def weight(weights: pd.Series | None,
               group: str) -> float:
        return float(weights.get(group, 0.0)) if weights is not None else 0.0

    ordered = sorted(rest, key=lambda group: (-weight(index, group), -weight(fund, group), group))

    return [*ordered, *last]
