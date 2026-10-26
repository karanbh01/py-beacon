# src/beacon/analysis/snapshots.py
"""
What a run's names looked like at each rebalance, kept with the run.

A rerun of a run's analysis months later should give the same answer even if
the data store has been refreshed since, so what the analysis groups and
weighs names by is read as it stood at each rebalance and kept with the run.

Each snapshot is a long table indexed by date, a row per name per rebalance,
with an ``IDENTIFIER`` column and a column per field. A name with no record in
force on a date has no row for it. Between rebalances, the latest snapshot on
or before a day is the one in force.
"""
# BN-366, the characteristics BN-367 and the traded values BN-368.
import numpy as np
import pandas as pd

from ..backtest.result import BacktestResult
from ..backtest.screens import DEFAULT_LIQUIDITY_DAYS
from ..data.base import count_through
from ..data.corporate_actions import DIVIDEND, TRAILING_YEAR, without_cancelled
from ..data.fetcher import DataFetcher
from ..data.prices import PRICE_COLUMN, converted, wide_columns
from ..data.returns import name_returns

SHARES_COLUMN = "SHARES_OUTSTANDING"
VOLUME_COLUMN = "VOLUME"

IDENTIFIER = "IDENTIFIER"

# The fields names are grouped by, where the reference data has them.
GROUPING_FIELDS = ("SECTOR", "GICS_INDUSTRY_GROUP", "GICS_INDUSTRY", "SUB_INDUSTRY",
                   "REGION", "COUNTRY_LISTING", "COUNTRY_DOMICILE", "CURRENCY",
                   "EXCHANGE")


def rebalance_dates(result: BacktestResult) -> list[pd.Timestamp]:
    """Every day the fund or its index rebalanced, within the run, sorted."""
    days = result.portfolio.nav.index
    dates = {pd.Timestamp(row.date) for row in result.rebalance_pricing}
    books = [result.index.target, result.index.optimised, result.benchmark]

    for book in books:
        if book is not None and book.source is not None:
            dates |= {pd.Timestamp(date) for date in book.source.constituent_snapshots}

    if len(days) == 0:
        return []

    return sorted(date for date in dates if days[0] <= date <= days[-1])


def reference_snapshots(source: DataFetcher,
                        names: list[str],
                        dates: list[pd.Timestamp]) -> pd.DataFrame:
    """The grouping fields of *names* as they stood on each of *dates*.

    Args:
        source: The data the run used.
        names: The names to describe.
        dates: The days to describe them on.

    Returns:
        pd.DataFrame: Indexed by date, with ``IDENTIFIER`` and each of
        `GROUPING_FIELDS` the reference data has, as text; missing where a
        name has no value for a field.
    """
    frames = []

    for date in dates:
        frame = source.fetch_reference_data(names, str(date.date()))

        if frame.empty:
            continue

        fields = [field for field in GROUPING_FIELDS if field in frame.columns]
        rows = frame[fields].copy()
        rows.insert(0, IDENTIFIER, rows.index.astype(str))
        rows.index = pd.DatetimeIndex([date] * len(rows))
        frames.append(rows)

    if not frames:
        return pd.DataFrame(columns=[IDENTIFIER])

    return pd.concat(frames)


# The characteristics of `characteristics_snapshots`, and the features that
# give the ratios.
MARKET_CAP = "MARKET_CAP"
PE_RATIO = "PE_RATIO"
PB_RATIO = "PB_RATIO"
DIVIDEND_YIELD = "DIVIDEND_YIELD"
MOMENTUM = "MOMENTUM"
CHARACTERISTICS = (MARKET_CAP, PE_RATIO, PB_RATIO, DIVIDEND_YIELD, MOMENTUM)
RATIO_FEATURES = {PE_RATIO: "pe_ratio", PB_RATIO: "pb_ratio"}

# Momentum is the return from a year before to a month before, in sessions.
_MOMENTUM_FROM, _MOMENTUM_TO = 252, 21

# How many sessions a close may be carried to a rebalance before a name is
# taken to have stopped trading.
_STALE_SESSIONS = 5

# How far before the first rebalance prices are read, for a year of momentum.
_HISTORY = pd.Timedelta(days=400)


def characteristics_snapshots(source: DataFetcher,
                              names: list[str],
                              dates: list[pd.Timestamp],
                              currency: str,
                              market: dict[str, pd.DataFrame] | None = None) -> pd.DataFrame:
    """Each name's characteristics as they stood on each of *dates*.

    - ``MARKET_CAP``: the close times shares outstanding, in *currency*.
    - ``PE_RATIO`` and ``PB_RATIO``: the ``pe_ratio`` and ``pb_ratio``
      features in force on the date, by `DataFetcher.fetch_feature`'s rules.
    - ``DIVIDEND_YIELD``: ordinary dividends over the trailing year over the
      close, as `DataFetcher.fetch_trailing_dividend_yield` gives it.
    - ``MOMENTUM``: the total return, in the name's own currency, from 252
      sessions before the date to 21 sessions before it.

    A value the data does not give is missing. A name only has a row while it
    trades: one whose last close is more than five sessions old by the date
    has stopped, and is left out.

    Args:
        source: The data the run used.
        names: The names to describe.
        dates: The days to describe them on.
        currency: The currency market caps are in.
        market: ``CLOSE`` and ``SHARES_OUTSTANDING`` by date and name, from at
            least 400 days before the first date, for a caller that has read
            them already (`market_history`); read from *source* when None.

    Returns:
        pd.DataFrame: Indexed by date, with ``IDENTIFIER`` and each of
        `CHARACTERISTICS`.
    """
    if not dates or not names:
        return pd.DataFrame(columns=[IDENTIFIER, *CHARACTERISTICS])

    start, end = str((dates[0] - _HISTORY).date()), str(dates[-1].date())
    market = market if market is not None else market_history(source, names, dates)
    local = market[PRICE_COLUMN]

    if local.empty:
        return pd.DataFrame(columns=[IDENTIFIER, *CHARACTERISTICS])

    names = list(local.columns)
    sessions = local.index.to_numpy()
    closes = local.ffill(limit=_STALE_SESSIONS)
    book = converted(source, local, currency).ffill(limit=_STALE_SESSIONS)
    shares = market[SHARES_COLUMN].reindex(index=local.index, columns=names).ffill()
    growth = (1.0 + name_returns(source, names, start, end, None, closes=local)
              .fillna(0.0)).cumprod().where(closes.notna())
    ratios = {column: source.features.table_as_of(names, feature, dates)
              for column, feature in RATIO_FEATURES.items()}
    trailing = _trailing_dividends(source, names, dates)
    frames = []

    for date in dates:
        count = count_through(sessions, np.datetime64(date))

        if count == 0:
            continue

        row = count - 1
        close = closes.iloc[row]
        values = pd.DataFrame({
            MARKET_CAP: book.iloc[row] * shares.iloc[row],
            PE_RATIO: ratios[PE_RATIO].loc[date],
            PB_RATIO: ratios[PB_RATIO].loc[date],
            DIVIDEND_YIELD: (trailing.loc[date] / close).where(close > 0),
            MOMENTUM: _momentum(growth, row),
        }, index=names)
        # A name that has stopped trading, or not yet started, is not
        # described, though a ratio published before may still be in force.
        values = values[close.notna()].dropna(how="all")
        values.insert(0, IDENTIFIER, values.index.astype(str))
        values.index = pd.DatetimeIndex([date] * len(values))
        frames.append(values)

    if not frames:
        return pd.DataFrame(columns=[IDENTIFIER, *CHARACTERISTICS])

    return pd.concat(frames)


def market_history(source: DataFetcher,
                   names: list[str],
                   dates: list[pd.Timestamp],
                   until: pd.Timestamp | None = None) -> dict[str, pd.DataFrame]:
    """The market data `characteristics_snapshots` reads, by column: closes,
    shares outstanding and volumes from 400 days before the first of *dates*
    to the last of them, or to *until* when later, in one read."""
    last = max([*dates, *([until] if until is not None else [])])

    return wide_columns(source, names, str((dates[0] - _HISTORY).date()), str(last.date()),
                        [PRICE_COLUMN, SHARES_COLUMN, VOLUME_COLUMN])


def _momentum(growth: pd.DataFrame,
              row: int) -> pd.Series:
    """The return from a year of sessions before *row* to a month before."""
    if row < _MOMENTUM_FROM:
        return pd.Series(np.nan, index=growth.columns)

    return growth.iloc[row - _MOMENTUM_TO] / growth.iloc[row - _MOMENTUM_FROM] - 1.0


def _trailing_dividends(source: DataFetcher,
                        names: list[str],
                        dates: list[pd.Timestamp]) -> pd.DataFrame:
    """Ordinary dividends per share over the year to each date, by date and
    name: ex-dates after a year before and on or before the date."""
    table = pd.DataFrame(0.0, index=pd.DatetimeIndex(dates), columns=names)
    actions = source.corporate_actions

    if actions.is_empty:
        return table

    frame = without_cancelled(actions.data.reset_index(drop=True))
    frame = frame[(frame["TYPE"].astype(str).str.upper() == DIVIDEND)
                  & frame["IDENTIFIER"].astype(str).isin(set(names))]

    if frame.empty:
        return table

    ex_dates = pd.to_datetime(frame["EX_DATE"])
    values = pd.to_numeric(frame["VALUE"], errors="coerce")
    identifiers = frame["IDENTIFIER"].astype(str)

    for date in dates:
        inside = (ex_dates > date - TRAILING_YEAR) & (ex_dates <= date)
        paid = values[inside].groupby(identifiers[inside]).sum()
        table.loc[date, paid.index] = paid.to_numpy()

    return table


# The column of `traded_value_snapshots`.
TRADED_VALUE = "TRADED_VALUE"


def liquidity_dates(result: BacktestResult) -> list[pd.Timestamp]:
    """Each rebalance and the last day of each month of the run, sorted."""
    days = result.portfolio.nav.index

    if len(days) == 0:
        return []

    month_ends = pd.Series(days, index=days).groupby(days.to_period("M")).max()

    return sorted(set(rebalance_dates(result)) | {pd.Timestamp(day) for day in month_ends})


def traded_value_snapshots(source: DataFetcher,
                           names: list[str],
                           dates: list[pd.Timestamp],
                           currency: str,
                           market: dict[str, pd.DataFrame] | None = None,
                           lookback_days: int = DEFAULT_LIQUIDITY_DAYS) -> pd.DataFrame:
    """Each name's average daily traded value on each of *dates*, as the
    liquidity screens measure it.

    The close times the volume, in *currency* converted day by day, averaged
    over the name's last *lookback_days* days with a volume in the window the
    screens read (`beacon.backtest.screens.average_traded`).

    Args:
        source: The data the run used.
        names: The names to describe.
        dates: The days to describe them on.
        currency: The currency traded values are in.
        market: ``CLOSE`` and ``VOLUME`` by date and name, from at least 400
            days before the first date (`market_history`); read from *source*
            when None.
        lookback_days: How many days with a volume to average.

    Returns:
        pd.DataFrame: Indexed by date, with ``IDENTIFIER`` and
        ``TRADED_VALUE``; a name with no volume in the window has no row.
    """
    if not dates or not names:
        return pd.DataFrame(columns=[IDENTIFIER, TRADED_VALUE])

    market = market if market is not None else market_history(source, names, dates)
    local = market[PRICE_COLUMN]

    if local.empty:
        return pd.DataFrame(columns=[IDENTIFIER, TRADED_VALUE])

    closes = converted(source, local, currency).to_numpy()
    volumes = market[VOLUME_COLUMN].reindex(index=local.index, columns=local.columns).to_numpy()
    sessions = local.index.to_numpy()
    frames = []

    for date in dates:
        # The window the screens read: enough calendar days for the trading
        # days asked for, both ends included.
        start = date - pd.Timedelta(days=int(lookback_days * 1.6) + 10)
        lo = int(np.searchsorted(sessions, np.datetime64(start), side="left"))
        hi = count_through(sessions, np.datetime64(date))
        reported = ~np.isnan(volumes[lo:hi])
        # The last *lookback_days* reported days of each name.
        kept = reported & (reported[::-1].cumsum(axis=0)[::-1] <= lookback_days)
        traded = np.where(kept, closes[lo:hi] * volumes[lo:hi], np.nan)
        counted = (~np.isnan(traded)).sum(axis=0)

        with np.errstate(invalid="ignore"):
            average = np.nansum(traded, axis=0) / counted

        found = counted > 0
        frame = pd.DataFrame({IDENTIFIER: local.columns[found].astype(str),
                              TRADED_VALUE: average[found]},
                             index=pd.DatetimeIndex([date] * int(found.sum())))
        frames.append(frame)

    return pd.concat(frames)
