# src/beacon/backtest/histories.py
"""
Each name's volume and converted closes, read once per run rather than daily.

Market impact, the liquidity screen and cap, and execution limits each ask of
a name, on a day, about its last few months of trading: its average traded
value, its volatility, its volume. They used to read and convert that window
from the market data every time they asked, one name and one day at a time,
recomputing windows that overlap almost entirely from one day to the next.

Here a name's whole history is read and converted once, and each question is
answered by slicing it to the window the question always used: the same
calendar span back from the day, then the same last bars. The answers are the
ones the windowed reads gave. The FX conversion is day by day, so converting
the whole history converts each day as converting its window did.
"""
# BN-338, from the backtest performance analysis: with market impact on, 95%
# of a 600-name run went to these reads.
import weakref
from dataclasses import dataclass
from functools import cached_property

import numpy as np
import pandas as pd

from ..data.base import MarketData
from ..data.fetcher import DataFetcher
from ..data.prices import PRICE_COLUMN, currencies_of
from ..exceptions import CalculationError


@dataclass(frozen=True)
class History:
    """One name's reported volume and its closes in one currency, by date."""
    volume: pd.Series | None
    closes: pd.Series

    def volume_between(self,
                       start: pd.Timestamp,
                       end: pd.Timestamp) -> pd.Series | None:
        """Its reported volume from *start* to *end*, both included; None
        when it reports no volume at all."""
        return None if self.volume is None else self.volume.loc[start:end]

    def closes_between(self,
                       start: pd.Timestamp,
                       end: pd.Timestamp) -> pd.Series:
        """Its closes from *start* to *end*, both included, blanks dropped."""
        return self.closes.loc[start:end]

    # The same series as arrays (BN-345). Slicing a Series costs about a
    # millisecond, which market impact paid on every holding of an ETF's
    # basket every week; a binary search on an array costs microseconds.
    @cached_property
    def _volume_arrays(self) -> tuple[np.ndarray, np.ndarray]:
        volume = self.volume if self.volume is not None else pd.Series(dtype=float)
        return volume.index.to_numpy(dtype="datetime64[ns]"), volume.to_numpy()

    @cached_property
    def _close_arrays(self) -> tuple[np.ndarray, np.ndarray]:
        return self.closes.index.to_numpy(dtype="datetime64[ns]"), self.closes.to_numpy()

    def average_traded(self,
                       start: pd.Timestamp,
                       end: pd.Timestamp,
                       days: int,
                       by_value: bool = True) -> float | None:
        """Its average daily traded value (close times volume) or share
        volume over its last *days* reported days from *start* to *end*;
        None when it has none."""
        if self.volume is None:
            return None

        dates, volumes = self._volume_arrays
        lo, hi = _bounds(dates, start, end)
        lo = max(lo, hi - days)

        if hi <= lo:
            return None

        volume = volumes[lo:hi]

        if not by_value:
            return float(volume.mean())

        close_dates, closes = self._close_arrays
        c_lo, c_hi = _bounds(close_dates, start, end)
        window, window_closes = close_dates[c_lo:c_hi], closes[c_lo:c_hi]
        at = np.searchsorted(window, dates[lo:hi])
        found = at < len(window)
        found[found] = window[at[found]] == dates[lo:hi][found]
        aligned = np.full(len(volume), np.nan)
        aligned[found] = window_closes[at[found]]
        traded = aligned * volume
        traded = traded[~np.isnan(traded)]

        return float(traded.mean()) if len(traded) else None

    def last_bar(self,
                 start: pd.Timestamp,
                 end: pd.Timestamp) -> tuple[pd.Timestamp, float] | None:
        """Its last close from *start* to *end*, both included, and its date;
        None when it printed none (BN-348)."""
        dates, closes = self._close_arrays
        lo, hi = _bounds(dates, start, end)

        return (pd.Timestamp(dates[hi - 1]), float(closes[hi - 1])) if hi > lo else None

    def last_close(self,
                   date: pd.Timestamp) -> float | None:
        """Its last close on or before *date*, or None (BN-346)."""
        dates, closes = self._close_arrays
        at = int(np.searchsorted(dates, np.datetime64(date, "ns"), side="right"))

        return float(closes[at - 1]) if at else None

    def returns(self,
                start: pd.Timestamp,
                end: pd.Timestamp,
                days: int) -> np.ndarray:
        """Its last *days* daily returns from its closes from *start* to
        *end*."""
        dates, closes = self._close_arrays
        lo, hi = _bounds(dates, start, end)
        window = closes[lo:hi]

        return (window[1:] / window[:-1] - 1.0)[-days:] if len(window) > 1 else window[:0]


def _bounds(dates: np.ndarray,
            start: pd.Timestamp,
            end: pd.Timestamp) -> tuple[int, int]:
    """The slice of sorted *dates* from *start* to *end*, both included."""
    return (int(np.searchsorted(dates, np.datetime64(start, "ns"), side="left")),
            int(np.searchsorted(dates, np.datetime64(end, "ns"), side="right")))


# By market data, so a refresh that replaces it starts afresh, and the
# histories go when the data does.
_HISTORIES: "weakref.WeakKeyDictionary[MarketData, dict[tuple[str, str | None], History]]" = (
    weakref.WeakKeyDictionary())


def history(fetcher: DataFetcher,
            asset_id: str,
            currency: str | None) -> History:
    """*asset_id*'s history in *currency*, read on the first ask in a run."""
    held = _HISTORIES.setdefault(fetcher.market, {})
    key = (asset_id, currency.upper() if currency else None)
    found = held.get(key)

    if found is None:
        found = held[key] = _read(fetcher, asset_id, currency)

    return found


def _read(fetcher: DataFetcher,
          asset_id: str,
          currency: str | None) -> History:
    frame = fetcher.fetch_market_data(asset_id)
    volume = (frame["VOLUME"].dropna() if not frame.empty and "VOLUME" in frame.columns
              else None)

    return History(volume=volume, closes=_converted(fetcher, asset_id, frame, currency))


def _converted(fetcher: DataFetcher,
               asset_id: str,
               frame: pd.DataFrame,
               currency: str | None) -> pd.Series:
    """*asset_id*'s closes from its own rows, converted into *currency* day by
    day as `fetch_prices` converts them, blanks dropped.

    From the name's rows rather than through `fetch_prices`, whose read by a
    list of one makes pandas check the whole store's index for uniqueness on
    every call: 88 s over the 5,168 names of a 6,000-name run (BN-345).
    """
    if frame.empty or PRICE_COLUMN not in frame.columns:
        return pd.Series(dtype=float)

    closes = frame[PRICE_COLUMN]

    if currency is not None:
        target = currency.upper()
        quoted = currencies_of(fetcher, [asset_id], default=target)[asset_id]

        if quoted != target:
            rates = fetcher.fx_rates_on(quoted, target, closes.index)

            if rates is None:
                raise CalculationError(
                    calculation_name="Prices",
                    details=(f"no {quoted}/{target} rate, so {asset_id} cannot be priced "
                             f"in {target}. Leaving it in {quoted} would compare it with "
                             f"the rest on magnitude alone. Load the pair, or leave this "
                             f"name out."))

            closes = closes.mul(rates, axis=0)

    return closes.dropna()
