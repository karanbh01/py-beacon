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

import pandas as pd

from ..data.base import MarketData
from ..data.fetcher import DataFetcher


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
    prices = fetcher.fetch_prices([asset_id], currency=currency)
    closes = (prices[asset_id].dropna() if asset_id in prices.columns
              else pd.Series(dtype=float))

    return History(volume=volume, closes=closes)
