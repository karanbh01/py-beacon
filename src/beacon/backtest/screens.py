# src/beacon/backtest/screens.py
"""
Screens: which names a backtest may hold, decided at each rebalance.

    Implementation(screens=[
        LiquidityScreen(min_traded_value=5e6, exit=4e6),
        MarketCapScreen(min_cap=1e9),
        ExclusionScreen(sectors=["Tobacco"]),
    ])

A screen runs after the index's target weights and before any trade is
generated. A name it rejects leaves the target, so it is sold if held and
never bought, and its weight is redistributed (see `Implementation`). The
index itself is untouched.

## Buffers

A threshold screen takes an entry level and an optional exit level. A name
not held must clear the entry level to come in; a held name stays until it
falls below the exit level. With the exit below the entry, a name hovering
near the threshold does not flip in and out at every rebalance. Without an
exit level, both are the entry level.

## Currency

Money thresholds (market cap, traded value, price) are in the book's
currency, and every value is converted into it before it is compared, so a
threshold means the same money for every name.

A name a screen cannot value (no price, no share count) is rejected unless
the screen is built with `on_missing=True`.
"""
# BN-264, phase 2 of decisions/0006. ExpressionScreen was a trade modifier
# that only trimmed a failing holding (BN-255); it is a screen here.
import logging
from abc import ABC, abstractmethod
from collections.abc import Iterable
from dataclasses import dataclass

import pandas as pd

from ..data.fetcher import DataFetcher
from ..expressions import data
from ..expressions.core import Expression
from ..expressions.resolve import resolve, value_of
from ..expressions.validation import require_valid

logger = logging.getLogger(__name__)

# How far back a price is looked for when the rebalance day has none.
PRICE_LOOKBACK_DAYS = 10

# Trading days in a liquidity average: about three months.
DEFAULT_LIQUIDITY_DAYS = 63


@dataclass(frozen=True)
class ScreenContext:
    """What a screen is told about the run it screens for.

    Attributes:
        fetcher: The data the run reads.
        currency: The book's currency, which money thresholds are in.
    """
    fetcher: DataFetcher
    currency: str


class Screen(ABC):
    """Decides whether a name may be held on a rebalance date."""

    @property
    def name(self) -> str:
        """How the run's record names this screen."""
        return type(self).__name__

    def prepare(self,  # noqa: B027 -- an optional hook, not part of the interface
                fetcher: DataFetcher) -> None:
        """Check the screen against the data before the run starts."""

    @abstractmethod
    def admits(self,
               asset_id: str,
               date: pd.Timestamp,
               held: bool,
               context: ScreenContext) -> bool:
        """Whether *asset_id* may be held on *date*.

        Args:
            asset_id: The name.
            date: The rebalance date.
            held: Whether the book holds it now, which decides whether a
                buffered screen applies its entry or its exit level.
            context: The run's data and book currency.
        """


class ThresholdScreen(Screen):
    """A minimum on a value, with an optional lower exit level.

    Args:
        entry: The level a name not held must reach.
        exit: The level a held name must stay at or above. None uses
            *entry*. Must not be above *entry*.
        on_missing: Whether a name with no value passes.

    Raises:
        ValueError: If *exit* is above *entry*.
    """

    def __init__(self,
                 entry: float,
                 exit: float | None = None,
                 on_missing: bool = False):
        if exit is not None and exit > entry:
            raise ValueError(f"{type(self).__name__}: the exit level {exit:g} "
                             f"is above the entry level {entry:g}. A buffer "
                             f"lets a held name stay below the entry level, "
                             f"not above it.")

        self.entry = entry
        self.exit = exit
        self.on_missing = on_missing

    @abstractmethod
    def value(self,
              asset_id: str,
              date: pd.Timestamp,
              context: ScreenContext) -> float | None:
        """The value compared against the levels, or None when unknown."""

    def admits(self,
               asset_id: str,
               date: pd.Timestamp,
               held: bool,
               context: ScreenContext) -> bool:
        value = self.value(asset_id, date, context)

        if value is None:
            return self.on_missing

        level = self.exit if held and self.exit is not None else self.entry

        return value >= level


class MarketCapScreen(ThresholdScreen):
    """A minimum market cap, in the book's currency.

    Args:
        min_cap: The entry level.
        exit: The exit level for a held name. None uses *min_cap*.
        float_adjusted: Measure the free-float market cap instead.
        on_missing: Whether a name with no market cap passes.
    """

    def __init__(self,
                 min_cap: float,
                 exit: float | None = None,
                 float_adjusted: bool = False,
                 on_missing: bool = False):
        super().__init__(min_cap, exit, on_missing)
        self.float_adjusted = float_adjusted

    def value(self,
              asset_id: str,
              date: pd.Timestamp,
              context: ScreenContext) -> float | None:
        field = (data.market.free_float_market_cap if self.float_adjusted
                 else data.market.market_cap)
        cap = value_of(field, asset_id, date, context.fetcher,
                       currency=context.currency)

        return None if cap is None else float(cap)


class MinimumPriceScreen(ThresholdScreen):
    """A minimum price, in the book's currency.

    Args:
        min_price: The entry level.
        exit: The exit level for a held name. None uses *min_price*.
        on_missing: Whether a name with no recent price passes.
    """

    def __init__(self,
                 min_price: float,
                 exit: float | None = None,
                 on_missing: bool = False):
        super().__init__(min_price, exit, on_missing)

    def value(self,
              asset_id: str,
              date: pd.Timestamp,
              context: ScreenContext) -> float | None:
        start = date - pd.Timedelta(days=PRICE_LOOKBACK_DAYS)
        prices = _converted_closes(asset_id, start, date, context)

        return None if prices.empty else float(prices.iloc[-1])


class LiquidityScreen(ThresholdScreen):
    """A minimum average traded value (book currency) or volume (shares).

    Exactly one of *min_traded_value* and *min_volume* is given.

    Args:
        min_traded_value: The entry level for the average of close times
            volume, converted into the book's currency day by day.
        min_volume: The entry level for the average share volume.
        exit: The exit level for a held name, on the same measure.
        lookback_days: How many of the name's most recent trading days are
            averaged.
        on_missing: Whether a name with no volume passes.

    Raises:
        ValueError: Unless exactly one minimum is given.
    """

    def __init__(self,
                 min_traded_value: float | None = None,
                 min_volume: float | None = None,
                 exit: float | None = None,
                 lookback_days: int = DEFAULT_LIQUIDITY_DAYS,
                 on_missing: bool = False):
        if (min_traded_value is None) == (min_volume is None):
            raise ValueError("LiquidityScreen takes exactly one of "
                             "min_traded_value and min_volume.")

        entry = min_traded_value if min_traded_value is not None else min_volume
        assert entry is not None
        super().__init__(entry, exit, on_missing)
        self.by_value = min_traded_value is not None
        self.lookback_days = lookback_days

    def value(self,
              asset_id: str,
              date: pd.Timestamp,
              context: ScreenContext) -> float | None:
        # Calendar days enough to hold the trading days asked for.
        start = date - pd.Timedelta(days=int(self.lookback_days * 1.6) + 10)
        frame = context.fetcher.fetch_market_data(asset_id,
                                                  start.strftime("%Y-%m-%d"),
                                                  date.strftime("%Y-%m-%d"))

        if frame.empty or "VOLUME" not in frame.columns:
            return None

        volume = frame["VOLUME"].dropna().tail(self.lookback_days)

        if volume.empty:
            return None

        if not self.by_value:
            return float(volume.mean())

        closes = _converted_closes(asset_id, start, date, context)
        traded = (closes.reindex(volume.index) * volume).dropna()

        return None if traded.empty else float(traded.mean())


class ListingAgeScreen(Screen):
    """A minimum time since a name's first price in the data.

    Args:
        min_days: Calendar days since the first price.
    """

    def __init__(self,
                 min_days: int):
        self.min_days = min_days
        self._first: dict[str, pd.Timestamp | None] = {}

    def admits(self,
               asset_id: str,
               date: pd.Timestamp,
               held: bool,
               context: ScreenContext) -> bool:
        if asset_id not in self._first:
            frame = context.fetcher.fetch_market_data(asset_id)
            self._first[asset_id] = (None if frame.empty
                                     else pd.Timestamp(frame.index.min()))

        first = self._first[asset_id]

        return first is not None and (date - first).days >= self.min_days


class ExclusionScreen(Screen):
    """Names, sectors or regions a backtest may not hold.

    Sector and region are read from the reference record in force on the
    rebalance date, so a name that moved sector is judged by the sector it
    was in then.

    Args:
        identifiers: Names excluded outright.
        sectors: `SECTOR` values excluded.
        regions: `REGION` values excluded.
    """

    def __init__(self,
                 identifiers: Iterable[str] = (),
                 sectors: Iterable[str] = (),
                 regions: Iterable[str] = ()):
        self.identifiers = frozenset(identifiers)
        self.sectors = frozenset(sectors)
        self.regions = frozenset(regions)

    def admits(self,
               asset_id: str,
               date: pd.Timestamp,
               held: bool,
               context: ScreenContext) -> bool:
        if asset_id in self.identifiers:
            return False

        if not self.sectors and not self.regions:
            return True

        record = context.fetcher.fetch_reference_data(asset_id,
                                                      date.strftime("%Y-%m-%d"))

        if record.empty:
            return True

        return not (_any_in(record, "SECTOR", self.sectors)
                    or _any_in(record, "REGION", self.regions))


class ExpressionScreen(Screen):
    """Names that pass an expression, resolved at each rebalance.

    Money fields in the expression, such as `data.market.market_cap`, are in
    the book's currency.

    Args:
        expression: The screen.
        on_missing: Whether a name with no value for a field passes.
    """

    def __init__(self,
                 expression: Expression,
                 on_missing: bool = False):
        self.expression = expression
        self.on_missing = on_missing

    def prepare(self,
                fetcher: DataFetcher) -> None:
        """Refuse a field the data does not have, before the run starts.

        Raises:
            ExpressionError: Naming each field that does not resolve.
        """
        require_valid(self.expression, fetcher, "ExpressionScreen")

    def admits(self,
               asset_id: str,
               date: pd.Timestamp,
               held: bool,
               context: ScreenContext) -> bool:
        return resolve(self.expression, asset_id, date, context.fetcher,
                       on_missing=self.on_missing, currency=context.currency)


def _converted_closes(asset_id: str,
                      start: pd.Timestamp,
                      end: pd.Timestamp,
                      context: ScreenContext) -> pd.Series:
    """*asset_id*'s closes in the book's currency, blanks dropped."""
    prices = context.fetcher.fetch_prices([asset_id], start.strftime("%Y-%m-%d"),
                                          end.strftime("%Y-%m-%d"),
                                          currency=context.currency)

    if asset_id not in prices.columns:
        return pd.Series(dtype=float)

    return prices[asset_id].dropna()


def _any_in(record: pd.DataFrame,
            column: str,
            excluded: frozenset[str]) -> bool:
    """Whether the record's *column* holds any excluded value."""
    if not excluded or column not in record.columns:
        return False

    return bool(record[column].isin(excluded).any())
