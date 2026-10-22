# src/beacon/portfolio/history.py
"""
The portfolio's books over time: positions, cash and NAV, event by event.

The recorder is part of the Portfolio, composed rather than inherited, and
nothing outside the portfolio layer writes to one. Read the history through
`Portfolio.positions`, `Portfolio.cash` and `Portfolio.nav`.
"""
# Separated from `beacon.portfolio.base` only for size.
import logging
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Long-form, like market data: one row per asset per recorded date. Weight is
# stored, not derived on read — see `PortfolioHistory.record`.
POSITION_COLUMNS = ("DATE", "ASSET_ID", "QUANTITY", "PRICE", "MARKET_VALUE", "WEIGHT")


class MarkedHolding(Protocol):
    """The three fields the recorder reads off a holding.

    A structural type rather than an import of
    :class:`beacon.portfolio.base.Holding`, which would make the two modules
    import each other.
    """
    quantity: float
    current_price: float | None
    market_value: float | None


def _or_nan(value: float | None) -> float:
    """A missing mark as NaN, so the panel's columns stay numeric."""
    return float("nan") if value is None else value


@dataclass(frozen=True)
class _Snapshot:
    """One date's positions, column by column (BN-344): a dict per row cost
    more to build than the row was worth, on every event of a long run."""
    asset_ids: list[str]
    quantity: np.ndarray
    price: np.ndarray
    market_value: np.ndarray
    weight: np.ndarray


def _empty_positions() -> pd.DataFrame:
    """The positions panel for a portfolio that has never held anything.

    Empty but fully typed: a caller filtering or joining on `DATE` gets the
    same dtypes it would from a populated panel, instead of the object-dtype
    columns a bare `DataFrame()` would hand back.
    """
    return pd.DataFrame({
        "DATE": pd.Series(dtype="datetime64[ns]"),
        "ASSET_ID": pd.Series(dtype="object"),
        "QUANTITY": pd.Series(dtype="float64"),
        "PRICE": pd.Series(dtype="float64"),
        "MARKET_VALUE": pd.Series(dtype="float64"),
        "WEIGHT": pd.Series(dtype="float64"),
    })


class PortfolioHistory:
    """Records what the books said, on every event that changed them.

    Written as plain dicts during a run and converted to a DataFrame or Series
    on read. There is deliberately no `Position` class: the panel rests as a
    frame and lookups return frame slices.

    Keyed by date, and the **last** write for a date wins. A mark followed by
    a trade on the same day therefore leaves the post-trade state on record,
    which is what end-of-day books mean; the intermediate state is not
    something that survived the day.
    """
    def __init__(self) -> None:
        self._positions: dict[pd.Timestamp, _Snapshot] = {}
        self._cash: dict[pd.Timestamp, float] = {}
        self._nav: dict[pd.Timestamp, float] = {}

        # Built on read, discarded on write. The frame is the expensive part
        # of a long run, and callers read it far less often than the run
        # writes it.
        self._frame: pd.DataFrame | None = None

    def record(self,
               date: pd.Timestamp,
               holdings: Mapping[str, MarkedHolding],
               cash: float) -> None:
        """Write the current state of the books, dated *date*.

        The weight on each row is computed here and stored, rather than
        derived on read from the market value and the NAV. That is the point
        of the panel: it records what the books said at the time, so a later
        change to the derivation, or to the price data underneath, cannot
        silently restate history. A holding with no market value is recorded
        with weight 0.0 and left out of the NAV.

        Args:
            date: The date the state belongs to. Re-recording a date replaces
                what was there.
            holdings: The live holdings, read but not retained.
            cash: The cash balance to record beside them.
        """
        key = pd.Timestamp(date)

        # Summed here rather than through `Portfolio.get_total_value()`: the
        # holdings are being walked anyway, and that method warns once per
        # unpriced holding, which would become a warning per holding per
        # event.
        holdings_value = 0.0
        for holding in holdings.values():
            if holding.market_value is not None:
                holdings_value += holding.market_value

        total = holdings_value + cash

        count = len(holdings)
        values = list(holdings.values())
        marks = [holding.market_value for holding in values]
        unvalued = np.fromiter((mark is None for mark in marks), dtype=bool, count=count)
        market_value = np.fromiter((_or_nan(mark) for mark in marks),
                                   dtype="float64", count=count)

        # The same weight as before, element by element: the value over the
        # total, and 0.0 for a holding with no value or a total of zero.
        with np.errstate(divide="ignore", invalid="ignore"):
            weight = (np.where(unvalued, 0.0, market_value / total)
                      if total != 0 else np.zeros(count))

        self._positions[key] = _Snapshot(
            asset_ids=list(holdings),
            quantity=np.fromiter((float(holding.quantity) for holding in values),
                                 dtype="float64", count=count),
            price=np.fromiter((_or_nan(holding.current_price) for holding in values),
                              dtype="float64", count=count),
            market_value=market_value,
            weight=weight)
        self._cash[key] = cash
        self._nav[key] = total
        self._frame = None

    @property
    def positions(self) -> pd.DataFrame:
        """Every recorded position, long-form and sorted by date."""
        if self._frame is None:
            self._frame = self._build_positions()

        return self._frame

    @property
    def cash(self) -> pd.Series:
        """The cash balance on each recorded date."""
        return self._series(self._cash, "CASH")

    @property
    def nav(self) -> pd.Series:
        """The total value of the books on each recorded date."""
        return self._series(self._nav, "NAV")

    def _build_positions(self) -> pd.DataFrame:
        """Flatten the per-date snapshots into one frame, oldest date first."""
        snapshots = [(date, self._positions[date]) for date in sorted(self._positions)]

        if not any(snapshot.asset_ids for _, snapshot in snapshots):
            return _empty_positions()

        frame = pd.DataFrame({
            "DATE": pd.to_datetime([date for date, snapshot in snapshots
                                    for _ in snapshot.asset_ids]),
            "ASSET_ID": [name for _, snapshot in snapshots for name in snapshot.asset_ids],
            "QUANTITY": np.concatenate([snapshot.quantity for _, snapshot in snapshots]),
            "PRICE": np.concatenate([snapshot.price for _, snapshot in snapshots]),
            "MARKET_VALUE": np.concatenate([snapshot.market_value for _, snapshot in snapshots]),
            "WEIGHT": np.concatenate([snapshot.weight for _, snapshot in snapshots]),
        }, columns=list(POSITION_COLUMNS))

        return frame.reset_index(drop=True)

    def _series(self,
                points: dict[pd.Timestamp, float],
                name: str) -> pd.Series:
        """One of the scalar books as a date-indexed Series."""
        dates = sorted(points)

        return pd.Series([points[date] for date in dates],
                         index=pd.DatetimeIndex(dates, name="DATE"),
                         name=name,
                         dtype="float64")
