"""
PriceGapsMixin: a held name with no bar is valued at its last close.
"""
import logging

import pandas as pd

from ...data.fetcher import DataFetcher
from ..result import PriceGap

logger = logging.getLogger(__name__)


class PriceGapsMixin:
    """Carry a missing bar forward and record it, mixed into IndexCalculator."""

    # Provided by the IndexCalculator that mixes this in.
    data: DataFetcher
    price_column: str
    _last_bars: dict[str, tuple[pd.Timestamp, float]]
    _price_gaps: dict[tuple[str, pd.Timestamp], PriceGap]

    def remember_bars(self,
                      prices: dict[str, float | None],
                      date: pd.Timestamp) -> None:
        """Note today's closes, so a gap tomorrow can be valued without a read."""
        for ticker, price in prices.items():
            if price is not None:
                self._last_bars[ticker] = (date, price)

    def carried_price(self,
                      ticker: str,
                      date: pd.Timestamp) -> float | None:
        """*ticker*'s last close before *date*, recorded as a price gap.

        The run walks forward and remembers every close it values, so this is
        usually a lookup. A name with no remembered close, or one remembered
        from after *date* (a caller asking about the past), is read back from
        the data instead.

        Returns:
            float | None: The unconverted close, or None when the name has
            never printed a bar on or before *date*.
        """
        # BN-251: this used to be 0.0, so the level dropped by the name's
        # weight for the day and recovered when the bar returned, while the
        # backtest engine carried the price and the two disagreed.
        bar = self._last_bars.get(ticker)

        if bar is None or bar[0] >= date:
            bar = self._read_back(ticker, date)

        if bar is None:
            return None

        priced_from, price = bar
        key = (ticker, date)

        if key not in self._price_gaps:
            self._price_gaps[key] = PriceGap(date=date, asset_id=ticker,
                                             priced_from=priced_from)
            logger.warning("[%s] %s has no bar on a session the calendar says "
                           "was open; valued at its %s close.",
                           date.date(), ticker, priced_from.date())

        return price

    def recorded_gaps(self) -> list[PriceGap]:
        """Every gap recorded so far, in date order."""
        return sorted(self._price_gaps.values(),
                      key=lambda gap: (gap.date, gap.asset_id))

    def _read_back(self,
                   ticker: str,
                   date: pd.Timestamp) -> tuple[pd.Timestamp, float] | None:
        """The last bar *ticker* printed strictly before *date*, or None."""
        before = (date - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        frame = self.data.fetch_market_data(ticker, None, before)

        if not isinstance(frame, pd.DataFrame) or self.price_column not in frame:
            return None

        closes = frame[self.price_column].dropna()

        if closes.empty:
            return None

        return pd.Timestamp(closes.index[-1]), float(closes.iloc[-1])
