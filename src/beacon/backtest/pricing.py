# src/beacon/backtest/pricing.py
"""
PricingMixin — what a price is, to the engine: a date, a session, a currency.

Split out of `engine.py` (BN-183), which had grown past the size this project
keeps a module to once the exact-date read became a resolved one. The grouping
is not arbitrary: everything here answers one question — *what is this name
worth, in the book's currency, on this day* — and the engine above it decides
what to do with the answer.

Two conversions sit between a stored bar and a usable price, and both used to
be missing:

* **Currency.** Prices are stored as the company is quoted; a portfolio has
  one currency (BN-128).
* **Date.** A requested date is not always a session. Resolving it is BN-179's
  rule applied to the engine — backfill within the data, refuse beyond it —
  and BN-180's required calendar is what lets a closed market be told apart
  from a hole in the data, which are the same absence with opposite meanings.
"""
import logging

import pandas as pd

from ..data.fetcher import DataFetcher
from ..exceptions import CalculationError
from ..index.result import IndexResult
from ..index.schedule import is_session
from .result import PriceGap

logger = logging.getLogger(__name__)


class PricingMixin:
    """Price resolution and currency conversion, mixed into BacktestEngine."""

    # Provided by the BacktestEngine that mixes this in.
    data_provider: DataFetcher
    index_result: IndexResult
    price_column: str
    currency: str
    calendar: str | None
    _currencies: dict[str, str]
    _last_bars: dict[str, tuple[pd.Timestamp, float]]
    _price_gaps: list[PriceGap]
    _gaps_seen: set[tuple[str, pd.Timestamp]]
    _delistings: dict[str, pd.Timestamp]

    def _fetch_price(self,
                     asset_id: str,
                     date: pd.Timestamp) -> float | None:
        """One closing price for *asset_id* on *date*, in the book's currency.

        The conversion is the point. Prices are stored as the company is
        quoted -- yen in Tokyo, sterling in London -- while a portfolio has a
        single currency, and `IndexCalculator` has always converted its market
        values before comparing them. Returning the raw close here made the
        engine value a 300 yen share as 300 dollars: every non-domestic weight
        was wrong by its exchange rate, and against the single-currency
        universe that existed until BN-128 the error was invisible because
        every rate was 1.0.

        ## A missing bar is no longer simply a missing price (BN-183)

        This read used to ask for one exact date and answer None for anything
        else, which made a rebalance scheduled on a closed day trade nothing:
        both instruction builders return None without a price, so the record
        kept a rebalance dated to a day on which nothing moved, and attribution
        and the weights history then read those snapshots. The date resolves to
        a session instead, exactly as BN-179 made the weighting scheme resolve
        one -- and past the data's coverage it refuses in the same words,
        because there a carried price would answer a different question in this
        one's date.

        Which session, and whether anyone is told, is what BN-180's required
        calendar finally allows this to distinguish. A day the calendar says
        was **closed** resolves back silently: the market was shut, and the
        portfolio genuinely held the position through it. A day it says was
        **open** with no bar is a hole in the data; the price is carried
        forward -- a stale quote beats killing a five-hundred-name backtest
        over one bad day -- and recorded on the result as a
        :class:`~beacon.backtest.result.PriceGap`, so it is published rather
        than absorbed.

        Raises:
            CalculationError: If *date* falls outside the data's coverage
                entirely and the name had no bar on it.
        """
        raw = self._close_on(asset_id, date)

        if raw is not None:
            self._last_bars[asset_id] = (date, raw)

            return raw * self._rate_for(asset_id, date)

        # A name whose listing has ended has no price rather than a stale one.
        # Its holding is settled by `_dispose_delisted` from reference data,
        # which is a different fact from a market being shut.
        last_listed = self._delistings.get(asset_id)

        if last_listed is not None and date > last_listed:
            return None

        session = self._resolved_session(date)

        if session is None:
            return None

        bar = self._last_bar(asset_id, date)

        if bar is None:
            return None

        bar_date, bar_price = bar

        if self._market_was_open(date, session):
            self._record_gap(asset_id, date, bar_date)

        return bar_price * self._rate_for(asset_id, date)

    def _close_on(self,
                  asset_id: str,
                  date: pd.Timestamp) -> float | None:
        """The unconverted close *asset_id* printed on *date*, or None.

        The exact-date read, kept exact: resolution is the caller's decision,
        and a helper that quietly looked back would put it out of reach.

        Through `fetch_price` since BN-212, which consults the session panel;
        `fetch_market_data` does not, and this is the engine's hottest read --
        once per holding per day, 171,800 times over a 200-name three-year run,
        each one slicing the whole market frame to return a single row.
        """
        date_str = date.strftime("%Y-%m-%d")

        try:
            return self.data_provider.fetch_price(asset_id, date_str,
                                                  self.price_column)
        except Exception as error:
            logger.error(f"Error fetching price for {asset_id} on {date_str}: {error}")

            return None

    def _last_close(self,
                    frame: pd.DataFrame) -> tuple[pd.Timestamp | None, float | None]:
        """The final usable close in a date-indexed frame, and its date."""
        if not isinstance(frame, pd.DataFrame) or frame.empty:
            return None, None

        if self.price_column not in frame.columns:
            return None, None

        usable = frame[self.price_column].dropna()

        if usable.empty:
            return None, None

        stamp = usable.index[-1]

        return (pd.Timestamp(stamp) if isinstance(stamp, pd.Timestamp) else None,
                float(usable.iloc[-1]))

    def _last_bar(self,
                  asset_id: str,
                  date: pd.Timestamp) -> tuple[pd.Timestamp, float] | None:
        """The most recent bar *asset_id* printed on or before *date*.

        Incremental rather than a fresh history scan each time: the window
        starts at the last bar already known for the name, so a long run of
        missing days costs one short read apiece instead of one read of
        everything ever printed.

        A cached bar *later* than the date asked about is the one case the
        shortcut cannot take -- the run walks forward, so this means somebody
        asked about the past, and answering from a later close would be a
        look-ahead price. That scans from the beginning and leaves the cache
        where the run put it.
        """
        known = self._last_bars.get(asset_id)

        if known is not None and known[0] > date:
            return self._scan_back(asset_id, self._data_start(), date)

        start = known[0] if known is not None else self._data_start()
        found = self._scan_back(asset_id, start, date)

        if found is None:
            return known

        self._last_bars[asset_id] = found

        return found

    def _scan_back(self,
                   asset_id: str,
                   start: pd.Timestamp | None,
                   date: pd.Timestamp) -> tuple[pd.Timestamp, float] | None:
        """The last bar *asset_id* printed in [*start*, *date*], or None."""
        if start is None or start > date:
            return None

        try:
            frame = self.data_provider.fetch_market_data(asset_id,
                                                         start.strftime("%Y-%m-%d"),
                                                         date.strftime("%Y-%m-%d"))
        except Exception as error:
            logger.error("Could not read %s back to %s: %s", asset_id, start, error)

            return None

        bar_date, price = self._last_close(frame)

        if bar_date is None or price is None:
            return None

        return bar_date, price

    def _data_start(self) -> pd.Timestamp | None:
        """The first date the market data covers, when it can say."""
        try:
            first, _ = self.data_provider.date_range
        except Exception:
            return None

        return first if isinstance(first, pd.Timestamp) and pd.notna(first) else None

    def _resolved_session(self,
                          date: pd.Timestamp) -> pd.Timestamp | None:
        """The session *date* reads from, or a refusal past the data (BN-179).

        The shared primitive, reached defensively because the provider is an
        interface rather than a class: a stand-in that cannot resolve a session
        answers None here and leaves the read where it was, while a real
        fetcher saying None means the date is outside the data on one side or
        the other, and that refuses.

        Raises:
            CalculationError: If *date* lies outside the data's coverage.
        """
        resolver = getattr(self.data_provider, "resolve_session", None)

        if resolver is None:
            return None

        session = resolver(date)

        if isinstance(session, pd.Timestamp):
            return session

        if session is not None:
            return None

        first, last = self.data_provider.date_range

        raise CalculationError(
            calculation_name=f"backtest of '{self.index_result.index_id}'",
            details=(f"cannot price at {date:%Y-%m-%d}: the market data runs "
                     f"{first:%Y-%m-%d} to {last:%Y-%m-%d}, so that date lies "
                     f"outside it. Inside the range a date with no bar is a "
                     f"closed market and resolves back to the last session on "
                     f"or before it; outside it nothing is known, and carrying "
                     f"a price forward would answer a different question in "
                     f"this one's date. Ask for a date on or before "
                     f"{last:%Y-%m-%d}, or refresh the store."))

    def _market_was_open(self,
                         date: pd.Timestamp,
                         session: pd.Timestamp) -> bool:
        """Whether *date* was a day the market was supposed to trade.

        The calendar answers it properly. Without one — an engine assembled by
        hand rather than by a definition — the data's own sessions stand in:
        a date the store resolves to itself is one other names printed on, so
        this name's silence on it is the name's, not the market's.
        """
        if self.calendar is not None:
            return is_session(date, self.calendar)

        return bool(session == date)

    def _record_gap(self,
                    asset_id: str,
                    date: pd.Timestamp,
                    priced_from: pd.Timestamp) -> None:
        """Note that *asset_id* was marked at a carried price on *date*."""
        key = (asset_id, date)

        if key in self._gaps_seen:
            return

        self._gaps_seen.add(key)
        self._price_gaps.append(PriceGap(date=date, asset_id=asset_id,
                                         priced_from=priced_from))

        logger.warning(
            "[%s] %s has no bar on a session the calendar says was open; "
            "marked at its %s close.", date.date(), asset_id, priced_from.date())

    def _rate_for(self,
                  asset_id: str,
                  date: pd.Timestamp) -> float:
        """FX from an asset's listing currency into the book's, on *date*.

        Resolving *which* currencies are involved is the engine's own business
        — it knows the book's currency and the asset's listing currency — and
        the rate lookup is not. Since BN-206 that half delegates to
        `DataFetcher.fx_rate_on`, which is the library's one conversion.

        This method used to carry its own copy: its own per-pair cache, its
        own date walk, and its own answers to the two questions that matter.
        Both answers were wrong. An unknown pair returned **1.0**, valuing a
        foreign holding at parity and putting the NAV out by the whole
        exchange rate with a WARNING as the only record (BN-205) — the
        substitution BN-188 deleted everywhere it looked, in the one folder it
        did not look in. A date before the series returned the series' *first*
        rate, which is look-ahead (BN-204).

        Caching moved with the lookup: `fx_rate_on` holds one series per
        ordered pair on the fetcher, so a per-day rate is still seven slices
        for a global universe rather than one per holding per day, and there
        is now one cache rather than two that can disagree.

        Using a single fixed rate instead would be worse than it sounds. A
        constant scale factor cancels out of the weight arithmetic entirely --
        the engine sizes a position by value, so `quantity x price x rate` is
        the target value whatever the rate is -- and the conversion would look
        correct while changing nothing. It is the *drift* in the rate that a
        foreign holding actually experiences, and that only appears if the
        rate moves. That cancellation is also why neither defect above showed
        up in a NAV-level test: a one-name book comes out identical whether
        the rate is right, wrong, or invented.

        Raises:
            CalculationError: If no rate exists on or before *date*. A book
                that cannot convert a holding cannot state its NAV in its own
                currency, and saying so is the answer -- the same argument
                BN-198 settled for a calendar that cannot cover a window.
        """
        currency = self._currency_of(asset_id)

        if currency is None or currency == self.currency:
            return 1.0

        rate = self.data_provider.fx_rate_on(currency, self.currency, date)

        if rate is not None:
            return rate

        raise CalculationError(
            calculation_name="BacktestEngine",
            details=(f"no {currency.upper()}/{self.currency.upper()} rate on "
                     f"or before {date:%Y-%m-%d}, so {asset_id} cannot be "
                     f"valued in the book's currency. Valuing it unconverted "
                     f"— the old answer — carries a {currency.upper()} holding "
                     f"as though it were {self.currency.upper()}, so the NAV "
                     f"is wrong by the whole exchange rate and nothing in the "
                     f"result looks odd. Load the pair, or run the backtest "
                     f"in {currency.upper()}."))

    def _currency_of(self,
                     asset_id: str) -> str | None:
        """The currency an identifier is quoted in, from reference data."""
        if asset_id in self._currencies:
            return self._currencies[asset_id]

        resolved: str | None = None

        try:
            frame = self.data_provider.fetch_reference_data(asset_id)

            if not frame.empty and "CURRENCY" in frame.columns:
                value = frame["CURRENCY"].iloc[0]

                if pd.notna(value):
                    resolved = str(value).upper()
        except Exception as error:
            logger.error("Could not resolve the currency of %s: %s",
                         asset_id, error)

        self._currencies[asset_id] = resolved or self.currency

        return self._currencies[asset_id]
