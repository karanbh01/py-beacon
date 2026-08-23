"""
DataFetcher — unified interface for querying market and reference data.

Accepts single identifiers or lists and passes through column names as-is.
"""


from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from .base import MarketData, ReferenceData
from .corporate_actions import CorporateActions
from .features import FeatureData

# The datasets a fetcher can report freshness for. Named here so the server
# and the fetcher cannot drift apart on the spelling.
MARKET_DATASET = "market"
REFERENCE_DATASET = "reference"
ACTIONS_DATASET = "corporate_actions"
FEATURES_DATASET = "features"
DATASETS = (MARKET_DATASET, REFERENCE_DATASET, ACTIONS_DATASET,
            FEATURES_DATASET)

# How often a dataset is expected to change. This is the engine's answer to
# what "stale" means, and it belongs here rather than in a client: a UI holding
# its own 24h/7d thresholds is guessing at a property of the data, and guesses
# diverge from the engine the moment either changes.
DAILY = "daily"
STATIC = "static"
EVENT = "event"

FREQUENCY_FOR_DATASET = {
    MARKET_DATASET: DAILY,
    # Names, sectors and listings change, but not on a schedule worth
    # refreshing against. Reference data that is a month old is not stale.
    REFERENCE_DATASET: STATIC,
    # Driven by announcements, not by a clock. A quiet week is not staleness.
    ACTIONS_DATASET: EVENT,
    # Also event-driven, and for the same reason the table exists: a
    # fundamental arrives when it is published, not on a cadence. A feature
    # table untouched for two months between reporting seasons is current,
    # not stale.
    FEATURES_DATASET: EVENT,
}

# Seconds after which a dataset of each frequency should be treated as stale.
# Published alongside the frequency so a client renders "stale" without
# encoding the mapping itself — which is the hardcoded threshold in another
# place. None means the question does not apply.
STALE_AFTER_SECONDS: dict[str, float | None] = {
    DAILY: 60 * 60 * 24,
    STATIC: None,
    EVENT: 60 * 60 * 24 * 7,
}

# The classification column read when none is named. Sector is the one every
# reference dataset carries and the one group constraints are usually built on.
DEFAULT_SCHEME = "SECTOR"

# Where instruments with no classification are collected, rather than dropped.
UNCLASSIFIED = "UNCLASSIFIED"


class DataFetcher:
    """Unified query interface over MarketData and ReferenceData.

    Args:
        market_data: Time-series market data container.
        reference_data: Reference data container.
        corporate_actions: Action history. Absent means an empty history
            rather than None, so callers never have to check before asking —
            "this instrument paid nothing" and "we hold no action data" give
            the same answer to every question this class can be asked.
    """

    def __init__(self,
                 market_data: MarketData,
                 reference_data: ReferenceData | None = None,
                 corporate_actions: CorporateActions | None = None,
                 features: FeatureData | None = None):
        self._market = market_data
        self._reference = reference_data
        self._actions = (corporate_actions if corporate_actions is not None
                         else CorporateActions.empty())
        # Empty rather than None, on the same terms as the actions above: a
        # dataset without features is still a dataset, and callers should be
        # able to ask it what it holds without checking for None first.
        self._features = (features if features is not None
                          else FeatureData.empty())

        # Loading is a refresh. Stamping construction rather than leaving this
        # empty is what makes an age meaningful from the first request: a
        # freshly started server holds data that is genuinely seconds old, and
        # reporting "unknown" until someone happens to sync would be less true,
        # not more careful.
        now = datetime.now(UTC)
        self._refreshed: dict[str, datetime | None] = {
            MARKET_DATASET: now,
            REFERENCE_DATASET: now if reference_data is not None else None,
            ACTIONS_DATASET: now if not self._actions.is_empty else None,
            FEATURES_DATASET: now if not self._features.is_empty else None,
        }

        # Where this data was loaded from, stamped by whatever built the
        # fetcher. None for one assembled in-process: saying "local" would
        # claim a provenance it does not have.
        self._source: str | None = None
        self._store_path: Path | None = None

    # -- properties ----------------------------------------------------------

    @property
    def identifiers(self) -> list[str]:
        """Unique identifiers present in market data."""
        return self._market.identifiers

    @property
    def market_columns(self) -> list[str]:
        """Column names in the market data."""
        return self._market.columns

    @property
    def reference_identifiers(self) -> list[str] | None:
        """Unique identifiers in the reference data, or None if not loaded."""
        if self._reference is None:
            return None

        return self._reference.identifiers

    @property
    def reference_columns(self) -> list[str] | None:
        """Column names in the reference data, or None if not loaded."""
        if self._reference is None:
            return None
        return self._reference.columns

    @property
    def date_range(self) -> tuple[pd.Timestamp, pd.Timestamp]:
        """(earliest, latest) timestamps in the market data."""
        return self._market.date_range

    @property
    def corporate_actions(self) -> CorporateActions:
        """The action history. Empty rather than None when none was loaded."""
        return self._actions

    @property
    def features(self) -> FeatureData:
        """The feature table. Empty rather than None when none was loaded.

        Exposed for persistence and for discovery, on the same terms as
        `market`. Point-in-time reads go through `fetch_features` (BN-135),
        not through this.
        """
        return self._features

    @property
    def market(self) -> MarketData:
        """The market-data container itself.

        Exposed for persistence (`beacon.data.store`): writing a fetcher to
        disk means reading back everything it holds, and the summarising
        properties above cannot reconstruct a frame. Query through
        ``fetch_market_data`` instead — this is the whole dataset, not an
        answer to a question.
        """
        return self._market

    @property
    def source(self) -> str | None:
        """Where this data was loaded from, or None if nothing recorded it.

        Describes the *load*, not every row: a later sync merges rows from
        somewhere else without changing where the store came from. Modelling
        mixed provenance would need a source per row, which nothing asks for.
        """
        return self._source

    @property
    def store_path(self) -> Path | None:
        """The store this was loaded from, if it came from one."""
        return self._store_path

    def record_origin(self,
                      source: str,
                      path: Path | None = None) -> None:
        """Note where this fetcher's data was loaded from."""
        self._source = source
        self._store_path = path

    def delisting_dates(self) -> dict[str, pd.Timestamp]:
        """The last date each identifier is listed, for those whose life ends.

        Resolved in one pass rather than per identifier per day: an index over
        five thousand names and ten years would otherwise make twelve million
        point-in-time lookups to find a few hundred delistings.

        A name is treated as still listed if *any* of its records is
        open-ended, which is checked before taking the maximum -- `max` over a
        column containing NaT would silently ignore the open record and retire
        a name that never left.

        Returns:
            dict: identifier -> last listed date. Names that never leave are
            absent, so an empty mapping means a constant universe and callers
            can skip the work entirely.
        """
        if self._reference is None:
            return {}

        frame = self._reference.data.reset_index()

        if not {"IDENTIFIER", "DATE_TO"} <= set(frame.columns):
            return {}

        ends: dict[str, pd.Timestamp] = {}

        for identifier, values in frame.groupby("IDENTIFIER")["DATE_TO"]:
            if values.isna().any():
                continue

            ends[str(identifier)] = pd.Timestamp(values.max())

        return ends

    @property
    def reference(self) -> ReferenceData | None:
        """The reference-data container, or None if none was loaded.

        Exposed for persistence, on the same terms as :attr:`market`.
        """
        return self._reference

    # -- freshness -----------------------------------------------------------

    def record_refresh(self,
                       dataset: str,
                       when: datetime | None = None) -> None:
        """Note that a dataset has just been refreshed.

        Args:
            dataset: MARKET_DATASET or REFERENCE_DATASET.
            when: The moment. None uses now, which is what a real sync wants;
                tests pass an explicit time so an age can be asserted rather
                than approximated.

        Raises:
            ValueError: If the dataset is not one this fetcher holds.
        """
        if dataset not in DATASETS:
            raise ValueError(
                f"unknown dataset '{dataset}'. Known: {', '.join(DATASETS)}.")

        self._refreshed[dataset] = when if when is not None else datetime.now(UTC)

    def last_refreshed(self,
                       dataset: str) -> datetime | None:
        """When a dataset was last loaded or synced.

        Returns:
            datetime or None: The moment, or None when the dataset is not
            loaded at all — which is a different statement from "loaded and
            never refreshed" and should not be collapsed into it.
        """
        if dataset not in DATASETS:
            raise ValueError(
                f"unknown dataset '{dataset}'. Known: {', '.join(DATASETS)}.")

        return self._refreshed[dataset]

    def age_seconds(self,
                    dataset: str,
                    now: datetime | None = None) -> float | None:
        """How long ago a dataset was last refreshed, in seconds.

        Args:
            dataset: Which dataset.
            now: The reference moment, for tests.

        Returns:
            float or None: The age, or None when the dataset is not loaded.
            Never negative: a clock adjustment between the two readings would
            otherwise report data refreshed in the future, which is noise
            rather than information.
        """
        stamped = self.last_refreshed(dataset)
        if stamped is None:
            return None

        elapsed = ((now if now is not None else datetime.now(UTC)) - stamped)

        return max(elapsed.total_seconds(), 0.0)

    # -- ingestion -----------------------------------------------------------

    def merge_market_data(self,
                          frame: pd.DataFrame) -> int:
        """Fold freshly ingested rows into the market data.

        Newly fetched rows win where they overlap an existing identifier and
        date. A re-sync of a window is a correction — a restated close, a
        backfilled volume — so keeping the older value would make the sync
        pointless.

        The swap at the end is a single assignment, so a reader either sees the
        whole old dataset or the whole new one. This process is single-threaded
        and cooperatively scheduled, so there is no torn state to guard
        against; a reader that started before the swap simply finishes against
        the data it began with.

        Args:
            frame: Long-form rows carrying ``IDENTIFIER`` and ``DATE``.

        Returns:
            int: Rows added, counting only genuinely new identifier/date pairs
            — a re-sync that restates existing rows returns 0, which is the
            truthful answer to "how much did this add".
        """
        if frame.empty:
            return 0

        existing = self._market.data.reset_index()
        combined = pd.concat([existing, frame], ignore_index=True)
        combined["DATE"] = pd.to_datetime(combined["DATE"])

        before = len(existing)
        combined = combined.drop_duplicates(subset=["IDENTIFIER", "DATE"],
                                            keep="last")

        self._market = MarketData.from_dataframe(combined)
        self.record_refresh(MARKET_DATASET)

        return len(combined) - before

    def merge_reference_data(self,
                             frame: pd.DataFrame) -> int:
        """Fold freshly ingested reference records in.

        Args:
            frame: Rows carrying ``IDENTIFIER`` and ``DATE_FROM``.

        Returns:
            int: Records added.
        """
        if frame.empty:
            return 0

        if self._reference is None:
            self._reference = ReferenceData.from_dataframe(frame)
            self.record_refresh(REFERENCE_DATASET)

            return len(frame)

        existing = self._reference.data.reset_index()
        combined = pd.concat([existing, frame], ignore_index=True)

        before = len(existing)
        combined = combined.drop_duplicates(subset=["IDENTIFIER", "DATE_FROM"],
                                            keep="last")

        self._reference = ReferenceData.from_dataframe(combined)
        self.record_refresh(REFERENCE_DATASET)

        return len(combined) - before

    # -- corporate actions ---------------------------------------------------

    def fetch_corporate_actions(self,
                                identifier: str,
                                start_date: str | pd.Timestamp | None = None,
                                end_date: str | pd.Timestamp | None = None,
                                types: list[str] | None = None) -> pd.DataFrame:
        """Corporate actions for one identifier over a window.

        Args:
            identifier: The instrument.
            start_date: Earliest ex-date, inclusive.
            end_date: Latest ex-date, inclusive.
            types: Restrict to these action types.

        Returns:
            pd.DataFrame: Matching actions, oldest first; empty when there are
            none.
        """
        return self._actions.get(identifier, start_date, end_date, types)

    def fetch_trailing_dividend(self,
                                identifier: str,
                                as_of: str | pd.Timestamp) -> float:
        """Ordinary dividends per share over the trailing twelve months."""
        return self._actions.trailing_dividend(identifier, as_of)

    def fetch_trailing_dividend_yield(self,
                                      identifier: str,
                                      as_of: str | pd.Timestamp,
                                      price: float | None = None) -> float | None:
        """Trailing dividend yield, priced off the market data by default.

        Args:
            identifier: The instrument.
            as_of: End of the trailing window.
            price: Price to divide by. None reads the close on or before
                *as_of* from the market data.

        Returns:
            float or None: The yield, or None when no price is available — a
            missing price is a reason to say nothing rather than to guess.
        """
        if price is None:
            price = self._close_on_or_before(identifier, as_of)

        if price is None or price <= 0.0:
            return None

        return self._actions.trailing_dividend_yield(identifier, as_of, price)

    def _close_on_or_before(self,
                            identifier: str,
                            as_of: str | pd.Timestamp) -> float | None:
        """The most recent close at or before *as_of*, or None."""
        frame = self._market.get(identifier, end_date=str(pd.Timestamp(as_of).date()))
        if frame.empty or "CLOSE" not in frame.columns:
            return None

        closes = frame["CLOSE"].dropna()

        return float(closes.iloc[-1]) if len(closes) else None

    # -- market data ---------------------------------------------------------

    def fetch_market_data(self,
                          identifier: str | list[str],
                          start_date: str | None = None,
                          end_date: str | None = None,
                          columns: list[str] | None = None) -> pd.DataFrame:
        """Fetch time-series market data for one or more identifiers.

        Args:
            identifier: One identifier or a list of identifiers.
            start_date: Date string to filter the start of the date range.
            end_date: Date string to filter the end of the date range.
            columns: Subset of columns to return.

        Returns:
            pd.DataFrame: Single identifier: indexed by ``DATE``. Multiple
            identifiers: MultiIndexed by ``(IDENTIFIER, DATE)``. Empty
            DataFrame if no matching data is found.
        """
        return self._market.get(identifier, start_date, end_date, columns)

    # -- auxiliary market data -----------------------------------------------
    #
    # Shares outstanding, free-float factors and FX rates are all sourced from
    # the market-data container, which is the single home for these series:
    #   * shares outstanding  -> a per-(identifier, date) column
    #     (``SHARES_OUTSTANDING`` by default)
    #   * free-float factor    -> a per-(identifier, date) column
    #     (``FREE_FLOAT`` by default)
    #   * FX rates             -> a currency pair stored as its own identifier,
    #     named ``"{FROM}{TO}"`` (e.g. ``GBPUSD``)
    # Each accessor returns ``None`` / an empty series when the backing column
    # or identifier is absent, so callers can fall back gracefully.

    def fetch_shares_outstanding(self,
                                 identifier: str,
                                 date: str,
                                 column: str = "SHARES_OUTSTANDING") -> float | None:
        """Return shares outstanding for *identifier* on *date*.

        Sourced from the *column* market-data field. Returns ``None`` if the
        column is not present or there is no value on that date.
        """
        return self._market_scalar(identifier, date, column)

    def fetch_free_float_factor(self,
                                identifier: str,
                                date: str,
                                column: str = "FREE_FLOAT") -> float | None:
        """Return the free-float factor for *identifier* on *date*.

        Sourced from the *column* market-data field. Returns ``None`` if the
        column is not present or there is no value on that date.
        """
        return self._market_scalar(identifier, date, column)

    def _market_scalar(self,
                       identifier: str,
                       date: str,
                       column: str) -> float | None:
        """Read a single market-data value for *identifier* on *date*."""
        if column not in self._market.columns:
            return None
        df = self._market.get(identifier, date, date, columns=[column])
        if df.empty:
            return None
        val = df[column].iloc[0]
        return float(val) if pd.notna(val) else None

    def fetch_fx_rates(self,
                       from_currency: str,
                       to_currency: str,
                       start_date: str | None = None,
                       end_date: str | None = None,
                       column: str = "RATE") -> pd.Series:
        """Return the FX rate series converting *from_currency* into *to_currency*.

        The pair is looked up as a market-data identifier named
        ``f"{from_currency}{to_currency}"`` (upper-cased). The *column* field is
        used if present, otherwise the first data column. Returns an empty
        Series if the pair is not found.
        """
        pair = f"{from_currency}{to_currency}".upper()
        if pair not in self._market.identifiers:
            return pd.Series(dtype=float)
        df = self._market.get(pair, start_date, end_date)
        if df.empty:
            return pd.Series(dtype=float)
        rate_col = column if column in df.columns else df.columns[0]
        return df[rate_col]

    # -- reference data ------------------------------------------------------

    def fetch_reference_data(self,
                             identifier: str | list[str],
                             date: str | None = None,
                             columns: list[str] | None = None) -> pd.DataFrame:
        """Fetch reference data for one or more identifiers.

        Args:
            identifier: One identifier or a list of identifiers.
            date: Point-in-time date. Only rows valid at this date are
                returned.
            columns: Subset of columns to return.

        Returns:
            pd.DataFrame: Indexed by ``IDENTIFIER``. Empty DataFrame if no
            reference data is loaded or identifier is not found.
        """
        if self._reference is None:
            return pd.DataFrame()

        return self._reference.get(identifier, date, columns)

    def fetch_classification(self,
                             identifier: str,
                             date: str | pd.Timestamp | None = None,
                             scheme: str = DEFAULT_SCHEME) -> str | None:
        """One instrument's classification as it stood on a date.

        Reference data already carries validity ranges, so a name that moved
        from Industrials to Technology has two rows and this returns whichever
        was in force. That matters for anything historical: attributing a 2021
        return to a sector the company only joined in 2023 is a real way to get
        a breakdown wrong.

        Args:
            identifier: The instrument.
            date: The as-of date. None takes the currently-active record — the
                one with no end date — falling back to the latest start date if
                every record has been closed off.
            scheme: Which column to read, e.g. ``"SECTOR"``, ``"INDUSTRY"``,
                ``"COUNTRY"``. Free-form, because which columns a client loads
                is its own business.

        Returns:
            str or None: The classification, or None when it is unknown: no
            reference data, no such instrument, no such column, or no record
            valid on that date.
        """
        if self._reference is None:
            return None

        frame = self._reference.get(identifier,
                                    str(date) if date is not None else None)
        if frame.empty or scheme not in frame.columns:
            return None

        if date is None:
            frame = self._current_record(frame)

        value = frame[scheme].iloc[0]

        return None if pd.isna(value) else str(value)

    @staticmethod
    def _current_record(frame: pd.DataFrame) -> pd.DataFrame:
        """The record in force today: open-ended, else the most recent."""
        open_ended = frame[frame["DATE_TO"].isna()]
        if not open_ended.empty:
            return open_ended

        return frame.sort_values("DATE_FROM").tail(1)

    def fetch_classifications(self,
                              identifiers: list[str],
                              date: str | pd.Timestamp | None = None,
                              scheme: str = DEFAULT_SCHEME) -> dict[str, str | None]:
        """Classifications for several instruments at once.

        Every identifier appears, with None where the classification is
        unknown, so a caller can see what is missing rather than finding it
        silently absent.

        Args:
            identifiers: The instruments.
            date: As-of date, as for :meth:`fetch_classification`.
            scheme: Which column to read.

        Returns:
            dict: Identifier to classification.
        """
        return {identifier: self.fetch_classification(identifier, date, scheme)
                for identifier in identifiers}

    def group_by_classification(self,
                                identifiers: list[str],
                                date: str | pd.Timestamp | None = None,
                                scheme: str = DEFAULT_SCHEME) -> dict[str, list[str]]:
        """Instruments grouped by classification, ready for GroupBounds.

        Unclassified instruments are collected under UNCLASSIFIED rather than
        dropped. A name missing from every bucket is how a constraint set
        quietly stops covering part of the universe.

        Args:
            identifiers: The instruments.
            date: As-of date.
            scheme: Which column to read.

        Returns:
            dict: Classification to the identifiers carrying it, each list in
            the order the identifiers were given.
        """
        grouped: dict[str, list[str]] = {}

        for identifier in identifiers:
            label = self.fetch_classification(identifier, date, scheme)
            grouped.setdefault(label or UNCLASSIFIED, []).append(identifier)

        return grouped
