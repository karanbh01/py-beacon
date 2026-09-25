"""
`DataFetcher`: one query interface over market, reference, corporate-action
and feature data.

Accepts single identifiers or lists and passes through column names as-is.
"""


import logging
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from . import fx
from .base import MarketData, ReferenceData, as_of_position
from .corporate_actions import CorporateActions
from .features import MAX_AGE_DAYS, FeatureData
from .free_float import (
    DEFAULT_FREE_FLOAT_BACKFILL_DAYS,
    carried_forward,
    validated_window,
)
from .session import SessionPanel

logger = logging.getLogger(__name__)

# The datasets a fetcher can report freshness for. Named here so the server
# and the fetcher cannot drift apart on the spelling.
# The column an FX pair carries its rate in, and the marker that makes a
# market identifier a pair rather than an instrument.
RATE_COLUMN = "RATE"

MARKET_DATASET = "market"
REFERENCE_DATASET = "reference"
ACTIONS_DATASET = "corporate_actions"
FEATURES_DATASET = "features"
# Reported as a dataset of its own, though the rows live in the market frame:
# "do we hold exchange rates" is a question with its own answer, and a client
# deciding whether to offer an unhedged comparison needs it (BN-144).
FX_DATASET = "fx"

# Everything the freshness and coverage machinery knows about. FX is in here
# although it has no store file of its own -- its rows live in the market
# frame -- because it is reported and stamped like the rest.
DATASETS = (MARKET_DATASET, REFERENCE_DATASET, ACTIONS_DATASET,
            FEATURES_DATASET, FX_DATASET)

# The subset that is persisted as its own file. `store.save` walks this, so
# adding FX to DATASETS above must not make the store look for an fx file
# that will never exist.
STORED_DATASETS = (MARKET_DATASET, REFERENCE_DATASET, ACTIONS_DATASET,
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
    # Rates move daily, like the market rows they are stored beside.
    FX_DATASET: DAILY,
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

# How an FX rate is read on a day the pair did not print one (BN-207).
#
# A modelling assumption rather than an implementation detail, which is why it
# is selectable and why it lives in one place. The two answers are both
# defensible and they disagree about real money:
#
# CARRY_FORWARD treats a rate as in force until the next one is published, so
# a day with no bar is valued at the last rate that existed. That is what a
# market participant would actually have transacted at, and it is what four of
# the five conversion sites did before they were merged.
#
# EXACT_DAY refuses unless the pair printed a rate on that very date. That is
# the stricter reading and the right one when a conversion stands for a real
# cash event on a specific day -- a dividend, most obviously -- where an
# approximate rate is worse than being told the rate is unknown. It is what
# the corporate-action path did, alone and silently, before this was a setting.
#
# Neither is a default the library gets to choose on a user's behalf without
# saying so, hence `fx_policy` on the fetcher and the value published on
# `/health`. A number converted under one assumption is a different number
# under the other, and nothing in a result looks odd either way.
FX_CARRY_FORWARD = "CARRY_FORWARD"
FX_EXACT_DAY = "EXACT_DAY"
FX_POLICIES = (FX_CARRY_FORWARD, FX_EXACT_DAY)

# What an installation converts at when nobody chooses. Carry-forward, because
# it is what every path but one already did, so adopting the setting changes no
# existing number until somebody asks it to.
DEFAULT_FX_POLICY = FX_CARRY_FORWARD

# How far back the cheap first stage of a batch "last known bar" read reaches.
# Nearly every name is answered from it; the ones that are not are read again
# without a lower bound, which is rare and is the whole point of the split.
RECENT_DAYS = 30

# The classification column read when none is named. Sector is the one every
# reference dataset carries and the one group constraints are usually built on.
DEFAULT_SCHEME = "SECTOR"

# Where instruments with no classification are collected, rather than dropped.
UNCLASSIFIED = "UNCLASSIFIED"



def _last_row_each(frame: pd.DataFrame) -> pd.DataFrame:
    """Reduce a multi-identifier frame to one row per name: its latest.

    Done once with a grouped tail rather than by slicing per name downstream.
    That slicing is what made an unbounded read 32x slower -- `xs` over a
    1.37-million-row frame is cheap once and ruinous five hundred times, and
    only one row of each name's history is ever wanted.
    """
    if frame.empty or not isinstance(frame.index, pd.MultiIndex):
        return frame

    return frame.sort_index().groupby(level="IDENTIFIER", sort=False).tail(1)


def _identifiers_in(frame: pd.DataFrame) -> set[str]:
    """Which names a multi-identifier frame actually carries rows for."""
    if frame.empty or not isinstance(frame.index, pd.MultiIndex):
        return set()

    return {str(name) for name in frame.index.get_level_values("IDENTIFIER")}


class DataFetcher:
    """One query interface over market, reference, corporate-action and
    feature data.

    Args:
        market_data: Time-series market data container. FX pairs live in it
            as their own identifiers, named ``"{FROM}{TO}"``, with the rate in
            a ``RATE`` column.
        reference_data: Reference data container, or None if there is none.
        corporate_actions: Action history. Absent means an empty history
            rather than None, so callers never have to check before asking:
            "this instrument paid nothing" and "we hold no action data" give
            the same answer to every question this class can be asked.
        features: Feature table. Absent means an empty table, on the same
            terms.
        fx_policy: How an FX rate is read on a day the pair did not print
            one. ``"CARRY_FORWARD"`` (the default) uses the last rate
            published; ``"EXACT_DAY"`` answers only when the pair printed a
            rate on that very day. It applies to every conversion made
            through this fetcher.
        max_price_staleness_days: How many calendar days a name may go
            without trading before `stale_identifiers` reports it as not worth
            holding. None (the default) keeps every name, whenever it last
            traded.
        free_float_backfill_days: How many calendar days a reported free
            float carries forward over blank cells (default 90; 0 turns
            carrying off). See `beacon.data.free_float`.

    Raises:
        ValueError: If `fx_policy` is not a known policy,
            `max_price_staleness_days` is below 1, or
            `free_float_backfill_days` is negative or not an integer.
    """

    def __init__(self,
                 market_data: MarketData,
                 reference_data: ReferenceData | None = None,
                 corporate_actions: CorporateActions | None = None,
                 features: FeatureData | None = None,
                 fx_policy: str = DEFAULT_FX_POLICY,
                 max_price_staleness_days: int | None = None,
                 free_float_backfill_days: int = DEFAULT_FREE_FLOAT_BACKFILL_DAYS):
        if fx_policy not in FX_POLICIES:
            raise ValueError(
                f"Unknown fx_policy: {fx_policy!r}. "
                f"Supported values: {list(FX_POLICIES)}.")

        # One assumption for every conversion in the library (BN-207). Held
        # here rather than passed per call because it is a property of how this
        # dataset is being read, not of any one question asked of it -- and
        # because a per-call default is how five conversion sites came to
        # disagree in the first place.
        self.fx_policy = fx_policy

        if (max_price_staleness_days is not None
                and max_price_staleness_days < 1):
            raise ValueError(
                f"max_price_staleness_days must be at least 1 day, got "
                f"{max_price_staleness_days!r}. Pass None to keep every name "
                f"regardless of when it last traded.")

        # How long a name may go without trading before it stops being worth
        # holding (BN-211). None keeps everything, which is what this library
        # always did -- so adopting the setting changes no index and no
        # backtest until somebody asks it to.
        #
        # A modelling choice rather than a data property, and global rather
        # than per-index on Karan's call: one answer for the whole
        # installation, reaching index construction and backtests alike.
        self.max_price_staleness_days = max_price_staleness_days

        # How many days a free float carries forward over blank cells
        # (BN-219). The third global setting, on the same terms as the two
        # above: it changes numbers, so it is chosen once and published. See
        # `beacon.data.free_float` for why 90 and why there is no unlimited.
        self.free_float_backfill_days = validated_window(free_float_backfill_days)

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
            # The pairs are market rows, so they are as fresh as the market
            # data is. A separate stamp would drift from it for no reason.
            FX_DATASET: now,
        }

        # Where this data was loaded from, stamped by whatever built the
        # fetcher. None for one assembled in-process: saying "local" would
        # claim a provenance it does not have.
        self._source: str | None = None
        self._store_path: Path | None = None

        # Rate series, one per ordered pair, behind `fx_rate_on`. A run
        # converts every foreign name on every day and each fetch slices the
        # whole market frame, so an uncached lookup made an eighty-name global
        # index take longer than the rest of the suite put together. Cleared
        # whenever the market data underneath it is replaced.
        self._fx_series: dict[tuple[str, str], pd.Series] = {}
        # How each cached pair was found: direct, inverse, or a cross (BN-235).
        self._fx_routes: dict[tuple[str, str], str | None] = {}
        # Each name's observed free floats, blanks dropped, read once when a
        # blank day first needs carrying over (BN-219). Cleared with the FX
        # series on a merge, for the same reason.
        self._free_float_history: dict[tuple[str, str], pd.Series] = {}

        # The session a methodology is currently walking a universe over, read
        # in one slice. One panel rather than a growing map of them: the reads
        # that repeat are the ones inside a single rebalance -- a selection
        # rule prices every name and the weighting then prices the survivors
        # again -- and they are over at the moment the date moves on (BN-190).
        # Cleared whenever the market data underneath it is replaced.
        self._session_panel: SessionPanel | None = None

    # -- properties ----------------------------------------------------------

    @property
    def identifiers(self) -> list[str]:
        """Unique identifiers present in market data."""
        return self._market.identifiers

    @property
    def fx_pairs(self) -> list[str]:
        """Currency pairs held in the market data.

        A pair is stored as an ordinary market identifier named
        ``f"{from}{to}"``, so nothing in the frame separates it from an
        instrument except what it carries: `RATE` is populated on a pair and
        null on everything else. That is the discriminator, rather than a
        name pattern: an instrument legitimately called `EURUSD` would be
        misfiled by a six-letter rule, and a store may hold pairs for
        currencies its reference data never mentions.
        """
        # Pair storage as a market identifier is BN-128.
        if RATE_COLUMN not in self._market.columns:
            return []

        rates = self._market.data[RATE_COLUMN]
        present = rates.notna().groupby(level="IDENTIFIER").any()

        return sorted(present[present].index)

    @property
    def instrument_identifiers(self) -> list[str]:
        """Market identifiers that are instruments rather than currency pairs.

        What a universe or a search should offer: a pair is a rate series, not
        something anybody holds.
        """
        pairs = set(self.fx_pairs)

        return [name for name in self.identifiers if name not in pairs]

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

    def resolve_session(self,
                        date: str | pd.Timestamp) -> pd.Timestamp | None:
        """The market session *date* resolves to, backfilling inside the data.

        A date the data has no bar for but which sits inside its coverage is a
        day the market was shut. The last session on or before it is the one
        that was actually in force through the closure: reading it is not an
        approximation, it is what the day was. Past the last bar nothing is
        known, so that answers None rather than a stale print wearing a
        current date; the bound is the data's own coverage rather than a day
        count, because no day count can tell a long closure from the unknown
        future.

        Args:
            date: The date asked about.

        Returns:
            pd.Timestamp | None: The session, or None when *date* falls
            outside the data's coverage on either side.
        """
        as_of = pd.Timestamp(date)
        first, last = self.date_range

        if pd.isna(first) or as_of < first or as_of > last:
            return None

        return self._market.last_session_on_or_before(as_of)

    @property
    def corporate_actions(self) -> CorporateActions:
        """The action history. Empty rather than None when none was loaded."""
        return self._actions

    def fetch_feature(self,
                      identifier: str,
                      field: str,
                      date: str | pd.Timestamp | None = None,
                      feature_type: str | None = None,
                      max_age_days: int | None = MAX_AGE_DAYS) -> float | None:
        """One feature value, as it was knowable on a date.

        The point-in-time read. A value published after `date` is invisible,
        which is what keeps a backtest from screening on numbers nobody had.

        Returns:
            float | None: The value, or None when nothing is knowable. An
            instrument with no coverage is an ordinary answer, not an error:
            most datasets cover most names most of the time and not all of
            them all of it.
        """
        return self._features.value_as_of(identifier, field, date,
                                          feature_type, max_age_days)

    def fetch_features(self,
                       identifiers: list[str],
                       fields: list[str],
                       date: str | pd.Timestamp | None = None,
                       feature_type: str | None = None,
                       max_age_days: int | None = MAX_AGE_DAYS
                       ) -> dict[str, dict[str, float | None]]:
        """Several features for several instruments, on one date.

        The batch form, on the same argument the reference batch endpoint
        made: a client that has to fan out per name will, and moving the
        fan-out inside the server only relocates the cost.

        Returns:
            dict: identifier -> field -> value. Every requested pair is
            present, null where nothing is knowable, so a caller reads a value
            rather than testing for a key.
        """
        return {identifier: {field: self.fetch_feature(identifier, field,
                                                       date, feature_type,
                                                       max_age_days)
                             for field in fields}
                for identifier in identifiers}

    def replace_features(self,
                         features: FeatureData) -> None:
        """Swap the feature table, stamping the refresh.

        The only mutating method on a fetcher, and it exists because an import
        has to land somewhere the next request can see. It replaces rather
        than edits: `merged_with` builds the new table, so a read in flight
        keeps the frame it started with instead of watching rows appear under
        it.
        """
        self._features = features
        self._refreshed[FEATURES_DATASET] = datetime.now(UTC)

    def feature_types(self) -> list[str]:
        """Datasets the loaded features carry, for discovery."""
        return self._features.types

    def feature_fields(self,
                       feature_type: str | None = None) -> list[str]:
        """Fields the loaded features carry, optionally within one dataset."""
        return self._features.fields(feature_type)

    @property
    def features(self) -> FeatureData:
        """The feature table. Empty rather than None when none was loaded.

        Exposed for persistence and for discovery, on the same terms as
        `market`. Point-in-time reads go through `fetch_feature` and
        `fetch_features`, not through this.
        """
        # The point-in-time accessors are BN-135.
        return self._features

    @property
    def market(self) -> MarketData:
        """The market-data container itself.

        Exposed for persistence (`beacon.data.store`): writing a fetcher to
        disk means reading back everything it holds, and the summarising
        properties cannot reconstruct a frame. Query through
        ``fetch_market_data`` instead: this is the whole dataset, not an
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
        open-ended, even when another record for it has an end date.

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
            dataset: One of `DATASETS`: ``"market"``, ``"reference"``,
                ``"corporate_actions"``, ``"features"`` or ``"fx"``.
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
            loaded at all, which is a different statement from "loaded and
            never refreshed" and should not be collapsed into it. Loading
            counts as a refresh.

        Raises:
            ValueError: If the dataset is not one of `DATASETS`.
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
        date. A re-sync of a window is a correction (a restated close, a
        backfilled volume), so keeping the older value would make the sync
        pointless.

        The swap at the end is a single assignment, so a reader either sees the
        whole old dataset or the whole new one. This process is single-threaded
        and cooperatively scheduled, so there is no torn state to guard
        against; a reader that started before the swap simply finishes against
        the data it began with.

        Args:
            frame: Long-form rows carrying ``IDENTIFIER`` and ``DATE``.

        Returns:
            int: Rows added, counting only genuinely new identifier/date
            pairs. A re-sync that restates existing rows returns 0, which is
            the truthful answer to "how much did this add".
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
        # The pairs are market rows, so a merge can add or restate them; a
        # cache held over the swap would answer out of the old frame. The
        # session panel is the same story one day wide.
        self._fx_series.clear()
        self._fx_routes.clear()
        self._free_float_history.clear()
        self._session_panel = None
        self.record_refresh(MARKET_DATASET)

        return len(combined) - before

    def merge_reference_data(self,
                             frame: pd.DataFrame) -> int:
        """Fold freshly ingested reference records in.

        A record matching an existing identifier and ``DATE_FROM`` replaces
        it.

        Args:
            frame: Rows carrying ``IDENTIFIER`` and ``DATE_FROM``.

        Returns:
            int: Records added, counting only genuinely new identifier and
            ``DATE_FROM`` pairs.
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
            float or None: The yield, or None when no positive price is
            available: a missing price is a reason to say nothing rather than
            to guess.
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

    def warm_session(self,
                     identifiers: list[str],
                     date: str | pd.Timestamp) -> None:
        """Read one session's rows for *identifiers* in a single slice.

        A hint, not a contract: every read this serves answers identically
        without it, only slower. Without it, a methodology walking a universe
        makes one frame slice per name per column, each costing what the
        whole frame costs rather than what one row does, so a preview's cost
        per name climbs with the size of its universe.

        A selection rule prices every candidate and the weighting scheme then
        prices the survivors, the same names on the same day. The second warm
        is a subset of the first, so it keeps the panel rather than
        rebuilding it, and the reads that follow are free.

        Nothing goes stale under it: the panel answers only for the identifiers
        it was built with, only on its own session, and a merge clears it.
        Calling this with a different session or a name it does not hold
        replaces it, so the caller never has to say when it is done.

        Args:
            identifiers: The instruments about to be read one at a time.
            date: The session they will be read on. Resolve it first with
                :meth:`resolve_session`, since a panel for a closed day holds
                nothing and every read would fall back to the frame.
        """
        # BN-190.
        session = pd.Timestamp(date)
        held = self._session_panel

        if held is not None and held.session == session and held.covers(identifiers):
            return

        # Straight from the column index, with no DataFrame built on the way
        # (BN-213). `get` filtered the whole frame to find one day's rows,
        # which is 13.5 ms against 0.017.
        values, present = self._market.session_columns(session)

        self._session_panel = SessionPanel.from_columns(session, values, present)

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
        """Return the free-float factor in force for *identifier* on *date*.

        That day's value when there is one. Otherwise the last value before
        it, if no older than `free_float_backfill_days`: free float moves on
        corporate events and reviews, so a blank cell means nothing was
        reported, not that the float changed. Never a later value.

        Returns ``None`` if the column is absent, or nothing was reported
        within the window. Callers refuse through
        :func:`~beacon.data.free_float.require_free_float` rather than
        choosing a fallback of their own.
        """
        today = self._market_scalar(identifier, date, column)

        if today is not None or self.free_float_backfill_days == 0:
            return today

        return carried_forward(self._free_float_series(identifier, column),
                               pd.Timestamp(date),
                               self.free_float_backfill_days)

    def _free_float_series(self,
                           identifier: str,
                           column: str) -> pd.Series:
        """One name's reported free floats, blanks dropped, sorted by date."""
        key = (identifier, column)

        if key not in self._free_float_history:
            if column not in self._market.columns:
                history = pd.Series(dtype=float)
            else:
                frame = self._market.get(identifier, None, None, columns=[column])
                history = (frame[column].dropna().sort_index()
                           if not frame.empty else pd.Series(dtype=float))

            self._free_float_history[key] = history

        return self._free_float_history[key]

    def prices_on(self,
                  identifiers: list[str],
                  date: str,
                  column: str = "CLOSE") -> dict[str, float | None]:
        """:meth:`fetch_price` for many names on one day, in one read.

        The batch face of the same answer: warm the day's session once, take
        the whole column from it, and convert NaN to None, exactly what
        `fetch_price` returns name by name. The daily valuation asks this for
        every holding every day.

        A name the session panel does not answer for goes through
        `fetch_price` rather than being assumed absent, so a panel that
        somehow missed it costs a slower read instead of a wrong one.

        Returns:
            dict: identifier -> price, or None where there is none.
        """
        # BN-218. Before it, every name paid for a chain of four calls to
        # reach a dict lookup, 200 times a day.
        # Through `getattr`, like every other use of the hint: warming is an
        # optimisation a provider may lack, and a batch read must still answer
        # without it -- one name at a time, as `fetch_price` always did.
        warm = getattr(self, "warm_session", None)

        if callable(warm):
            warm(identifiers, date)

        panel = self._session_panel

        if panel is None or panel.stamp != date:
            return {name: self.fetch_price(name, date, column)
                    for name in identifiers}

        stored = panel.column(column)
        prices: dict[str, float | None] = {}

        for name in identifiers:
            if not panel.answers(name, date):
                prices[name] = self.fetch_price(name, date, column)
                continue

            value = stored.get(name)
            prices[name] = None if value is None or pd.isna(value) else float(value)  # type: ignore[arg-type]

        return prices

    def fetch_price(self,
                    identifier: str,
                    date: str,
                    column: str = "CLOSE") -> float | None:
        """Return *identifier*'s price on *date*, or None if it did not print.

        The scalar form of the single-day, single-name fetch a methodology
        makes for every name in a universe. It reads the same value
        ``fetch_market_data(identifier, date, date)`` does (the same column of
        the same row) and returns it rather than a one-row frame to slice,
        which is what lets a warmed session serve it. Returns None too when
        the column is absent from the market data.
        """
        # BN-190.
        return self._market_scalar(identifier, date, column)

    def _market_scalar(self,
                       identifier: str,
                       date: str,
                       column: str) -> float | None:
        """Read a single market-data value for *identifier* on *date*."""
        # Only when the warmed panel is for this exact session and genuinely
        # holds this name. Anything else -- another date, a name outside the
        # set it was built for -- goes to the data, because a panel that
        # answered beyond what it was built from would be guessing.
        panel = self._session_panel

        if panel is not None and panel.answers(identifier, date):
            # No column check first: the panel answers None for a column it
            # does not hold, which is the same answer the check gave (BN-215).
            # Checking anyway was 358,000 membership tests a run to protect
            # the 200 reads that miss the panel -- 99.94% of them asked a
            # question whose answer this line already knew.
            return panel.value(identifier, column)

        # Here the check is load-bearing. `get` selects columns strictly and
        # raises `KeyError` for one the store lacks, and optional columns --
        # FREE_FLOAT, SHARES_OUTSTANDING -- are allowed to be absent. That
        # `KeyError` is exactly how #215 became a bare 500.
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
        """Return the stored FX rate series converting *from_currency* into *to_currency*.

        The pair is looked up as a market-data identifier named
        ``f"{from_currency}{to_currency}"`` (upper-cased). The *column* field is
        used if present, otherwise the first data column. Returns an empty
        Series if that exact pair is not stored: this does not invert or cross
        rates. :meth:`fx_rate_on` and :meth:`fx_rates_on` do.
        """
        pair = f"{from_currency}{to_currency}".upper()
        if pair not in self._market.identifiers:
            return pd.Series(dtype=float)
        df = self._market.get(pair, start_date, end_date)
        if df.empty:
            return pd.Series(dtype=float)
        rate_col = column if column in df.columns else df.columns[0]
        return df[rate_col]

    def fx_route(self,
                 from_currency: str,
                 to_currency: str) -> str | None:
        """How a rate for this pair is found: direct, inverse or a cross.

        Returns:
            str | None: "direct" for a stored pair, "inverse" for one over the
            stored reverse pair, "cross via USD" for a rate built from two
            legs, "same currency" when no conversion is needed, and None when
            no rate can be found.
        """
        if from_currency.upper() == to_currency.upper():
            return "same currency"

        pair = (from_currency.upper(), to_currency.upper())
        self._rate_series(*pair)

        return self._fx_routes[pair]

    def _rate_series(self,
                     source: str,
                     target: str) -> pd.Series:
        """The rate series for a pair, found once and cached (BN-235).

        The one place a pair is resolved, for both :meth:`fx_rate_on` and
        :meth:`fx_rates_on`: the stored pair, else its inverse, else a cross
        through USD. See `beacon.data.fx`.
        """
        pair = (source, target)

        if pair not in self._fx_series:
            series, route = fx.rate_series(
                lambda a, b: self.fetch_fx_rates(a, b).sort_index(),
                source, target, exact_day=self.fx_policy == FX_EXACT_DAY)

            self._fx_series[pair] = series
            self._fx_routes[pair] = route

            # Once per pair, not per day: a derived rate is correct but worth
            # knowing about, since it is not a quote anyone published.
            if route not in (None, fx.DIRECT):
                logger.info("No %s%s pair is stored; converting %s to %s by "
                            "%s.", source, target, source, target, route)

        return self._fx_series[pair]

    def fx_rate_on(self,
                   from_currency: str,
                   to_currency: str,
                   date: str | pd.Timestamp) -> float | None:
        """The rate converting *from_currency* into *to_currency* on *date*.

        The single currency conversion in the library, so the number
        displayed and the number weighted by are the same quantity. The pair
        is found as stored, else as the inverse of the stored reverse pair,
        else as a cross through USD (see `beacon.data.fx` and
        :meth:`fx_route`).

        The series is fetched once per ordered pair and cached, because a run
        asks this on every foreign name on every day and each fetch slices the
        whole market frame.

        Args:
            from_currency: The currency being converted out of.
            to_currency: The currency being converted into.
            date: The date the rate is wanted on.

        Returns:
            float | None: The rate in force on *date*, or None when the pair
            is unknown **or when its history begins after *date***. Under the
            default ``CARRY_FORWARD`` policy the last rate on or before *date*
            is used; under ``EXACT_DAY`` only a rate printed on *date* itself,
            else None. Callers treat None as "cannot convert" rather than as a
            rate of one. Nothing here invents parity on a caller's behalf: a
            rate of 1.0 is a claim about two currencies, and the only one this
            makes is that a currency converts into itself.

            Carried **forward** only, never backward: a rate dated after the
            day it would be applied to is look-ahead. As with
            `resolve_session`, there is no earlier observation to be in force,
            so there is no answer rather than a substitute for one.
        """
        # BN-188 made this the single conversion. There were three, and they
        # disagreed: the index calculator carried a rate forward and refused
        # when the pair was unknown, the reference endpoint substituted 1.0
        # and reported the local number under a dollar heading, and the
        # market-cap weighting did not convert at all, which is how a yen name
        # came to carry fifteen times the weight it should. One lookup means
        # the number displayed and the number weighted by are the same
        # quantity, which is the half of this that nothing was checking.
        #
        # BN-204: a date before the series starts used to answer with the
        # series' first rate, a rate dated after the day it was applied to.
        if from_currency.upper() == to_currency.upper():
            return 1.0

        pair = (from_currency.upper(), to_currency.upper())

        series = self._rate_series(*pair)

        if series.empty:
            return None

        stamp = pd.Timestamp(date)

        if self.fx_policy == FX_EXACT_DAY:
            # No carry at all: the rate must have printed on this very day.
            # `.get` rather than a search, because "the rate for this date" is
            # a lookup under this policy rather than a question about ordering.
            value = series.get(stamp)

            return None if value is None or pd.isna(value) else float(value)

        # Through `as_of_position` rather than a local search (BN-208): it
        # returns None where the raw form returns -1, and -1 is a legal pandas
        # index meaning the *last* element. That collision is what put a March
        # rate on a January valuation here, twice.
        position = as_of_position(series.index, stamp)

        return None if position is None else float(series.iloc[position])

    def latest_rows(self,
                    identifiers: list[str],
                    as_of: pd.Timestamp,
                    columns: list[str] | None = None,
                    recent_days: int = RECENT_DAYS) -> pd.DataFrame:
        """One row per name: its most recent bar at or before *as_of*.

        The batch "what is the last thing we know about these names" read,
        shared by the reference endpoint and the staleness gate.

        Read in two stages: the last *recent_days* first, which answers
        almost every name, then only the names it missed again without a
        lower bound. The result is reduced to one row per name once, with a
        grouped tail, rather than sliced per name downstream.

        Args:
            identifiers: Names to look up.
            as_of: The date to look back from, inclusive.
            columns: Columns to read, or None for all of them.
            recent_days: How far back the cheap first stage reaches.

        Returns:
            pd.DataFrame: MultiIndexed as the market data is, holding at most
            one row per identifier. Names with no bar at or before *as_of* are
            absent rather than present-and-empty.
        """
        # BN-211. Two stages because the obvious version is thirty times
        # slower. Measured over 500 names and ten years of daily bars: reading
        # the recent window costs 650 ms and reading the whole history costs
        # 20.6 seconds. The fetch is not what differs (identifier selection
        # dominates it either way); it is that every per-name slice afterwards
        # then cuts a 1.37-million-row frame. The single grouped tail is what
        # makes even an all-stale store cheap: 782 ms against 19.8 seconds.
        end_str = pd.Timestamp(as_of).strftime("%Y-%m-%d")
        recent = (pd.Timestamp(as_of)
                  - pd.DateOffset(days=recent_days)).strftime("%Y-%m-%d")

        frame = _last_row_each(
            self.fetch_market_data(identifiers, recent, end_str, columns))
        seen = _identifiers_in(frame)
        missing = [name for name in identifiers if name not in seen]

        if not missing:
            return frame

        older = _last_row_each(
            self.fetch_market_data(missing, None, end_str, columns))

        if older.empty:
            return frame

        if frame.empty:
            return older

        return pd.concat([frame, older]).sort_index()

    def last_priced_on(self,
                       identifiers: list[str],
                       as_of: pd.Timestamp) -> dict[str, pd.Timestamp]:
        """When each name last printed a bar at or before *as_of*.

        Args:
            identifiers: Names to look up.
            as_of: The date to look back from.

        Returns:
            dict: identifier -> the date of its last bar. A name with no bar
            at all is absent from the mapping, which is a different thing from
            one whose last bar is old.
        """
        frame = self.latest_rows(identifiers, as_of)

        if frame.empty or not isinstance(frame.index, pd.MultiIndex):
            return {}

        names = frame.index.get_level_values("IDENTIFIER")
        dates = frame.index.get_level_values("DATE")

        return {str(name): pd.Timestamp(date)
                for name, date in zip(names, dates, strict=True)}

    def stale_identifiers(self,
                          identifiers: list[str],
                          as_of: pd.Timestamp) -> set[str]:
        """Which names have not traded recently enough to be worth holding.

        Empty when no threshold is set, which is the default: staleness is
        something an installation opts into with `max_price_staleness_days`,
        and until it does this costs one comparison and reads nothing.

        A name with **no** price at all is not reported here. That is a
        different condition with a different remedy (the weighting already
        refuses it by name), and folding the two together would quietly
        excuse a missing instrument as a quiet one.

        Args:
            identifiers: Names to test.
            as_of: The date staleness is measured from.

        Returns:
            set: Identifiers whose last bar is more than
            `max_price_staleness_days` calendar days before *as_of*.
        """
        # BN-211.
        if self.max_price_staleness_days is None:
            return set()

        stamp = pd.Timestamp(as_of)
        priced = self.last_priced_on(identifiers, stamp)

        return {name for name, date in priced.items()
                if (stamp - date).days > self.max_price_staleness_days}

    def fx_rates_on(self,
                    from_currency: str,
                    to_currency: str,
                    days: pd.Index) -> pd.Series | None:
        """:meth:`fx_rate_on` over many days at once, as a Series.

        The vectorised face of the same rule, for a caller that needs a rate
        for every day of a run rather than one date, such as chained levels.
        Asking per day would be one search per day per currency where a
        single reindex answers the lot. The policy, the carry semantics, the
        routes and the meaning of "no rate" are the same as
        :meth:`fx_rate_on`; the two methods differ only in how many answers
        they return.

        Args:
            from_currency: The currency being converted out of.
            to_currency: The currency being converted into.
            days: The dates wanted, ascending.

        Returns:
            pd.Series | None: One rate per day, indexed by *days*, or None when
            the pair is unknown entirely. Individual days the policy cannot
            answer for are NaN: a day is missing, not the pair.
        """
        # BN-207. Two implementations of the lookup is how the library came to
        # have five of them, so the rule lives here once. The NaN-for-a-day
        # versus None-for-the-pair distinction is one `_rate_series` in
        # chaining depends on.
        if from_currency.upper() == to_currency.upper():
            return pd.Series(1.0, index=days)

        pair = (from_currency.upper(), to_currency.upper())

        series = self._rate_series(*pair)

        if series.empty:
            return None

        if self.fx_policy == FX_EXACT_DAY:
            return series.astype(float).reindex(days)

        # `ffill` is the carry, and it leaves NaN before the first rate rather
        # than back-filling it — which is the vectorised statement of BN-204:
        # forward over a gap, never backward into one.
        return series.astype(float).reindex(days, method="ffill")

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
            date: The as-of date. None takes the currently-active record (the
                one with no end date), falling back to the latest start date if
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
