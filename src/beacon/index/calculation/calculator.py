# src/beacon/index/calculation/calculator.py
"""
Module for the IndexCalculator, responsible for the logic of
constituent selection, weighting, index level calculation, and corporate action adjustments.
"""
import logging
from typing import Any

import pandas as pd

from ...asset.base import Asset
from ...asset.equity import Equity
from ...data.fetcher import DataFetcher
from ...exceptions import CalculationError, UnexpectedCalculationError
from ..capping import CapReport, apply_cap
from ..constructor import IndexDefinition
from ..context import IndexContext
from ..requirements import require_columns
from ..result import IndexResult, daily_weights_frame
from ..schedule import calendar_coverage, describe_bounds, effective_date, sessions
from .corporate_actions import CorporateActionsMixin
from .deletions import DeletionMixin
from .market_values import MarketValuesMixin
from .selection import (
    UNIVERSE_POSITION,
    SelectionResult,
    SelectionStep,
    select_with_provenance,
)
from .total_return import REINVESTING, TotalReturnMixin, withholding_for

logger = logging.getLogger(__name__)


def weight_rows(date: pd.Timestamp,
                units: dict[Asset, float],
                values: dict[Asset, float]) -> list[dict[str, object]]:
    """One record per constituent for one day: what was held, and its share.

    The weights are *realised* shares of the day's aggregate — value over
    total — so they drift with prices between rebalances, and they renormalise
    the moment a name is deleted. That is why the panel is recorded here
    rather than derived later from the rebalance snapshot: the two only agree
    on the rebalance date itself.

    Plain dicts appended during the run and converted once at the end, the
    pattern the backtest engine already uses for its own records — there is no
    per-day object to build and throw away.

    Args:
        date: The calculation day.
        units: What the index holds after the day's events.
        values: Those holdings valued for *date*, from
            :meth:`~.market_values.MarketValuesMixin.holding_values`.

    Returns:
        list: Records keyed by :data:`~beacon.index.result.DAILY_WEIGHT_COLUMNS`.
        A day whose holdings are worth nothing at all records nothing, because
        it has no weights to record — the level is carried forward on such a
        day, and a row of zeros would read as "held nothing" rather than
        "could not be valued".
    """
    total = sum(values.values())

    if total <= 0.0:
        return []

    return [{"DATE": date,
             "IDENTIFIER": asset.asset_id,
             "AMOUNT": units[asset],
             "WEIGHT": values[asset] / total}
            for asset in units]


class IndexCalculator(MarketValuesMixin, DeletionMixin,
                      TotalReturnMixin, CorporateActionsMixin):
    """
    Stateless index calculator. Accepts an IndexDefinition and DataFetcher,
    and provides methods for constituent selection, weighting, index level
    calculation, and corporate action adjustments. All state is passed
    through method parameters and return values.
    """
    def __init__(self,
                 index_definition: IndexDefinition,
                 data_provider: DataFetcher,
                 price_column: str = "CLOSE"):
        """
        Initializes the IndexCalculator.

        Args:
            index_definition: The IndexDefinition object that specifies the index rules.
            data_provider: A DataFetcher instance to access market and asset data.
            price_column: Market-data column read as the constituent price when
                computing market values. Defaults to ``"CLOSE"``.
        """
        if not index_definition:
            raise ValueError("index_definition must be provided.")
        if not data_provider:
            raise ValueError("data_provider must be provided.")

        self.definition: IndexDefinition = index_definition
        self.data: DataFetcher = data_provider
        self.price_column: str = price_column

        # What the methodology gets to know about the index it is running
        # inside. Rules and schemes have always taken this argument and it was
        # never supplied, so a market-cap weighting could not tell which
        # currency to compare its caps in and added them as though every
        # currency's unit were the same size (BN-188).
        self.context: IndexContext = IndexContext(currency=self.definition.currency)

        logger.info(f"IndexCalculator initialized for index '{self.definition.index_name}'.")


    def _reference_rows(self,
                        identifiers: list[str],
                        date_str: str) -> dict[str, dict[str, Any]]:
        """The reference record in force on *date_str*, keyed by identifier.

        One read for the whole universe rather than one per name (BN-192).
        ``fetch_reference_data`` has always taken a list, and the point-in-time
        filter it applies is per row — ``DATE_FROM <= date`` and ``DATE_TO``
        open or on/after it — so a batch answers each name with the record that
        was valid for *that* name on that date, exactly as the per-name read
        did. Nothing here narrows by position.

        The join back is by identifier, for the reason BN-190 batched the
        market reads that way: a frame holding many names in one block is one
        ordering assumption away from attaching one company's currency to
        another, and since BN-188 a currency decides an FX conversion, so a
        misattributed one is a wrong weight rather than a cosmetic error.

        Args:
            identifiers: Every name the definition asks for.
            date_str: The point-in-time date, ``YYYY-MM-DD``.

        Returns:
            dict: identifier -> its row as a plain mapping. A name the
            reference data does not know is simply absent, which is what the
            caller reports. Where a name carries several valid records the
            first is kept, which is the record the per-name read's ``iloc[0]``
            took.
        """
        frame = self.data.fetch_reference_data(list(identifiers), date_str)

        if frame.empty:
            return {}

        rows: dict[str, dict[str, Any]] = {}

        # One pass over the frame rather than a slice per name: `iloc` per row
        # costs what a frame read costs, which is the shape being removed.
        for label, row in zip(frame.index, frame.to_dict("records"),
                              strict=True):
            rows.setdefault(str(label), row)

        return rows

    def _get_universe(self,
                      date: pd.Timestamp) -> list[Asset]:
        """Resolve universe_identifiers from the IndexDefinition into Asset objects.

        Reads ``self.data.fetch_reference_data`` **once** for the whole
        universe and constructs an :class:`Equity` per identifier from that
        name's own row (BN-192). An identifier the reference data does not know
        is skipped with a warning; a lookup that *fails* is left to fail
        (BN-184).

        Args:
            date: Point-in-time date for reference data lookup.

        Returns:
            A list of Asset objects corresponding to resolvable identifiers,
            in the order the definition names them.

        Raises:
            CalculationError: If the definition names no universe at all. This
                used to return an empty universe, which calculates an index
                over nothing and publishes it — a definition missing its
                universe and one whose names all fell away were spelled the
                same way (BN-184).
        """
        identifiers = self.definition.universe_identifiers
        if identifiers is None:
            raise CalculationError(
                calculation_name="UniverseResolution",
                details=(f"index '{self.definition.index_name}' has no "
                         f"universe_identifiers, so there is nothing to select "
                         f"constituents from. An index cannot be calculated "
                         f"over an unspecified universe."))

        assets: list[Asset] = []
        date_str = date.strftime('%Y-%m-%d')
        rows = self._reference_rows(identifiers, date_str)

        for identifier in identifiers:
            row = rows.get(identifier)

            # BN-184, triaged as report-not-refuse: a universe naming a name
            # the reference data has never heard of is ordinary, and skipping
            # it is the right arithmetic. What is missing is that the result
            # says nothing about having computed over fewer names than the
            # definition asked for. The count below reaches a log and nobody
            # else; carrying it on IndexResult is the open half of #197.
            #
            # Batching the read (BN-192) moved where this is decided, not what
            # it decides: a name absent from the batch is a name the read
            # returned nothing for, which is the empty frame the per-name read
            # produced.
            if row is None:
                logger.warning(
                    f"_get_universe: No reference data for '{identifier}' on "
                    f"{date_str}. Skipping.")
                continue

            asset = Equity(
                name=row.get("NAME", identifier),
                currency=row.get("CURRENCY", self.definition.currency),
                ticker=identifier,
                exchange=row.get("EXCHANGE", "UNKNOWN"),
            )
            assets.append(asset)

        logger.info(
            f"_get_universe: Resolved {len(assets)}/{len(identifiers)} identifiers "
            f"for '{self.definition.index_name}' on {date_str}."
        )
        return assets

    def resolve_universe(self,
                         date: pd.Timestamp) -> list[Asset]:
        """Resolve the definition's universe identifiers into Asset objects.

        The public entry point for universe resolution, for callers outside
        the calculation loop — the constituent preview, for one. Delegates to
        the internal implementation, so anything that stubs that also governs
        this.

        Args:
            date: Point-in-time date for the reference-data lookup.

        Returns:
            list[Asset]: Assets for every identifier that resolved.
        """
        return self._get_universe(date)

    def select_constituents(self,
                            universe: list[Asset],
                            current_date: pd.Timestamp) -> list[Asset]:
        """
        Selects index constituents from a given universe based on eligibility rules.

        A thin projection of :meth:`select_with_provenance`: the survivors, with
        the record of which rule removed each excluded name discarded. Callers
        wanting that record — the preview waterfall, anything answering "why is
        this name missing" — should use the fuller method rather than repeating
        the walk, which is what BN-102 existed to stop.

        Args:
            universe: A list of potential Asset objects to consider for inclusion.
            current_date: The date for which selection is being made.

        Returns:
            A list of Asset objects that are eligible for the index.
        """
        return self.select_with_provenance(universe, current_date).survivors

    def select_with_provenance(self,
                               universe: list[Asset],
                               current_date: pd.Timestamp) -> SelectionResult:
        """Select constituents, keeping the record of how the universe narrowed.

        Args:
            universe: A list of potential Asset objects to consider for inclusion.
            current_date: The date for which selection is being made.

        Returns:
            SelectionResult: Survivors, one step per rule, and the position of
            the rule that excluded each removed asset.
        """
        logger.info(
            f"[{current_date.strftime('%Y-%m-%d')}] Selecting constituents for "
            f"'{self.definition.index_name}'. Universe size: {len(universe)}")

        # BN-184, triaged as leave. No universe means no survivors, which is
        # the arithmetic rather than a stand-in for it, and the provenance
        # record says so explicitly: a universe step of zero remaining. The
        # condition is not lost downstream either — `run` refuses a base date
        # with no constituents outright, in `_require_a_base_composition`.
        if not universe:
            logger.warning("Constituent selection called with an empty universe.")

            return SelectionResult(survivors=[],
                                   steps=[SelectionStep(position=UNIVERSE_POSITION,
                                                        remaining=0)])

        result = select_with_provenance(universe,
                                        self.definition.eligibility_rules,
                                        current_date,
                                        self.data,
                                        self.context)

        logger.info(
            f"Selected {len(result.survivors)} constituents for "
            f"'{self.definition.index_name}'.")

        return result

    def calculate_constituent_weights(self,
                                      constituents: list[Asset],
                                      current_date: pd.Timestamp) -> dict[Asset, float]:
        """
        Calculates the weights for the given constituents based on the index's weighting scheme.

        Args:
            constituents: A list of Asset objects that are part of the index.
            current_date: The date for which weights are calculated.

        Returns:
            A dictionary mapping each Asset to its float weight. Sum of weights should be 1.0.

        Raises:
            CalculationError: If the scheme refuses — an unpriced constituent,
                an unknown share count, a market cap of zero — in which case it
                propagates exactly as the scheme raised it, remedy and all
                (BN-196). Also if its weights do not sum to 1: that used to be
                silently renormalised with a warning, and a scheme's own output
                rescaled is the scheme not being applied, which is the BN-179
                argument exactly (BN-184).
            UnexpectedCalculationError: If the scheme raises anything else. A
                crash, not a decision, and it carries its own published code so
                a client does not read it as a refusal (BN-194).
        """
        date_str = current_date.strftime('%Y-%m-%d')
        if not constituents:
            logger.warning(
                f"[{date_str}] Calculating weights for an empty list of "
                f"constituents for '{self.definition.index_name}'.")
            return {}

        logger.info(
            f"[{date_str}] Calculating weights for {len(constituents)} "
            f"constituents of '{self.definition.index_name}'.")

        try:
            weights = self.definition.weighting_scheme.calculate_weights(
                constituents, current_date, self.data, self.context
            )

        # A scheme's own refusal, passed through as it was raised (BN-196).
        # BN-194 wrapped this on the premise that a guard in *this* file refuses
        # deliberately while a scheme only ever faults — which BN-179, BN-188
        # and BN-191 had already made false: every substitution they removed
        # became a `CalculationError` raised from inside a scheme, each naming
        # its remedy. Wrapping them relabelled the whole class as a crash, so
        # "unpriced constituent" published as "the engine broke" — the exact
        # inversion BN-194 set out to fix. What reaches the `except` below is
        # what that premise actually described: an exception the scheme never
        # meant to raise.
        except CalculationError:
            raise

        except Exception as e:
            logger.error(
                f"Error applying weighting scheme "
                f"{self.definition.weighting_scheme.scheme_name}: {e}")

            # The `WeightingScheme-` prefix survives as a *name* only — nothing
            # may branch on it, which is why the class differs.
            raise UnexpectedCalculationError(
                calculation_name=f"WeightingScheme-{self.definition.weighting_scheme.scheme_name}",
                cause=e) from e

        # A scheme that does not return weights summing to 1 has not produced
        # the weighting it names. Rescaling them here published a *different*
        # allocation under the scheme's name and said so only in a log, which
        # is the same substitution BN-179 removed from the market-cap path.
        # Raised outside the try above so it reaches the caller rather than
        # being re-wrapped as a scheme failure.
        weight_sum = sum(weights.values())

        if weights and abs(weight_sum - 1.0) > 1e-9:
            raise CalculationError(
                calculation_name=(f"WeightingScheme-"
                                  f"{self.definition.weighting_scheme.scheme_name}"),
                details=(f"the {len(weights)} weights it returned for "
                         f"{date_str} sum to {weight_sum!r}, not 1. Rescaling "
                         f"them would publish an allocation the scheme did "
                         f"not produce under the scheme's own name."))

        logger.info(f"Weights calculated for '{self.definition.index_name}'.")
        return weights

    def cap_weights(self,
                    weights: dict[Asset, float]) -> tuple[dict[Asset, float], CapReport]:
        """Apply the definition's cap, returning the weights and a report.

        Capping happens here rather than inside a weighting scheme so that it
        composes with every scheme, and it returns its report rather than
        storing one so the calculator stays stateless and `run()` stays
        idempotent.

        Args:
            weights: Normalised weights keyed by Asset.

        Returns:
            tuple: The capped weights and a CapReport. With no cap configured
            the weights are returned unchanged and the report is empty.
        """
        cap = self.definition.max_constituent_weight
        if cap is None or not weights:
            return weights, CapReport(cap=cap)

        # apply_cap works on identifiers so its report can name constituents
        # without depending on the asset classes.
        by_id = {asset.asset_id: weight for asset, weight in weights.items()}
        capped, report = apply_cap(by_id, cap)

        return {asset: capped[asset.asset_id] for asset in weights}, report

    def _require_a_base_composition(self,
                                    resolved: list[Asset],
                                    constituents: list[Asset],
                                    base_date: pd.Timestamp) -> None:
        """Refuse a base date on which the index would hold nothing.

        `initialize_divisor` refuses this too, one step later, by way of a
        zero market value — but it can only say the aggregate was zero, which
        is the symptom. Here the cause is still in hand: how many identifiers
        the definition named, how many the data resolved, and how many
        survived the rules. BN-178's discipline is that a refusal names what
        to fix, and an empty index is almost always a universe that did not
        resolve rather than a market genuinely worth nothing.

        Raises:
            CalculationError: If *constituents* is empty.
        """
        if constituents:
            return

        named = self.definition.universe_identifiers or []

        raise CalculationError(
            calculation_name="BaseDateComposition",
            details=(f"index '{self.definition.index_name}' holds nothing on "
                     f"its base date {base_date:%Y-%m-%d}, so there is no "
                     f"composition to anchor a level to. Of the "
                     f"{len(named)} universe_identifiers the definition "
                     f"names, {len(resolved)} resolved against the reference "
                     f"data and {len(constituents)} passed the eligibility "
                     f"rules. Calculating on would publish a level series "
                     f"for an index with no constituents."))

    def initialize_divisor(self,
                           initial_total_market_value: float) -> float:
        """
        Calculates the initial divisor for the index on its base_date.
        Divisor = Initial Total Market Value / Base Index Value.

        Args:
            initial_total_market_value: The sum of (price * shares * fx_rate *
                free_float_if_applicable) for all base constituents on the
                base_date, expressed in index currency.

        Returns:
            The initial divisor as a float.
        """
        if initial_total_market_value <= 0:
            logger.error("Initial total market value must be positive to initialize divisor.")
            raise CalculationError(
                "DivisorInitialization",
                f"the base constituents are worth {initial_total_market_value} "
                f"on the base date, so there is no scale to anchor the index "
                f"to. Any divisor chosen here would produce a level series "
                f"that is coherent and means nothing.")
        if self.definition.base_value <= 0:
            logger.error("Base index value must be positive to initialize divisor.")
            raise CalculationError("DivisorInitialization", "Base index value is non-positive.")

        divisor = initial_total_market_value / self.definition.base_value
        logger.info(
            f"Divisor for '{self.definition.index_name}' initialized to: {divisor:.4f} "
            f"(Initial Market Value: {initial_total_market_value:.2f}, "
            f"Base Value: {self.definition.base_value})")
        return divisor

    @staticmethod
    def adjust_divisor_for_rebalance(old_divisor: float,
                                     old_market_value: float,
                                     new_market_value: float) -> float:
        """Adjust the divisor to maintain index level continuity across a rebalance.

        When index composition or weights change, the total market value shifts.
        To prevent an artificial jump in the index level the divisor is scaled:

            new_divisor = old_divisor * (new_market_value / old_market_value)

        This guarantees: level_before == level_after.

        Args:
            old_divisor: The divisor in effect before the rebalance.
            old_market_value: Aggregate market value under the **old** composition.
            new_market_value: Aggregate market value under the **new** composition.

        Returns:
            The adjusted divisor.

        Raises:
            ValueError: If *old_divisor*, *old_market_value* or *new_market_value*
                is zero or negative.
        """
        if old_divisor <= 0:
            raise ValueError(f"old_divisor must be positive, got {old_divisor}")
        if old_market_value <= 0:
            raise ValueError(f"old_market_value must be positive, got {old_market_value}")
        if new_market_value <= 0:
            raise ValueError(f"new_market_value must be positive, got {new_market_value}")

        new_divisor = old_divisor * (new_market_value / old_market_value)

        logger.info(
            f"Divisor adjusted for rebalance: {old_divisor:.6f} -> {new_divisor:.6f} "
            f"(old_mv={old_market_value:.2f}, new_mv={new_market_value:.2f})"
        )
        return new_divisor

    #todo: run() is currently iterating through all dates, this function should be
    # vectorised for efficiency.
    def run(self,
            start_date: str | None = None,
            end_date: str | None = None) -> IndexResult:
        """Run the full index calculation over a date range.

        Iterates the index's own trading sessions from *start_date* to
        *end_date* — the definition's calendar, not Monday to Friday (BN-186)
        — handling three day types:

        1. **Base date** – resolve universe, select constituents, compute
           weights, initialise divisor, set level = base_value. Rolled forward
           to the first session when the base date itself was not one.
        2. **Rebalance date** – reconstitute (re-resolve universe, re-select,
           re-weight) and adjust divisor for continuity.
        3. **Regular day** – compute index level using current constituents
           and weights.

        The method is idempotent: it carries no state between calls.

        Args:
            start_date: First calculation date (YYYY-MM-DD).  Defaults to
                ``definition.base_date``.
            end_date: Last calculation date (YYYY-MM-DD).  Required.

        Returns:
            An :class:`IndexResult` containing index levels, divisor history,
            constituent snapshots, weight snapshots, and the daily weights
            panel — one row per constituent per day, recorded as the loop
            goes, since the state it holds each day is path-dependent and
            cannot be reconstructed from the rebalance snapshots afterwards.

        Raises:
            ValueError: If *end_date* is not provided or precedes the base date.
            CalculationError: If the dataset lacks a column the definition
                reads -- checked before any work, rather than discovered at the
                first read and reported as one company's problem (BN-217).
        """
        self.require_columns()

        base_date = self.definition.base_date
        pd_start = pd.Timestamp(start_date) if start_date else base_date
        if end_date is None:
            raise ValueError("end_date must be provided.")
        pd_end = pd.Timestamp(end_date)

        if pd_end < base_date:
            raise ValueError(
                f"end_date ({pd_end.strftime('%Y-%m-%d')}) precedes "
                f"base_date ({base_date.strftime('%Y-%m-%d')})."
            )

        # Ensure start is not before base_date
        if pd_start < base_date:
            pd_start = base_date

        # What the calendar can actually speak for, asked before the sessions
        # are built (BN-198). A window the calendar cannot reach used to
        # produce an empty index *successfully*: good data, a real request, a
        # result with no levels in it and a WARNING as the only record.
        coverage = calendar_coverage(pd_start, pd_end, self.definition.calendar)

        if coverage.is_empty and self.definition.calendar is not None:
            raise CalculationError(
                calculation_name="IndexCalculator",
                details=(
                    f"cannot schedule {pd_start:%Y-%m-%d} to "
                    f"{pd_end:%Y-%m-%d}: {describe_bounds(coverage)}, so it "
                    f"can speak for none of that window. An index cannot have "
                    f"levels on days its calendar does not know about, and "
                    f"publishing an empty one would report that as a result "
                    f"rather than as a problem. Move the dates inside those "
                    f"bounds, or name a calendar that covers the period."))

        if coverage.is_partial:
            logger.warning(
                "%s narrowed this run: %s. The index is calculated over the "
                "covered range and carries both in its result.",
                self.definition.calendar, coverage.describe_window())

        # The index's own sessions, not Monday to Friday (BN-186). A level is
        # a statement that the market traded and the constituents were worth
        # something; on 25 December neither is true, and iterating business
        # days published one anyway whenever the store happened to carry a bar.
        trading_days = sessions(pd_start, pd_end, self.definition.calendar)

        # Still reachable with a full cover: a window inside the calendar's
        # bounds that holds no session at all, a single weekend being the
        # smallest case. That is a real answer to a narrow question rather
        # than a calendar failing to reach, so it stays an empty result.
        if trading_days.empty:
            logger.warning("No trading days in the requested range.")
            return IndexResult(
                index_id=self.definition.index_id,
                index_levels=pd.Series(dtype=float),
                divisor_history=pd.Series(dtype=float),
                constituent_snapshots={},
                weight_snapshots={},
                calendar_coverage=coverage if coverage.is_partial else None,
            )

        # A base date the market was shut on is initialised on the first
        # session that follows it. Rolling forward is the only direction
        # available -- there is no index before its base date to roll back to
        # -- and refusing would make 1 January, the commonest base date there
        # is, unusable. Only when the run starts at the base date: a run
        # starting later never meets it, which is the behaviour it always had.
        if pd_start <= base_date and base_date not in trading_days:
            logger.warning(
                "Base date %s is not a session on %s; the index is based on "
                "%s, the first session after it.",
                base_date.date(), self.definition.calendar,
                trading_days[0].date())
            base_date = pd.Timestamp(trading_days[0])

        # Pre-compute rebalance dates (excluding base date which is handled separately)
        rebalance_dates_list = self.definition.get_rebalance_dates(
            pd_start.strftime('%Y-%m-%d'),
            pd_end.strftime('%Y-%m-%d'),
        )
        # Announced on one date, in force on another. With no lag the two
        # coincide and this mapping is the identity, which is what keeps every
        # index defined before BN-126 producing identical levels.
        # The panel is built only when a lag actually applies. An index
        # without one does no calendar work at all, which keeps this change
        # free for every index defined before it.
        lag = self.definition.effective_lag_sessions
        panel = (sessions(pd_start, pd_end, self.definition.calendar)
                 if lag > 0 else pd.DatetimeIndex([]))
        effective_for = {announced: effective_date(announced, lag, panel)
                         for announced in rebalance_dates_list
                         if announced != base_date}

        # Keyed by the date the composition is *applied*, since that is the day
        # the loop has to act on. Two announcements landing on one effective
        # date would be a schedule shorter than its own lag; the later wins,
        # which is the one a reader would expect to be in force.
        announced_for = {effective: announced
                         for announced, effective in effective_for.items()}
        rebalance_dates = set(announced_for)
        rebalance_dates.discard(base_date)

        # Accumulators
        index_levels: dict[pd.Timestamp, float] = {}
        divisor_values: dict[pd.Timestamp, float] = {}
        constituent_snapshots: dict[pd.Timestamp, list[str]] = {}
        weight_snapshots: dict[pd.Timestamp, dict[str, float]] = {}
        # Only rebalances where the cap actually bound get an entry, so an
        # uncapped index carries an empty mapping rather than noise.
        cap_reports: dict[pd.Timestamp, CapReport] = {}
        # Effective date -> announcement date, populated only where the two
        # differ. An index with no lag carries an empty mapping, so its
        # presence is itself the signal that a lag applies.
        announcements: dict[pd.Timestamp, pd.Timestamp] = {}
        # One record per constituent per day. The loop already holds the true
        # state of the index on every date and used to discard it, keeping
        # only levels, divisors and the rebalance snapshots.
        #
        # Held as dicts until the run ends and converted once, which is the
        # pattern the backtest engine uses. The cost is at the peak rather
        # than at rest: a pending record measures ~240 bytes against ~19 in
        # the frame, so a 6,000-name decade would hold ~3.6 GB here before
        # collapsing to ~286 MB. Chunked conversion is the fix if that ever
        # binds; nothing in this repository runs at that size yet.
        daily_records: list[dict[str, object]] = []

        # Cash distributions, loaded once. A price index skips this entirely,
        # so it costs nothing and reads no action history — which is what keeps
        # every index defined before BN-125 producing identical levels.
        reinvesting = self.definition.return_type in REINVESTING
        distributions = self.cash_distribution_schedule() if reinvesting else {}

        # When each name stops being listed, resolved once. An index over a
        # universe where nothing is ever delisted gets an empty mapping and
        # pays for one `if` per day.
        delistings = self.delisting_schedule()
        previous_date = base_date
        withholding = withholding_for(self.definition.return_type,
                                      self.definition.withholding_tax_rate)

        # Running state. `units` is what the index actually holds: fixed
        # between rebalances, so weights drift with relative performance
        # instead of being reset every day.
        constituents: list[Asset] = []
        weights: dict[Asset, float] = {}
        units: dict[Asset, float] = {}
        divisor: float = 0.0
        level: float = self.definition.base_value

        for date in trading_days:
            # One slice for the day, then every per-name read below is served
            # from it (BN-212). The panel has existed since BN-190 and was
            # wired into selection and weighting, which run at rebalances --
            # roughly forty times over this loop's eight hundred. The daily
            # valuation is the one that runs every session, and it was the one
            # still reading name by name.
            self._warm_holdings(units, date)

            # Today's holdings, valued. Empty on a day the index has no
            # holdings to value, which records no weights.
            values: dict[Asset, float] = {}

            if date == base_date:
                # --- Base date initialisation ---
                constituents_raw = self._get_universe(date)
                constituents = self.select_constituents(constituents_raw, date)

                self._require_a_base_composition(constituents_raw,
                                                 constituents, date)

                weights = self.calculate_constituent_weights(constituents, date)
                weights, cap_report = self.cap_weights(weights)
                if cap_report.was_capped:
                    cap_reports[date] = cap_report

                # The aggregate the index represents is still the constituents'
                # total market value, which keeps the divisor's magnitude and
                # meaning unchanged. What changes is that the holdings are now
                # units derived from the weights, rather than shares
                # outstanding — so the methodology actually drives the level.
                mv_map = self._get_constituent_market_values(weights, date)
                total_mv = sum(mv_map.values())
                units = self.index_units(weights, total_mv, date)
                values = self.holding_values(units, date)

                # Straight through to `initialize_divisor`, which refuses a
                # non-positive aggregate. A zero market value used to be
                # caught here and turned into a divisor of 1.0, which
                # bypassed that refusal and published a level series scaled
                # by an arbitrary constant: internally coherent, and a
                # measure of nothing (BN-184).
                divisor = self.initialize_divisor(total_mv)

                level = self.definition.base_value

                # Record snapshots
                constituent_snapshots[date] = [a.asset_id for a in constituents]
                weight_snapshots[date] = {a.asset_id: w for a, w in weights.items()}

            elif date in rebalance_dates:
                # --- Rebalance date ---
                # Value the outgoing holdings at today's prices. This is the
                # level the new composition has to start from, which is what
                # the divisor adjustment preserves.
                old_aggregate = self.aggregate_value(units, date)

                # The outgoing holdings are the ones that went ex today, so the
                # reinvestment belongs to them and has to happen before the
                # composition changes. Adjusting the divisor here composes with
                # the continuity adjustment below: that one preserves whatever
                # level is in force, which now includes the distribution.
                if reinvesting:
                    paid = distributions.get(date, {})
                    divisor = self.reinvest(
                        divisor, old_aggregate,
                        self.distribution_received(
                            units, paid, withholding,
                            self.distribution_rates(
                                paid, units, date,
                                self.definition.currency)))

                # Reconstitute as of the *announcement*: the constituent list
                # and target weights are what was published, even though they
                # are implemented at today's prices. Selecting on the effective
                # date instead would let a name that qualified when the index
                # was announced be dropped by a price move in between, which is
                # not what a published composition means.
                announced_on = announced_for.get(date, date)

                constituents_raw = self._get_universe(announced_on)
                constituents = self.select_constituents(constituents_raw,
                                                        announced_on)
                weights = self.calculate_constituent_weights(constituents,
                                                             announced_on)
                weights, cap_report = self.cap_weights(weights)
                if cap_report.was_capped:
                    cap_reports[date] = cap_report

                if announced_on != date:
                    announcements[date] = announced_on

                # Rebuild the holdings to the new weights, scaled to the
                # constituents' total market value so the divisor keeps the
                # magnitude it has always had.
                new_mv_map = self._get_constituent_market_values(weights, date)
                new_total_mv = sum(new_mv_map.values())
                units = self.index_units(weights, new_total_mv, date)
                values = self.holding_values(units, date)
                new_aggregate = float(sum(values.values()))

                # Adjust divisor for continuity
                if old_aggregate > 0 and new_aggregate > 0:
                    divisor = self.adjust_divisor_for_rebalance(
                        divisor, old_aggregate, new_aggregate
                    )
                elif new_aggregate > 0:
                    divisor = new_aggregate / level if level > 0 else 1.0

                # Compute level with adjusted divisor
                level = new_aggregate / divisor if divisor > 0 else level

                # Record snapshots
                constituent_snapshots[date] = [a.asset_id for a in constituents]
                weight_snapshots[date] = {a.asset_id: w for a, w in weights.items()}

            else:
                # --- Regular trading day ---
                if not constituents or divisor <= 0:
                    # Before base date initialisation or no constituents
                    pass
                else:
                    # Before anything else: a holding that stopped being
                    # listed cannot be valued, and leaving it in would report
                    # its whole weight as a loss on the day it went.
                    units, divisor, deleted = self.apply_deletions(
                        units, divisor, date, delistings, previous_date)

                    if deleted:
                        constituents = [asset for asset in constituents
                                        if asset.asset_id not in set(deleted)]
                        weights = {asset: weight
                                   for asset, weight in weights.items()
                                   if asset.asset_id not in set(deleted)}

                    # Valued once, then used three times over: to reinvest
                    # into, to set the level, and to record the day's weights.
                    # A price lookup per holding is the run's dominant cost.
                    values = self.holding_values(units, date)
                    aggregate = float(sum(values.values()))

                    if reinvesting:
                        paid = distributions.get(date, {})
                        divisor = self.reinvest(
                            divisor,
                            aggregate,
                            self.distribution_received(
                                units, paid, withholding,
                                self.distribution_rates(
                                paid, units, date,
                                self.definition.currency)))

                    level = self.level_from_units(
                        units=units,
                        divisor=divisor,
                        current_date=date,
                        previous_index_level=level,
                        values=values,
                    )

            index_levels[date] = level
            divisor_values[date] = divisor
            daily_records.extend(weight_rows(date, units, values))

            # Deletions are valued on the last day the leaver still had a
            # price, so the loop has to remember which day that was.
            previous_date = date

        logger.info(
            f"run() completed for '{self.definition.index_name}': "
            f"{len(trading_days)} trading days, "
            f"{len(constituent_snapshots)} rebalance(s), "
            f"{len(daily_records)} daily weight record(s)."
        )

        return IndexResult(
            index_id=self.definition.index_id,
            index_levels=pd.Series(index_levels),
            divisor_history=pd.Series(divisor_values),
            constituent_snapshots=constituent_snapshots,
            weight_snapshots=weight_snapshots,
            cap_reports=cap_reports,
            announcement_dates=announcements,
            daily_weights=daily_weights_frame(daily_records),
            calendar_coverage=coverage if coverage.is_partial else None,
        ).with_data(self.data)

    def require_columns(self) -> None:
        """Refuse up front if the dataset cannot support this definition.

        Public because the constituent preview runs a definition without
        calling `run`, and deserves the same answer: a preview of a
        market-cap index over a store with no share counts should say so,
        not fail on the first name it tries to price.

        Raises:
            CalculationError: Naming each missing column and what needs it.
        """
        require_columns(self.definition, self.data, self.price_column)


    def _warm_holdings(self,
                       units: dict[Asset, float],
                       date: pd.Timestamp) -> None:
        """Read one session's rows for everything held, in a single slice.

        A hint, and deliberately forgiving: a provider that does not implement
        `warm_session` is a hand-assembled double, and the reads it serves all
        answer identically without it. What it must not do is change an
        answer, so nothing here is allowed to fail loudly -- and nothing here
        can fail quietly either, because the panel only ever *replaces* a
        slower read of the same rows.
        """
        if not units:
            return

        warm = getattr(self.data, "warm_session", None)

        if warm is None:
            return

        warm([asset.asset_id for asset in units], date)


    def run_daily_calculation(self,
                              current_date: pd.Timestamp,
                              constituents: list[Asset],
                              weights: dict[Asset, float],
                              previous_index_level: float,
                              previous_divisor: float) -> tuple[float, float]:
        """
        Runs a single day's index calculation process.

        Args:
            current_date: The date for which to perform calculations.
            constituents: Current index constituents.
            weights: Current constituent weights.
            previous_index_level: Index level from the previous period.
            previous_divisor: Divisor from the previous period.

        Returns:
            Tuple of (new_index_level, new_divisor).
        """
        divisor = previous_divisor

        if divisor is None or divisor <= 0:
            if current_date == self.definition.base_date:
                if not constituents:
                    raise ValueError(
                        "Base date calculation: Constituents not provided. "
                        "Cannot initialize divisor.")
                base_day_values = self._get_constituent_market_values(
                    constituents_with_weights=dict.fromkeys(constituents, 0),
                    current_date=current_date
                )
                initial_mv = sum(base_day_values.values())
                if initial_mv > 0:
                    divisor = self.initialize_divisor(initial_mv)
                else:
                    raise ValueError(
                        f"Cannot initialize divisor on base date {current_date} due to "
                        "zero or negative market value.")
            else:
                raise ValueError("Divisor not initialized for index calculation.")

        new_level, final_divisor = self.calculate_index_level(
            current_date=current_date,
            constituents=constituents,
            weights=weights,
            divisor=divisor,
            previous_index_level=previous_index_level,
        )

        return new_level, final_divisor
