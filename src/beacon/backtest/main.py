# src/beacon/backtest/main.py
"""
Backtest: the one-call front door composing calculator, cache and engine.

The class a user, the fund and the server all call. The construction holds the
modelling assumptions (capital, costs, currency, the benchmark of record) and
each :meth:`run` supplies the subject: an :class:`IndexDefinition` and a window.

Inside a run the passes are strictly sequential: fingerprint the calculation
and reuse a cached :class:`IndexResult` when the cache allows it, otherwise
calculate and store; optionally optimise the published weights; then hand the
schedule to :class:`BacktestEngine` to simulate. Calculate-then-simulate
produces numbers identical to a fused daily loop while keeping the
calculation a separable, cacheable artifact.

An optimised run is the same composition one derivation deeper: the
definition-plus-config becomes an **ephemeral**
:class:`~beacon.index.derived.OptimisedIndexDefinition`, calculated through
the same cache-assisted path a stored one takes (the parent's calculation is
one cache entry shared with every plain run of the same definition, and the
derived calculation is another when the whole chain can be keyed), and the
engine receives the solved calculation as a real IndexResult. Ad-hoc and
stored optimised indices therefore produce bit-identical numbers.

scipy is required only when a solve actually runs, and the default cache
location needs `platformdirs`; without it a Backtest simply runs uncached, so
the front door works on the core install exactly as it does on a full one.
"""
# Decisions 17 and 19 of the backtester redesign made this the one front door.
# The fused daily loop is recorded (decision 20) as the shape of a future
# walk-forward mode, not built here. Optimised runs are BN-167; `beacon.optimise`
# has imported scipy-free since BN-166.
import copy
import logging
from collections.abc import Mapping, Sequence
from typing import Any, Literal

import pandas as pd

from .. import sources
from ..assumptions import ModellingAssumptions, apply
from ..data.fetcher import DataFetcher
from ..exceptions import CalculationError
from ..index import cache as index_cache
from ..index.cache import IndexResultCache
from ..index.calculation import IndexCalculator
from ..index.calculation.total_return import REINVESTING
from ..index.derived import (
    AnyIndexDefinition,
    OptimisedIndexDefinition,
    calculate_derived_index,
)
from ..index.result import IndexResult
from ..optimise import OptimisationConfig
from ..strategy.active import ActiveStep, ActiveStrategy
from ..strategy.base import StrategyContext
from ..strategy.futures import FuturesReplication, IndexFutures
from ..strategy.synthetic import SyntheticReplication
from ..strategy.tracking import IndexTracking, ReplicationStep
from .dividends import REINVEST
from .engine import BacktestEngine
from .flows import Flows
from .implementation import Implementation
from .result import BacktestResult
from .rules import BacktestModifier
from .settings import book_currency
from .share_classes import ShareClass
from .vehicle import Vehicle

logger = logging.getLogger(__name__)


def _default_cache() -> IndexResultCache:
    """A cache at the default root."""
    return IndexResultCache()


def _rejecting_empty(definition: AnyIndexDefinition,
                     result: IndexResult) -> IndexResult:
    """The result, unless the calculation came back empty — then fail loudly.

    A definition whose universe resolves to nothing used to sail straight
    through: a dead level series, zero trades, no error — a backtest of
    nothing, presented as a result. The check lives here rather than in the
    calculator because the calculator's tolerance of an empty day is behaviour
    other callers depend on; the front door is where "simulate this" is the
    stated intent, so an index that never held anything is refused.

    Args:
        definition: What was calculated, for the message.
        result: The calculation to inspect.

    Returns:
        IndexResult: *result*, when there is something to simulate.

    Raises:
        CalculationError: When the level series is empty or all zero, or the
            index never held a single constituent.
    """
    levels = result.index_levels
    lifeless = levels.empty or bool((levels == 0.0).all())

    if not lifeless and not result.daily_weights.empty:
        return result

    universe = definition.universe_identifiers
    described = ("universe_identifiers is not set"
                 if universe is None
                 else f"none of its {len(universe)} universe_identifiers "
                      f"produced a holding")

    raise CalculationError(
        "Backtest",
        f"the calculation for index '{definition.index_id}' is empty "
        f"({'no level was computed' if levels.empty else 'it never held a single constituent'}), "
        f"so there is nothing to simulate. Check the definition's "
        f"universe_identifiers ({described}): an empty or unresolvable "
        f"universe gives an index with no constituents.")


class Backtest:
    """One-call backtests: assumptions on the object, the index per run.

    The constructor mirrors :class:`BacktestEngine`'s parameters (what stays
    fixed across runs) and :meth:`run` takes the definition and window, so a
    parameter sweep is one object per assumption set over one shared (cached)
    calculation::

        bt = Backtest(initial_capital=1_000_000, transaction_cost_bps=5.0)
        result = bt.run(definition, start="2023-01-03", end="2023-12-29")

    Args:
        initial_capital: The starting capital for each run. With share
            classes, each class's by name.
        transaction_cost_bps: Transaction cost in basis points applied to
            each trade's notional value. Defaults to 0 (no cost).
        price_column: Market-data column both the calculator and the engine
            read. Defaults to ``"CLOSE"``.
        currency: The simulated book's currency. None, the default, uses the
            index's currency. Pass another to keep the book in a different
            currency from the index.
        modelling_assumptions: What the run takes as given about markets and
            data, laid over the process-wide default (see
            :func:`beacon.use_modelling_assumptions`). The index calculation
            reads data under the same assumptions as the simulation.
        dividends: What happens to cash distributions: ``"reinvest"`` (the
            default), ``"cash"`` or ``"distribute"``. See
            :mod:`beacon.backtest.dividends`.
        implementation: The screens and redistribution rule applied at each
            rebalance. None screens nothing beyond stale prices.
        share_classes: The fund's share classes, sharing one book, each with
            its own fee, flows and distribution policy (see
            :mod:`beacon.backtest.share_classes`). None runs one class.
        flows: Money arriving and leaving: one scenario or several, added
            together (see :mod:`beacon.backtest.flows`). None for a fixed
            amount of capital.
        vehicle: The fund's fee and unit price at launch. None charges no
            fee.
        modifiers: Optional hooks that can skip rebalances or adjust trades.
        benchmark: The benchmark of record, stored on every result this
            object produces.
        data_provider: Data source for both the calculation and the
            simulation. None resolves the process's ambient source
            (:func:`beacon.sources.resolve`) at each run. Resolution is per
            run, not at construction, so ``beacon.use()`` after construction
            is honoured.
        cache: Where calculated IndexResults are kept between runs. None
            uses the default location when `platformdirs` is available and
            degrades to no caching when it is not. False turns caching off,
            so every run calculates its index afresh.
    """

    def __init__(self,
                 initial_capital: float | Mapping[str, float],
                 transaction_cost_bps: float = 0.0,
                 price_column: str = "CLOSE",
                 currency: str | None = None,
                 modelling_assumptions: ModellingAssumptions | None = None,
                 dividends: str = REINVEST,
                 implementation: Implementation | None = None,
                 modifiers: list[BacktestModifier] | None = None,
                 benchmark: IndexResult | pd.Series | None = None,
                 data_provider: DataFetcher | None = None,
                 cache: IndexResultCache | Literal[False] | None = None,
                 flows: Flows | list[Flows] | Mapping[str, Flows | list[Flows]] | None = None,
                 vehicle: Vehicle | None = None,
                 share_classes: Sequence[ShareClass] | None = None):
        self.initial_capital: float | Mapping[str, float] = initial_capital
        self.transaction_cost_bps: float = transaction_cost_bps
        self.price_column: str = price_column
        self.currency: str | None = currency
        self.modelling_assumptions: ModellingAssumptions | None = modelling_assumptions
        self.implementation: Implementation | None = implementation
        self.flows: Flows | list[Flows] | Mapping[str, Flows | list[Flows]] | None = flows
        self.share_classes: tuple[ShareClass, ...] | None = (
            tuple(share_classes) if share_classes else None)
        self.vehicle: Vehicle | None = vehicle
        self.dividends: str = dividends
        self.modifiers: list[BacktestModifier] | None = modifiers
        self.benchmark: IndexResult | pd.Series | None = benchmark
        self.data_provider: DataFetcher | None = data_provider
        # False switches caching off (BN-262): None already meant "the
        # default location", so there was no way to ask for none.
        self.cache: IndexResultCache | None = (
            None if cache is False
            else cache if cache is not None
            else _default_cache())

        logger.info("Backtest initialised: capital %.2f, cost %.1f bps, "
                    "cache %s.",
                    (sum(initial_capital.values()) if isinstance(initial_capital, Mapping)
                     else initial_capital), transaction_cost_bps,
                    "off" if self.cache is None else f"at {self.cache.root}")

    def run(self,
            definition: AnyIndexDefinition | IndexTracking | ActiveStrategy,
            start: str | None = None,
            end: str | None = None,
            optimised: bool = False,
            optimisation_config: OptimisationConfig | None = None) -> BacktestResult:
        """Calculate (or reuse) the index, then simulate tracking it.

        Args:
            definition: The index to calculate and track: a plain
                :class:`IndexDefinition`, or a stored
                :class:`~beacon.index.derived.OptimisedIndexDefinition`,
                which always fills both index books: its parent's calculation
                as ``index.target`` and its own as ``index.optimised``. Or an
                :class:`~beacon.strategy.IndexTracking`, which holds the index
                through a replication: the index is calculated and measured
                against, and the replicated weights are traded. Or an
                :class:`~beacon.strategy.ActiveStrategy`, which builds its own
                portfolio and is measured against the index as its benchmark.
            start: First date (YYYY-MM-DD). Defaults to the definition's
                base date.
            end: Last date (YYYY-MM-DD). Required.
            optimised: Ad-hoc optimisation of *definition*: build an
                ephemeral derived index over it from *optimisation_config*
                (same solve, same chained levels as a stored one) and trade
                that. Requires the config; needs scipy only on this path.
            optimisation_config: What the ad-hoc derivation is asked to do
                (objective, constraints, reserved risk model). Only
                meaningful, and only allowed, with ``optimised=True``.

        Returns:
            BacktestResult: The engine's result: portfolio kept whole, books
            filled, data bound to the run's own source.

        Raises:
            ValueError: If *end* is not provided, or *optimised* and
                *optimisation_config* contradict each other (a flag with no
                config, or a config with no flag).
            CalculationError: If a calculation comes back empty (no level
                was computed, every level is zero, or the index never held a
                constituent), the objective is unknown, or an optimisation is
                infeasible.
            DataSourceError: If no data source is bound and the process has
                no ambient one.
        """
        if end is None:
            raise ValueError("end must be provided.")

        # BN-269 and BN-280: a strategy holds the index through a
        # replication, or builds its own portfolio against it as a benchmark.
        strategy = (definition if isinstance(definition, IndexTracking | ActiveStrategy)
                    else None)

        if strategy is not None and optimised:
            raise ValueError("a strategy is already a way of holding the index; "
                             "use OptimisedReplication rather than "
                             "optimised=True.")

        index_definition: AnyIndexDefinition = (
            definition.index if isinstance(definition, IndexTracking | ActiveStrategy)
            else definition)
        if optimised and optimisation_config is None:
            raise ValueError(
                "optimised=True needs an optimisation_config saying what to "
                "solve for.")
        if not optimised and optimisation_config is not None:
            raise ValueError(
                "an optimisation_config was given but optimised is False; "
                "pass optimised=True to use it, or drop the config.")

        fetcher = (self.data_provider if self.data_provider is not None
                   else sources.resolve())

        # One set of assumptions for the whole run, resolved once: the index
        # calculation and the simulation read data the same way (BN-276).
        assumptions = (self.modelling_assumptions
                       or ModellingAssumptions()).effective()
        fetcher = apply(assumptions, fetcher)

        logger.info("Backtest run for '%s' from %s to %s.",
                    index_definition.index_id, start or index_definition.base_date.date(), end)

        derived = self._derived_definition(index_definition, optimised,
                                           optimisation_config)

        if derived is None:
            index_result = self._calculated(index_definition, fetcher, start, end)
            target_index = None
        else:
            # The parent's calculation is cache-assisted exactly as a plain
            # run's is — and it is the same entry, so an optimised run over a
            # warm index_definition solves without recalculating the parent. The
            # derived calculation caches too, when the whole chain keys.
            target_index = self._calculated(derived.source, fetcher, start, end)
            index_result = self._calculated(derived, fetcher, start, end,
                                            parent_result=target_index)

        first = start if start is not None else str(index_definition.base_date.date())
        context = StrategyContext(fetcher, book_currency(self.currency, index_result))
        strategy = self._with_basket(strategy, fetcher, start, end)
        replication: list[ReplicationStep] = (
            self._replicated(strategy, index_result, context, first, end)
            if isinstance(strategy, IndexTracking) else [])
        active: list[ActiveStep] = (strategy.steps(index_result, first, end, context)
                                    if isinstance(strategy, ActiveStrategy) else [])
        steps: list[ReplicationStep] | list[ActiveStep] = replication or active

        engine = BacktestEngine(
            start_date=first,
            end_date=end,
            initial_capital=self.initial_capital,
            data_provider=fetcher,
            index_result=index_result,
            price_column=self.price_column,
            currency=self.currency,
            modelling_assumptions=assumptions,
            dividends=self.dividends,
            implementation=self._implementation_for(index_definition),
            flows=self.flows,
            share_classes=self.share_classes,
            vehicle=self.vehicle,
            transaction_cost_bps=self.transaction_cost_bps,
            modifiers=self.modifiers,
            benchmark=self.benchmark,
            target_index=target_index,
            # The definition's own, resolved through a derivation to its
            # parent's: the engine needs it to tell a market holiday apart
            # from a hole in the data when a bar is missing (BN-183).
            calendar=index_definition.calendar,
            schedule=({step.date: step.weights for step in steps}
                      if strategy is not None else None),
            swap=(strategy.replication.terms
                  if isinstance(strategy, IndexTracking)
                  and isinstance(strategy.replication, SyntheticReplication)
                  else None),
            futures=(self._priced(strategy.replication.futures, index_definition)
                     if isinstance(strategy, IndexTracking)
                     and isinstance(strategy.replication, FuturesReplication)
                     else None))

        result = engine.run()
        result.replication = replication
        result.active = active

        return result

    def _with_basket(self,
                     strategy: IndexTracking | ActiveStrategy | None,
                     fetcher: DataFetcher,
                     start: str | None,
                     end: str) -> IndexTracking | ActiveStrategy | None:
        """A synthetic replication whose basket is an index, with the basket
        calculated over the run (and cached like any index); any other
        strategy as it is."""
        if not (isinstance(strategy, IndexTracking)
                and isinstance(strategy.replication, SyntheticReplication)
                and strategy.replication.basket_index is not None):
            return strategy

        basket = self._calculated(strategy.replication.basket_index, fetcher, start, end)

        return IndexTracking(strategy.index, strategy.replication.with_basket(basket))

    def _implementation_for(self,
                            definition: AnyIndexDefinition) -> Implementation | None:
        """This backtest's implementation, with any futures equitising its
        buffer priced for *definition* (a copy; the object given is left as
        it is)."""
        implementation = self.implementation

        if implementation is None or implementation.equitise is None:
            return implementation

        priced = copy.copy(implementation)
        priced.equitise = self._priced(implementation.equitise, definition)

        return priced

    @staticmethod
    def _priced(futures: IndexFutures,
                definition: AnyIndexDefinition) -> IndexFutures:
        """*futures* with no dividend yield on an index that reinvests its
        dividends, whose level already counts them; as given otherwise."""
        reinvesting = getattr(definition, "return_type", None) in REINVESTING

        if futures.dividend_yield is None and reinvesting:
            return IndexFutures(roll_days=futures.roll_days,
                                roll_cost_bps=futures.roll_cost_bps, dividend_yield=0.0)

        return futures

    @staticmethod
    def _replicated(strategy: IndexTracking,
                    index_result: IndexResult,
                    context: StrategyContext,
                    start: str,
                    end: str) -> list[ReplicationStep]:
        """The strategy's replication at each rebalance inside the run."""
        window = {date: weights
                  for date, weights in index_result.weight_snapshots.items()
                  if pd.Timestamp(start) <= date <= pd.Timestamp(end)}

        return strategy.replicated(window, context)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _derived_definition(definition: AnyIndexDefinition,
                            optimised: bool,
                            config: OptimisationConfig | None
                            ) -> OptimisedIndexDefinition | None:
        """The derivation this run trades, or None for a plain passive run.

        A stored optimised definition *is* the derivation. The ad-hoc flag
        wraps whatever was given — plain or already optimised, which is how a
        chain is tried without minting one — in an ephemeral derived
        definition built from the config, so both lifetimes calculate through
        the identical path and produce identical numbers.
        """
        if optimised and config is not None:
            return OptimisedIndexDefinition.from_config(
                index_id=f"{definition.index_id}::optimised",
                index_name=f"{definition.index_name} (optimised)",
                source=definition,
                config=config)

        if isinstance(definition, OptimisedIndexDefinition):
            return definition

        return None

    def calculate(self,
                  definition: AnyIndexDefinition,
                  start: str | None = None,
                  end: str | None = None) -> IndexResult:
        """Calculate (or reuse) an index alone, as a run would calculate it.

        The same calculation `run` starts from, with this backtest's data and
        modelling assumptions, through the same cache: an index calculated
        here is reused by a run over the same dates, and the other way round.
        No portfolio is simulated.

        Args:
            definition: The index: a plain :class:`IndexDefinition` or a
                derived one, whose source is calculated first.
            start: First date (YYYY-MM-DD). Defaults to the definition's
                base date.
            end: Last date (YYYY-MM-DD). Required.

        Returns:
            IndexResult: Its levels, divisors and the weights at each
            rebalance.

        Raises:
            ValueError: If *end* is not provided.
            CalculationError: If the calculation comes back empty.
        """
        # BN-316: an index's figures without a backtest.
        if end is None:
            raise ValueError("end must be provided.")

        fetcher = (self.data_provider if self.data_provider is not None
                   else sources.resolve())
        fetcher = apply(self._assumptions(), fetcher)

        if isinstance(definition, OptimisedIndexDefinition):
            parent = self._calculated(definition.source, fetcher, start, end)

            return self._calculated(definition, fetcher, start, end, parent_result=parent)

        return self._calculated(definition, fetcher, start, end)

    def _assumptions(self) -> ModellingAssumptions:
        """This backtest's assumptions over the process-wide default."""
        return (self.modelling_assumptions or ModellingAssumptions()).effective()

    def _calculated(self,
                    definition: AnyIndexDefinition,
                    fetcher: DataFetcher,
                    start: str | None,
                    end: str,
                    parent_result: IndexResult | None = None) -> IndexResult:
        """The IndexResult for this run: cached when possible, fresh otherwise.

        An empty calculation is rejected *before* it can be stored, so the
        cache only ever holds results worth reusing. A derived definition
        calculates through :func:`calculate_derived_index`, reusing
        *parent_result* when the caller already holds the source's
        calculation; its fingerprint folds in the parent's recursively, so a
        hit here is honest about every link of the chain.
        """
        key, parts = self._cache_key(definition, fetcher, start, end)

        cached = self.cache.get(key) if key is not None and self.cache is not None else None
        if cached is not None:
            logger.info("Index cache hit for '%s' (%s): reusing the "
                        "calculation.", definition.index_id, key)

            # The cache returns results unbound; rebound here so asset-level
            # views on the run read what the simulation reads (decision 16).
            return _rejecting_empty(definition, cached.with_data(fetcher))

        if key is not None:
            logger.info("Index cache miss for '%s' (%s): calculating.",
                        definition.index_id, key)

        if isinstance(definition, OptimisedIndexDefinition):
            calculated = calculate_derived_index(
                definition, fetcher, start_date=start, end_date=end,
                price_column=self.price_column, parent_result=parent_result,
                modelling_assumptions=self._assumptions())
        else:
            calculated = IndexCalculator(
                definition, fetcher, price_column=self.price_column,
                modelling_assumptions=self._assumptions()).run(start_date=start,
                                                              end_date=end)

        result = _rejecting_empty(definition, calculated)

        if key is not None and self.cache is not None:
            self.cache.put(key, result, parts)

        return result

    def _cache_key(self,
                   definition: AnyIndexDefinition,
                   fetcher: DataFetcher,
                   start: str | None,
                   end: str) -> tuple[str | None, dict[str, Any] | None]:
        """The run's fingerprint and its clear-text parts, or (None, None).

        No key means no cache — the safety rule of `beacon.index.cache` — and
        the disposition is INFO-logged once per run, never per day. Anything
        the key builder cannot handle (a hand-rolled or mock fetcher, say) is
        treated as uncacheable rather than allowed to fail the run: caching
        is a convenience and must never cost the calculation.
        """
        if self.cache is None:
            logger.info("Backtest for '%s' runs uncached: no cache is "
                        "configured.", definition.index_id)

            return None, None

        try:
            parts = index_cache.key_parts(definition, fetcher, start, end)
            if parts is None:
                reason = index_cache.explain_uncacheable(definition, fetcher,
                                                         start, end)
                logger.info("Index result for '%s' is uncacheable: %s",
                            definition.index_id, reason)

                return None, None

            return index_cache.fingerprint(definition, fetcher, start, end), parts
        except Exception as error:
            logger.info("Index result for '%s' is uncacheable: the key could "
                        "not be built (%s).", definition.index_id, error)

            return None, None
