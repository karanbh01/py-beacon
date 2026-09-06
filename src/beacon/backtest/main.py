# src/beacon/backtest/main.py
"""
Backtest — the one-call front door composing calculator, cache and engine.

The class a user, the fund and the server all call (decisions 17 and 19 of
the backtester redesign). The construction holds the modelling assumptions —
capital, costs, currency, the benchmark of record — and each :meth:`run`
supplies the subject: an :class:`IndexDefinition` and a window.

Inside a run the passes are strictly sequential: fingerprint the calculation
and reuse a cached :class:`IndexResult` when the store allows it, otherwise
calculate and store; optionally optimise the published weights; then hand the
schedule to :class:`BacktestEngine` to simulate. Calculate-then-simulate
produces numbers identical to a fused daily loop while keeping the
calculation a separable, cacheable artifact; the fused loop is recorded
(decision 20) as the shape of a future walk-forward mode, not built here.

An optimised run (BN-167) is the same composition one derivation deeper: the
definition-plus-config becomes an **ephemeral**
:class:`~beacon.index.derived.OptimisedIndexDefinition`, calculated through
the same cache-assisted path a stored one takes — the parent's calculation is
one cache entry shared with every plain run of the same definition, the
derived calculation is another when the whole chain keys — and the engine
receives the solved calculation as a real IndexResult. Ad-hoc and stored are
one thing in two lifetimes, which is what makes their numbers bit-identical.

scipy is required only when a solve actually runs (`beacon.optimise` imports
scipy-free since BN-166), and the default cache location needs
`platformdirs`; without it a Backtest simply runs uncached, because caching
is a convenience and the front door must work on the core install exactly as
it does on a full one.
"""
import logging
from typing import Any

import pandas as pd

from .. import sources
from ..data.fetcher import DataFetcher
from ..exceptions import CalculationError, MissingDependencyError
from ..index import cache as index_cache
from ..index.cache import IndexResultCache
from ..index.calculation import IndexCalculator
from ..index.derived import (
    AnyIndexDefinition,
    OptimisedIndexDefinition,
    calculate_derived_index,
)
from ..index.result import IndexResult
from ..optimise import OptimisationConfig
from .engine import BacktestEngine
from .result import BacktestResult
from .rules import BacktestModifier

logger = logging.getLogger(__name__)


def _default_cache() -> IndexResultCache | None:
    """A cache at the default root, or None when `platformdirs` is missing.

    Degrading beats raising: a missing optional package should cost the
    convenience it provides — reuse across runs — not the run itself.
    """
    try:
        return IndexResultCache()
    except MissingDependencyError:
        logger.info("platformdirs is not installed; backtests run uncached.")

        return None


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
        f"the calculation for index '{definition.index_id}' is empty — "
        f"{'no level was computed' if levels.empty else 'it never held a single constituent'} "
        f"— so there is nothing to simulate. The likely omission is the "
        f"definition's universe_identifiers ({described}): an empty or "
        f"unresolvable universe would otherwise backtest silently to a dead "
        f"level and zero trades.")


class Backtest:
    """One-call backtests: assumptions on the object, the index per run.

    The constructor mirrors :class:`BacktestEngine`'s parameters — what stays
    fixed across runs — and :meth:`run` takes the definition and window, so a
    parameter sweep is one object per assumption set over one shared (cached)
    calculation::

        bt = Backtest(initial_capital=1_000_000, transaction_cost_bps=5.0)
        result = bt.run(definition, start="2023-01-03", end="2023-12-29")

    Args:
        initial_capital: The starting capital for each run.
        transaction_cost_bps: Transaction cost in basis points applied to
            each trade's notional value. Defaults to 0 (no cost).
        price_column: Market-data column both the calculator and the engine
            read. Defaults to ``"CLOSE"``.
        currency: The simulated book's currency. Defaults to ``"USD"``.
        modifiers: Optional hooks that can skip rebalances or adjust trades.
        benchmark: The benchmark of record, stored on every result this
            object produces.
        data_provider: Data source for both the calculation and the
            simulation. None resolves the process's ambient source
            (:func:`beacon.sources.resolve`) at each run — resolution is per
            run, not at construction, so ``beacon.use()`` after construction
            is honoured.
        cache: Where calculated IndexResults are kept between runs. None
            uses the default location when `platformdirs` is available and
            degrades to no caching when it is not.
    """

    def __init__(self,
                 initial_capital: float,
                 transaction_cost_bps: float = 0.0,
                 price_column: str = "CLOSE",
                 currency: str = "USD",
                 modifiers: list[BacktestModifier] | None = None,
                 benchmark: IndexResult | pd.Series | None = None,
                 data_provider: DataFetcher | None = None,
                 cache: IndexResultCache | None = None):
        self.initial_capital: float = initial_capital
        self.transaction_cost_bps: float = transaction_cost_bps
        self.price_column: str = price_column
        self.currency: str = currency
        self.modifiers: list[BacktestModifier] | None = modifiers
        self.benchmark: IndexResult | pd.Series | None = benchmark
        self.data_provider: DataFetcher | None = data_provider
        self.cache: IndexResultCache | None = (cache if cache is not None
                                               else _default_cache())

        logger.info("Backtest initialised: capital %.2f, cost %.1f bps, "
                    "cache %s.",
                    initial_capital, transaction_cost_bps,
                    "off" if self.cache is None else f"at {self.cache.root}")

    def run(self,
            definition: AnyIndexDefinition,
            start: str | None = None,
            end: str | None = None,
            optimised: bool = False,
            optimisation_config: OptimisationConfig | None = None) -> BacktestResult:
        """Calculate (or reuse) the index, then simulate tracking it.

        Args:
            definition: The index to calculate and track — a plain
                :class:`IndexDefinition`, or a stored
                :class:`~beacon.index.derived.OptimisedIndexDefinition`,
                which always fills both index books: its parent's calculation
                as ``index.target`` and its own as ``index.optimised``.
            start: First date (YYYY-MM-DD). Defaults to the definition's
                base date.
            end: Last date (YYYY-MM-DD). Required.
            optimised: Ad-hoc optimisation of *definition*: build an
                ephemeral derived index over it from *optimisation_config* —
                same solve, same chained levels as a stored one — and trade
                that. Requires the config; needs scipy only on this path.
            optimisation_config: What the ad-hoc derivation is asked to do
                (objective, constraints, reserved risk model). Only
                meaningful — and only allowed — with ``optimised=True``.

        Returns:
            BacktestResult: The engine's result — portfolio kept whole, books
            filled, data bound to the run's own source.

        Raises:
            ValueError: If *end* is not provided, or *optimised* and
                *optimisation_config* contradict each other (a flag with no
                config, or a config with no flag).
            CalculationError: If a calculation comes back empty (see
                :func:`_rejecting_empty`), the objective is unknown, or an
                optimisation is infeasible.
            DataSourceError: If no data source is bound and the process has
                no ambient one.
        """
        if end is None:
            raise ValueError("end must be provided.")
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

        logger.info("Backtest run for '%s' from %s to %s.",
                    definition.index_id, start or definition.base_date.date(), end)

        derived = self._derived_definition(definition, optimised,
                                           optimisation_config)

        if derived is None:
            index_result = self._calculated(definition, fetcher, start, end)
            target_index = None
        else:
            # The parent's calculation is cache-assisted exactly as a plain
            # run's is — and it is the same entry, so an optimised run over a
            # warm definition solves without recalculating the parent. The
            # derived calculation caches too, when the whole chain keys.
            target_index = self._calculated(derived.source, fetcher, start, end)
            index_result = self._calculated(derived, fetcher, start, end,
                                            parent_result=target_index)

        engine = BacktestEngine(
            start_date=start if start is not None else str(definition.base_date.date()),
            end_date=end,
            initial_capital=self.initial_capital,
            data_provider=fetcher,
            index_result=index_result,
            price_column=self.price_column,
            currency=self.currency,
            transaction_cost_bps=self.transaction_cost_bps,
            modifiers=self.modifiers,
            benchmark=self.benchmark,
            target_index=target_index)

        return engine.run()

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
            calculated = calculate_derived_index(definition, fetcher,
                                                 start_date=start,
                                                 end_date=end,
                                                 price_column=self.price_column,
                                                 parent_result=parent_result)
        else:
            calculated = IndexCalculator(
                definition, fetcher,
                price_column=self.price_column).run(start_date=start,
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
