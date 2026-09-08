# src/beacon/index/derived.py
"""
Optimised indices — an index derived from an index you already built.

The owner's framing (planning/optimised_index_design.md rev 2): create an
index, then optimise it — objective function, constraints — and that creates a
NEW index. The derivation stores exactly three things — the **source** index,
the **objective** and the **constraints** — plus the usual identity attributes.
No weights are stored anywhere: definitions are rules, weights are calculated,
and calculations are cached (BN-160).

Two consequences shape this module:

* **Rebalancing follows the parent.** The child solves exactly at the parent's
  published snapshots; a frequency of its own would have no parent weights at
  the extra dates. The derived definition therefore declares no schedule.
* **The calculation is a normal IndexResult.** Solve the parent's weights at
  each rebalance, then chain the solved weights into daily levels — same
  calendar as the parent, level path equal to compounding the solved-weight
  portfolio's returns from the parent's price data.

:class:`OptimisedIndexDefinition` is deliberately a sibling of
:class:`~beacon.index.constructor.IndexDefinition` rather than a subclass: a
subclass would have to invent eligibility rules, a weighting scheme and a
rebalancing frequency it does not have, and the calculator — which stays
untouched — must never receive one by accident.

The level chaining itself lives in `chaining.py` (BN-171): it is arithmetic
over identifiers that needs nothing from a derivation but the index's
identity, currency and base value, so it takes those as arguments and stays
usable by anything else with weights to chain.

scipy enters only when a solve actually runs (`beacon.optimise` imports
scipy-free since BN-166); this module is importable on the core install, and
the core-import test holds it to that.
"""
import logging
from collections.abc import Sequence
from typing import Union

import pandas as pd

from ..data.fetcher import DataFetcher
from ..exceptions import CalculationError
from ..optimise.config import MIN_TRACKING_ERROR, OptimisationConfig
from ..optimise.constraints import Constraint
from ..optimise.result import OptimisationResult
from ..optimise.solver import minimise_tracking_error
from ..risk.model import RiskModel
from .calculation import IndexCalculator
from .chaining import chain_levels
from .constructor import IndexDefinition
from .result import IndexResult

logger = logging.getLogger(__name__)

#: The objectives a derived calculation knows how to solve. Only tracking-error
#: distance ships first; the field is the extension point for covariance-aware
#: objectives (design record, "Not in scope yet").
OBJECTIVES = (MIN_TRACKING_ERROR,)

#: Any definition a calculation or a fingerprint can be asked about: a plain
#: index, or a derivation on one (recursively).
AnyIndexDefinition = Union[IndexDefinition, "OptimisedIndexDefinition"]


class OptimisedIndexDefinition:
    """An optimised index: a derivation on a source index, plus identity.

    The source stays first-class — referenced, never copied — so editing the
    parent changes its optimised children at their next calculation, which is
    what "optimise the index I built" means. Chained optimisation (a source
    that is itself optimised) falls out of the recursion for free.

    Args:
        index_id: A unique identifier for the derived index.
        index_name: The common name of the derived index.
        source: The parent — a plain :class:`IndexDefinition`, or another
            :class:`OptimisedIndexDefinition` for a chain.
        objective: What to minimise. Only ``"min_tracking_error"`` exists
            today; an unknown value fails the calculation loudly, naming the
            accepted ones.
        constraints: What the solved weights must satisfy, as
            :class:`~beacon.optimise.constraints.Constraint` instances. Empty
            means the solver's default of full investment alone.
        base_date: First calculation date (YYYY-MM-DD). None inherits the
            source's, which is the usual case: the child lives on the parent's
            calendar.
        base_value: The level the chained path starts at. None inherits the
            source's.
        currency: The derived index's currency. None inherits the source's.
        description: Optional textual description.
        risk_model: RESERVED — carried but unused, mirroring
            :class:`~beacon.optimise.config.OptimisationConfig`. Setting one
            makes the calculation uncacheable (it cannot be keyed yet) and
            changes no result.
    """

    def __init__(self,
                 index_id: str,
                 index_name: str,
                 source: AnyIndexDefinition,
                 objective: str = MIN_TRACKING_ERROR,
                 constraints: Sequence[Constraint] = (),
                 base_date: str | None = None,
                 base_value: float | None = None,
                 currency: str | None = None,
                 description: str | None = None,
                 risk_model: RiskModel | None = None):
        if not index_id:
            raise ValueError("index_id cannot be empty.")
        if not index_name:
            raise ValueError("index_name cannot be empty.")
        if source is None:
            raise ValueError("source must be provided.")
        if base_value is not None and base_value <= 0:
            raise ValueError("base_value, when provided, must be positive.")

        self.index_id: str = index_id
        self.index_name: str = index_name
        self.source: AnyIndexDefinition = source
        self.objective: str = objective
        self.constraints: tuple[Constraint, ...] = tuple(constraints)
        self.description: str | None = description
        self.risk_model: RiskModel | None = risk_model

        self._base_date: pd.Timestamp | None = (pd.Timestamp(base_date)
                                                if base_date is not None else None)
        self._base_value: float | None = base_value
        self._currency: str | None = currency.upper() if currency is not None else None

        logger.info("OptimisedIndexDefinition '%s' created over source '%s' "
                    "with %d constraint(s).",
                    index_id, source.index_id, len(self.constraints))

    @classmethod
    def from_config(cls,
                    index_id: str,
                    index_name: str,
                    source: AnyIndexDefinition,
                    config: OptimisationConfig) -> "OptimisedIndexDefinition":
        """The derivation an :class:`OptimisationConfig` describes.

        One vocabulary for ad-hoc and stored runs (owner decision): the config
        is the stored derivation minus the source, so an ad-hoc `Backtest.run`
        builds an ephemeral definition through here and calculates it exactly
        as a stored one would be.
        """
        return cls(index_id=index_id,
                   index_name=index_name,
                   source=source,
                   objective=config.objective,
                   constraints=config.constraints,
                   risk_model=config.risk_model)

    # -- inherited identity --------------------------------------------------

    @property
    def base_date(self) -> pd.Timestamp:
        """The first calculation date: own when given, else the source's."""
        return self._base_date if self._base_date is not None else self.source.base_date

    @property
    def base_value(self) -> float:
        """The starting level: own when given, else the source's."""
        return self._base_value if self._base_value is not None else self.source.base_value

    @property
    def currency(self) -> str:
        """The index currency: own when given, else the source's."""
        return self._currency if self._currency is not None else self.source.currency

    @property
    def universe_identifiers(self) -> list[str] | None:
        """The investable universe, which is always the source's.

        The derivation holds no universe of its own — it reallocates over
        exactly the names the parent published — so the answer resolves
        through the chain to the root definition's.
        """
        return self.source.universe_identifiers

    def __repr__(self) -> str:
        return (f"OptimisedIndexDefinition(index_id='{self.index_id}', "
                f"index_name='{self.index_name}', "
                f"source='{self.source.index_id}', "
                f"objective='{self.objective}', "
                f"constraints={len(self.constraints)})")


def calculate_derived_index(definition: OptimisedIndexDefinition,
                            data_provider: DataFetcher,
                            start_date: str | None = None,
                            end_date: str | None = None,
                            price_column: str = "CLOSE",
                            parent_result: IndexResult | None = None) -> IndexResult:
    """Calculate an optimised index into a standard :class:`IndexResult`.

    The three-step workflow of the design record: calculate the parent (or
    accept a pre-supplied calculation — the Backtest integration passes its
    cached one), solve the parent's published weights at every rebalance under
    the definition's constraints, then chain the solved weights into the
    derived index's own daily levels.

    Args:
        definition: The derivation to calculate.
        data_provider: Data source for the parent calculation, prices and FX.
        start_date: First date (YYYY-MM-DD). Defaults to the definition's
            base date. Ignored when *parent_result* is supplied, whose own
            window governs.
        end_date: Last date (YYYY-MM-DD). Required unless *parent_result* is
            supplied.
        price_column: Market-data column read as the price.
        parent_result: The source's calculation, when the caller already has
            it. None calculates the source here — recursively, when the
            source is itself optimised.

    Returns:
        IndexResult: Daily levels, divisor history, constituent and weight
        snapshots at exactly the parent's rebalance dates, and the daily
        weights panel — a normal index result, data-bound to *data_provider*.

    Raises:
        CalculationError: If the objective is unknown, the parent produced no
            rebalance snapshots to solve, or a solve is infeasible — the
            solver's own message names the binding conflict.
        ValueError: If no window end is available to calculate the parent.
    """
    check_objective(definition)

    parent = (parent_result if parent_result is not None
              else calculate_source(definition.source, data_provider,
                                    start_date, end_date, price_column))

    if not parent.weight_snapshots:
        raise CalculationError(
            "DerivedIndex",
            f"the source index '{definition.source.index_id}' produced no "
            f"rebalance snapshots, so there are no weights to optimise for "
            f"'{definition.index_id}'.")

    solved = _solved_schedule(definition, parent)

    return chain_levels(definition.index_id,
                        definition.base_value,
                        definition.currency,
                        parent,
                        solved,
                        data_provider,
                        price_column).with_data(data_provider)


def check_objective(definition: OptimisedIndexDefinition) -> None:
    """Refuse an objective this module cannot solve, naming the accepted set.

    Checked before anything expensive happens — a full parent calculation, in
    the usual case — so a typo in a stored derivation fails in the time it
    takes to read the document rather than after a minute of arithmetic.

    Raises:
        CalculationError: If the objective is not one of :data:`OBJECTIVES`.
    """
    if definition.objective not in OBJECTIVES:
        raise CalculationError(
            "DerivedIndex",
            f"unknown objective '{definition.objective}' on index "
            f"'{definition.index_id}'. Accepted objectives: "
            f"{', '.join(OBJECTIVES)}.")


def calculate_source(source: AnyIndexDefinition,
                     data_provider: DataFetcher,
                     start_date: str | None = None,
                     end_date: str | None = None,
                     price_column: str = "CLOSE") -> IndexResult:
    """The parent's calculation — recursive when the parent is itself derived.

    Public because the parent's published weights are what a derivation *is*
    defined against, so anything reasoning about a derivation — the
    calculation below, the server's preview — needs them, and needs them from
    one place. A chain resolves here rather than at each caller.

    Args:
        source: The parent definition: rule-driven, or another derivation.
        data_provider: Data source for prices, reference data and FX.
        start_date: First date (YYYY-MM-DD). None uses the source's base date.
        end_date: Last date (YYYY-MM-DD). Required by the calculator.
        price_column: Market-data column read as the price.

    Returns:
        IndexResult: The source's own calculation over that window.
    """
    if isinstance(source, OptimisedIndexDefinition):
        return calculate_derived_index(source, data_provider,
                                       start_date=start_date,
                                       end_date=end_date,
                                       price_column=price_column)

    return IndexCalculator(source, data_provider,
                           price_column=price_column).run(start_date=start_date,
                                                          end_date=end_date)


def solve_snapshot(definition: OptimisedIndexDefinition,
                   source_weights: dict[str, float]) -> OptimisationResult:
    """Solve one of the parent's snapshots under the derivation's constraints.

    The single solve of this module, so the schedule below and any caller
    asking "what would this derivation do at that date" — the server's preview
    — cannot disagree about what the answer is. The whole
    :class:`~beacon.optimise.result.OptimisationResult` is returned rather than
    only its weights, because which constraints bound and how much room the
    rest had left is the interesting half of the answer, and re-deriving it
    from the weights afterwards would be a second implementation of the rules.

    scipy is required inside the solve (BN-166); an infeasible constraint set
    raises there with a message naming the binding conflict, which is exactly
    the loud failure the design demands — nothing is caught here.

    Args:
        definition: The derivation supplying the objective and constraints.
        source_weights: The parent's published weights at one rebalance.

    Returns:
        OptimisationResult: Solved weights, binding constraints, every
        constraint's slack, and the solver's diagnostics.

    Raises:
        CalculationError: If the objective is unknown, or the solve is
            infeasible or fails to converge.
    """
    check_objective(definition)

    return minimise_tracking_error(source_weights,
                                   constraints=list(definition.constraints))


def _solved_schedule(definition: OptimisedIndexDefinition,
                     parent: IndexResult) -> dict[pd.Timestamp, dict[str, float]]:
    """The parent's snapshots, each solved under the derivation's constraints."""
    schedule: dict[pd.Timestamp, dict[str, float]] = {}

    for date in sorted(parent.weight_snapshots):
        result = solve_snapshot(definition, parent.weight_snapshots[date])
        schedule[date] = {str(asset): float(value)
                          for asset, value in result.weights.items()}

    logger.info("Solved %d rebalance(s) of '%s' under %d constraint(s) for "
                "'%s'.",
                len(schedule), definition.source.index_id,
                len(definition.constraints), definition.index_id)

    return schedule
