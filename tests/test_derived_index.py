# tests/test_derived_index.py
"""BN-167: the optimised index as a derivation — definition, calculation, run.

Three claims carry the weight. First, the calculation is economically honest:
the solved weights satisfy the constraints at every one of the PARENT's
rebalance dates, and the level path equals compounding the solved-weight
portfolio's returns from the parent's price data — recomputed here
independently, segment by segment, from the raw prices. Second, ad-hoc and
stored are one thing in two lifetimes: `run(optimised=True, config)` on a
plain definition produces bit-identical numbers to the equivalent stored
`OptimisedIndexDefinition`. Third, the solve is deterministic — same inputs,
identical weights — which is what makes caching the derived result honest.
"""
import logging

import pandas as pd
import pytest

from beacon.backtest.main import Backtest
from beacon.exceptions import CalculationError
from beacon.index.constructor import IndexDefinition
from beacon.index.derived import OptimisedIndexDefinition, calculate_derived_index
from beacon.index.methodology import EqualWeighted
from beacon.index.result import IndexResult
from beacon.optimise import (
    FullInvestment,
    OptimisationConfig,
    PositionBounds,
)
from beacon.testing import dataset

pytest.importorskip("scipy")

# A short window inside the canonical dataset: three single-currency names,
# one quarter, three monthly rebalances. AAA/BBB/CCC are all USD, so the
# independent level recomputation below needs no FX arithmetic.
UNIVERSE = ["AAA", "BBB", "CCC"]
START = "2024-01-02"
END = "2024-03-28"
CAPITAL = 1_000_000.0

# Forcing 40% into one name of an equal-weight three: genuinely binding, so
# the solved book differs from the parent and the constraint has teeth.
BINDING = (FullInvestment(),
           PositionBounds(minimum=0.4, maximum=1.0, assets=["AAA"]))


def build_parent(universe: list[str] | None = None) -> IndexDefinition:
    return IndexDefinition(
        index_id="PARENT-IX",
        index_name="Parent Index",
        base_date=START,
        base_value=1000.0,
        currency="USD",
        eligibility_rules=[],
        weighting_scheme=EqualWeighted(),
        rebalancing_frequency="MONTHLY",
        universe_identifiers=universe if universe is not None else UNIVERSE)


def build_derived(constraints=BINDING,
                  source: IndexDefinition | OptimisedIndexDefinition | None = None,
                  **kwargs) -> OptimisedIndexDefinition:
    return OptimisedIndexDefinition(
        index_id=kwargs.pop("index_id", "DERIVED-IX"),
        index_name=kwargs.pop("index_name", "Derived Index"),
        source=source if source is not None else build_parent(),
        constraints=constraints,
        **kwargs)


def calculated(definition: OptimisedIndexDefinition,
               **kwargs) -> IndexResult:
    """One derived calculation with the pipeline's INFO chatter suppressed."""
    logging.disable(logging.ERROR)
    try:
        return calculate_derived_index(definition,
                                       dataset.data_fetcher(),
                                       start_date=kwargs.pop("start_date", START),
                                       end_date=kwargs.pop("end_date", END),
                                       **kwargs)
    finally:
        logging.disable(logging.NOTSET)


@pytest.fixture(scope="module")
def derived_result() -> IndexResult:
    """One real derived calculation, shared read-only across the module."""
    return calculated(build_derived())


class TestDerivedDefinition:
    """Identity inherits from the source unless overridden; misuse is loud."""

    def test_identity_defaults_inherit_the_source(self):
        derived = build_derived()

        assert derived.base_date == pd.Timestamp(START)
        assert derived.base_value == 1000.0
        assert derived.currency == "USD"
        assert derived.universe_identifiers == UNIVERSE

    def test_explicit_identity_wins_over_the_source(self):
        derived = build_derived(base_date="2024-02-01",
                                base_value=500.0,
                                currency="usd")

        assert derived.base_date == pd.Timestamp("2024-02-01")
        assert derived.base_value == 500.0
        assert derived.currency == "USD"

    def test_construction_rejects_the_obvious_mistakes(self):
        with pytest.raises(ValueError, match="index_id"):
            build_derived(index_id="")

        with pytest.raises(ValueError, match="index_name"):
            build_derived(index_name="")

        with pytest.raises(ValueError, match="base_value"):
            build_derived(base_value=-1.0)

        with pytest.raises(ValueError, match="source"):
            OptimisedIndexDefinition(index_id="X", index_name="X", source=None)

    def test_from_config_maps_the_whole_derivation(self):
        config = OptimisationConfig(constraints=list(BINDING))

        derived = OptimisedIndexDefinition.from_config(
            index_id="FROM-CFG", index_name="From Config",
            source=build_parent(), config=config)

        assert derived.objective == config.objective
        assert derived.constraints == tuple(BINDING)
        assert derived.risk_model is None

    def test_repr_names_the_source(self):
        assert "PARENT-IX" in repr(build_derived())


class TestDerivedCalculation:
    """Solved at the parent's dates, chained into an honest level path."""

    def test_an_unknown_objective_is_refused_naming_the_accepted(self):
        derived = build_derived(objective="min_variance")

        with pytest.raises(CalculationError) as raised:
            calculated(derived)

        message = str(raised.value)

        assert "min_variance" in message
        assert "min_tracking_error" in message

    def test_rebalances_are_exactly_the_parents(self,
                                                derived_result):
        """The child inherits the parent's schedule: same snapshot dates, no
        schedule of its own."""
        logging.disable(logging.ERROR)
        try:
            from beacon.index.calculation import IndexCalculator

            parent = IndexCalculator(build_parent(),
                                     dataset.data_fetcher()).run(
                start_date=START, end_date=END)
        finally:
            logging.disable(logging.NOTSET)

        assert (sorted(derived_result.weight_snapshots)
                == sorted(parent.weight_snapshots))
        pd.testing.assert_index_equal(derived_result.index_levels.index,
                                      parent.index_levels.index)

    def test_solved_weights_satisfy_the_constraints_at_every_rebalance(self,
                                                                       derived_result):
        for date, weights in derived_result.weight_snapshots.items():
            assert weights["AAA"] >= 0.4 - 1e-6, f"floor broken on {date}"
            assert sum(weights.values()) == pytest.approx(1.0, abs=1e-6)

    def test_levels_compound_the_solved_portfolio_returns(self,
                                                          derived_result):
        """The economic consistency claim, recomputed independently: from the
        base value, each inter-rebalance segment multiplies the level by the
        solved-weight portfolio's return, read straight off the raw prices."""
        prices = dataset.prices()[UNIVERSE]
        rebalances = sorted(derived_result.weight_snapshots)

        expected = 1000.0
        for position, rebalance in enumerate(rebalances):
            weights = derived_result.weight_snapshots[rebalance]
            segment_end = (rebalances[position + 1]
                           if position + 1 < len(rebalances)
                           else derived_result.index_levels.index[-1])

            segment = derived_result.index_levels.loc[rebalance:segment_end]
            for date in segment.index:
                relative = sum(weights[asset]
                               * prices.at[date, asset] / prices.at[rebalance, asset]
                               for asset in weights)

                assert segment[date] == pytest.approx(expected * relative,
                                                      rel=1e-9), date

            expected *= sum(weights[asset]
                            * prices.at[segment_end, asset] / prices.at[rebalance, asset]
                            for asset in weights)

    def test_the_level_starts_at_the_base_value(self,
                                                derived_result):
        assert float(derived_result.index_levels.iloc[0]) == pytest.approx(1000.0)

    def test_the_divisor_is_the_identity_of_the_chained_path(self,
                                                             derived_result):
        """The aggregate this index represents is its own portfolio value, so
        the divisor initialises to 1.0 and never moves."""
        assert (derived_result.divisor_history == 1.0).all()

    def test_the_daily_panel_records_weights_that_sum_to_one(self,
                                                             derived_result):
        panel = derived_result.daily_weights

        assert not panel.empty

        totals = panel.groupby("DATE", observed=True)["WEIGHT"].sum()

        assert (abs(totals - 1.0) < 1e-5).all()

    def test_a_presupplied_parent_result_changes_nothing(self,
                                                         derived_result):
        """The Backtest integration hands over its cached parent calculation;
        the answer must be what a self-calculated parent gives, bit for bit."""
        logging.disable(logging.ERROR)
        try:
            from beacon.index.calculation import IndexCalculator

            parent = IndexCalculator(build_parent(),
                                     dataset.data_fetcher()).run(
                start_date=START, end_date=END)
        finally:
            logging.disable(logging.NOTSET)

        reused = calculated(build_derived(), parent_result=parent)

        assert reused.index_levels.equals(derived_result.index_levels)
        assert reused.weight_snapshots == derived_result.weight_snapshots

    def test_two_independent_solves_are_bit_identical(self,
                                                      derived_result):
        """Determinism, pinned: the cache's honesty depends on the solver
        returning the identical vector for identical inputs."""
        again = calculated(build_derived())

        assert again.weight_snapshots == derived_result.weight_snapshots
        assert again.index_levels.equals(derived_result.index_levels)
        pd.testing.assert_frame_equal(again.daily_weights,
                                      derived_result.daily_weights)

    def test_infeasible_constraints_fail_loudly(self):
        """Caps totalling 60% cannot reach full investment; the solver's
        message names the conflict rather than returning a fudged answer."""
        derived = build_derived(constraints=[FullInvestment(),
                                             PositionBounds(minimum=0.0,
                                                            maximum=0.2)])

        with pytest.raises(CalculationError, match="cannot reach"):
            calculated(derived)

    def test_a_parent_with_no_snapshots_is_refused(self):
        hollow = IndexResult(index_id="HOLLOW",
                             index_levels=pd.Series(dtype=float),
                             divisor_history=pd.Series(dtype=float),
                             constituent_snapshots={},
                             weight_snapshots={})

        with pytest.raises(CalculationError, match="no rebalance snapshots"):
            calculated(build_derived(), parent_result=hollow)


class TestChainedOptimisation:
    """A derivation on a derivation calculates through the same recursion."""

    def test_a_two_level_chain_calculates(self):
        inner = build_derived(constraints=[FullInvestment(),
                                           PositionBounds(minimum=0.0,
                                                          maximum=0.4)],
                              index_id="INNER-IX", index_name="Inner")
        outer = build_derived(constraints=list(BINDING), source=inner,
                              index_id="OUTER-IX", index_name="Outer")

        result = calculated(outer)

        assert result.index_id == "OUTER-IX"
        assert float(result.index_levels.iloc[0]) == pytest.approx(1000.0)
        assert len(result.weight_snapshots) == 3

        for weights in result.weight_snapshots.values():
            assert weights["AAA"] >= 0.4 - 1e-6
            assert sum(weights.values()) == pytest.approx(1.0, abs=1e-6)


class TestOptimisedBacktest:
    """The owner's table: benchmark, index.target, index.optimised, portfolio."""

    def backtest(self,
                 benchmark: pd.Series | None = None) -> Backtest:
        return Backtest(initial_capital=CAPITAL,
                        data_provider=dataset.data_fetcher(),
                        benchmark=benchmark)

    def run(self,
            backtest: Backtest,
            definition,
            **kwargs):
        logging.disable(logging.ERROR)
        try:
            return backtest.run(definition, start=START, end=END, **kwargs)
        finally:
            logging.disable(logging.NOTSET)

    def test_a_stored_optimised_definition_fills_all_four_books(self):
        benchmark = pd.Series([100.0, 101.0, 102.5],
                              index=pd.bdate_range(START, periods=3))

        result = self.run(self.backtest(benchmark), build_derived())

        assert result.benchmark is not None
        assert result.index.target is not None
        assert result.index.optimised is not None
        assert not result.portfolio.nav.empty
        assert result.index.tracked is result.index.optimised

    def test_the_books_carry_the_right_calculations(self):
        """`index.target` is the parent's own calculation; `index.optimised`
        is the solved chain — different books, because the constraint binds."""
        result = self.run(self.backtest(), build_derived())

        assert result.index.target.source.index_id == "PARENT-IX"
        assert result.index.optimised.source.index_id == "DERIVED-IX"
        assert not result.index.target.levels.equals(
            result.index.optimised.levels)

        # The optimised book holds the floor the target book does not.
        for weights in result.index.optimised.source.weight_snapshots.values():
            assert weights["AAA"] >= 0.4 - 1e-6

    def test_an_ad_hoc_run_equals_the_stored_definition_run_bit_identically(self):
        """One thing in two lifetimes: the ephemeral derivation and the stored
        one calculate through the identical path."""
        config = OptimisationConfig(constraints=list(BINDING))

        ad_hoc = self.run(self.backtest(), build_parent(),
                          optimised=True, optimisation_config=config)
        stored = self.run(self.backtest(), build_derived())

        assert ad_hoc.trading_nav.equals(stored.trading_nav)
        assert ad_hoc.index.optimised.levels.equals(
            stored.index.optimised.levels)
        assert ad_hoc.index.target.levels.equals(stored.index.target.levels)
        assert (ad_hoc.index.optimised.source.weight_snapshots
                == stored.index.optimised.source.weight_snapshots)

    def test_the_flag_without_a_config_and_the_reverse_both_raise(self):
        backtest = self.backtest()

        with pytest.raises(ValueError, match="optimisation_config"):
            backtest.run(build_parent(), start=START, end=END, optimised=True)

        with pytest.raises(ValueError, match="optimised"):
            backtest.run(build_parent(), start=START, end=END,
                         optimisation_config=OptimisationConfig())

    def test_a_plain_run_leaves_the_optimised_book_empty(self):
        result = self.run(self.backtest(), build_parent())

        assert result.index.target is not None
        assert result.index.optimised is None
