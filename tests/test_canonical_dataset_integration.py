# tests/test_canonical_dataset_integration.py
"""BN-95: the whole pipeline driven off the canonical dataset.

The second independent consumer of `beacon.testing.dataset` — the acceptance
criterion asks for two, and one of them being a real end-to-end run rather than
another unit test is what proves the fixture is actually usable rather than
merely self-consistent.

It also earns its place as a test: index construction, backtesting, risk
estimation, optimisation and attribution all run against one dataset here, so a
change that quietly breaks the seam between two of them fails in this file even
when both of their own test modules still pass.
"""
import pandas as pd
import pytest

from beacon.analysis import attribute, drifted_weights
from beacon.backtest.engine import BacktestEngine
from beacon.index.calculation import IndexCalculator
from beacon.index.constructor import IndexDefinition
from beacon.index.methodology import EqualWeighted, MarketCapWeighted
from beacon.optimise import (
    FullInvestment,
    GroupBounds,
    PositionBounds,
    minimise_tracking_error,
)
from beacon.risk import estimate_risk_model
from beacon.testing import dataset

BASE_VALUE = 1000.0
INITIAL_CAPITAL = 10_000_000.0


def _definition(weighting) -> IndexDefinition:
    """An index over the canonical universe."""
    return IndexDefinition(
        index_id="CANON",
        index_name="Canonical Test Index",
        base_date=dataset.START,
        base_value=BASE_VALUE,
        currency=dataset.BASE_CURRENCY,
        eligibility_rules=[],
        weighting_scheme=weighting,
        rebalancing_frequency="QUARTERLY",
        universe_identifiers=list(dataset.UNIVERSE),
    )


@pytest.fixture(scope="module")
def fetcher():
    return dataset.data_fetcher()


@pytest.fixture(scope="module")
def index_result(fetcher):
    return IndexCalculator(_definition(EqualWeighted()), fetcher).run(
        end_date=dataset.END)


@pytest.fixture(scope="module")
def backtest_result(index_result,
                    fetcher):
    return BacktestEngine(
        start_date=dataset.START,
        end_date=dataset.END,
        initial_capital=INITIAL_CAPITAL,
        data_provider=fetcher,
        index_result=index_result,
        transaction_cost_bps=0.0,
    ).run()


@pytest.fixture(scope="module")
def risk_model():
    return estimate_risk_model(dataset.returns(), intensity=0.1)


class TestIndexConstruction:

    def test_the_index_runs_over_the_whole_span(self,
                                                index_result):
        levels = index_result.index_levels

        assert len(levels) > 700
        assert levels.index[0] == pd.Timestamp(dataset.START)

    def test_it_starts_at_the_base_value(self,
                                         index_result):
        assert index_result.index_levels.iloc[0] == pytest.approx(BASE_VALUE, rel=1e-9)

    def test_every_constituent_is_included(self,
                                           index_result):
        first_rebalance = min(index_result.weight_snapshots)
        weights = index_result.weight_snapshots[first_rebalance]

        assert set(weights) == set(dataset.UNIVERSE)

    def test_weights_sum_to_one_at_every_rebalance(self,
                                                   index_result):
        for date, weights in index_result.weight_snapshots.items():
            assert sum(weights.values()) == pytest.approx(1.0, abs=1e-9), date

    def test_the_level_is_positive_throughout(self,
                                              index_result):
        assert (index_result.index_levels > 0).all()

    def test_market_cap_weighting_gives_a_different_index(self,
                                                          fetcher,
                                                          index_result):
        """The two schemes must disagree, or the weighting is not driving the level.

        This is the invariant BN-103 existed to restore, checked here on the
        canonical data rather than on a bespoke fixture.
        """
        capped = IndexCalculator(_definition(MarketCapWeighted()), fetcher).run(
            end_date=dataset.END)

        assert not capped.index_levels.equals(index_result.index_levels)


class TestBacktest:

    def test_the_nav_series_spans_the_run(self,
                                          backtest_result):
        assert len(backtest_result.trading_nav) > 700

    def test_it_starts_at_the_initial_capital(self,
                                              backtest_result):
        assert backtest_result.trading_nav.iloc[0] == pytest.approx(
            INITIAL_CAPITAL, rel=1e-6)

    def test_trading_actually_happened(self,
                                       backtest_result):
        assert len(backtest_result.portfolio.transactions) > 0

    def test_it_tracks_the_index_closely_at_zero_cost(self,
                                                      backtest_result):
        """Bound measured on this dataset, not assumed.

        Index level and backtest NAV are different constructions and agree
        exactly only when price paths are proportional, so the tolerance is
        empirical — but it is tight enough that a real tracking break would
        show up here.
        """
        tracking_error = backtest_result.get_tracking_error()

        assert tracking_error < 0.05


class TestRiskAndOptimisation:

    def test_the_risk_model_covers_the_universe(self,
                                                risk_model):
        assert set(risk_model.asset_ids) == set(dataset.UNIVERSE)

    def test_the_optimiser_can_track_the_index(self,
                                               risk_model):
        """Equal weights are feasible, so a solve with room reproduces them."""
        target = dataset.equal_weights()

        result = minimise_tracking_error(
            target, [FullInvestment(), PositionBounds(0.0, 1.0)],
            risk_model=risk_model)

        assert result.tracking_error() < 1e-6

    def test_a_sector_cap_forces_a_real_trade_off(self,
                                                  risk_model):
        """The technology pair is 2 of 6 names, so a 20% sector cap binds."""
        target = dataset.equal_weights()
        technology = dataset.sectors()["Technology"]

        result = minimise_tracking_error(
            target,
            [FullInvestment(), PositionBounds(0.0, 1.0),
             GroupBounds("Technology", technology, maximum=0.20)],
            risk_model=risk_model)

        assert result.weights[technology].sum() == pytest.approx(0.20, abs=1e-6)
        assert result.tracking_error() > 0.0

    def test_the_constrained_answer_is_still_fully_invested(self,
                                                            risk_model):
        result = minimise_tracking_error(
            dataset.equal_weights(),
            [FullInvestment(), PositionBounds(0.0, 0.25)],
            risk_model=risk_model)

        assert result.weights.sum() == pytest.approx(1.0, abs=1e-9)


class TestAttribution:

    @pytest.fixture(scope="class")
    def decomposition(self,
                      index_result):
        """The index's own return, decomposed over its drifting weights."""
        prices = dataset.prices()
        weights = drifted_weights(index_result.weight_snapshots, prices)
        asset_returns = prices.pct_change().reindex(weights.index)
        period_returns = (weights.shift(1) * asset_returns).sum(axis=1)

        return attribute(period_returns, weights, asset_returns)

    def test_contributions_reconcile_to_the_index_return(self,
                                                         decomposition):
        """The Carino-linked identity, on the canonical dataset."""
        assert decomposition.reconciles()
        assert abs(decomposition.residual) < 1e-12

    def test_every_constituent_is_accounted_for(self,
                                                decomposition):
        assert {item.asset_id for item in decomposition.contributions} == set(
            dataset.UNIVERSE)

    def test_the_strongest_performer_contributes_most(self,
                                                      decomposition):
        """AAA has the highest total return in the dataset, equally weighted."""
        largest = max(decomposition.contributions,
                      key=lambda item: item.contribution)

        assert largest.asset_id == "AAA"
