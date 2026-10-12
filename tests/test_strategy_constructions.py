# tests/test_strategy_constructions.py
"""BN-281: more ways to build an active portfolio.

One rebalance at a time, on test_strategy_tracking's eighteen names, against
an equal-weighted benchmark. Each construction is checked for what defines it:
minimum variance for the least risk, maximum diversification for the highest
diversification ratio, risk parity for its risk shares, maximum Sharpe for
its ratio, Black-Litterman for holding the benchmark when there are no views,
factor tilts for reaching their targets, and the top-N rule for its names.
"""
import numpy as np
import pandas as pd
import pytest

from beacon.backtest import Backtest
from beacon.strategy import (
    ActiveStrategy,
    BlackLitterman,
    FactorTilts,
    FunctionSignal,
    MaximumDiversification,
    MaximumSharpe,
    MinimumVariance,
    RiskParity,
    StrategyContext,
    TopN,
)
from beacon.strategy.base import covariance_of, estimated_risk, trailing_returns
from test_strategy_tracking import DATA, END, NAMES, START, definition

DATE = pd.Timestamp(START)
BENCHMARK = {name: 1.0 / len(NAMES) for name in NAMES}
CONTEXT = StrategyContext(DATA, "USD")
RANK = {name: position for position, name in enumerate(NAMES)}
SIGNAL = FunctionSignal(lambda name, date, fetcher: float(RANK.get(name, 0)))
FLAT = FunctionSignal(lambda name, date, fetcher: 1.0)


def step(construction, signal=SIGNAL, **settings):
    strategy = ActiveStrategy(benchmark=definition(), signal=signal,
                              construction=construction, **settings)

    return strategy.rebalanced(DATE, dict(BENCHMARK), {}, CONTEXT)


def covariance() -> np.ndarray:
    returns = trailing_returns(NAMES, DATE, CONTEXT, 252)

    return covariance_of(estimated_risk(returns, NAMES), NAMES)


SIGMA = covariance()


def vector(weights: dict[str, float]) -> np.ndarray:
    return np.array([weights.get(name, 0.0) for name in NAMES])


def volatility(weights: dict[str, float]) -> float:
    w = vector(weights)
    return float(np.sqrt(w @ SIGMA @ w))


def diversification(weights: dict[str, float]) -> float:
    w = vector(weights)
    return float(np.sqrt(np.diag(SIGMA)) @ w) / volatility(weights)


ALL = [MinimumVariance(), MaximumDiversification(), RiskParity(), MaximumSharpe(),
       BlackLitterman(), FactorTilts({"rank": SIGNAL}, {"rank": 0.3}), TopN(5)]


class TestEveryConstruction:

    @pytest.mark.parametrize("construction", ALL, ids=lambda c: c.name)
    def test_it_is_long_only_and_fully_invested(self, construction):
        weights = step(construction).weights

        assert min(weights.values()) >= -1e-9
        assert sum(weights.values()) == pytest.approx(1.0, abs=1e-6)

    @pytest.mark.parametrize("construction", ALL, ids=lambda c: c.name)
    def test_its_risk_shares_add_up(self, construction):
        shares = step(construction).risk_contributions

        assert sum(shares.values()) == pytest.approx(1.0, abs=1e-6)

    def test_the_benchmark_free_ones_need_no_signal(self):
        for construction in (MinimumVariance(), MaximumDiversification(), RiskParity()):
            assert step(construction, signal=None).weights

    def test_the_others_do(self):
        with pytest.raises(ValueError, match="signal"):
            ActiveStrategy(benchmark=definition(), construction=TopN(5))


class TestMinimumVariance:

    def test_it_is_the_least_risky(self):
        least = volatility(step(MinimumVariance()).weights)

        assert least < volatility(BENCHMARK)
        assert least < volatility(step(MaximumDiversification()).weights)
        assert least < volatility(step(RiskParity()).weights)


class TestMaximumDiversification:

    def test_it_is_the_most_diversified(self):
        most = diversification(step(MaximumDiversification()).weights)

        assert most >= diversification(BENCHMARK)
        assert most >= diversification(step(MinimumVariance()).weights)


class TestRiskParity:

    def test_every_name_carries_an_equal_share_of_risk(self):
        shares = step(RiskParity()).risk_contributions

        assert len(shares) == len(NAMES)
        assert max(shares.values()) - min(shares.values()) < 2e-3

    def test_budgets_set_each_share(self):
        budgets = {name: (2.0 if position < 3 else 1.0) for position, name in enumerate(NAMES)}
        shares = step(RiskParity(budgets)).risk_contributions
        total = sum(budgets.values())

        for name, budget in budgets.items():
            assert shares[name] == pytest.approx(budget / total, abs=2e-3)

    @pytest.mark.parametrize("budgets", [{"TEC0": -1.0}, {"TEC0": 0.0}])
    def test_budgets_must_be_shares(self, budgets):
        with pytest.raises(ValueError, match="budget"):
            RiskParity(budgets)


class TestMaximumSharpe:

    def test_no_portfolio_tried_has_a_better_ratio(self):
        chosen = step(MaximumSharpe(base_return=0.05, alpha_per_score=0.02))
        expected = vector({name: 0.05 + 0.02 * score
                           for name, score in chosen.scores.items()})

        def sharpe(weights):
            return float(expected @ vector(weights)) / volatility(weights)

        for other in (BENCHMARK, step(MinimumVariance()).weights,
                      step(RiskParity()).weights):
            assert sharpe(chosen.weights) >= sharpe(other) - 1e-6

    def test_settings_out_of_range_are_refused(self):
        with pytest.raises(ValueError, match="MaximumSharpe"):
            MaximumSharpe(base_return=0.0)


class TestBlackLitterman:

    def test_with_no_views_it_holds_the_benchmark(self):
        weights = step(BlackLitterman(), signal=FLAT).weights

        for name in NAMES:
            assert weights.get(name, 0.0) == pytest.approx(BENCHMARK[name], abs=2e-3)

    def test_views_tilt_it_toward_the_better_scores(self):
        weights = step(BlackLitterman(confidence=4.0)).weights

        assert weights.get(NAMES[-1], 0.0) > BENCHMARK[NAMES[-1]]
        assert weights.get(NAMES[0], 0.0) < BENCHMARK[NAMES[0]]

    def test_settings_must_be_positive(self):
        with pytest.raises(ValueError, match="positive"):
            BlackLitterman(tau=0.0)


class TestFactorTilts:

    def test_it_reaches_its_target_exposure(self):
        tilted = step(FactorTilts({"rank": SIGNAL}, {"rank": 0.3}, strength=10.0))

        assert tilted.exposures["rank"] == pytest.approx(0.3, abs=0.02)

    def test_with_no_tilt_it_holds_the_benchmark(self):
        held = step(FactorTilts({"rank": SIGNAL}, {}))

        assert held.exposures["rank"] == pytest.approx(0.0, abs=1e-3)
        assert held.tracking_error < 1e-3

    def test_a_target_must_name_a_factor(self):
        with pytest.raises(ValueError, match="value"):
            FactorTilts({"rank": SIGNAL}, {"value": 0.2})


class TestTopN:

    def test_it_holds_the_top_names_equally(self):
        weights = step(TopN(5)).weights

        assert set(weights) == set(NAMES[-5:])
        assert all(weight == pytest.approx(0.2, abs=1e-6) for weight in weights.values())

    def test_signal_weighting_gives_more_to_the_better_scores(self):
        weights = step(TopN(5, weighting="signal")).weights
        held = [weights[name] for name in NAMES[-5:] if name in weights]

        assert held == sorted(held)

    @pytest.mark.parametrize("settings", [{"n": 0}, {"weighting": "loudest"}])
    def test_settings_out_of_range_are_refused(self, settings):
        with pytest.raises(ValueError):
            TopN(**settings)


class TestInABacktest:

    def test_risk_parity_runs_through_a_backtest(self):
        result = Backtest(initial_capital=1e7, data_provider=DATA, cache=False).run(
            ActiveStrategy(benchmark=definition(), construction=RiskParity(),
                           rebalancing="QUARTERLY"), start=START, end=END)

        assert len(result.active) == 4
        assert all(step.risk_contributions for step in result.active)
