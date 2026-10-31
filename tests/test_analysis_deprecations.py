# tests/test_analysis_deprecations.py
"""The analysis helpers deprecated in 0.10.0 keep working, warn once each
call, and name the formula that replaces them."""
import warnings

import numpy as np
import pandas as pd
import pytest

from beacon.analysis import (
    RiskMetricsCalculator,
    calculate_max_drawdown,
    calculate_sharpe_ratio,
    calculate_volatility,
    simple_performance_attribution,
)
from beacon.analysis.performance import formulas

DATES = pd.bdate_range("2024-01-01", periods=60)
RETURNS = pd.Series(np.random.default_rng(3).normal(0.0005, 0.01, len(DATES)), index=DATES)
PRICES = 100.0 * (1.0 + RETURNS).cumprod()
BENCHMARK = RETURNS * 0.8


class TestTheFunctionsWarn:
    def test_volatility(self):
        with pytest.warns(DeprecationWarning, match="formulas.volatility"):
            found = calculate_volatility(PRICES, 252)

        assert found == formulas.volatility(PRICES.pct_change().dropna(), 252)

    def test_sharpe_ratio(self):
        with pytest.warns(DeprecationWarning, match="formulas.sharpe_ratio"):
            found = calculate_sharpe_ratio(RETURNS, 0.02, 252)

        assert found == formulas.sharpe_ratio(RETURNS, 252, 0.02)

    def test_max_drawdown(self):
        with pytest.warns(DeprecationWarning, match="formulas.max_drawdown"):
            found = calculate_max_drawdown(PRICES)

        assert found == pytest.approx(formulas.max_drawdown(PRICES), abs=1e-15)

    def test_simple_performance_attribution(self):
        with pytest.warns(DeprecationWarning, match="compounded_return"):
            found = simple_performance_attribution(RETURNS, BENCHMARK)

        assert found["active_return"] == pytest.approx(
            formulas.compounded_return(RETURNS) - formulas.compounded_return(BENCHMARK),
            abs=1e-15)

    def test_each_names_the_release_it_goes_in(self):
        with pytest.warns(DeprecationWarning, match="removed in 0.11.0"):
            calculate_max_drawdown(PRICES)


class TestTheCalculator:
    def test_warns_when_made(self):
        with pytest.warns(DeprecationWarning, match="RiskMetricsCalculator"):
            RiskMetricsCalculator()

    def test_its_methods_do_not_warn_again(self):
        with pytest.warns(DeprecationWarning):
            calculator = RiskMetricsCalculator()

        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)

            assert calculator.calculate_volatility(PRICES) == pytest.approx(
                formulas.volatility(PRICES.pct_change().dropna(), 252))
            assert calculator.calculate_sharpe_ratio(RETURNS, 0.0) == pytest.approx(
                formulas.sharpe_ratio(RETURNS, 252))
            assert calculator.calculate_max_drawdown(PRICES) <= 0.0


def test_importing_them_does_not_warn():
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)

        import beacon.analysis.risk  # noqa: F401
