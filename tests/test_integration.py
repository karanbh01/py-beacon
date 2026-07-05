# tests/test_integration.py
"""End-to-end integration test for the full Beacon pipeline.

Exercises the whole stack with synthetic data only:
Environment/DataFetcher -> IndexDefinition -> IndexCalculator.run() (IndexResult)
-> BacktestEngine.run() (BacktestResult) -> tracking comparison.

Both the calculator and the engine drive off a single plain DataFetcher: prices
come from the CLOSE market-data column and shares outstanding from the
SHARES_OUTSTANDING market-data column.
"""
import pandas as pd
import pytest

from beacon.environment.config import Environment
from beacon.data.base import MarketData, ReferenceData
from beacon.data.fetcher import DataFetcher
from beacon.index.constructor import IndexDefinition
from beacon.index.methodology import EqualWeighted
from beacon.index.calculation import IndexCalculator
from beacon.index.result import IndexResult
from beacon.backtest.engine import BacktestEngine
from beacon.backtest.result import BacktestResult
from beacon.portfolio.base import Transaction


# ---------------------------------------------------------------------------
# Synthetic universe: 3 assets, ~6 months of daily prices, quarterly rebalance
# ---------------------------------------------------------------------------
ASSETS = ["AAA", "BBB", "CCC"]
BASE_DATE = "2024-01-01"
END_DATE = "2024-06-28"
BASE_VALUE = 1000.0
INITIAL_CAPITAL = 1_000_000.0

TRADING_DAYS = pd.bdate_range(start=BASE_DATE, end=END_DATE)
N_DAYS = len(TRADING_DAYS)

# Base prices and modest, divergent total returns over the window (so the
# quarterly rebalance actually trades) plus fixed share counts.
_BASE_PRICE = {"AAA": 100.0, "BBB": 50.0, "CCC": 200.0}
_TOTAL_RETURN = {"AAA": 0.08, "BBB": 0.02, "CCC": 0.05}
_SHARES = {"AAA": 1000, "BBB": 2000, "CCC": 500}


def _price(asset_id: str, day: pd.Timestamp) -> float:
    """Deterministic geometric price path for *asset_id* on *day*."""
    frac = TRADING_DAYS.get_loc(day) / (N_DAYS - 1)
    return _BASE_PRICE[asset_id] * ((1 + _TOTAL_RETURN[asset_id]) ** frac)


def _market_dataframe() -> pd.DataFrame:
    """Long-form market data with IDENTIFIER, DATE, CLOSE, SHARES_OUTSTANDING."""
    rows = [
        {
            "IDENTIFIER": a, "DATE": d.strftime("%Y-%m-%d"),
            "CLOSE": _price(a, d), "SHARES_OUTSTANDING": _SHARES[a],
        }
        for a in ASSETS
        for d in TRADING_DAYS
    ]
    return pd.DataFrame(rows)


def _reference_dataframe() -> pd.DataFrame:
    """Reference data with the fields IndexCalculator resolves a universe from."""
    rows = [
        {
            "IDENTIFIER": a, "NAME": f"Asset {a}", "CURRENCY": "USD",
            "EXCHANGE": "NYSE", "DATE_FROM": "2020-01-01", "DATE_TO": pd.NaT,
        }
        for a in ASSETS
    ]
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Pipeline fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def environment():
    """Environment holding the synthetic data and transaction-cost setting."""
    env = Environment()
    env.set_environment(
        MARKET_DATA=_market_dataframe(),
        REFERENCE_DATA=_reference_dataframe(),
        TRANSACTION_COST=50.0,  # basis points, used by the costed run
    )
    return env


@pytest.fixture(scope="module")
def fetcher(environment):
    market = MarketData.from_dataframe(environment.data_source.MARKET_DATA)
    reference = ReferenceData.from_dataframe(environment.data_source.REFERENCE_DATA)
    return DataFetcher(market, reference)


@pytest.fixture(scope="module")
def definition():
    return IndexDefinition(
        index_id="ITEST",
        index_name="Integration Test EW",
        base_date=BASE_DATE,
        base_value=BASE_VALUE,
        currency="USD",
        eligibility_rules=[],
        weighting_scheme=EqualWeighted(),
        rebalancing_frequency="QUARTERLY",
        universe_identifiers=ASSETS,
    )


@pytest.fixture(scope="module")
def index_result(definition, fetcher):
    return IndexCalculator(definition, fetcher).run(end_date=END_DATE)


def _backtest(index_result, fetcher, cost_bps):
    return BacktestEngine(
        start_date=BASE_DATE,
        end_date=END_DATE,
        initial_capital=INITIAL_CAPITAL,
        data_provider=fetcher,
        target_index_result=index_result,
        transaction_cost_bps=cost_bps,
    ).run()


@pytest.fixture(scope="module")
def zero_cost_result(index_result, fetcher):
    return _backtest(index_result, fetcher, 0.0)


@pytest.fixture(scope="module")
def costed_result(index_result, fetcher, environment):
    return _backtest(index_result, fetcher, environment.simulation.TRANSACTION_COST)


# ---------------------------------------------------------------------------
# Setup / types
# ---------------------------------------------------------------------------

class TestPipelineSetup:

    def test_environment_and_fetcher_types(self, environment, fetcher):
        assert isinstance(environment, Environment)
        assert isinstance(fetcher, DataFetcher)
        assert environment.simulation.TRANSACTION_COST == 50.0

    def test_index_result_shape_and_types(self, index_result):
        assert isinstance(index_result, IndexResult)
        assert isinstance(index_result.index_levels, pd.Series)
        assert len(index_result.index_levels) == N_DAYS
        # Base level equals the configured base value.
        assert index_result.index_levels.iloc[0] == pytest.approx(BASE_VALUE)
        # Quarterly over ~6 months => base rebalance + one more.
        assert len(index_result.weight_snapshots) == 2
        for weights in index_result.weight_snapshots.values():
            assert set(weights) == set(ASSETS)
            assert sum(weights.values()) == pytest.approx(1.0)

    def test_backtest_result_shape_and_types(self, zero_cost_result):
        r = zero_cost_result
        assert isinstance(r, BacktestResult)
        assert isinstance(r.portfolio_nav, pd.Series)
        assert isinstance(r.cash_history, pd.Series)
        assert isinstance(r.actual_weight_history, pd.DataFrame)
        assert len(r.portfolio_nav) == N_DAYS
        assert len(r.cash_history) == N_DAYS
        assert r.transactions and all(isinstance(t, Transaction) for t in r.transactions)
        assert r.target_index_result is not None


# ---------------------------------------------------------------------------
# Tracking: zero-cost and non-zero-cost scenarios
# ---------------------------------------------------------------------------

class TestTracking:

    def test_zero_cost_tracking_error_below_1bp(self, zero_cost_result):
        te = zero_cost_result.get_tracking_error()
        assert te is not None
        assert te < 1e-4  # < 1 basis point (annualised)

    def test_nonzero_cost_tracking_difference_negative(self, costed_result):
        td = costed_result.get_tracking_difference()
        assert td is not None
        assert td < 0.0

    def test_costs_worsen_tracking(self, zero_cost_result, costed_result):
        # Trading costs drag on the portfolio: lower final NAV and a more
        # negative tracking difference than the zero-cost run.
        assert costed_result.portfolio_nav.iloc[-1] < zero_cost_result.portfolio_nav.iloc[-1]
        assert costed_result.get_tracking_difference() < zero_cost_result.get_tracking_difference()

    def test_summary_reports_metrics(self, zero_cost_result):
        summary = zero_cost_result.summary()
        for key in ("total_return", "annualised_return", "volatility",
                    "sharpe_ratio", "max_drawdown", "tracking_error",
                    "tracking_difference"):
            assert key in summary
            assert summary[key] is not None


# ---------------------------------------------------------------------------
# Full-pipeline sanity
# ---------------------------------------------------------------------------

class TestPipelineConsistency:

    def test_transactions_only_on_rebalance_dates(self, zero_cost_result, index_result):
        rebalance_dates = set(index_result.weight_snapshots.keys())
        txn_dates = {t.transaction_date for t in zero_cost_result.transactions}
        assert txn_dates.issubset(rebalance_dates)

    def test_nav_starts_at_initial_capital(self, zero_cost_result):
        assert zero_cost_result.portfolio_nav.iloc[0] == pytest.approx(INITIAL_CAPITAL, rel=1e-6)

    def test_index_and_portfolio_share_dates(self, zero_cost_result, index_result):
        assert list(zero_cost_result.portfolio_nav.index) == list(index_result.index_levels.index)
