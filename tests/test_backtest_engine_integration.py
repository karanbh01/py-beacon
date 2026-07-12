# tests/test_backtest_engine_integration.py
"""Integration test for the full BacktestEngine pipeline.

Feeds a synthetic ``IndexResult`` into the engine through a real
``DataFetcher`` (backed by ``MarketData.from_dataframe``) — no mocks — and
verifies that portfolio NAV tracks the index, that transaction costs create a
tracking difference, and that trades fire on rebalance dates with sells before
buys.
"""
import pytest
import pandas as pd
import numpy as np

from beacon.backtest.engine import BacktestEngine
from beacon.backtest.result import BacktestResult
from beacon.data.base import MarketData
from beacon.data.fetcher import DataFetcher
from beacon.index.result import IndexResult


# ---------------------------------------------------------------------------
# Synthetic universe: 2 assets, equal weight, monthly rebalance, ~3 months
# ---------------------------------------------------------------------------
ASSETS = ["ASSET_A", "ASSET_B"]

BASE_DATE = "2024-01-02"   # First business day of Jan 2024
END_DATE = "2024-03-29"    # ~3 months of business days
BASE_VALUE = 1000.0
INITIAL_CAPITAL = 1000.0   # matches BASE_VALUE so NAV and index level are comparable

# Monthly rebalance dates (all business days within the range)
REBALANCE_DATES = [pd.Timestamp(d) for d in ("2024-01-02", "2024-02-01", "2024-03-01")]

TRADING_DAYS = pd.bdate_range(start=BASE_DATE, end=END_DATE, freq="B")


def _price(asset_id: str,
           i: int,
           n: int) -> float:
    """Deterministic geometric price path for day index *i* of *n*.

    ASSET_A drifts +10% over the window, ASSET_B drifts +20%. The differing
    trajectories mean equal-weighting and rebalancing genuinely matter.
    """
    frac = i / (n - 1)
    if asset_id == "ASSET_A":
        return 100.0 * (1.10 ** frac)
    return 50.0 * (1.20 ** frac)


def _build_price_lookup() -> dict:
    """Return {(asset_id, Timestamp) -> price} for every trading day."""
    n = len(TRADING_DAYS)
    return {
        (asset_id, day): _price(asset_id, i, n)
        for asset_id in ASSETS
        for i, day in enumerate(TRADING_DAYS)
    }


PRICE_LOOKUP = _build_price_lookup()


def _make_fetcher() -> DataFetcher:
    """Build a real DataFetcher over synthetic CLOSE prices."""
    rows = [
        {"IDENTIFIER": asset_id, "DATE": day.strftime("%Y-%m-%d"), "CLOSE": price}
        for (asset_id, day), price in PRICE_LOOKUP.items()
    ]
    market = MarketData.from_dataframe(pd.DataFrame(rows))
    return DataFetcher(market)


def _reference_index_levels() -> pd.Series:
    """Compute an equal-weighted, monthly-rebalanced index off the same prices.

    This mirrors a zero-cost fractional-share rebalanced portfolio, so the
    zero-cost backtest NAV should track it almost exactly. It is an independent
    reference — it does not use BacktestEngine.
    """
    rebal = set(REBALANCE_DATES)
    weight = 1.0 / len(ASSETS)
    units = {a: 0.0 for a in ASSETS}
    value = BASE_VALUE
    levels = {}

    for day in TRADING_DAYS:
        # Mark to market with the units carried in from the prior day.
        if any(units[a] != 0.0 for a in ASSETS):
            value = sum(units[a] * PRICE_LOOKUP[(a, day)] for a in ASSETS)
        # Rebalance back to equal weight on rebalance dates.
        if day in rebal:
            for a in ASSETS:
                units[a] = (value * weight) / PRICE_LOOKUP[(a, day)]
            value = sum(units[a] * PRICE_LOOKUP[(a, day)] for a in ASSETS)
        levels[day] = value

    series = pd.Series(levels, dtype=float)
    series.index.name = "Date"
    return series


def _make_index_result() -> IndexResult:
    """Build an IndexResult with equal weights on each rebalance date."""
    levels = _reference_index_levels()
    weight_snapshots = {
        d: {a: 1.0 / len(ASSETS) for a in ASSETS} for d in REBALANCE_DATES
    }
    constituent_snapshots = {d: list(ASSETS) for d in REBALANCE_DATES}
    return IndexResult(
        index_id="TEST_EW_IDX",
        index_levels=levels,
        divisor_history=pd.Series(1.0, index=levels.index),
        constituent_snapshots=constituent_snapshots,
        weight_snapshots=weight_snapshots,
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def index_result():
    return _make_index_result()


@pytest.fixture
def fetcher():
    return _make_fetcher()


@pytest.fixture
def zero_cost_result(index_result,
                     fetcher):
    engine = BacktestEngine(
        start_date=BASE_DATE,
        end_date=END_DATE,
        initial_capital=INITIAL_CAPITAL,
        data_provider=fetcher,
        target_index_result=index_result,
        transaction_cost_bps=0.0,
    )
    return engine.run()


@pytest.fixture
def costed_result(index_result,
                  fetcher):
    engine = BacktestEngine(
        start_date=BASE_DATE,
        end_date=END_DATE,
        initial_capital=INITIAL_CAPITAL,
        data_provider=fetcher,
        target_index_result=index_result,
        transaction_cost_bps=50.0,  # 0.5% per trade
    )
    return engine.run()


# ---------------------------------------------------------------------------
# Full pipeline against an IndexResult
# ---------------------------------------------------------------------------

class TestPipelineAgainstIndex:

    def test_returns_backtest_result(self,
                                     zero_cost_result):
        assert isinstance(zero_cost_result, BacktestResult)

    def test_covers_all_trading_days(self,
                                     zero_cost_result):
        assert len(zero_cost_result.portfolio_nav) == len(TRADING_DAYS)

    def test_first_nav_matches_base_value(self,
                                          zero_cost_result):
        """Fully invested on day 0 -> NAV equals initial capital."""
        assert zero_cost_result.portfolio_nav.iloc[0] == pytest.approx(BASE_VALUE, rel=1e-9)

    def test_zero_cost_nav_tracks_index(self,
                                        zero_cost_result,
                                        index_result):
        """Zero-cost NAV should closely replicate the index level path."""
        nav = zero_cost_result.portfolio_nav
        levels = index_result.index_levels.reindex(nav.index)
        rel_dev = ((nav - levels) / levels).abs()
        assert rel_dev.max() < 1e-6

    def test_zero_cost_tracking_error_near_zero(self,
                                                zero_cost_result):
        te = zero_cost_result.get_tracking_error()
        assert te is not None
        assert te == pytest.approx(0.0, abs=1e-6)

    def test_zero_cost_tracking_difference_near_zero(self,
                                                     zero_cost_result):
        td = zero_cost_result.get_tracking_difference()
        assert td is not None
        assert td == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# Transaction costs create tracking difference
# ---------------------------------------------------------------------------

class TestTransactionCosts:

    def test_costed_nav_lags_zero_cost(self,
                                       costed_result,
                                       zero_cost_result):
        """Costs drag on performance -> final NAV must be lower."""
        assert costed_result.portfolio_nav.iloc[-1] < zero_cost_result.portfolio_nav.iloc[-1]

    def test_costed_nav_lags_index(self,
                                   costed_result,
                                   index_result):
        assert costed_result.portfolio_nav.iloc[-1] < index_result.index_levels.iloc[-1]

    def test_costed_tracking_difference_negative(self,
                                                 costed_result):
        td = costed_result.get_tracking_difference()
        assert td is not None
        assert td < 0.0

    def test_transaction_costs_recorded(self,
                                        costed_result):
        total_cost = sum(t.transaction_cost for t in costed_result.transactions)
        assert total_cost > 0.0


# ---------------------------------------------------------------------------
# Transaction timing and ordering
# ---------------------------------------------------------------------------

class TestTransactionTimingAndOrdering:

    def test_transactions_only_on_rebalance_dates(self,
                                                  zero_cost_result):
        txn_dates = {t.transaction_date for t in zero_cost_result.transactions}
        assert txn_dates.issubset(set(REBALANCE_DATES))

    def test_every_rebalance_date_has_transactions(self,
                                                   zero_cost_result):
        txn_dates = {t.transaction_date for t in zero_cost_result.transactions}
        assert txn_dates == set(REBALANCE_DATES)

    def test_first_rebalance_is_all_buys(self,
                                         zero_cost_result):
        """Investing from cash on day 0 produces buys for both assets."""
        first = [t for t in zero_cost_result.transactions
                 if t.transaction_date == REBALANCE_DATES[0]]
        assert len(first) == 2
        assert all(t.transaction_type == "BUY" for t in first)
        assert {t.asset_id for t in first} == set(ASSETS)

    def test_sells_before_buys_on_each_rebalance(self,
                                                 zero_cost_result):
        """Within any rebalance date, all sells are appended before buys."""
        txns = zero_cost_result.transactions
        for rdate in REBALANCE_DATES:
            positions = [i for i, t in enumerate(txns) if t.transaction_date == rdate]
            sides = [txns[i].transaction_type for i in positions]
            sell_idx = [i for i, s in enumerate(sides) if s == "SELL"]
            buy_idx = [i for i, s in enumerate(sides) if s == "BUY"]
            if sell_idx and buy_idx:
                assert max(sell_idx) < min(buy_idx), f"buy before sell on {rdate}"

    def test_later_rebalance_rotates_weights(self,
                                             zero_cost_result):
        """Price drift makes ASSET_B overweight -> sell B, buy A at rebalance."""
        second = [t for t in zero_cost_result.transactions
                  if t.transaction_date == REBALANCE_DATES[1]]
        sells = [t for t in second if t.transaction_type == "SELL"]
        buys = [t for t in second if t.transaction_type == "BUY"]
        assert any(t.asset_id == "ASSET_B" for t in sells)
        assert any(t.asset_id == "ASSET_A" for t in buys)


# ---------------------------------------------------------------------------
# Custom weight dict as target (not an IndexResult)
# ---------------------------------------------------------------------------

class TestCustomWeightTarget:

    def test_runs_with_custom_weights(self,
                                      fetcher):
        target_weights = {
            d: {a: 1.0 / len(ASSETS) for a in ASSETS} for d in REBALANCE_DATES
        }
        engine = BacktestEngine(
            start_date=BASE_DATE,
            end_date=END_DATE,
            initial_capital=INITIAL_CAPITAL,
            data_provider=fetcher,
            target_weights=target_weights,
        )
        result = engine.run()

        assert isinstance(result, BacktestResult)
        assert len(result.transactions) > 0
        # No index bound -> no target-relative metrics.
        assert result.target_index_result is None

    def test_custom_weights_match_index_when_equal(self,
                                                   fetcher):
        """A custom equal-weight schedule reproduces the equal-weight index NAV."""
        target_weights = {
            d: {a: 1.0 / len(ASSETS) for a in ASSETS} for d in REBALANCE_DATES
        }
        engine = BacktestEngine(
            start_date=BASE_DATE, end_date=END_DATE,
            initial_capital=INITIAL_CAPITAL, data_provider=fetcher,
            target_weights=target_weights,
        )
        result = engine.run()
        levels = _reference_index_levels().reindex(result.portfolio_nav.index)
        rel_dev = ((result.portfolio_nav - levels) / levels).abs()
        assert rel_dev.max() < 1e-6


# ---------------------------------------------------------------------------
# BacktestResult metrics
# ---------------------------------------------------------------------------

class TestResultMetrics:

    def test_summary_has_core_metrics(self,
                                      zero_cost_result):
        summary = zero_cost_result.summary()
        for key in ("total_return", "annualised_return", "volatility",
                    "sharpe_ratio", "max_drawdown"):
            assert key in summary
            assert summary[key] is not None

    def test_summary_includes_tracking_metrics_with_target(self,
                                                           zero_cost_result):
        summary = zero_cost_result.summary()
        assert "tracking_error" in summary
        assert "tracking_difference" in summary

    def test_summary_total_return_positive(self,
                                           zero_cost_result):
        """Both assets appreciate, so the tracking portfolio should gain."""
        assert zero_cost_result.summary()["total_return"] > 0.0

    def test_tracking_error_none_without_target(self,
                                                fetcher):
        target_weights = {
            d: {a: 1.0 / len(ASSETS) for a in ASSETS} for d in REBALANCE_DATES
        }
        engine = BacktestEngine(
            start_date=BASE_DATE, end_date=END_DATE,
            initial_capital=INITIAL_CAPITAL, data_provider=fetcher,
            target_weights=target_weights,
        )
        result = engine.run()
        assert result.get_tracking_error() is None

    def test_tracking_difference_none_without_target(self,
                                                     fetcher):
        target_weights = {
            d: {a: 1.0 / len(ASSETS) for a in ASSETS} for d in REBALANCE_DATES
        }
        engine = BacktestEngine(
            start_date=BASE_DATE, end_date=END_DATE,
            initial_capital=INITIAL_CAPITAL, data_provider=fetcher,
            target_weights=target_weights,
        )
        result = engine.run()
        assert result.get_tracking_difference() is None

    def test_summary_omits_tracking_metrics_without_target(self,
                                                           fetcher):
        target_weights = {
            d: {a: 1.0 / len(ASSETS) for a in ASSETS} for d in REBALANCE_DATES
        }
        engine = BacktestEngine(
            start_date=BASE_DATE, end_date=END_DATE,
            initial_capital=INITIAL_CAPITAL, data_provider=fetcher,
            target_weights=target_weights,
        )
        summary = engine.run().summary()
        assert "tracking_error" not in summary
        assert "tracking_difference" not in summary
