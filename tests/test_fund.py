# tests/test_fund.py
"""Tests for the refactored Fund layer: IndexFund and ETF.

The refactored IndexFund composes an IndexCalculator (target weights) and a
BacktestEngine (portfolio simulation). These tests wire up a synthetic data
provider exposing the unified DataFetcher interface — fetch_market_data,
fetch_reference_data and fetch_shares_outstanding — so the full pipeline runs
end to end with no external data.
"""
from unittest.mock import MagicMock

import pandas as pd
import pytest

from beacon.data.base import MarketData
from beacon.data.fetcher import DataFetcher
from beacon.fund.base import IndexFund
from beacon.fund.etf import ETF
from beacon.index.calculation import IndexCalculator
from beacon.index.constructor import IndexDefinition
from beacon.index.methodology import EqualWeighted
from beacon.portfolio.base import Portfolio

# ---------------------------------------------------------------------------
# Synthetic universe: 2 assets, equal weight, monthly rebalance, ~3 months
# ---------------------------------------------------------------------------
ASSETS = ["ASSET_A", "ASSET_B"]
BASE_DATE = "2024-01-02"
END_DATE = "2024-03-29"
BASE_VALUE = 1000.0
INITIAL_CAPITAL = 1000.0

TRADING_DAYS = pd.bdate_range(start=BASE_DATE, end=END_DATE, freq="B")
N_DAYS = len(TRADING_DAYS)


def _price(asset_id: str,
           i: int) -> float:
    """Geometric price path: ASSET_A +10%, ASSET_B +20% over the window."""
    frac = i / (N_DAYS - 1)
    if asset_id == "ASSET_A":
        return 100.0 * (1.10 ** frac)
    return 50.0 * (1.20 ** frac)


PRICE_LOOKUP = {
    (asset_id, day): _price(asset_id, i)
    for asset_id in ASSETS
    for i, day in enumerate(TRADING_DAYS)
}


def _make_data_provider():
    """Provider satisfying both the IndexCalculator and BacktestEngine APIs."""
    rows = [
        {"IDENTIFIER": a, "DATE": d.strftime("%Y-%m-%d"), "CLOSE": p}
        for (a, d), p in PRICE_LOOKUP.items()
    ]
    fetcher = DataFetcher(MarketData.from_dataframe(pd.DataFrame(rows)))
    provider = MagicMock()

    def fetch_reference_data(identifier,
                             date=None,
                             columns=None):
        if identifier in ASSETS:
            return pd.DataFrame(
                {"NAME": [identifier], "CURRENCY": ["USD"], "EXCHANGE": ["NYSE"]},
                index=pd.Index([identifier], name="IDENTIFIER"),
            )
        return pd.DataFrame()

    provider.fetch_reference_data.side_effect = fetch_reference_data
    provider.fetch_market_data.side_effect = (
        lambda identifier, start=None, end=None, columns=None:
        fetcher.fetch_market_data(identifier, start, end, columns)
    )
    provider.fetch_shares_outstanding.side_effect = lambda ticker, date: 1000
    return provider


def _make_definition():
    return IndexDefinition(
        index_id="TEST_EW",
        index_name="Test Equal Weight",
        base_date=BASE_DATE,
        base_value=BASE_VALUE,
        currency="USD",
        eligibility_rules=[],
        weighting_scheme=EqualWeighted(),
        rebalancing_frequency="MONTHLY",
        universe_identifiers=ASSETS,
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def data_provider():
    return _make_data_provider()


@pytest.fixture
def definition():
    return _make_definition()


@pytest.fixture
def calculator(definition,
               data_provider):
    return IndexCalculator(definition, data_provider)


@pytest.fixture
def index_fund(definition,
               calculator,
               data_provider):
    return IndexFund(
        fund_id="FUND1",
        target_index_definition=definition,
        index_agent=calculator,
        portfolio=Portfolio("fund1_pf", initial_cash=INITIAL_CAPITAL),
        data_provider=data_provider,
        management_fee_bps=0,
    )


@pytest.fixture
def etf(definition,
        calculator,
        data_provider):
    return ETF(
        fund_id="ETF1",
        etf_ticker="TEST",
        target_index_definition=definition,
        index_agent=calculator,
        portfolio=Portfolio("etf1_pf", initial_cash=INITIAL_CAPITAL),
        data_provider=data_provider,
        management_fee_bps=0,
    )


# ---------------------------------------------------------------------------
# IndexFund — construction
# ---------------------------------------------------------------------------

class TestIndexFundConstruction:

    def test_valid_construction(self,
                                index_fund,
                                definition,
                                calculator):
        assert index_fund.fund_id == "FUND1"
        assert index_fund.target_index_definition is definition
        assert index_fund.index_agent is calculator
        assert index_fund.management_fee_bps == 0
        # No pipeline has run yet.
        assert index_fund.index_result is None
        assert index_fund.backtest_result is None

    def test_empty_fund_id_raises(self,
                                  definition,
                                  calculator,
                                  data_provider):
        with pytest.raises(ValueError, match="fund_id"):
            IndexFund("", definition, calculator,
                      Portfolio("p", initial_cash=100.0), data_provider)

    def test_missing_index_definition_raises(self,
                                             calculator,
                                             data_provider):
        with pytest.raises(ValueError, match="target_index_definition"):
            IndexFund("F", None, calculator,
                      Portfolio("p", initial_cash=100.0), data_provider)

    def test_missing_index_agent_raises(self,
                                        definition,
                                        data_provider):
        with pytest.raises(ValueError, match="index_agent"):
            IndexFund("F", definition, None,
                      Portfolio("p", initial_cash=100.0), data_provider)

    def test_negative_fee_raises(self,
                                 definition,
                                 calculator,
                                 data_provider):
        with pytest.raises(ValueError, match="management_fee_bps"):
            IndexFund("F", definition, calculator,
                      Portfolio("p", initial_cash=100.0), data_provider,
                      management_fee_bps=-1)


# ---------------------------------------------------------------------------
# IndexFund — NAV
# ---------------------------------------------------------------------------

class TestIndexFundNav:

    def test_run_backtest_populates_results(self,
                                            index_fund):
        result = index_fund.run_backtest(end_date=END_DATE)
        assert index_fund.backtest_result is result
        assert index_fund.index_result is not None
        assert len(result.trading_nav) == N_DAYS

    def test_nav_on_base_date_equals_initial_capital(self,
                                                     index_fund):
        index_fund.run_backtest(end_date=END_DATE)
        nav = index_fund.calculate_nav(pd.Timestamp(BASE_DATE))
        assert nav == pytest.approx(INITIAL_CAPITAL, rel=1e-9)

    def test_nav_matches_backtest_nav_series(self,
                                             index_fund):
        index_fund.run_backtest(end_date=END_DATE)
        nav = index_fund.calculate_nav(pd.Timestamp(END_DATE))
        expected = index_fund.backtest_result.trading_nav.iloc[-1]
        assert nav == pytest.approx(expected)

    def test_nav_grows_with_appreciating_assets(self,
                                                index_fund):
        index_fund.run_backtest(end_date=END_DATE)
        nav_end = index_fund.calculate_nav(pd.Timestamp(END_DATE))
        assert nav_end > INITIAL_CAPITAL

    def test_calculate_nav_lazily_runs_backtest(self,
                                                index_fund):
        """calculate_nav triggers the pipeline if it has not run yet."""
        assert index_fund.backtest_result is None
        nav = index_fund.calculate_nav(pd.Timestamp("2024-02-15"))
        assert nav > 0
        assert index_fund.backtest_result is not None

    def test_nav_before_base_date_returns_seed_capital(self,
                                                       index_fund):
        assert index_fund.calculate_nav(pd.Timestamp("2023-12-01")) == INITIAL_CAPITAL

    def test_fund_does_not_mutate_own_portfolio(self,
                                                index_fund):
        """Trading is delegated to the engine's internal portfolio."""
        index_fund.run_backtest(end_date=END_DATE)
        assert index_fund.portfolio.transactions == []
        assert index_fund.portfolio.cash_balance == INITIAL_CAPITAL


# ---------------------------------------------------------------------------
# IndexFund — management fee deduction
# ---------------------------------------------------------------------------

class TestIndexFundFee:

    def test_fee_reduces_nav(self,
                             definition,
                             calculator,
                             data_provider):
        no_fee = IndexFund("NF", definition, calculator,
                           Portfolio("nf", initial_cash=INITIAL_CAPITAL),
                           data_provider, management_fee_bps=0)
        with_fee = IndexFund("WF", definition, calculator,
                             Portfolio("wf", initial_cash=INITIAL_CAPITAL),
                             data_provider, management_fee_bps=100)
        no_fee.run_backtest(end_date=END_DATE)
        with_fee.run_backtest(end_date=END_DATE)

        gross = no_fee.calculate_nav(pd.Timestamp(END_DATE))
        net = with_fee.calculate_nav(pd.Timestamp(END_DATE))
        assert net < gross

    def test_fee_factor_matches_expected(self,
                                         definition,
                                         calculator,
                                         data_provider):
        fee_bps = 100
        fund = IndexFund("WF", definition, calculator,
                         Portfolio("wf", initial_cash=INITIAL_CAPITAL),
                         data_provider, management_fee_bps=fee_bps)
        fund.run_backtest(end_date=END_DATE)

        gross = fund.backtest_result.trading_nav.iloc[-1]
        net = fund.calculate_nav(pd.Timestamp(END_DATE))

        daily_rate = (fee_bps / 10000.0) / 252.0
        expected_factor = (1.0 - daily_rate) ** (N_DAYS - 1)
        assert net == pytest.approx(gross * expected_factor)

    def test_no_fee_on_base_date(self,
                                 definition,
                                 calculator,
                                 data_provider):
        """Zero days elapsed -> no fee accrued yet."""
        fund = IndexFund("WF", definition, calculator,
                         Portfolio("wf", initial_cash=INITIAL_CAPITAL),
                         data_provider, management_fee_bps=100)
        fund.run_backtest(end_date=END_DATE)
        nav = fund.calculate_nav(pd.Timestamp(BASE_DATE))
        assert nav == pytest.approx(INITIAL_CAPITAL, rel=1e-9)


# ---------------------------------------------------------------------------
# ETF — construction
# ---------------------------------------------------------------------------

class TestETFConstruction:

    def test_valid_construction(self,
                                etf):
        assert etf.fund_id == "ETF1"
        assert etf.etf_ticker == "TEST"
        assert etf.creation_unit_size == 50000
        assert etf.market_price is None

    def test_is_index_fund(self,
                           etf):
        assert isinstance(etf, IndexFund)

    def test_empty_ticker_raises(self,
                                 definition,
                                 calculator,
                                 data_provider):
        with pytest.raises(ValueError, match="etf_ticker"):
            ETF("E", "", definition, calculator,
                Portfolio("p", initial_cash=100.0), data_provider)

    def test_non_positive_creation_unit_raises(self,
                                               definition,
                                               calculator,
                                               data_provider):
        with pytest.raises(ValueError, match="creation_unit_size"):
            ETF("E", "TICK", definition, calculator,
                Portfolio("p", initial_cash=100.0), data_provider,
                creation_unit_size=0)


# ---------------------------------------------------------------------------
# ETF — tracking performance
# ---------------------------------------------------------------------------

class TestETFTrackingPerformance:

    def test_returns_metrics_from_backtest_result(self,
                                                  etf):
        result = etf.run_backtest(end_date=END_DATE)
        perf = etf.get_tracking_performance(result)
        assert set(perf) == {"tracking_error", "tracking_difference"}
        assert perf["tracking_error"] == pytest.approx(result.get_tracking_error())
        assert perf["tracking_difference"] == pytest.approx(result.get_tracking_difference())

    def test_metrics_are_finite_floats(self,
                                       etf):
        result = etf.run_backtest(end_date=END_DATE)
        perf = etf.get_tracking_performance(result)
        assert isinstance(perf["tracking_error"], float)
        assert isinstance(perf["tracking_difference"], float)
        assert pd.notna(perf["tracking_error"])
        assert pd.notna(perf["tracking_difference"])

    def test_error_when_no_target_index(self,
                                        etf,
                                        data_provider):
        """A backtest driven by a raw weight dict has no target index."""
        from beacon.backtest.engine import BacktestEngine
        weights = {pd.Timestamp(BASE_DATE): dict.fromkeys(ASSETS, 0.5)}
        result = BacktestEngine(
            start_date=BASE_DATE, end_date=END_DATE,
            initial_capital=INITIAL_CAPITAL, data_provider=data_provider,
            target_weights=weights,
        ).run()
        perf = etf.get_tracking_performance(result)
        assert "error" in perf

    def test_raises_on_none_result(self,
                                   etf):
        with pytest.raises(ValueError, match="BacktestResult must be provided"):
            etf.get_tracking_performance(None)


# ---------------------------------------------------------------------------
# ETF — market price simulation
# ---------------------------------------------------------------------------

class TestETFMarketPrice:

    def test_simulate_market_price_returns_nav(self,
                                               etf):
        etf.run_backtest(end_date=END_DATE)
        date = pd.Timestamp(END_DATE)
        price = etf.simulate_market_price(date)
        assert price == pytest.approx(etf.calculate_nav(date))

    def test_simulate_market_price_sets_attribute(self,
                                                  etf):
        etf.run_backtest(end_date=END_DATE)
        date = pd.Timestamp(END_DATE)
        price = etf.simulate_market_price(date)
        assert etf.market_price == pytest.approx(price)

    def test_market_price_on_base_date_equals_capital(self,
                                                      etf):
        etf.run_backtest(end_date=END_DATE)
        price = etf.simulate_market_price(pd.Timestamp(BASE_DATE))
        assert price == pytest.approx(INITIAL_CAPITAL, rel=1e-9)
