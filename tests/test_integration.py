"""End-to-end integration tests for the Beacon pipeline."""

import numpy as np
import pandas as pd

from beacon.backtest.engine import BacktestEngine
from beacon.data.base import MarketData, ReferenceData
from beacon.data.fetcher import DataFetcher


class SyntheticPipelineFetcher(DataFetcher):
    """DataFetcher adapter exposing legacy index-calculator helpers."""

    def fetch_prices(self, ticker, start_date, end_date):
        return self.fetch_market_data(ticker, start_date, end_date)

    def fetch_shares_outstanding(self, ticker, date):
        return 1_000_000.0

    def fetch_free_float_factor(self, ticker, date):
        return 1.0

    def fetch_fx_rates(self, from_currency, to_currency, start_date, end_date):
        return pd.Series([1.0], index=[pd.Timestamp(start_date)])
from beacon.index.calculation import IndexCalculator
from beacon.index.constructor import IndexDefinition
from beacon.index.methodology import EqualWeighted


def _synthetic_data_fetcher():
    dates = pd.bdate_range("2025-01-02", "2025-06-30")
    market_rows = []
    reference_rows = []
    price_specs = {
        "AAA": (100.0, 0.040),
        "BBB": (80.0, 0.025),
        "CCC": (60.0, 0.015),
    }

    for ticker, (start_price, six_month_return) in price_specs.items():
        reference_rows.append(
            {
                "IDENTIFIER": ticker,
                "DATE_FROM": "2024-01-01",
                "DATE_TO": None,
                "NAME": f"{ticker} Corp",
                "CURRENCY": "USD",
                "EXCHANGE": "SYNTH",
            }
        )
        for idx, date in enumerate(dates):
            trend = start_price * (1.0 + six_month_return * idx / (len(dates) - 1))
            seasonal = 0.10 * np.sin(idx / 7.0)
            close = trend + seasonal
            market_rows.append(
                {
                    "IDENTIFIER": ticker,
                    "DATE": date.strftime("%Y-%m-%d"),
                    "CLOSE": close,
                    "Adj Close": close,
                    "Volume": 1_000_000 + idx,
                }
            )

    market_data = MarketData.from_dataframe(pd.DataFrame(market_rows))
    reference_data = ReferenceData.from_dataframe(pd.DataFrame(reference_rows))
    return SyntheticPipelineFetcher(market_data, reference_data)


def _index_definition():
    return IndexDefinition(
        index_id="synthetic_equal_weight",
        index_name="Synthetic Equal Weight",
        base_date="2025-01-02",
        base_value=100.0,
        currency="USD",
        eligibility_rules=[],
        weighting_scheme=EqualWeighted(),
        rebalancing_frequency="QUARTERLY",
        universe_identifiers=["AAA", "BBB", "CCC"],
    )


def _run_pipeline(transaction_cost_bps=0.0):
    data_fetcher = _synthetic_data_fetcher()
    index_result = IndexCalculator(_index_definition(), data_fetcher).run(
        start_date="2025-01-02",
        end_date="2025-06-30",
    )
    backtest_result = BacktestEngine(
        "2025-01-02",
        "2025-06-30",
        100_000.0,
        data_fetcher,
        target_index_result=index_result,
        transaction_cost_bps=transaction_cost_bps,
    ).run()
    return index_result, backtest_result


def test_full_pipeline_zero_cost_tracks_index_closely():
    index_result, backtest_result = _run_pipeline(transaction_cost_bps=0.0)

    assert not index_result.index_levels.empty
    assert len(index_result.weight_snapshots) >= 2
    assert set(index_result.constituent_snapshots[index_result.index_levels.index[0]]) == {"AAA", "BBB", "CCC"}
    assert not backtest_result.portfolio_nav.empty
    assert backtest_result.transactions
    assert backtest_result.get_tracking_error() is not None
    assert backtest_result.get_tracking_error() < 1e-3
    assert abs(backtest_result.get_tracking_difference()) < 5e-3


def test_full_pipeline_nonzero_costs_create_tracking_drag():
    _, zero_cost_result = _run_pipeline(transaction_cost_bps=0.0)
    _, cost_result = _run_pipeline(transaction_cost_bps=25.0)

    assert cost_result.portfolio_nav.iloc[-1] < zero_cost_result.portfolio_nav.iloc[-1]
    assert cost_result.get_tracking_difference() < zero_cost_result.get_tracking_difference()
    assert cost_result.get_tracking_error() is not None
