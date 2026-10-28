# tests/test_liquidity_analytics.py
"""BN-392: days to sell, and the share of the book sellable in 1, 5 and 20
days, worked out by hand on a small book.

test_backtest_flows' two names each trade a million shares a day: FLAT at 50
(50,000,000 a day) and GROW from 100 (100,000,000 a day and rising). A fund of
200,000,000 holds 40% FLAT and 60% GROW at the start, selling 20% of a day's
traded value a day.
"""
import numpy as np
import pandas as pd
import pytest

from beacon.analysis.data import RunData
from beacon.analysis.panels import FUND_WEIGHTS
from beacon.analysis.risk.liquidity import LiquidityAnalytics
from test_backtest_flows import fetcher, run

START = pd.Timestamp("2024-01-02")


@pytest.fixture(scope="module")
def data():
    return RunData.from_result(run(capital=2e8))


class TestByHand:

    def test_days_to_sell_each_holding(self,
                                       data):
        table = LiquidityAnalytics(data).days_to_sell(START)

        assert table.loc["FLAT", "days"] == pytest.approx(0.4 * 2e8 / (0.2 * 5e7))
        assert table.loc["GROW", "days"] == pytest.approx(0.6 * 2e8 / (0.2 * 1e8))
        assert list(table.index) == ["FLAT", "GROW"]

    def test_the_share_sellable_in_each_horizon(self,
                                                data):
        first = LiquidityAnalytics(data).profile().loc[START]

        assert first["sellable_1d"] == pytest.approx((1e7 + 2e7) / 2e8)
        assert first["sellable_5d"] == pytest.approx((5e7 + 1e8) / 2e8)
        assert first["sellable_20d"] == pytest.approx(1.0)
        assert first["days_to_sell_all"] == pytest.approx(8.0)
        assert first["mean_days"] == pytest.approx(0.4 * 8 + 0.6 * 6)
        assert first["no_traded_value"] == 0.0

    def test_a_slower_participation_takes_longer(self,
                                                 data):
        slow = LiquidityAnalytics(data, participation=0.1).profile().loc[START]

        assert slow["days_to_sell_all"] == pytest.approx(16.0)

    def test_each_rebalance_and_month_end(self,
                                          data):
        dates = LiquidityAnalytics(data).dates()

        assert dates[0] == START and len(dates) == 12


class TestTradedValues:

    def test_measured_from_the_data_over_the_window_given(self,
                                                          data):
        stored = LiquidityAnalytics(data).traded_values()
        measured = LiquidityAnalytics(data, source=fetcher(), lookback_days=20).traded_values()

        assert measured["FLAT"].to_numpy() == pytest.approx(stored["FLAT"].to_numpy())
        # GROW's price rises, so its last 20 days trade more than its last 63.
        assert (measured["GROW"].iloc[4:] > stored["GROW"].iloc[4:]).all()

    def test_a_name_with_no_traded_value_cannot_be_sold(self,
                                                        data):
        tables = dict(data.tables)
        tables[FUND_WEIGHTS] = tables[FUND_WEIGHTS].assign(QUIET=0.1)
        held = RunData(currency="USD", returns=data.returns, nav=data.nav, units=data.units,
                       initial_capital=2e8, periods_per_year=252, risk_free_rate=0.0,
                       tables=tables)
        analytics = LiquidityAnalytics(held)

        assert analytics.days_to_sell(START).loc["QUIET", "days"] == np.inf
        assert analytics.profile().loc[START, "no_traded_value"] == pytest.approx(0.1 / 1.1)


class TestRefused:

    @pytest.mark.parametrize("participation", [0.0, 1.5])
    def test_a_participation_out_of_range(self,
                                          data,
                                          participation):
        with pytest.raises(ValueError, match="participation"):
            LiquidityAnalytics(data, participation=participation)

    def test_a_book_without_weights(self,
                                    data):
        with pytest.raises(KeyError, match="benchmark"):
            LiquidityAnalytics(data).days_to_sell(START, "benchmark")
