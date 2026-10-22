# tests/test_backtest_cash_band.py
"""BN-358: a fund's cash kept within a band around its buffer, every day.

Without a band, cash that arrives between rebalances (here a dividend kept as
cash) waits for the next rebalance. With one, cash above the upper edge is
invested back down to the buffer the same day, and cash below the lower edge
is raised back up to it.
"""
import pandas as pd
import pytest

from beacon.backtest import Implementation
from beacon.backtest.flows import DatedFlows
from test_backtest_dividends import EX
from test_backtest_dividends import run as run_dividends
from test_strategy_futures import FLOW_DAY, index, run


def cash_share(result) -> pd.Series:
    return result.portfolio.cash / result.trading_nav


class TestWithoutABand:

    def test_dividend_cash_waits_for_the_rebalance(self):
        result = run_dividends(dividends="cash")

        assert cash_share(result)[EX] > 0.01


class TestAboveTheBand:

    def test_dividend_cash_is_invested_down_to_the_buffer(self):
        result = run_dividends(dividends="cash",
                               implementation=Implementation(cash_band=(0.0, 0.002)))

        assert cash_share(result)[EX] < 0.001
        assert cash_share(result).max() <= 0.002 + 1e-9


class TestBelowTheBand:

    def test_an_outflow_is_made_good_by_selling_pro_rata(self):
        banded = run(index(), flows=DatedFlows({FLOW_DAY: -1e6}),
                     implementation=Implementation(cash_buffer=0.05, cash_band=(0.04, 0.06)))
        loose = run(index(), flows=DatedFlows({FLOW_DAY: -1e6}),
                    implementation=Implementation(cash_buffer=0.05))

        assert cash_share(loose)[FLOW_DAY] < 0.04
        assert cash_share(banded)[FLOW_DAY] == pytest.approx(0.05, abs=0.002)


class TestTheSetting:

    @pytest.mark.parametrize("band", [(0.03, 0.04), (0.06, 0.07), (-0.01, 0.06), (0.04, 1.0)])
    def test_a_band_must_hold_the_buffer(self, band):
        with pytest.raises(ValueError, match="cash_band"):
            Implementation(cash_buffer=0.05, cash_band=band)

    def test_it_is_off_by_default(self):
        assert Implementation().cash_band is None
