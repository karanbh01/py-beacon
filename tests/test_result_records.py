# tests/test_result_records.py
"""BN-459/BN-460: a run's recorded data under `result.fund` (the vehicle's)
and `result.trading` (the engine's), and the flat names before 0.11.0
reading them, warning, until 0.12.0.

The runs are test_strategy_futures' eight-name index held physically and
through futures, test_strategy_synthetic's synthetic fund, test_backtest_flows'
two names with a cash band and as an ETF.
"""
import warnings

import pandas as pd
import pytest

from beacon.backtest import Implementation
from beacon.backtest.fund_record import FundRecord
from beacon.backtest.result import MOVED
from beacon.backtest.trading_record import TradingRecord


@pytest.fixture(scope="module")
def plain():
    from test_strategy_futures import index, run

    return run(index(), transaction_cost_bps=5.0)


@pytest.fixture(scope="module")
def futures():
    from test_strategy_futures import futures as held_through_futures
    from test_strategy_futures import run

    return run(held_through_futures())


@pytest.fixture(scope="module")
def synthetic():
    from beacon.strategy import SyntheticReplication
    from test_strategy_synthetic import BASKET, run

    return run(SyntheticReplication(BASKET))


def same(found,
         expected) -> bool:
    """Whether two recorded values are the same."""
    if isinstance(found, pd.Series | pd.DataFrame):
        return bool(found.equals(expected))

    return bool(found == expected)


class TestTheFundsRecord:

    def test_its_nav_and_units(self,
                               plain):
        fund = plain.fund

        assert isinstance(fund, FundRecord) and fund.currency == "USD"
        assert fund.trading_nav.index[0] > plain.portfolio.nav.index[0]
        pd.testing.assert_series_equal(fund.aum, fund.trading_nav)
        pd.testing.assert_series_equal(fund.nav_per_unit, fund.trading_nav / fund.units_outstanding,
                                       check_names=False)
        assert fund.fees.paid == 0.0 and fund.dividends.entitled.columns.tolist() == [
            "GROSS", "WITHHELD"]

    def test_a_part_the_vehicle_lacks_is_none(self,
                                              plain):
        fund = plain.fund

        assert (fund.etf, fund.swap, fund.futures, fund.cash_band, fund.hedging) == (
            None, None, None, None, None)

    def test_a_fund_held_through_futures(self,
                                         futures):
        found = futures.fund.futures

        assert found is not None and found.trades and found.rolls is not None
        pd.testing.assert_series_equal(
            found.exposure, (found.notional.reindex(futures.fund.trading_nav.index).fillna(0.0)
                             / futures.fund.trading_nav), check_names=False)

    def test_a_synthetic_fund(self,
                              synthetic):
        swap = synthetic.fund.swap
        nav = synthetic.fund.trading_nav

        assert swap is not None and swap.resets
        pd.testing.assert_series_equal(
            swap.counterparty_exposure,
            swap.value.clip(lower=0.0).reindex(nav.index).fillna(0.0) / nav, check_names=False)

    def test_a_cash_band_and_an_etf(self):
        from test_backtest_etf import etf
        from test_backtest_flows import fetcher, run

        banded = run(data=fetcher(0.03), implementation=Implementation(
            cash_buffer=0.05, cash_band=(0.04, 0.06)), transaction_cost_bps=10.0)
        listed = run(vehicle=etf())

        assert banded.fund.cash_band is not None and not banded.fund.cash_band.events.empty
        assert listed.fund.etf is not None and not listed.fund.etf.market.empty


class TestTheTradingRecord:

    def test_what_each_rebalance_did(self,
                                     plain):
        trading = plain.trading

        assert isinstance(trading, TradingRecord)
        assert [step.date for step in trading.steps] == [row.date for row in trading.pricing]
        assert trading.total_costs == pytest.approx(sum(
            transaction.transaction_cost for transaction in plain.portfolio.transactions))
        assert trading.unfilled_value == 0.0

    def test_a_strategy_s_own_record_is_none_for_another(self,
                                                         plain,
                                                         futures):
        assert plain.trading.replication is None and plain.trading.active is None
        assert futures.trading.replication


class TestTheFlatNamesWarn:

    @pytest.mark.parametrize("name", list(MOVED))
    def test_each_reads_its_new_home(self,
                                     name,
                                     plain,
                                     futures):
        for result in (plain, futures):
            path, read = MOVED[name]

            with pytest.warns(DeprecationWarning, match=rf"BacktestResult\.{name} is deprecated "
                                                        rf"and will be removed in 0.12.0; "
                                                        rf"use result\.{path}\."):
                found = getattr(result, name)

            assert same(found, read(result))

    def test_a_missing_part_reads_empty_as_before(self,
                                                  plain):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)

            assert plain.swap_value.empty and plain.market.empty and plain.futures_trades == []
            assert plain.replication == [] and plain.active == []

    def test_the_new_paths_do_not_warn(self,
                                       plain):
        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)

            _ = (plain.fund.nav_per_unit, plain.trading.steps, plain.fund.fees.payable,
                 plain.trading.total_costs)
