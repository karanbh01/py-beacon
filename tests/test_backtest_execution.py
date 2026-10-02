# tests/test_backtest_execution.py
"""BN-266 and BN-252: costs that depend on size, how fast orders can trade,
and buys sized net of their costs.

BIG moves up and down 1% on alternate days and trades 10 million shares a
day. THIN is flat at 10 and trades 1,000 shares a day. The data starts a
quarter before the backtest, so impact has a volatility to work from.
"""
import numpy as np
import pandas as pd
import pytest

from beacon.backtest import (
    BacktestEngine,
    ExecutionLimit,
    Implementation,
    MarketImpact,
    ScreenContext,
)
from beacon.data.base import MarketData, ReferenceData
from beacon.data.fetcher import DataFetcher
from beacon.testing.weights import index_result_from_weights

HISTORY = pd.bdate_range("2023-10-02", "2024-02-29")
DAYS = HISTORY[HISTORY >= "2024-01-02"]
FIRST = DAYS[0]
FEBRUARY = pd.Timestamp("2024-02-01")


def big_prices() -> np.ndarray:
    moves = np.where(np.arange(len(HISTORY)) % 2 == 0, 1.01, 1 / 1.01)

    return 100.0 * np.cumprod(moves)


def fetcher() -> DataFetcher:
    rows = [{"IDENTIFIER": "BIG", "DATE": day, "CLOSE": price, "VOLUME": 1e7}
            for day, price in zip(HISTORY, big_prices(), strict=True)]
    rows += [{"IDENTIFIER": "THIN", "DATE": day, "CLOSE": 10.0, "VOLUME": 1e3}
             for day in HISTORY]
    reference = pd.DataFrame([
        {"IDENTIFIER": name, "DATE_FROM": "2020-01-01", "NAME": name,
         "CURRENCY": "USD", "EXCHANGE": "XNYS"} for name in ("BIG", "THIN")])

    return DataFetcher(MarketData.from_dataframe(pd.DataFrame(rows)),
                       ReferenceData.from_dataframe(reference))


def run(capital: float,
        schedule: dict | None = None,
        **engine_args):
    schedule = schedule or {FIRST: {"BIG": 1.0}}

    return BacktestEngine(start_date=str(DAYS[0].date()),
                          end_date=str(DAYS[-1].date()),
                          initial_capital=capital, data_provider=fetcher(),
                          index_result=index_result_from_weights(schedule),
                          calendar="XNYS", **engine_args).run()


def held(result,
         name: str,
         date: pd.Timestamp) -> float:
    positions = result.portfolio.positions
    row = positions[(positions["DATE"] == date) & (positions["ASSET_ID"] == name)]

    return float(row["QUANTITY"].iloc[0]) if not row.empty else 0.0


def total_costs(result) -> float:
    return sum(trade.transaction_cost for trade in result.portfolio.transactions)


class TestBuysAreSizedNetOfCosts:

    def test_a_costed_rebalance_fills_in_full(self):
        """BN-252: the last buy of every costed rebalance used to run short."""
        result = run(1e6, {FIRST: {"BIG": 0.6, "THIN": 0.4}},
                     transaction_cost_bps=25.0)

        assert result.unfilled == []
        assert result.portfolio.cash.loc[FIRST] == pytest.approx(0.0, abs=1e-6)

    def test_a_name_with_no_price_is_recorded(self):
        result = run(1e6, {FIRST: {"BIG": 0.5, "GHOST": 0.5}})

        assert [(order.asset_id, order.reason) for order in result.unfilled] == [
            ("GHOST", "no price")]
        assert result.unfilled[0].shortfall_value == pytest.approx(5e5)


class TestMarketImpact:

    def context(self) -> ScreenContext:
        return ScreenContext(fetcher(), "USD")

    def test_the_rate_follows_the_square_root_law(self):
        impact = MarketImpact(coefficient=0.5)
        date = DAYS[-1]
        closes = pd.Series(big_prices(), index=HISTORY).loc[:date]
        volatility = closes.pct_change().dropna().tail(63).std()
        traded = (closes * 1e7).tail(63).mean()

        rate = impact.rate("BIG", 1e8, date, self.context())

        assert rate == pytest.approx(0.5 * volatility * np.sqrt(1e8 / traded))

    def test_a_flat_name_has_no_impact(self):
        """No volatility, nothing to move."""
        assert MarketImpact().rate("THIN", 1e6, DAYS[-1], self.context()) == 0.0

    def test_a_larger_fund_pays_proportionally_more(self):
        """With impact a trade's cost grows faster than its size."""
        implementation = Implementation(impact=MarketImpact())
        small = run(1e6, implementation=implementation)
        large = run(1e9, implementation=implementation)

        assert (total_costs(large) / 1e9) > 10 * (total_costs(small) / 1e6)

    def test_without_impact_cost_is_proportional(self):
        small = run(1e6, transaction_cost_bps=10.0)
        large = run(1e9, transaction_cost_bps=10.0)

        assert total_costs(large) / 1e9 == pytest.approx(total_costs(small) / 1e6)


class TestExecutionLimits:

    def test_participation_works_an_order_over_several_days(self):
        """500 THIN shares at 10% of 1,000 a day take five sessions."""
        result = run(1e4, {FIRST: {"THIN": 0.5, "BIG": 0.5}},
                     implementation=Implementation(
                         execution=ExecutionLimit(participation=0.1)))

        assert [held(result, "THIN", day) for day in DAYS[:6]] == pytest.approx(
            [100.0, 200.0, 300.0, 400.0, 500.0, 500.0])
        assert result.unfilled == []

    def test_an_order_spread_over_days_trades_evenly(self):
        result = run(1e6, implementation=Implementation(
            execution=ExecutionLimit(days=4)))
        quantities = [trade.quantity for trade in result.portfolio.transactions]

        assert len(quantities) == 4
        assert quantities == pytest.approx([quantities[0]] * 4)

    def test_an_order_still_working_at_the_next_rebalance_is_recorded(self):
        """THIN cannot fill before February's rebalance replaces the order.
        February's orders are slow too and are still working at the end."""
        result = run(1e6, {FIRST: {"THIN": 1.0}, FEBRUARY: {"BIG": 1.0}},
                     implementation=Implementation(
                         execution=ExecutionLimit(participation=0.1)))
        working = [order for order in result.unfilled
                   if order.reason == "execution limit" and order.date == FIRST]

        assert [order.asset_id for order in working] == ["THIN"]
        assert working[0].filled_quantity < working[0].requested_quantity


class TestTheSettingsAreChecked:

    @pytest.mark.parametrize("build", [
        lambda: MarketImpact(coefficient=-1.0),
        ExecutionLimit,
        lambda: ExecutionLimit(participation=0.0),
        lambda: ExecutionLimit(days=0),
    ])
    def test_an_impossible_setting_is_refused(self,
                                              build):
        with pytest.raises(ValueError):
            build()
