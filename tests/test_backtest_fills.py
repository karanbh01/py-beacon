# tests/test_backtest_fills.py
"""Regression tests for BN-85: buys must not be silently dropped."""
from itertools import pairwise

import pandas as pd
import pytest

from beacon.backtest.engine import CASH_TOLERANCE, BacktestEngine, TradeInstruction
from beacon.backtest.result import UnfilledOrder
from beacon.data.base import MarketData
from beacon.data.fetcher import DataFetcher
from beacon.portfolio.base import Portfolio

ASSETS = ["AAA", "BBB"]
DATES = pd.bdate_range("2024-01-01", "2024-06-28")
REBALANCES = [pd.Timestamp("2024-01-01"), pd.Timestamp("2024-04-01")]

# AAA rises, BBB falls. Equal weight means each rebalance sells the winner and
# buys the loser, so a dropped buy leaves the portfolio concentrated in AAA and
# flatters the result — the mechanism this issue exists to fix.
GROWTH = {"AAA": 1.60, "BBB": 0.70}
BASE_PRICE = {"AAA": 100.0, "BBB": 100.0}


def build_fetcher() -> DataFetcher:
    """Two names on deterministic opposing geometric paths."""
    span = len(DATES) - 1
    rows = [
        {"IDENTIFIER": name,
         "DATE": date,
         "CLOSE": BASE_PRICE[name] * (GROWTH[name] ** (index / span))}
        for name in ASSETS
        for index, date in enumerate(DATES)
    ]

    return DataFetcher(MarketData.from_dataframe(pd.DataFrame(rows)))


def equal_weight_schedule() -> dict[pd.Timestamp, dict[str, float]]:
    """Equal weights on each rebalance date."""
    return {date: {"AAA": 0.5, "BBB": 0.5} for date in REBALANCES}


def run(cost_bps: float):
    """Run the two-name equal-weight backtest at a given cost."""
    engine = BacktestEngine(start_date=str(DATES[0].date()),
                            end_date=str(DATES[-1].date()),
                            initial_capital=1_000_000.0,
                            data_provider=build_fetcher(),
                            target_weights=equal_weight_schedule(),
                            transaction_cost_bps=cost_bps)

    return engine.run()


class TestCostsDoNotFlatterTheResult:
    """The headline defect: higher costs must not produce a better backtest."""

    def test_costs_do_not_increase_total_return(self):
        free = run(0.0).summary()["total_return"]
        costly = run(100.0).summary()["total_return"]

        assert costly <= free, (
            f"100bps costs produced a better return than zero cost "
            f"({costly:.6f} > {free:.6f}) — a leg was probably dropped.")

    def test_costs_are_monotonic_in_the_return(self):
        """More cost, never more return, across a range."""
        returns = [run(bps).summary()["total_return"] for bps in (0.0, 25.0, 50.0, 100.0)]

        assert all(later <= earlier + 1e-12
                   for earlier, later in pairwise(returns))

    def test_costs_actually_reduce_the_return(self):
        """Not merely 'not better' — costs are charged and show up."""
        assert run(100.0).summary()["total_return"] < run(0.0).summary()["total_return"]


class TestRebalanceLegsExecute:
    """A rebalance that sells then buys with the proceeds fills both legs."""

    def test_first_rebalance_only_buys(self):
        """Nothing to sell yet — the portfolio starts as cash."""
        result = run(0.0)
        sides = {t.transaction_type for t in result.portfolio.transactions
                 if t.transaction_date == REBALANCES[0]}

        assert sides == {"BUY"}

    def test_later_rebalance_sells_then_buys_with_the_proceeds(self):
        """The case the defect broke: the buy funded by the sell must execute."""
        result = run(0.0)
        second = REBALANCES[1]
        sides = {t.transaction_type for t in result.portfolio.transactions
                 if t.transaction_date == second}

        assert sides == {"BUY", "SELL"}, f"{second} only did {sides}"

        # AAA rose and BBB fell, so equal weight trims AAA and tops up BBB.
        bought = {t.asset_id for t in result.portfolio.transactions
                  if t.transaction_date == second and t.transaction_type == "BUY"}
        sold = {t.asset_id for t in result.portfolio.transactions
                if t.transaction_date == second and t.transaction_type == "SELL"}

        assert sold == {"AAA"}
        assert bought == {"BBB"}

    def test_nothing_is_unfilled_at_zero_cost(self):
        """The float-noise rejection showed up here before the fix."""
        result = run(0.0)

        assert result.unfilled == []
        assert result.total_unfilled_value == 0.0

    def test_weights_land_near_target_after_a_rebalance(self):
        result = run(0.0)
        second = REBALANCES[1]
        row = result.portfolio.weights.loc[second]

        assert row["AAA"] == pytest.approx(0.5, abs=0.02)
        assert row["BBB"] == pytest.approx(0.5, abs=0.02)


class TestPartialFill:
    """A partially affordable buy is sized down, not abandoned."""

    def _portfolio(self,
                   cash: float) -> Portfolio:
        return Portfolio(portfolio_id="t", initial_cash=cash)

    def _engine(self,
                cost_bps: float = 0.0) -> BacktestEngine:
        return BacktestEngine(start_date=str(DATES[0].date()),
                              end_date=str(DATES[-1].date()),
                              initial_capital=1_000.0,
                              data_provider=build_fetcher(),
                              target_weights=equal_weight_schedule(),
                              transaction_cost_bps=cost_bps)

    def test_exactly_affordable_buy_executes_in_full(self):
        portfolio = self._portfolio(1_000.0)
        trade = TradeInstruction("AAA", "BUY", 10.0, 100.0, 0.0)

        shortfall = self._engine()._execute_buy(portfolio, trade, DATES[0])

        assert shortfall is None
        assert portfolio.holdings["AAA"].quantity == pytest.approx(10.0)

    def test_buy_within_float_tolerance_executes_in_full(self):
        """The original defect: need == have, rejected on sub-cent noise."""
        portfolio = self._portfolio(1_000.0 * (1 - CASH_TOLERANCE / 2))
        trade = TradeInstruction("AAA", "BUY", 10.0, 100.0, 0.0)

        shortfall = self._engine()._execute_buy(portfolio, trade, DATES[0])

        assert shortfall is None
        assert portfolio.holdings["AAA"].quantity == pytest.approx(10.0)

    def test_partially_affordable_buy_is_sized_down(self):
        portfolio = self._portfolio(600.0)
        trade = TradeInstruction("AAA", "BUY", 10.0, 100.0, 0.0)

        shortfall = self._engine()._execute_buy(portfolio, trade, DATES[0])

        assert portfolio.holdings["AAA"].quantity == pytest.approx(6.0)
        assert shortfall is not None
        assert shortfall.filled_quantity == pytest.approx(6.0)
        assert shortfall.requested_quantity == pytest.approx(10.0)
        assert shortfall.shortfall_value == pytest.approx(400.0)

    def test_reduced_order_accounts_for_its_own_cost(self):
        """Cash must cover the notional AND the cost charged on it."""
        portfolio = self._portfolio(600.0)
        trade = TradeInstruction("AAA", "BUY", 10.0, 100.0, 10.0)

        self._engine(cost_bps=100.0)._execute_buy(portfolio, trade, DATES[0])

        # 600 = q * 100 * 1.01  ->  q = 5.9406
        assert portfolio.holdings["AAA"].quantity == pytest.approx(5.9406, abs=1e-4)
        assert portfolio.cash_balance == pytest.approx(0.0, abs=1e-6)

    def test_a_sized_down_order_never_overspends(self):
        portfolio = self._portfolio(600.0)
        trade = TradeInstruction("AAA", "BUY", 10.0, 100.0, 10.0)

        self._engine(cost_bps=100.0)._execute_buy(portfolio, trade, DATES[0])

        assert portfolio.cash_balance >= -1e-9

    def test_unaffordable_buy_is_recorded_with_nothing_filled(self):
        portfolio = self._portfolio(0.0)
        trade = TradeInstruction("AAA", "BUY", 10.0, 100.0, 0.0)

        shortfall = self._engine()._execute_buy(portfolio, trade, DATES[0])

        assert shortfall is not None
        assert shortfall.filled_quantity == 0.0
        assert shortfall.shortfall_value == pytest.approx(1_000.0)
        assert "AAA" not in portfolio.holdings

    def test_the_reduced_quantity_never_exceeds_the_request(self):
        """Ample cash must not inflate the order beyond what was asked for."""
        portfolio = self._portfolio(1_000_000.0)
        trade = TradeInstruction("AAA", "BUY", 10.0, 100.0, 0.0)

        self._engine()._execute_buy(portfolio, trade, DATES[0])

        assert portfolio.holdings["AAA"].quantity == pytest.approx(10.0)


class TestUnfilledRecord:

    def test_result_exposes_unfilled_orders(self):
        """A caller must be able to see this without reading the log."""
        result = run(0.0)

        assert isinstance(result.unfilled, list)
        assert result.total_unfilled_value == pytest.approx(0.0)

    def test_total_sums_the_shortfalls(self):
        orders = [UnfilledOrder(date=DATES[0], asset_id="AAA",
                                requested_quantity=10.0, filled_quantity=6.0,
                                price=100.0, shortfall_value=400.0),
                  UnfilledOrder(date=DATES[1], asset_id="BBB",
                                requested_quantity=5.0, filled_quantity=0.0,
                                price=50.0, shortfall_value=250.0)]
        result = run(0.0)
        result.unfilled.extend(orders)

        assert result.total_unfilled_value == pytest.approx(650.0)
