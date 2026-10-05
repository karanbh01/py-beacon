# tests/test_backtest_flows.py
"""BN-267: units, flows, the cash buffer and the management fee.

Two names over the first half of 2024, rebalanced to 60/40 at the start of
each month: GROW rises 0.05% a session and FLAT does not move. Both trade a
million shares a day. With no costs and inflows invested pro rata to the
holdings, a flow scales the book exactly, so NAV per unit must not notice it.
"""
import math
from itertools import pairwise

import numpy as np
import pandas as pd
import pytest

from beacon.backtest import (
    BacktestEngine,
    DatedFlows,
    ExecutionLimit,
    FlowContext,
    Implementation,
    PerformanceChasingFlows,
    PeriodicFlows,
    RandomFlows,
    Vehicle,
)
from beacon.backtest.flows import money_weighted_return
from beacon.data.base import MarketData, ReferenceData
from beacon.data.fetcher import DataFetcher
from beacon.index.schedule import sessions
from beacon.portfolio.cash_flows import FEE, REDEMPTION, SUBSCRIPTION
from beacon.testing.weights import index_result_from_weights

DAYS = sessions(pd.Timestamp("2024-01-02"), pd.Timestamp("2024-06-28"), "XNYS")
MONTH_STARTS = [day for previous, day in pairwise(DAYS)
                if day.month != previous.month]
SCHEDULE = {day: {"GROW": 0.6, "FLAT": 0.4} for day in [DAYS[0], *MONTH_STARTS]}
CAPITAL = 1_000_000.0
# A session that is not a rebalance, for flows dealt between rebalances.
QUIET = DAYS[10]


def fetcher(grow_rate: float = 0.0005) -> DataFetcher:
    rows = [{"IDENTIFIER": "GROW", "DATE": day,
             "CLOSE": 100.0 * (1.0 + grow_rate) ** position, "VOLUME": 1e6}
            for position, day in enumerate(DAYS)]
    rows += [{"IDENTIFIER": "FLAT", "DATE": day, "CLOSE": 50.0, "VOLUME": 1e6}
             for day in DAYS]
    reference = pd.DataFrame([
        {"IDENTIFIER": name, "DATE_FROM": "2020-01-01", "NAME": name,
         "CURRENCY": "USD", "EXCHANGE": "XNYS"} for name in ("GROW", "FLAT")])

    return DataFetcher(MarketData.from_dataframe(pd.DataFrame(rows)),
                       ReferenceData.from_dataframe(reference))


def run(capital: float = CAPITAL,
        data: DataFetcher | None = None,
        **engine_args):
    return BacktestEngine(start_date=str(DAYS[0].date()),
                          end_date=str(DAYS[-1].date()),
                          initial_capital=capital,
                          data_provider=data if data is not None else fetcher(),
                          index_result=index_result_from_weights(SCHEDULE),
                          calendar="XNYS", **engine_args).run()


def holdings_mode(**settings) -> Implementation:
    return Implementation(invest_flows="holdings", **settings)


def flows_of(result,
             kind: str) -> list:
    return [flow for flow in result.portfolio.cash_flows if flow.kind == kind]


def held(result,
         name: str,
         date: pd.Timestamp) -> float:
    positions = result.portfolio.positions
    row = positions[(positions["DATE"] == date) & (positions["ASSET_ID"] == name)]

    return float(row["QUANTITY"].iloc[0]) if not row.empty else 0.0


class TestWithoutFlowsNothingChanges:

    def test_the_units_never_move(self):
        result = run()

        assert (result.units_outstanding == CAPITAL).all()
        assert result.nav_per_unit.equals(result.trading_nav / CAPITAL)

    def test_a_scenario_that_never_flows_leaves_the_metrics_alone(self):
        plain = run()
        quiet = run(flows=DatedFlows({}))

        assert quiet.flows == []
        assert quiet.get_returns().equals(plain.get_returns())
        assert quiet.summary() == plain.summary()
        assert "money_weighted_return" not in plain.summary()


class TestUnits:

    def test_an_inflow_creates_units_at_the_days_nav_per_unit(self):
        result = run(flows=DatedFlows({QUIET: 500_000.0}))
        record = result.flows[0]

        assert record.date == QUIET
        assert record.units == pytest.approx(500_000.0 / record.nav_per_unit)
        assert result.units_outstanding[QUIET] == pytest.approx(
            CAPITAL + record.units)

    def test_a_flow_does_not_move_nav_per_unit(self):
        """At no cost, invested pro rata, an inflow scales the book."""
        plain = run(implementation=holdings_mode())
        flowed = run(flows=DatedFlows({QUIET: 500_000.0, DAYS[60]: -300_000.0}),
                     implementation=holdings_mode())

        assert flowed.trading_nav.iloc[-1] > plain.trading_nav.iloc[-1] * 1.1
        np.testing.assert_allclose(flowed.nav_per_unit, plain.nav_per_unit,
                                   rtol=1e-9)

    def test_the_returns_are_a_units(self):
        plain = run(implementation=holdings_mode())
        flowed = run(flows=DatedFlows({QUIET: 2e6}),
                     implementation=holdings_mode())

        assert flowed.summary()["total_return"] == pytest.approx(
            plain.summary()["total_return"], rel=1e-9)
        assert flowed.get_tracking_difference() == pytest.approx(
            plain.get_tracking_difference(), abs=1e-9)

    def test_performance_is_drawn_per_unit(self):
        """Charts and comparisons read a unit's NAV, so a flow is not a gain."""
        result = run(flows=DatedFlows({QUIET: 2e6}))

        assert result.performance_levels().equals(result.nav_per_unit)

    def test_an_outflow_cancels_units_and_is_paid_out(self):
        result = run(flows=DatedFlows({QUIET: -250_000.0}))
        record = result.flows[0]

        assert record.amount == pytest.approx(-250_000.0)
        assert record.units == pytest.approx(-250_000.0 / record.nav_per_unit)
        assert flows_of(result, REDEMPTION)[0].amount == pytest.approx(-250_000.0)

    def test_a_redemption_larger_than_the_fund_is_cut_to_it(self):
        result = run(flows=DatedFlows({QUIET: -1e9}))
        record = result.flows[0]

        assert record.amount > -2e6
        assert record.units == pytest.approx(-CAPITAL, rel=1e-9)
        assert result.units_outstanding[QUIET] == pytest.approx(0.0, abs=1e-6)

    def test_a_fund_can_launch_empty_and_be_funded_by_flows(self):
        result = run(capital=0.0, flows=DatedFlows({DAYS[0]: 1e6}),
                     vehicle=Vehicle(launch_price=10.0))

        assert result.flows[0].nav_per_unit == 10.0
        assert result.units_outstanding.iloc[-1] == pytest.approx(1e5)


class TestTheMoneyWeightedReturn:

    def test_a_known_rate(self):
        start = pd.Timestamp("2024-01-01")
        flows = [(start, -100.0), (start + pd.Timedelta(days=365), 110.0)]

        assert money_weighted_return(flows) == pytest.approx(0.10, abs=1e-9)

    def test_without_flows_it_is_the_calendar_annualised_return(self):
        result = run()
        days = (DAYS[-1] - result.portfolio.inception).days
        growth = result.trading_nav.iloc[-1] / CAPITAL

        assert result.money_weighted_return() == pytest.approx(
            growth ** (365.0 / days) - 1.0, rel=1e-6)

    def test_money_arriving_late_in_a_rise_earns_less(self):
        """The unit's return is the same; the investors' money is not."""
        result = run(flows=DatedFlows({DAYS[100]: 5e6}),
                     implementation=holdings_mode())
        summary = result.summary()
        days = (DAYS[-1] - result.portfolio.inception).days
        unit_rate = (1.0 + summary["total_return"]) ** (365.0 / days) - 1.0

        assert summary["money_weighted_return"] < unit_rate


class TestTheManagementFee:

    def test_it_accrues_daily_on_net_assets_act_365(self):
        """Flat prices and no costs: only the fee moves the NAV."""
        fee = 50.0
        result = run(data=fetcher(grow_rate=0.0),
                     vehicle=Vehicle(management_fee_bps=fee))
        dates = [result.portfolio.inception, *DAYS]
        expected = CAPITAL * math.prod(
            1.0 - fee / 10_000.0 * (later - earlier).days / 365.0
            for earlier, later in pairwise(dates))

        assert result.trading_nav.iloc[-1] == pytest.approx(expected, rel=1e-9)

    def test_what_is_owed_is_paid_from_cash(self):
        result = run(data=fetcher(grow_rate=0.0),
                     vehicle=Vehicle(management_fee_bps=50.0))
        paid = -sum(flow.amount for flow in flows_of(result, FEE))
        owed = result.fees_payable.iloc[-1]

        assert paid > 0.0
        assert paid + owed == pytest.approx(CAPITAL - result.trading_nav.iloc[-1],
                                            rel=1e-9)

    def test_without_a_vehicle_nothing_is_charged(self):
        assert flows_of(run(), FEE) == []


class TestTheCashBuffer:

    def test_a_rebalance_keeps_it_in_cash(self):
        result = run(implementation=Implementation(cash_buffer=0.05))

        assert result.rebalance_steps[0].cash_weight == pytest.approx(0.05)
        assert result.portfolio.cash[DAYS[0]] == pytest.approx(0.05 * CAPITAL)

    def test_an_outflow_within_it_sells_nothing(self):
        result = run(implementation=Implementation(cash_buffer=0.05),
                     flows=DatedFlows({QUIET: -20_000.0}))

        assert [trade for trade in result.portfolio.transactions
                if trade.transaction_date == QUIET] == []
        assert flows_of(result, REDEMPTION)[0].amount == pytest.approx(-20_000.0)

    def test_an_inflow_tops_it_up_first(self):
        result = run(implementation=Implementation(cash_buffer=0.05),
                     flows=DatedFlows({QUIET: 100_000.0}))
        nav = result.trading_nav[QUIET]

        assert result.portfolio.cash[QUIET] == pytest.approx(0.05 * nav, rel=1e-6)


class TestInvestingAnInflow:

    def test_toward_the_target_by_default(self):
        result = run(flows=DatedFlows({QUIET: 100_000.0}))
        bought = {name: held(result, name, QUIET) - held(result, name, DAYS[9])
                  for name in ("GROW", "FLAT")}
        grow_price = 100.0 * 1.0005 ** 10
        spent = {"GROW": bought["GROW"] * grow_price, "FLAT": bought["FLAT"] * 50.0}

        assert spent["GROW"] / spent["FLAT"] == pytest.approx(0.6 / 0.4)

    def test_or_pro_rata_to_the_holdings(self):
        result = run(flows=DatedFlows({QUIET: 100_000.0}),
                     implementation=holdings_mode())
        before = {name: held(result, name, DAYS[9]) for name in ("GROW", "FLAT")}
        after = {name: held(result, name, QUIET) for name in ("GROW", "FLAT")}

        assert after["GROW"] / before["GROW"] == pytest.approx(
            after["FLAT"] / before["FLAT"])

    def test_under_an_execution_limit_it_is_worked_over_days(self):
        """10% of a million shares a day is 100,000; buying 60% of 200
        million in GROW at about 100 needs well over a day."""
        result = run(flows=DatedFlows({QUIET: 2e8}),
                     implementation=Implementation(
                         execution=ExecutionLimit(participation=0.1)))
        bought = [held(result, "GROW", day) - held(result, "GROW", earlier)
                  for earlier, day in pairwise(DAYS[9:14])]

        assert bought == pytest.approx([1e5, 1e5, 1e5, 1e5])


class TestScenarios:

    def context(self,
                date: pd.Timestamp,
                previous: pd.Timestamp,
                aum: float = 1e6,
                history: pd.Series | None = None) -> FlowContext:
        return FlowContext(previous=previous, date=date, aum=aum,
                           nav_per_unit=history if history is not None
                           else pd.Series(dtype=float))

    def test_periodic_flows_arrive_at_each_new_month_after_the_first_day(self):
        result = run(flows=PeriodicFlows(amount=10_000.0))

        assert [flow.date for flow in result.flows] == MONTH_STARTS
        assert all(flow.amount == 10_000.0 for flow in result.flows)
        assert flows_of(result, SUBSCRIPTION)[0].amount == 10_000.0

    def test_a_periodic_share_of_the_assets(self):
        scenario = PeriodicFlows(fraction=-0.02, frequency="QUARTERLY")
        scenario.start(pd.Timestamp("2024-01-02"))

        assert scenario.amount(self.context(pd.Timestamp("2024-04-01"),
                                            pd.Timestamp("2024-03-28"))) == -2e4
        assert scenario.amount(self.context(pd.Timestamp("2024-04-02"),
                                            pd.Timestamp("2024-04-01"))) == 0.0

    def test_random_flows_repeat_with_their_seed(self):
        first = run(flows=RandomFlows(volatility=0.2, seed=7))
        again = run(flows=RandomFlows(volatility=0.2, seed=7))
        other = run(flows=RandomFlows(volatility=0.2, seed=8))

        assert [f.amount for f in first.flows] == [f.amount for f in again.flows]
        assert [f.amount for f in first.flows] != [f.amount for f in other.flows]

    def test_performance_chasing_follows_the_trailing_return(self):
        scenario = PerformanceChasingFlows(sensitivity=0.5, lookback_days=2,
                                           frequency="DAILY")
        scenario.start(pd.Timestamp("2024-01-02"))
        rising = pd.Series([1.0, 1.05, 1.1])
        today = pd.Timestamp("2024-01-05")
        before = pd.Timestamp("2024-01-04")

        assert scenario.amount(self.context(today, before, history=rising)) == (
            pytest.approx(0.5 * 0.1 * 1e6))
        assert scenario.amount(self.context(today, before,
                                            history=rising.iloc[:2])) == 0.0

    @pytest.mark.parametrize("build", [
        PeriodicFlows,
        lambda: PeriodicFlows(amount=1.0, fraction=0.1),
        lambda: PeriodicFlows(amount=1.0, frequency="HOURLY"),
        lambda: RandomFlows(volatility=-0.1),
        lambda: PerformanceChasingFlows(lookback_days=0),
        lambda: Vehicle(management_fee_bps=-1.0),
        lambda: Vehicle(launch_price=0.0),
        lambda: Implementation(cash_buffer=1.0),
        lambda: Implementation(invest_flows="somewhere"),
    ])
    def test_an_impossible_setting_is_refused(self,
                                              build):
        with pytest.raises(ValueError):
            build()
