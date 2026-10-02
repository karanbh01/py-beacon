# tests/test_backtest_execution.py
"""BN-266 and BN-252: costs that depend on size, how fast orders can trade,
and buys sized net of their costs.

BIG moves up and down 1% on alternate days and trades 10 million shares a
day. THIN is flat at 10 and trades 1,000 shares a day, unless a test says
otherwise. NOVOL is flat at 10 with no volume at all. The data starts a
quarter before the backtest, so impact has a volatility to work from.
"""
import logging

import numpy as np
import pandas as pd
import pytest

from beacon import ModellingAssumptions
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


def fetcher(thin_volume: dict | None = None) -> DataFetcher:
    """*thin_volume* replaces THIN's volume on the days it names."""
    thin_volume = thin_volume or {}
    rows = [{"IDENTIFIER": "BIG", "DATE": day, "CLOSE": price, "VOLUME": 1e7}
            for day, price in zip(HISTORY, big_prices(), strict=True)]
    rows += [{"IDENTIFIER": "THIN", "DATE": day, "CLOSE": 10.0,
              "VOLUME": thin_volume.get(day, 1e3)} for day in HISTORY]
    rows += [{"IDENTIFIER": "NOVOL", "DATE": day, "CLOSE": 10.0}
             for day in HISTORY]
    reference = pd.DataFrame([
        {"IDENTIFIER": name, "DATE_FROM": "2020-01-01", "NAME": name,
         "CURRENCY": "USD", "EXCHANGE": "XNYS"}
        for name in ("BIG", "THIN", "NOVOL")])

    return DataFetcher(MarketData.from_dataframe(pd.DataFrame(rows)),
                       ReferenceData.from_dataframe(reference))


def run(capital: float,
        schedule: dict | None = None,
        data: DataFetcher | None = None,
        **engine_args):
    schedule = schedule or {FIRST: {"BIG": 1.0}}

    return BacktestEngine(start_date=str(DAYS[0].date()),
                          end_date=str(DAYS[-1].date()),
                          initial_capital=capital,
                          data_provider=data if data is not None else fetcher(),
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


class TestTheDaysVolume:
    """Which volume a participation limit holds a name to."""

    LIMIT = ExecutionLimit(participation=0.1, lookback_days=20)
    DAY = DAYS[10]

    def volume(self,
               thin_volume: dict,
               backfill_days: int = 5,
               name: str = "THIN") -> float | None:
        return self.LIMIT.volume(name, self.DAY, fetcher(thin_volume),
                                 backfill_days)

    def test_the_days_volume_when_reported(self):
        assert self.volume({self.DAY: 2500.0}) == 2500.0

    def test_a_volume_of_zero_stands(self):
        """Nothing traded, perhaps a one-off holiday: not a gap to fill."""
        assert self.volume({self.DAY: 0.0}) == 0.0

    def test_a_blank_takes_the_last_reported_within_the_backfill(self):
        yesterday = DAYS[9]

        assert self.volume({yesterday: 700.0, self.DAY: np.nan}) == 700.0

    def test_past_the_backfill_a_blank_takes_the_average(self):
        """The last report was the day before; with no backfill the average
        of the 20 reported days before stands in."""
        volumes = {day: float(i) for i, day in enumerate(HISTORY)}
        volumes[self.DAY] = np.nan
        reported = [v for day, v in volumes.items() if day < self.DAY]

        assert self.volume(volumes, backfill_days=0) == pytest.approx(
            np.mean(reported[-20:]))

    def test_a_name_with_no_volume_has_none(self):
        assert self.volume({}, name="NOVOL") is None

    def test_the_backfill_defaults_to_five_days(self):
        assert ModellingAssumptions().resolved(
            fetcher()).volume_backfill_days == 5

    def test_a_day_with_zero_volume_trades_nothing(self):
        result = run(1e4, {FIRST: {"THIN": 0.5, "BIG": 0.5}},
                     data=fetcher({DAYS[1]: 0.0}),
                     implementation=Implementation(
                         execution=ExecutionLimit(participation=0.1)))

        assert [held(result, "THIN", day) for day in DAYS[:3]] == pytest.approx(
            [100.0, 100.0, 200.0])

    def test_the_run_takes_its_backfill_from_the_assumptions(self):
        """A blank on day two: carried, 1,000 traded at 10%; not carried,
        the average of the 1,000s and the 5,000s before it."""
        spikes = dict.fromkeys(HISTORY[:-len(DAYS)], 5e3)
        data = {**spikes, DAYS[1]: np.nan}
        args = {"implementation": Implementation(
            execution=ExecutionLimit(participation=0.1))}
        carried = run(1e4, {FIRST: {"THIN": 0.5, "BIG": 0.5}},
                      data=fetcher(data), **args)
        averaged = run(1e4, {FIRST: {"THIN": 0.5, "BIG": 0.5}},
                       data=fetcher(data),
                       modelling_assumptions=ModellingAssumptions(
                           volume_backfill_days=0), **args)

        assert held(carried, "THIN", DAYS[1]) == pytest.approx(200.0)
        assert held(averaged, "THIN", DAYS[1]) > 200.0

    def test_a_name_with_no_volume_is_not_held_back_and_warns_once(self,
                                                                    caplog):
        with caplog.at_level(logging.WARNING, logger="beacon.backtest.execution"):
            result = run(1e4, {FIRST: {"NOVOL": 0.5, "BIG": 0.5}},
                         implementation=Implementation(
                             execution=ExecutionLimit(participation=0.1)))

        assert held(result, "NOVOL", FIRST) == pytest.approx(500.0)
        assert sum("no volume" in record.message
                   for record in caplog.records) == 1


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
