# tests/test_backtest_capacity.py
"""BN-265: capacity caps that scale with the fund.

The capping arithmetic is checked on `plan` with weight caps, which need no
data. The data-driven caps run through the engine over three names: BIG
(free-float cap 80 billion, traded 1 billion a day), MID (4 billion, 50
million a day) and SMALL (8 million, 10 thousand a day).
"""
import pandas as pd
import pytest

from beacon.backtest import (
    BacktestEngine,
    Implementation,
    LiquidityCap,
    MinimumPosition,
    OwnershipCap,
    ScreenContext,
    WeightCap,
)
from beacon.backtest.implementation import plan
from beacon.data.base import MarketData, ReferenceData
from beacon.data.fetcher import DataFetcher
from beacon.index.calculation import IndexCalculator
from beacon.index.constructor import IndexDefinition
from beacon.index.methodology import EqualWeighted

DAYS = pd.bdate_range("2024-01-02", "2024-03-28")
FIRST = pd.Timestamp("2024-01-02")
NAMES = ["BIG", "MID", "SMALL"]
PRICE = {"BIG": 100.0, "MID": 50.0, "SMALL": 10.0}
SHARES = {"BIG": 1e9, "MID": 1e8, "SMALL": 1e6}
VOLUME = {"BIG": 1e7, "MID": 1e6, "SMALL": 1e3}
FREE_FLOAT = 0.8


def fetcher() -> DataFetcher:
    rows = [{"IDENTIFIER": name, "DATE": day, "CLOSE": PRICE[name],
             "VOLUME": VOLUME[name], "SHARES_OUTSTANDING": SHARES[name],
             "FREE_FLOAT": FREE_FLOAT}
            for name in NAMES for day in DAYS]
    reference = pd.DataFrame([
        {"IDENTIFIER": name, "DATE_FROM": "2020-01-01", "NAME": name,
         "CURRENCY": "USD", "EXCHANGE": "XNYS"} for name in NAMES])

    return DataFetcher(MarketData.from_dataframe(pd.DataFrame(rows)),
                       ReferenceData.from_dataframe(reference))


def stepped(implementation: Implementation,
            target: dict[str, float],
            book_value: float = 1e6):
    return plan(implementation, target, FIRST, held=set(), stale=set(),
                context=ScreenContext(fetcher(), "USD"),
                book_value=book_value)


def run(capital: float,
        implementation: Implementation):
    data = fetcher()
    definition = IndexDefinition(index_id="three", index_name="Three",
                                 base_date=str(DAYS[0].date()),
                                 base_value=100.0, currency="USD",
                                 eligibility_rules=[],
                                 weighting_scheme=EqualWeighted(),
                                 rebalancing_frequency="MONTHLY",
                                 calendar="XNYS", universe_identifiers=NAMES)
    index = IndexCalculator(definition, data).run(end_date=str(DAYS[-1].date()))

    return BacktestEngine(start_date=str(DAYS[0].date()),
                          end_date=str(DAYS[-1].date()),
                          initial_capital=capital, data_provider=data,
                          index_result=index,
                          implementation=implementation).run()


class TestTheCappingArithmetic:

    def test_the_excess_is_spread_until_no_name_is_over(self):
        """A at 0.6 is cut to 0.4; its 0.2 lifts B to 0.45, over the cap
        too, so B is cut and C takes the rest."""
        step = stepped(Implementation(caps=[WeightCap(0.4)]),
                       {"A": 0.6, "B": 0.3, "C": 0.1})

        assert step.weights == pytest.approx({"A": 0.4, "B": 0.4, "C": 0.2})
        assert step.capped == pytest.approx({"A": 0.4, "B": 0.4})

    def test_under_the_cash_rule_the_excess_is_cash(self):
        step = stepped(Implementation(caps=[WeightCap(0.4)],
                                      redistribution="cash"),
                       {"A": 0.6, "B": 0.3, "C": 0.1})

        assert step.weights == pytest.approx({"A": 0.4, "B": 0.3, "C": 0.1})
        assert step.cash_weight == pytest.approx(0.2)

    def test_when_every_name_is_capped_the_rest_is_cash(self):
        step = stepped(Implementation(caps=[WeightCap(0.25)]),
                       {"A": 0.5, "B": 0.3, "C": 0.2})

        assert step.weights == pytest.approx({"A": 0.25, "B": 0.25, "C": 0.25})
        assert step.cash_weight == pytest.approx(0.25)

    def test_the_tightest_cap_applies(self):
        step = stepped(Implementation(caps=[WeightCap(0.5), WeightCap(0.4)]),
                       {"A": 0.6, "B": 0.4})

        assert step.weights["A"] == pytest.approx(0.4)

    def test_a_position_too_small_is_dropped_and_redistributed(self):
        step = stepped(Implementation(
            minimum_position=MinimumPosition(weight=0.15)),
            {"A": 0.6, "B": 0.3, "C": 0.1})

        assert step.removed == {"C": "MinimumPosition"}
        assert step.weights == pytest.approx({"A": 0.6 / 0.9, "B": 0.3 / 0.9})

    def test_a_minimum_by_value_depends_on_the_book(self):
        target = {"A": 0.9, "B": 0.1}
        implementation = Implementation(
            minimum_position=MinimumPosition(value=50_000))

        assert stepped(implementation, target, book_value=1e6).removed == {}
        assert stepped(implementation, target,
                       book_value=1e5).removed == {"B": "MinimumPosition"}

    def test_dropping_a_small_position_respects_the_caps(self):
        """C's weight goes to A and B, but A is already at its cap."""
        step = stepped(Implementation(caps=[WeightCap(0.5)],
                                      minimum_position=MinimumPosition(weight=0.1)),
                       {"A": 0.5, "B": 0.42, "C": 0.08})

        assert step.weights == pytest.approx({"A": 0.5, "B": 0.5})

    def test_no_caps_changes_nothing(self):
        target = {"A": 0.6, "B": 0.4}

        assert stepped(Implementation(), target).weights == target


class TestCapsFromTheData:

    def test_ownership_limits_a_small_name_in_a_large_fund(self):
        """SMALL's free float is 8 million; 1% of it is 80 thousand, a weight
        of 0.008 in a book of 10 million."""
        result = run(1e7, Implementation(caps=[OwnershipCap(0.01)]))
        step = result.rebalance_steps[0]

        assert step.capped == pytest.approx({"SMALL": 8e4 / 1e7})
        assert step.weights["BIG"] == pytest.approx(step.weights["MID"])
        assert sum(step.weights.values()) == pytest.approx(1.0)

    def test_the_cap_shrinks_smoothly_as_the_fund_grows(self):
        small_fund = run(1e7, Implementation(caps=[OwnershipCap(0.01)]))
        large_fund = run(1e8, Implementation(caps=[OwnershipCap(0.01)]))

        small = small_fund.rebalance_steps[0].weights["SMALL"]
        large = large_fund.rebalance_steps[0].weights["SMALL"]

        assert large == pytest.approx(small / 10)

    def test_liquidity_limits_by_days_and_participation(self):
        """SMALL trades 10 thousand a day: two days at 20% is 4 thousand."""
        result = run(1e7, Implementation(caps=[LiquidityCap(days=2)]))

        assert result.rebalance_steps[0].capped == pytest.approx(
            {"SMALL": 4e3 / 1e7})

    def test_the_book_holds_what_the_cap_allows(self):
        result = run(1e7, Implementation(caps=[OwnershipCap(0.01)]))
        held = result.portfolio.positions
        small = held[(held["DATE"] == FIRST) & (held["ASSET_ID"] == "SMALL")]

        assert small["MARKET_VALUE"].iloc[0] == pytest.approx(8e4, rel=1e-6)


class TestTheCapsAreChecked:

    @pytest.mark.parametrize("build", [
        lambda: OwnershipCap(0.0),
        lambda: LiquidityCap(days=0),
        lambda: LiquidityCap(days=1, participation=1.5),
        lambda: WeightCap(1.2),
        MinimumPosition,
    ])
    def test_an_impossible_setting_is_refused(self,
                                              build):
        with pytest.raises(ValueError):
            build()
