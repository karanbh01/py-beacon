# tests/test_backtest_stages.py
"""BN-264: the backtest as stages: target, screens, redistribution, trades.

Three names, equally weighted and rebalanced monthly. BIG is large and liquid
at 100. MID is mid-sized at 50. SMALL is small and thinly traded, and its
price falls from 10 to 2, so a price screen admits it at the first rebalance
and rejects it later.
"""
import numpy as np
import pandas as pd
import pytest

from beacon.backtest import (
    Backtest,
    BacktestEngine,
    ExclusionScreen,
    ExpressionScreen,
    Implementation,
    LiquidityScreen,
    ListingAgeScreen,
    MarketCapScreen,
    MinimumPriceScreen,
    ScreenContext,
)
from beacon.data.base import MarketData, ReferenceData
from beacon.data.fetcher import DataFetcher
from beacon.exceptions import ExpressionError
from beacon.expressions import data
from beacon.index.calculation import IndexCalculator
from beacon.index.constructor import IndexDefinition
from beacon.index.methodology import EqualWeighted
from beacon.testing.weights import index_result_from_weights

DAYS = pd.bdate_range("2024-01-02", "2024-04-30")
NAMES = ["BIG", "MID", "SMALL"]
CAPITAL = 1_000_000.0
SECTORS = {"BIG": "Technology", "MID": "Tobacco", "SMALL": "Retail"}

# The first rebalance and a later one, on XNYS sessions.
FIRST = pd.Timestamp("2024-01-02")
LATER = pd.Timestamp("2024-04-01")


def prices() -> dict[str, np.ndarray]:
    n = len(DAYS)

    return {"BIG": np.full(n, 100.0),
            "MID": np.full(n, 50.0),
            "SMALL": np.linspace(10.0, 2.0, n)}


def fetcher(late_listing: str | None = None) -> DataFetcher:
    """The three names; *late_listing* starts trading in March."""
    shares = {"BIG": 1e9, "MID": 1e8, "SMALL": 1e6}
    volume = {"BIG": 1e7, "MID": 1e6, "SMALL": 1e3}
    rows = []

    for name, path in prices().items():
        for day, price in zip(DAYS, path, strict=True):
            if name == late_listing and day < pd.Timestamp("2024-03-01"):
                continue

            rows.append({"IDENTIFIER": name, "DATE": day, "CLOSE": price,
                         "VOLUME": volume[name],
                         "SHARES_OUTSTANDING": shares[name]})

    reference = pd.DataFrame([
        {"IDENTIFIER": name, "DATE_FROM": "2020-01-01", "NAME": name,
         "CURRENCY": "USD", "EXCHANGE": "XNYS", "SECTOR": SECTORS[name]}
        for name in NAMES])

    return DataFetcher(MarketData.from_dataframe(pd.DataFrame(rows)),
                       ReferenceData.from_dataframe(reference))


def definition() -> IndexDefinition:
    return IndexDefinition(index_id="three", index_name="Three",
                           base_date=str(DAYS[0].date()), base_value=100.0,
                           currency="USD", eligibility_rules=[],
                           weighting_scheme=EqualWeighted(),
                           rebalancing_frequency="MONTHLY", calendar="XNYS",
                           universe_identifiers=NAMES)


def run(implementation: Implementation | None = None,
        data: DataFetcher | None = None,
        **engine_args):
    data = data if data is not None else fetcher()
    index = IndexCalculator(definition(), data).run(
        end_date=str(DAYS[-1].date()))

    return BacktestEngine(start_date=str(DAYS[0].date()),
                          end_date=str(DAYS[-1].date()),
                          initial_capital=CAPITAL, data_provider=data,
                          index_result=index, implementation=implementation,
                          **engine_args).run()


def held_on(result,
            date: pd.Timestamp) -> set[str]:
    positions = result.portfolio.positions
    day = positions[positions["DATE"] == date]

    return set(day.loc[day["QUANTITY"] > 0, "ASSET_ID"])


def step_on(result,
            date: pd.Timestamp):
    return next(step for step in result.rebalance_steps if step.date == date)


class TestWithoutAnImplementation:

    def test_every_rebalance_is_recorded_and_nothing_removed(self):
        result = run()

        assert len(result.rebalance_steps) == 4
        assert all(step.removed == {} for step in result.rebalance_steps)
        assert all(step.weights == pytest.approx(step.target)
                   for step in result.rebalance_steps)

    def test_an_empty_implementation_changes_nothing(self):
        plain = run()
        empty = run(Implementation())

        pd.testing.assert_series_equal(plain.trading_nav, empty.trading_nav)


class TestScreens:

    def test_a_screened_name_is_never_held_and_its_weight_spread(self):
        result = run(Implementation(screens=[MarketCapScreen(min_cap=1e8)]))
        step = step_on(result, FIRST)

        assert step.removed == {"SMALL": "MarketCapScreen"}
        assert step.weights == pytest.approx({"BIG": 0.5, "MID": 0.5})
        assert "SMALL" not in held_on(result, FIRST)

    def test_redistribution_to_cash_leaves_the_weight_in_cash(self):
        result = run(Implementation(screens=[MarketCapScreen(min_cap=1e8)],
                                    redistribution="cash"))
        step = step_on(result, FIRST)

        assert step.weights == pytest.approx({"BIG": 1 / 3, "MID": 1 / 3})
        assert step.cash_weight == pytest.approx(1 / 3)
        assert (result.portfolio.cash.loc[FIRST]
                == pytest.approx(CAPITAL / 3, rel=1e-6))

    def test_a_holding_that_starts_failing_is_sold(self):
        """BN-255: the old ExpressionScreen only trimmed a failing holding."""
        result = run(Implementation(screens=[MinimumPriceScreen(min_price=5.0)]))

        assert "SMALL" in held_on(result, FIRST)
        assert step_on(result, LATER).removed == {"SMALL": "MinimumPriceScreen"}
        assert "SMALL" not in held_on(result, LATER)

    def test_an_expression_screen_removes_the_name_it_rejects(self):
        result = run(Implementation(
            screens=[ExpressionScreen(data.reference.sector != "Tobacco")]))

        assert step_on(result, FIRST).removed == {"MID": "ExpressionScreen"}

    def test_an_expression_screen_with_a_typo_refuses_the_run(self):
        with pytest.raises(ExpressionError, match="sectr"):
            run(Implementation(
                screens=[ExpressionScreen(data.reference.sectr == "Retail")]))

    def test_liquidity_by_volume_and_by_traded_value(self):
        by_volume = run(Implementation(
            screens=[LiquidityScreen(min_volume=1e4)]))
        by_value = run(Implementation(
            screens=[LiquidityScreen(min_traded_value=1e8)]))

        assert step_on(by_volume, FIRST).removed == {"SMALL": "LiquidityScreen"}
        # BIG trades 1e9 a day, MID 5e7, SMALL about 1e4.
        assert set(step_on(by_value, FIRST).removed) == {"MID", "SMALL"}

    def test_a_late_listing_waits_for_its_age(self):
        """A schedule built by hand, since an index cannot weight a name
        before it has a price."""
        data = fetcher(late_listing="MID")
        march = pd.Timestamp("2024-03-01")
        equal = dict.fromkeys(NAMES, 1 / 3)
        index = index_result_from_weights({FIRST: equal, march: equal,
                                           LATER: equal})

        result = BacktestEngine(
            start_date=str(DAYS[0].date()), end_date=str(DAYS[-1].date()),
            initial_capital=CAPITAL, data_provider=data, index_result=index,
            implementation=Implementation(
                screens=[ListingAgeScreen(min_days=30)])).run()

        # Every name's data starts on FIRST, so all are too young then.
        assert set(step_on(result, FIRST).removed) == set(NAMES)
        assert step_on(result, march).removed == {"MID": "ListingAgeScreen"}
        assert "MID" not in step_on(result, LATER).removed

    def test_exclusions_by_name_and_by_sector(self):
        result = run(Implementation(screens=[
            ExclusionScreen(identifiers=["SMALL"], sectors=["Tobacco"])]))

        assert step_on(result, FIRST).removed == {
            "SMALL": "ExclusionScreen", "MID": "ExclusionScreen"}
        assert step_on(result, FIRST).weights == {"BIG": pytest.approx(1.0)}

    def test_the_first_screen_to_reject_is_the_one_recorded(self):
        result = run(Implementation(screens=[
            MinimumPriceScreen(min_price=20.0), MarketCapScreen(min_cap=1e8)]))

        assert step_on(result, FIRST).removed == {"SMALL": "MinimumPriceScreen"}


class TestBuffers:

    def context(self) -> ScreenContext:
        return ScreenContext(fetcher(), "USD")

    def test_a_held_name_stays_until_the_exit_level(self):
        """SMALL is at about 4.6 on 1 April: under the entry, over the exit."""
        screen = MinimumPriceScreen(min_price=5.0, exit=3.0)

        assert screen.admits("SMALL", LATER, True, self.context())
        assert not screen.admits("SMALL", LATER, False, self.context())

    def test_without_an_exit_both_levels_are_the_entry(self):
        screen = MinimumPriceScreen(min_price=5.0)

        assert not screen.admits("SMALL", LATER, True, self.context())

    def test_the_buffer_keeps_a_holding_in_a_run(self):
        result = run(Implementation(
            screens=[MinimumPriceScreen(min_price=5.0, exit=3.0)]))

        assert "SMALL" not in step_on(result, LATER).removed
        assert "SMALL" in held_on(result, LATER)

    def test_an_exit_above_the_entry_is_refused(self):
        with pytest.raises(ValueError, match="above the entry level"):
            MinimumPriceScreen(min_price=5.0, exit=6.0)


class TestTheFrontDoor:

    def test_a_backtest_passes_its_implementation_to_the_engine(self):
        result = Backtest(initial_capital=CAPITAL, data_provider=fetcher(),
                          implementation=Implementation(
                              screens=[MarketCapScreen(min_cap=1e8)])).run(
            definition(), end=str(DAYS[-1].date()))

        assert step_on(result, FIRST).removed == {"SMALL": "MarketCapScreen"}

    def test_a_screen_passed_as_a_modifier_says_where_it_goes(self):
        with pytest.raises(TypeError, match="Implementation"):
            run(modifiers=[MarketCapScreen(min_cap=1e8)])

    def test_an_unknown_redistribution_is_refused(self):
        with pytest.raises(ValueError, match="redistribution"):
            Implementation(redistribution="evenly")
