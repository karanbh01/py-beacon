# tests/test_splits.py
"""BN-244: a split between rebalances leaves the index level and NAV alone.

On a split's ex-date the stored close falls by the ratio. What holds a fixed
number of shares (the index's units, the backtest's position) has to rise by
the same ratio that day, or the level and NAV fall with the price.

Prices are flat apart from the split, so any movement is the mechanism under
test rather than the market.
"""
import numpy as np
import pandas as pd
import pytest

from beacon.backtest.engine import BacktestEngine
from beacon.data.base import MarketData, ReferenceData
from beacon.data.corporate_actions import CorporateActions, ratios_between
from beacon.data.fetcher import DataFetcher
from beacon.exceptions import CalculationError
from beacon.index.calculation import IndexCalculator
from beacon.index.chaining import chain_levels
from beacon.index.constructor import IndexDefinition
from beacon.index.methodology import MarketCapWeighted
from beacon.portfolio.base import Portfolio

START = "2021-01-04"
END = "2021-03-31"
EX_DATE = "2021-02-16"   # mid-quarter: no rebalance near it


def build_fetcher(ratio: float = 2.0,
                  status: str = "paid") -> DataFetcher:
    """AAA flat at 100; BBB flat at 50 until it splits by *ratio*."""
    rows = []

    for date in pd.bdate_range(START, END):
        split = date >= pd.Timestamp(EX_DATE)

        for identifier, price, shares in (
                ("AAA", 100.0, 1_000_000.0),
                ("BBB", 50.0 / ratio if split else 50.0,
                 1_000_000.0 * ratio if split else 1_000_000.0)):
            rows.append({"IDENTIFIER": identifier, "DATE": date,
                         "CLOSE": price, "SHARES_OUTSTANDING": shares,
                         "FREE_FLOAT": 1.0})

    reference = pd.DataFrame([
        {"IDENTIFIER": name, "DATE_FROM": START, "NAME": name,
         "CURRENCY": "USD", "EXCHANGE": "XNAS"} for name in ("AAA", "BBB")])

    actions = pd.DataFrame([
        {"IDENTIFIER": "BBB", "EX_DATE": EX_DATE,
         "TYPE": "SPLIT" if ratio >= 1 else "REVERSE_SPLIT",
         "VALUE": ratio, "STATUS": status}])

    return DataFetcher(MarketData.from_dataframe(pd.DataFrame(rows)),
                       ReferenceData.from_dataframe(reference),
                       CorporateActions.from_dataframe(actions))


def definition() -> IndexDefinition:
    return IndexDefinition(index_id="S", index_name="Splits", base_date=START,
                           base_value=1000.0, currency="USD",
                           eligibility_rules=[],
                           weighting_scheme=MarketCapWeighted(),
                           rebalancing_frequency="QUARTERLY",
                           calendar="XNYS",
                           universe_identifiers=["AAA", "BBB"])


def run_index(fetcher: DataFetcher):
    return IndexCalculator(definition(), fetcher).run(start_date=START,
                                                      end_date=END)


class TestTheIndex:

    @pytest.mark.parametrize("ratio", [2.0, 0.5, 1.05])
    def test_the_level_is_flat_across_the_ex_date(self,
                                                  ratio):
        """A split, a reverse split and a stock dividend."""
        levels = run_index(build_fetcher(ratio)).index_levels

        assert np.allclose(levels.to_numpy(), 1000.0), (
            f"level moved on a flat market: min {levels.min():.2f}, "
            f"max {levels.max():.2f}")

    def test_the_units_move_and_the_divisor_does_not(self):
        result = run_index(build_fetcher())
        amounts = result.daily_weights.pivot(index="DATE", columns="IDENTIFIER",
                                             values="AMOUNT")

        before = amounts.loc[:pd.Timestamp(EX_DATE)].iloc[-2]["BBB"]
        after = amounts.loc[pd.Timestamp(EX_DATE), "BBB"]

        assert after == pytest.approx(2 * before)
        assert result.divisor_history.nunique() == 1

    def test_a_cancelled_split_is_not_applied(self):
        """The prices here still split, so ignoring the action shows as a
        step in the level: the step is the evidence it was left out."""
        levels = run_index(build_fetcher(status="cancelled")).index_levels

        assert levels.min() < 1000.0 - 1.0


class TestAChainedIndex:
    """An optimised index holds units too, chained over the solved weights."""

    def test_the_level_is_flat_across_the_ex_date(self):
        fetcher = build_fetcher()
        parent = run_index(fetcher)
        first = parent.index_levels.index[0]

        levels = chain_levels("C", 1000.0, "USD", parent,
                              {first: {"AAA": 0.5, "BBB": 0.5}},
                              fetcher, "CLOSE").index_levels

        assert np.allclose(levels.to_numpy(), 1000.0)


class TestTheBacktest:

    @pytest.mark.parametrize("ratio", [2.0, 0.5])
    def test_the_nav_is_flat_across_the_ex_date(self,
                                                ratio):
        fetcher = build_fetcher(ratio)
        engine = BacktestEngine(start_date=START,
                                end_date=END,
                                initial_capital=1_000_000.0,
                                data_provider=fetcher,
                                index_result=run_index(fetcher),
                                calendar="XNYS")

        nav = engine.run().portfolio.nav

        assert np.allclose(nav.to_numpy(), 1_000_000.0, rtol=1e-9), (
            f"NAV moved on a flat market: min {nav.min():.2f}, "
            f"max {nav.max():.2f}")

    def test_the_share_count_moves_without_a_trade(self):
        portfolio = Portfolio(portfolio_id="p", initial_cash=1_000.0)
        portfolio.execute_buy("BBB", 10.0, 50.0)
        trades = len(portfolio.transactions)

        portfolio.apply_ratio("BBB", 2.0)
        holding = portfolio.holdings["BBB"]

        assert holding.quantity == 20.0
        assert holding.average_cost_price == 25.0
        assert holding.current_price == 25.0
        assert len(portfolio.transactions) == trades


class TestTheSchedule:

    def test_ratios_on_one_name_and_day_compound(self):
        actions = CorporateActions.from_dataframe(pd.DataFrame([
            {"IDENTIFIER": "BBB", "EX_DATE": EX_DATE, "TYPE": "SPLIT",
             "VALUE": 2.0},
            {"IDENTIFIER": "BBB", "EX_DATE": EX_DATE, "TYPE": "STOCK_DIVIDEND",
             "VALUE": 1.5}]))

        assert actions.ratio_schedule() == {pd.Timestamp(EX_DATE): {"BBB": 3.0}}

    def test_an_ex_date_the_loop_skipped_applies_on_the_next_day(self):
        schedule = {pd.Timestamp("2021-02-13"): {"BBB": 2.0}}   # a Saturday

        assert ratios_between(schedule, pd.Timestamp("2021-02-12"),
                              pd.Timestamp("2021-02-15")) == {"BBB": 2.0}
        assert ratios_between(schedule, pd.Timestamp("2021-02-15"),
                              pd.Timestamp("2021-02-16")) == {}

    def test_a_non_positive_ratio_is_refused(self):
        actions = CorporateActions.from_dataframe(pd.DataFrame([
            {"IDENTIFIER": "BBB", "EX_DATE": EX_DATE, "TYPE": "SPLIT",
             "VALUE": 0.0}]))

        with pytest.raises(CalculationError, match="non-positive split ratio"):
            actions.ratio_schedule()
