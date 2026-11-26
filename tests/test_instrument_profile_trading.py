# tests/test_instrument_profile_trading.py
"""BN-473: a name's trades and events through the run,
`instrument.profile(name)`'s `trades` and `events`.

The run is test_instrument_profile's: test_strategy_futures' eight names
held through 2024, each capped at 30% and positions under 5.5% dropped, so
N0 is capped at every rebalance and N7 screened out until the last.
"""
import dataclasses

import pandas as pd
import pytest

from beacon.analysis.analytics import Analytics
from beacon.analysis.analytics.profile import columns_of
from beacon.analysis.panels import ACTIONS, FUND_WEIGHTS, TRADES
from beacon.analysis.reports.statistics import RUN_PREDATES, Unavailable
from test_instrument_profile import CAP, SMALLEST

FIRST = pd.Timestamp("2024-01-02")


@pytest.fixture(scope="module")
def result():
    from beacon.backtest import Implementation
    from beacon.backtest.capacity import MinimumPosition, WeightCap
    from test_strategy_futures import index, run

    implementation = Implementation(caps=[WeightCap(CAP)],
                                    minimum_position=MinimumPosition(weight=SMALLEST))

    return run(index(), implementation=implementation, transaction_cost_bps=5.0)


def replaced(result,
             **tables) -> Analytics:
    """The run's analytics with some of its tables replaced, or left out
    where given None."""
    data = result.analytics.run_data
    kept = {name: table for name, table in {**data.tables, **tables}.items()
            if table is not None}

    return Analytics(dataclasses.replace(data, tables=kept))


class TestItsTrades:

    def test_every_trade_in_it(self,
                               result):
        found = result.analytics.instrument.profile("N0").trades
        kept = result.analytics.run_data.tables[TRADES]
        mine = kept.loc[kept["IDENTIFIER"] == "N0"]

        assert list(found.columns) == ["side", "quantity", "price", "notional", "cost",
                                       "impact", "reason", "tags"]
        assert list(found.index) == list(mine.index)
        assert found["notional"].tolist() == pytest.approx(mine["VALUE"].tolist())
        assert set(found["reason"]) == {"rebalance"}

    def test_over_a_window(self,
                           result):
        found = result.analytics.instrument.profile("N0", start="2024-06-01").trades

        assert (found.index >= pd.Timestamp("2024-06-01")).all() and len(found)

    def test_their_units(self,
                         result):
        columns = {column.name: column
                   for column in columns_of(result.analytics.instrument.profile("N0").trades)}

        assert columns["notional"].unit == "money" and columns["side"].unit is None

    def test_a_run_kept_before_its_trades_were(self,
                                               result):
        with pytest.raises(Unavailable) as raised:
            _ = replaced(result, **{TRADES: None}).instrument.profile("N0").trades

        assert raised.value.code == RUN_PREDATES


class TestItsEvents:

    def test_added_and_capped(self,
                              result):
        found = result.analytics.instrument.profile("N0").events
        steps = [step.date for step in result.rebalance_steps]

        assert found.loc[FIRST, "event"].tolist() == ["added_to_fund", "added_to_index",
                                                      "capped"]
        capped = found.loc[found["event"] == "capped"]
        assert list(capped.index) == steps
        assert capped["amount"].tolist() == pytest.approx([CAP] * len(steps))

    def test_screened_out_until_the_fund_bought_it(self,
                                                    result):
        found = result.analytics.instrument.profile("N7").events
        steps = result.rebalance_steps
        screened = found.loc[found["event"] == "screened_out"]
        bought = found.index[found["event"] == "added_to_fund"]

        assert list(screened.index) == [step.date for step in steps if "N7" in step.removed]
        assert set(screened["detail"]) == {steps[0].removed["N7"]}
        assert len(bought) == 1 and bought[0] > screened.index[-1]

    def test_sold_out_of_the_fund(self,
                                  result):
        weights = result.analytics.run_data.tables[FUND_WEIGHTS].copy()
        weights.loc[weights.index >= "2024-08-01", "N1"] = 0.0
        found = replaced(result, **{FUND_WEIGHTS: weights}).instrument.profile("N1").events
        sold = found.index[found["event"] == "removed_from_fund"]

        assert list(sold) == [weights.index[weights.index >= "2024-08-01"][0]]

    def test_a_corporate_action(self,
                                result):
        actions = result.analytics.run_data.tables[ACTIONS]
        split = pd.DataFrame({"IDENTIFIER": ["N0"], "EX_DATE": [pd.Timestamp("2024-06-03")],
                              "TYPE": ["SPLIT"], "VALUE": [2.0]})
        found = replaced(result, **{ACTIONS: pd.concat([actions, split])}).instrument.profile(
            "N0").events

        assert found.loc[found["event"] == "corporate_action", ["detail", "amount"]
                         ].values.tolist() == [["SPLIT", 2.0]]

    def test_each_kind_is_labelled(self,
                                   result):
        events = columns_of(result.analytics.instrument.profile("N0").events)[0]

        assert events.labels["screened_out"] == "Screened out"
        assert result.analytics.instrument.profile("N0").events["detail"].iloc[0] is None
