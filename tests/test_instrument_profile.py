# tests/test_instrument_profile.py
"""BN-468: a name's profile over the run, `instrument.profile(name)`: how
each rebalance weighted it and how its weight drifted each day.

The run is test_strategy_futures' eight-name market-cap index held through
2024 with each name capped at 30% and positions under 5.5% after the cap
dropped, so N0 is capped and N7 removed at the first rebalance; its total-return version is
the benchmark.
"""
import dataclasses

import pandas as pd
import pytest

from beacon.analysis import RunData
from beacon.analysis.analytics import Analytics
from beacon.analysis.analytics.profile import ACTIVE, LABELS, Profile, columns_of
from beacon.analysis.panels import FUND_WEIGHTS, INDEX_WEIGHTS, REBALANCE_WEIGHTS
from beacon.analysis.reports.statistics import RUN_PREDATES, Unavailable
from beacon.backtest import Implementation
from beacon.backtest.breakdown import REBALANCE_COLUMNS, rebalance_weights
from beacon.backtest.capacity import MinimumPosition, WeightCap

CAP, SMALLEST = 0.3, 0.055


@pytest.fixture(scope="module")
def result():
    from beacon.index.calculation import IndexCalculator
    from test_strategy_futures import DATA, END, START, index, run

    benchmark = IndexCalculator(index("TOTAL_RETURN"), DATA).run(start_date=START, end_date=END)
    implementation = Implementation(caps=[WeightCap(CAP)],
                                    minimum_position=MinimumPosition(weight=SMALLEST))

    return run(index(), implementation=implementation, benchmark=benchmark,
               transaction_cost_bps=5.0)


@pytest.fixture(scope="module")
def tables(result):
    return result.analytics._context.run.tables


class TestTheRunKeepsItsRebalances:

    def test_a_row_for_each_name_each_stage_saw(self,
                                                result):
        found = rebalance_weights(result)
        first = result.rebalance_steps[0]
        mine = found.loc[found.index == first.date].set_index("IDENTIFIER")

        assert tuple(found.columns) == REBALANCE_COLUMNS
        assert set(mine.index) == set(first.target) | set(first.removed)
        assert mine.loc["N0", "CAPPED"] == pytest.approx(CAP)
        assert mine.loc["N7", "REMOVED"] == first.removed["N7"]
        assert mine.loc["N7", "WEIGHT"] == 0.0

    def test_it_is_one_of_the_run_s_tables(self,
                                           result,
                                           tables):
        pd.testing.assert_frame_equal(tables[REBALANCE_WEIGHTS], rebalance_weights(result))


class TestTheRebalances:

    @pytest.fixture(scope="class")
    def capped(self,
               result):
        return result.analytics.instrument.profile("N0").rebalance_weights

    def test_each_stage_s_weight(self,
                                 result,
                                 capped):
        steps = result.rebalance_steps

        assert list(capped.index) == [step.date for step in steps]
        assert capped["target"].tolist() == pytest.approx([step.target["N0"] for step in steps])
        assert capped["screened"].tolist() == pytest.approx([step.screened["N0"]
                                                             for step in steps])
        assert capped["weight"].tolist() == pytest.approx([CAP] * len(steps))
        assert capped["capped"].tolist() == pytest.approx([CAP] * len(steps))
        assert capped["removed"].isna().all()

    def test_each_comparator_s_weight_that_day(self,
                                               tables,
                                               capped):
        index = tables[INDEX_WEIGHTS]["N0"]

        assert list(capped.columns[-2:]) == ["index", "benchmark"]
        assert capped["index"].tolist() == pytest.approx(index.reindex(capped.index).tolist())

    def test_a_removed_name(self,
                            result):
        removed = result.analytics.instrument.profile("N7").rebalance_weights
        first = result.rebalance_steps[0]

        assert removed.loc[first.date, "removed"] == first.removed["N7"]
        assert removed.loc[first.date, "weight"] == 0.0
        assert removed.loc[first.date, "target"] == pytest.approx(first.target["N7"])

    def test_over_a_window(self,
                           result):
        found = result.analytics.instrument.profile("N0", start="2024-03-01").rebalance_weights

        assert found.index[0] == pd.Timestamp("2024-04-01")


class TestTheDays:

    def test_the_fund_s_drifted_weight_and_its_active_weight(self,
                                                              result,
                                                              tables):
        found = result.analytics.instrument.profile("N0").daily_weights
        fund = tables[FUND_WEIGHTS]["N0"].fillna(0.0)

        assert list(found.columns) == ["fund", "index", "benchmark", ACTIVE]
        pd.testing.assert_series_equal(found["fund"], fund.reindex(found.index),
                                       check_names=False)
        pd.testing.assert_series_equal(found[ACTIVE], found["fund"] - found["index"],
                                       check_names=False)
        assert found.attrs == {"against": "index"}
        # The cap holds the fund under the index each day.
        assert (found[ACTIVE] < 0).all()

    def test_against_the_benchmark(self,
                                   result):
        found = result.analytics.instrument.profile("N0", against="benchmark").daily_weights

        assert found.attrs == {"against": "benchmark"}
        pd.testing.assert_series_equal(found[ACTIVE], found["fund"] - found["benchmark"],
                                       check_names=False)

    def test_a_day_a_portfolio_did_not_hold_it_is_0(self,
                                                    result):
        found = result.analytics.instrument.profile("N7").daily_weights
        first, second = (step.date for step in result.rebalance_steps[:2])
        out = found.loc[(found.index >= first) & (found.index < second)]

        # Removed at the first rebalance, held by the index throughout.
        assert (out["fund"] == 0.0).all() and (out["index"] > 0.0).all()


class TestItsParts:

    def test_each_column_s_label_and_unit(self,
                                          result):
        profile = result.analytics.instrument.profile("N0")
        columns = {column.name: column for column in columns_of(profile.rebalance_weights)}

        assert columns["target"].unit == "fraction" and columns["removed"].unit is None
        assert columns["capped"].label == LABELS["capped"]
        assert Profile.parts == ("rebalance_weights", "daily_weights")
        assert repr(profile) == "Profile('N0': rebalance_weights, daily_weights)"

    def test_a_stored_run_s_is_the_same(self,
                                        result,
                                        tmp_path):
        result.analytics.run_data.write(tmp_path)
        stored = Analytics(RunData.from_record(tmp_path)).instrument.profile("N7")
        finished = result.analytics.instrument.profile("N7")

        pd.testing.assert_frame_equal(stored.rebalance_weights, finished.rebalance_weights)
        pd.testing.assert_frame_equal(stored.daily_weights, finished.daily_weights)

    def test_a_run_kept_before_its_rebalances_were(self,
                                                   result,
                                                   tables):
        older = dataclasses.replace(result.analytics.run_data,
                                    tables={name: table for name, table in tables.items()
                                            if name != REBALANCE_WEIGHTS})
        profile = Analytics(older).instrument.profile("N0")

        with pytest.raises(Unavailable) as raised:
            _ = profile.rebalance_weights

        assert raised.value.code == RUN_PREDATES
        assert not profile.daily_weights.empty


class TestRefusals:

    def test_a_name_no_portfolio_held(self,
                                      result):
        with pytest.raises(ValueError, match="No portfolio of the run held ZZZ"):
            result.analytics.instrument.profile("ZZZ")

    def test_a_comparator_the_run_lacks(self,
                                        result):
        with pytest.raises(Unavailable):
            result.analytics.instrument.profile("N0", against="optimised")
