# tests/test_instrument_profile_data.py
"""BN-472: a name's style, key data and corporate actions through the run,
`instrument.profile(name)`'s `style`, `key_data` and `corporate_actions`.

The run is test_strategy_futures' eight-name market-cap index held through
2024, rebalanced each quarter; its dividends all went ex in 2023, so a split
is added to what it kept to see one in the window.
"""
import dataclasses

import pandas as pd
import pytest

from beacon.analysis import RunData
from beacon.analysis.analytics import Analytics
from beacon.analysis.analytics.profile import columns_of
from beacon.analysis.panels import ACTIONS, CHARACTERISTICS, REFERENCE
from beacon.analysis.reports.statistics import RUN_PREDATES, Unavailable

NAME = "N0"
SPLIT = pd.Timestamp("2024-06-03")


@pytest.fixture(scope="module")
def result():
    from test_strategy_futures import index, run

    return run(index(), transaction_cost_bps=5.0)


@pytest.fixture(scope="module")
def profile(result):
    return result.analytics.instrument.profile(NAME)


def replaced(result,
             **tables) -> Analytics:
    """The run's analytics with some of its tables replaced, or left out
    where given None."""
    data = result.analytics.run_data
    kept = {name: table for name, table in {**data.tables, **tables}.items()
            if table is not None}

    return Analytics(dataclasses.replace(data, tables=kept))


class TestItsStyle:

    def test_a_row_each_rebalance_kept(self,
                                       result,
                                       profile):
        found = profile.style
        kept = result.analytics.run_data.tables[CHARACTERISTICS]

        assert list(found.index) == sorted(pd.unique(kept.index))
        assert list(found.columns) == ["size_position", "style_position", "size", "style",
                                       "moved"]
        assert set(found["size"]) <= {"large", "mid", "small"}

    def test_the_last_is_where_the_run_s_style_box_puts_it(self,
                                                           result,
                                                           profile):
        placed = result.analytics.instrument.style_positions().set_index("name")
        last = profile.style.iloc[-1]

        assert last["size_position"] == pytest.approx(placed.loc[NAME, "size"], rel=1e-12)
        assert last["style_position"] == pytest.approx(placed.loc[NAME, "style"], rel=1e-12)

    def test_moved_says_when_its_cell_changed(self,
                                              profile):
        found = profile.style
        boxes = found["size"] + " " + found["style"]

        assert not found["moved"].iloc[0]
        assert found["moved"].iloc[1:].tolist() == (boxes.iloc[1:].to_numpy()
                                                    != boxes.iloc[:-1].to_numpy()).tolist()

    def test_a_run_kept_before_its_characteristics_were(self,
                                                        result):
        older = replaced(result, **{CHARACTERISTICS: None})

        with pytest.raises(Unavailable) as raised:
            _ = older.instrument.profile(NAME).style

        assert raised.value.code == RUN_PREDATES


class TestItsKeyData:

    def test_its_reference_and_market_cap_each_rebalance(self,
                                                         result,
                                                         profile):
        found = profile.key_data
        caps = result.analytics.run_data.tables[CHARACTERISTICS]
        caps = caps.loc[caps["IDENTIFIER"] == NAME, "MARKET_CAP"]

        assert list(found.index) == sorted(pd.unique(
            result.analytics.run_data.tables[REFERENCE].index))
        assert (found["currency"] == "USD").all() and (found["exchange"] == "XNYS").all()
        assert found["market_cap"].tolist() == pytest.approx(caps.reindex(found.index).tolist())

    def test_its_name_and_free_float_with_the_data_store(self,
                                                         profile):
        found = profile.key_data

        assert found.columns[0] == "name" and (found["name"] == NAME).all()
        assert "free_float" in found.columns

    def test_a_stored_run_without_the_data_store(self,
                                                 result,
                                                 profile,
                                                 tmp_path):
        result.analytics.run_data.write(tmp_path)
        stored = Analytics(RunData.from_record(tmp_path)).instrument.profile(NAME).key_data

        assert "name" not in stored.columns and "free_float" not in stored.columns
        pd.testing.assert_frame_equal(stored, profile.key_data[stored.columns],
                                      check_freq=False)

    def test_each_column_s_label_and_unit(self,
                                          profile):
        columns = {column.name: column for column in columns_of(profile.key_data)}

        assert (columns["exchange"].label, columns["exchange"].unit) == ("Exchange", None)
        assert columns["market_cap"].unit == "money"


class TestItsCorporateActions:

    @pytest.fixture(scope="class")
    def split(self,
              result):
        actions = result.analytics.run_data.tables[ACTIONS]
        added = pd.DataFrame({"IDENTIFIER": [NAME], "EX_DATE": [SPLIT], "TYPE": ["SPLIT"],
                              "VALUE": [2.0]})

        return replaced(result, **{ACTIONS: pd.concat([actions, added], ignore_index=True)})

    def test_each_action_in_the_window(self,
                                       split):
        found = split.instrument.profile(NAME).corporate_actions

        assert list(found.index) == [SPLIT]
        assert found.loc[SPLIT].tolist() == ["SPLIT", 2.0]

    def test_none_before_the_window_or_on_other_names(self,
                                                      profile,
                                                      split):
        assert profile.corporate_actions.empty
        assert split.instrument.profile("N1").corporate_actions.empty
        assert split.instrument.profile(NAME, start="2024-07-01").corporate_actions.empty

    def test_a_run_kept_before_its_actions_were(self,
                                                result):
        older = replaced(result, **{ACTIONS: None})

        with pytest.raises(Unavailable) as raised:
            _ = older.instrument.profile(NAME).corporate_actions

        assert raised.value.code == RUN_PREDATES
