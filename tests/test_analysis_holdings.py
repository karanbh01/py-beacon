# tests/test_analysis_holdings.py
"""BN-403: daily weights worked out again from holdings, closes and a patch.

Hand-made holdings over a week: a name bought, one resized, one sold and
bought back.
"""
import numpy as np
import pandas as pd

from beacon.analysis.holdings import (
    AMOUNT,
    HELD,
    NAME,
    WEIGHT,
    held_amounts,
    holding_changes,
    patch_of,
    patched,
)

DAYS = pd.bdate_range("2024-03-04", periods=5)


def amounts() -> pd.DataFrame:
    return pd.DataFrame({"AAA": [10.0, 10.0, 12.0, 12.0, 12.0],
                         "BBB": [np.nan, 5.0, 5.0, 5.0, 5.0],
                         "CCC": [3.0, 3.0, np.nan, np.nan, 4.0]}, index=DAYS)


class TestHoldings:

    def test_only_the_days_a_holding_changed_are_kept(self):
        changes = holding_changes(amounts())

        assert len(changes) == 6
        assert changes.loc[DAYS[2]].set_index(NAME)[HELD].to_dict() == {"AAA": True, "CCC": False}

    def test_the_holdings_come_back_on_every_day(self):
        frame = amounts()

        back = held_amounts(holding_changes(frame), DAYS, list(frame.columns))

        np.testing.assert_array_equal(back.to_numpy(), frame.to_numpy())

    def test_a_holding_that_ends_is_missing_until_it_starts_again(self):
        back = held_amounts(holding_changes(amounts()), DAYS, ["CCC"])

        assert back["CCC"].isna().tolist() == [False, False, True, True, False]
        assert back.loc[DAYS[4], "CCC"] == 4.0

    def test_no_changes_hold_nothing(self):
        empty = holding_changes(amounts().iloc[:0])

        assert held_amounts(empty, DAYS, ["AAA"]).isna().all().all()
        assert set(empty.columns) == {NAME, AMOUNT, HELD}


class TestPatch:

    def test_only_cells_that_differ_are_kept_and_put_back(self):
        recorded = amounts() / 100
        rebuilt = recorded.copy()
        rebuilt.iloc[1, 0] = np.nextafter(rebuilt.iloc[1, 0], 1.0)
        rebuilt.iloc[2, 2] = 0.0

        patch = patch_of(rebuilt, recorded)

        assert len(patch) == 2
        assert patched(rebuilt, patch).equals(recorded)

    def test_identical_tables_need_no_patch(self):
        recorded = amounts()

        assert patch_of(recorded.copy(), recorded).empty
        assert patched(recorded, patch_of(recorded, recorded)) is recorded

    def test_a_patch_keeps_the_recorded_precision(self):
        recorded = (amounts() / 7).astype("float32")
        rebuilt = (amounts().astype("float64") / 7).astype("float32")
        rebuilt.iloc[0, 0] = np.float32(0.5)

        patch = patch_of(rebuilt, recorded)

        assert patch[WEIGHT].dtype == np.float32
        assert patched(rebuilt, patch).equals(recorded)
