# tests/test_brinson_names.py
"""BN-415: Brinson-Fachler selection and interaction by name within each
group.

By hand, one day: the fund holds A1 (40%) and A2 (20%) in group A and B1
(40%) in B; the comparator holds A1 (30%) and A3 (20%) in A and B1 (50%) in
B. Within A the fund's shares are 2/3 and 1/3, the comparator's 0.6 and 0.4,
and the comparator's return in A is 0.6 x 2% + 0.4 x 5% = 3.2%:

    d(A1) = (2/3 - 0.6) x (0.02 - 0.032) = -0.0008
    d(A2) = (1/3 - 0)   x (0.01 - 0.032) = -0.007333...
    d(A3) = (0   - 0.4) x (0.05 - 0.032) = -0.0072

A3, which only the comparator held, shows what not holding it cost. Each
group's names add up to its own effects; over a run as well, with
test_run_attribution's thirty names in three sectors.
"""
import numpy as np
import pandas as pd
import pytest

from beacon.analysis.attribution import Attribution
from beacon.analysis.attribution.brinson import daily_effects
from beacon.analysis.attribution.brinson_names import name_effects
from beacon.analysis.attribution.groups import by_group
from test_factor_analytics import DAYS
from test_run_attribution import consistent

DAY = pd.DatetimeIndex([pd.Timestamp("2024-03-04")])
GROUPS = {pd.Timestamp("2024-03-01"): pd.Series({"A1": "A", "A2": "A", "A3": "A", "B1": "B"})}


def one_day(fund: dict[str, float],
            comparator: dict[str, float],
            returns: dict[str, float]) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    fund_weights = pd.DataFrame(fund, index=DAY)
    comparator_weights = pd.DataFrame(comparator, index=DAY)
    earned = pd.DataFrame(returns, index=DAY)
    fund_held = by_group(fund_weights, GROUPS)
    comparator_held = by_group(comparator_weights, GROUPS)
    fund_earned = by_group(fund_weights * earned.reindex(columns=fund_weights.columns), GROUPS)
    comparator_earned = by_group(comparator_weights
                                 * earned.reindex(columns=comparator_weights.columns), GROUPS)
    names = name_effects(fund_weights, comparator_weights, earned, GROUPS, fund_held,
                         comparator_held, comparator_earned, pd.Series(1.0, index=DAY))

    return names, daily_effects(fund_held, comparator_held, fund_earned, comparator_earned)


class TestByHand:

    @pytest.fixture(scope="class")
    def found(self):
        return one_day({"A1": 0.4, "A2": 0.2, "B1": 0.4},
                       {"A1": 0.3, "A3": 0.2, "B1": 0.5},
                       {"A1": 0.02, "A2": 0.01, "A3": 0.05, "B1": 0.015})

    def test_each_name_s_selection_and_interaction(self,
                                                   found):
        names, _ = found
        expected = {"A1": -0.0008, "A2": (1 / 3) * (0.01 - 0.032), "A3": -0.4 * 0.018}

        for name, share in expected.items():
            assert names.loc[("A", name), "selection"] == pytest.approx(0.5 * share)
            assert names.loc[("A", name), "interaction"] == pytest.approx(0.1 * share)

    def test_a_name_only_the_comparator_held_shows_what_missing_it_cost(self,
                                                                        found):
        names, _ = found

        assert names.loc[("A", "A3"), "fund_weight"] == 0.0
        assert names.loc[("A", "A3"), "total"] < 0.0

    def test_a_group_s_names_add_up_to_its_effects(self,
                                                   found):
        names, groups = found

        for group in ("A", "B"):
            for effect in ("selection", "interaction"):
                assert names.loc[group, effect].sum() == pytest.approx(
                    float(groups[effect][group].iloc[0]), abs=1e-15)

    def test_a_group_the_fund_does_not_hold_has_no_selection(self):
        names, groups = one_day({"B1": 1.0}, {"A1": 0.3, "A3": 0.2, "B1": 0.5},
                                {"A1": 0.03, "A3": -0.02, "B1": 0.01})

        for name in ("A1", "A3"):
            assert names.loc[("A", name), ["selection", "interaction"]].tolist() == [0.0, 0.0]

        assert float(groups["selection"]["A"].iloc[0]) == 0.0

    def test_a_group_the_comparator_does_not_hold_is_measured_against_its_whole(self):
        names, groups = one_day({"A1": 0.5, "B1": 0.5}, {"B1": 1.0},
                                {"A1": 0.03, "B1": 0.01})

        assert names.loc[("A", "A1"), "interaction"] == pytest.approx(
            float(groups["interaction"]["A"].iloc[0]))
        assert names.loc[("A", "A1"), "interaction"] == pytest.approx(0.5 * (0.03 - 0.01))


class TestOverARun:

    @pytest.fixture(scope="class")
    def found(self):
        return Attribution(consistent()).brinson("SECTOR")

    def test_each_sector_s_names_add_up_to_its_effects(self,
                                                       found):
        names = found.names

        for group in ("Tech", "Energy", "Health"):
            for effect in ("selection", "interaction"):
                assert names.loc[group, effect].sum() == pytest.approx(
                    found.groups.loc[group, effect], rel=1e-9, abs=1e-15), (group, effect)

    def test_every_name_held_by_either_book_is_there(self,
                                                     found):
        assert len(found.names) == 30
        assert set(found.names.index.get_level_values(0)) == {"Tech", "Energy", "Health"}

    def test_over_a_window_too(self):
        window = Attribution(consistent()).brinson("SECTOR", start=DAYS[100], end=DAYS[199])

        assert np.isclose(window.names["selection"].sum(), window.groups["selection"].sum(),
                          rtol=1e-9, atol=1e-15)
