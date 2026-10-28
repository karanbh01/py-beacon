# tests/test_brinson.py
"""BN-394: Brinson-Fachler attribution, by hand and over a run.

By hand, one day of two groups: the fund holds 60% in A, earning 2%, and 40%
in B, earning 1%; the comparator 50% in each, earning 1.5% and 1.2%, so
1.35% in all. The fund earns 1.6%, 0.25% more:

    allocation   A: 0.1 x (0.015 - 0.0135) =  0.00015   B: -0.1 x (0.012 - 0.0135) =  0.00015
    selection    A: 0.5 x (0.02 - 0.015)   =  0.0025    B:  0.5 x (0.01 - 0.012)   = -0.001
    interaction  A: 0.1 x (0.02 - 0.015)   =  0.0005    B: -0.1 x (0.01 - 0.012)   =  0.0002

The run is test_run_attribution's thirty names in three sectors.
"""
import math

import pandas as pd
import pytest

from beacon.analysis.attribution import Attribution
from beacon.analysis.attribution.brinson import CASH, active_carino, daily_effects
from beacon.analysis.data import RunData
from beacon.analysis.panels import CASH as CASH_TABLE
from beacon.analysis.panels import FUND_WEIGHTS
from test_factor_analytics import DAYS
from test_run_attribution import consistent

DAY = [pd.Timestamp("2024-03-04")]


def frame(values: dict[str, float]) -> pd.DataFrame:
    return pd.DataFrame([values], index=DAY)


class TestByHand:

    def test_one_day_s_effects(self):
        effects = daily_effects(frame({"A": 0.6, "B": 0.4}), frame({"A": 0.5, "B": 0.5}),
                                frame({"A": 0.012, "B": 0.004}),
                                frame({"A": 0.0075, "B": 0.006}))

        assert effects["allocation"].iloc[0].to_dict() == pytest.approx({"A": 0.00015,
                                                                         "B": 0.00015})
        assert effects["selection"].iloc[0].to_dict() == pytest.approx({"A": 0.0025,
                                                                        "B": -0.001})
        assert effects["interaction"].iloc[0].to_dict() == pytest.approx({"A": 0.0005,
                                                                          "B": 0.0002})
        assert sum(table.sum().sum() for table in effects.values()) == pytest.approx(0.0025)

    def test_a_group_the_fund_does_not_hold(self):
        effects = daily_effects(frame({"A": 1.0}), frame({"A": 0.5, "B": 0.5}),
                                frame({"A": 0.02}), frame({"A": 0.0075, "B": 0.006}))

        assert effects["selection"].loc[DAY[0], "B"] == 0.0
        assert effects["interaction"].loc[DAY[0], "B"] == 0.0
        assert effects["allocation"].loc[DAY[0], "B"] == pytest.approx(-0.5 * (0.012 - 0.0135))

    def test_a_group_the_comparator_does_not_hold_is_all_interaction(self):
        effects = daily_effects(frame({"A": 0.5, "C": 0.5}), frame({"A": 1.0}),
                                frame({"A": 0.01, "C": 0.02}), frame({"A": 0.015}))

        assert effects["allocation"].loc[DAY[0], "C"] == 0.0
        assert effects["selection"].loc[DAY[0], "C"] == 0.0
        assert effects["interaction"].loc[DAY[0], "C"] == pytest.approx(0.5 * (0.04 - 0.015))

    def test_the_linking_factor(self):
        assert active_carino(0.02, 0.02) == pytest.approx(1 / 1.02)
        assert active_carino(0.03, 0.01) == pytest.approx((math.log(1.03) - math.log(1.01)) / 0.02)


@pytest.fixture(scope="module")
def attribution():
    return Attribution(consistent())


class TestOverARun:

    def test_the_effects_add_up_to_the_active_return(self,
                                                     attribution):
        found = attribution.brinson("SECTOR")

        assert found.reconciles(1e-9)
        assert found.explained == pytest.approx(found.active_return)
        assert found.groups["total"].sum() == pytest.approx(found.explained)

    def test_each_sector_and_cash(self,
                                  attribution):
        groups = attribution.brinson("SECTOR").groups

        assert set(groups.index) == {"Tech", "Energy", "Health", CASH}
        assert groups.loc[CASH, ["fund_weight", "total"]].tolist() == pytest.approx([0.0, 0.0])
        assert groups["fund_weight"].sum() == pytest.approx(1.0)

    def test_over_a_window_and_by_period(self,
                                         attribution):
        window = attribution.brinson("SECTOR", start=DAYS[100], end=DAYS[199])
        years = attribution.brinson_by_period("SECTOR")

        assert window.reconciles(1e-9)
        assert list(years.index) == ["2023", "2024"]
        assert years["residual"].abs().max() < 1e-9

    def test_cash_earning_its_interest(self):
        data = consistent()
        tables = dict(data.tables)
        tables[FUND_WEIGHTS] = tables[FUND_WEIGHTS] * 0.9
        cash = pd.DataFrame({"cash": 1e5, "interest": 1e5 * 0.0002}, index=DAYS)
        cash.iloc[0, 1] = 0.0
        tables[CASH_TABLE] = cash
        held = RunData(currency="USD", returns=data.returns * 0.9 + 0.1 * 0.0002,
                       nav=data.nav, units=data.units, initial_capital=1e6,
                       periods_per_year=252, risk_free_rate=0.0, tables=tables,
                       comparators=data.comparators)

        found = Attribution(held).brinson("SECTOR")

        assert found.groups.loc[CASH, "fund_weight"] == pytest.approx(0.1)
        assert found.groups.loc[CASH, "fund_return"] == pytest.approx(1.0002 ** 299 - 1.0)
        assert found.reconciles(1e-9)
