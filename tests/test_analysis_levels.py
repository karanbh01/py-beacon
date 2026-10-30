# tests/test_analysis_levels.py
"""BN-401: the figures the server's views used to work out themselves, now
the library's: level series rebased and compared, the attribution of an
index's names over a window, and risk from prices."""
import numpy as np
import pandas as pd
import pytest

from beacon.analysis.attribution.schedule import attribute_schedule
from beacon.analysis.performance.levels import (
    annual_returns,
    compared,
    daily_returns,
    headline,
    rebased,
    total_return,
)
from beacon.exceptions import DataNotFoundError
from beacon.risk.prices import (
    MINIMUM_OBSERVATIONS,
    active_risk_of_prices,
    returns_of_prices,
    risk_of_prices,
)

DAYS = pd.bdate_range("2023-12-28", periods=120)


def path(daily: float) -> pd.Series:
    return pd.Series(50.0 * (1.0 + daily) ** np.arange(len(DAYS)), index=DAYS)


def prices() -> pd.DataFrame:
    wiggle = np.sin(np.arange(len(DAYS)) / 3.0) * 0.002

    return pd.DataFrame({"AAA": path(0.001) * (1.0 + wiggle),
                         "BBB": path(-0.0005) * (1.0 - wiggle),
                         "CCC": path(0.0003) * (1.0 + 2.0 * wiggle)})


class TestLevels:

    def test_rebased_starts_at_100_and_keeps_its_shape(self):
        level = rebased(path(0.01))

        assert level.iloc[0] == 100.0
        assert level.iloc[1] == pytest.approx(101.0)

    def test_rebased_leaves_an_empty_or_zero_led_series(self):
        empty = pd.Series(dtype=float)
        zero = pd.Series([0.0, 1.0], index=DAYS[:2])

        assert rebased(empty).empty
        assert rebased(zero).tolist() == [0.0, 1.0]

    def test_returns_and_total(self):
        level = path(0.01)

        assert daily_returns(level).iloc[0] == pytest.approx(0.01)
        assert total_return(level) == pytest.approx(1.01 ** 119 - 1.0)
        assert total_return(pd.Series(dtype=float)) == 0.0

    def test_the_headline_compounds_to_the_total(self):
        figures = headline(path(0.001), 252, 0.0)

        assert figures["total_return"] == pytest.approx(1.001 ** 119 - 1.0)
        assert figures["max_drawdown"] == 0.0

    def test_annual_returns_compound_to_the_total(self):
        level = path(0.002)
        years = annual_returns(level)

        assert list(years) == ["2023", "2024"]
        assert np.prod([1.0 + value for value in years.values()]) == pytest.approx(
            total_return(level) + 1.0)


class TestCompared:

    def test_only_the_dates_both_have(self):
        long, short = path(0.001), path(0.002).iloc[10:].drop(DAYS[20])
        found = compared({"long": long, "short": short})

        assert len(found.window) == len(DAYS) - 11
        assert all(series.iloc[0] == 100.0 for series in found.levels.values())
        assert found.total_returns["short"] == pytest.approx(1.002 ** 109 - 1.0)

    def test_series_that_share_no_date(self):
        with pytest.raises(DataNotFoundError):
            compared({"a": path(0.0).iloc[:5], "b": path(0.0).iloc[5:]})


class TestSchedule:

    def test_names_add_up_to_the_index_and_the_drags(self):
        weights = {DAYS[0]: {"AAA": 0.5, "BBB": 0.3, "CCC": 0.2}}
        uncapped = {DAYS[0]: {"AAA": 0.6, "BBB": 0.25, "CCC": 0.15}}
        found = attribute_schedule(weights, uncapped, prices(), None, None,
                                   total_costs=100.0, initial_capital=1e5)

        assert found.reconciles()
        assert found.cost_drag == pytest.approx(-0.001)
        assert found.cap_drag is not None

    def test_no_drags_without_a_cap_or_costs(self):
        weights = {DAYS[0]: {"AAA": 0.5, "BBB": 0.5}}
        found = attribute_schedule(weights, weights, prices(), "2024-02-01", None)

        assert found.cap_drag is None and found.cost_drag is None
        assert found.start >= "2024-02-01"

    def test_an_empty_window(self):
        weights = {DAYS[0]: {"AAA": 1.0}}

        with pytest.raises(DataNotFoundError, match="2030-01-01"):
            attribute_schedule(weights, weights, prices(), "2030-01-01", None)


class TestRiskOfPrices:

    def test_contributions_add_up_to_the_volatility(self):
        found = risk_of_prices({"AAA": 0.5, "BBB": 0.5}, prices())

        assert found is not None
        assert sum(found.contribution.values()) == pytest.approx(found.volatility)

    def test_tracking_error_against_a_benchmark(self):
        found = active_risk_of_prices({"AAA": 0.6, "BBB": 0.4},
                                      {"AAA": 0.4, "BBB": 0.4, "CCC": 0.2}, prices())

        assert found is not None
        assert set(found.contribution) == {"AAA", "BBB", "CCC"}

    def test_too_few_days(self):
        short = prices().iloc[:MINIMUM_OBSERVATIONS]

        assert len(returns_of_prices(short)) == MINIMUM_OBSERVATIONS - 1
        assert risk_of_prices({"AAA": 1.0}, short) is None
        assert active_risk_of_prices({"AAA": 1.0}, {"BBB": 1.0}, short) is None
