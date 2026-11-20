# tests/test_risk_rating.py
"""The risk rating: the factsheet's 1 to 7 scale on weekly volatility over the
last five years, a tracking error band, and the days spent at each.

Runs are made of seeded normal daily returns: a fund whose volatility is set
by period, and an index the fund tracks with a set daily spread.
"""
import numpy as np
import pandas as pd
import pytest

from beacon.analysis.data import RunData
from beacon.analysis.performance import formulas
from beacon.analysis.performance.risk_rating import (
    RiskRatingSettings,
    bands,
    ratings,
    risk_distribution,
    risk_rating,
    weekly,
)
from beacon.analysis.reports.catalogue import CATALOGUE
from beacon.analysis.reports.context import Context
from beacon.analysis.reports.options import AnalysisOptions
from beacon.analysis.reports.statistics import TOO_SHORT, Unavailable

SETTINGS = RiskRatingSettings()
YEAR = 252


def days(count: int) -> pd.DatetimeIndex:
    return pd.bdate_range("2015-01-05", periods=count)


def noise(index: pd.DatetimeIndex,
          annual: float,
          seed: int) -> pd.Series:
    """Daily returns with about *annual* annualised volatility."""
    draws = np.random.default_rng(seed).standard_normal(len(index))

    return pd.Series(draws * annual / np.sqrt(YEAR), index=index)


def levels(returns: pd.Series) -> pd.Series:
    return (1.0 + returns).cumprod() * 100.0


class TestTheScale:

    def test_a_bound_takes_the_higher_rating(self):
        found = ratings(np.array([0.004, 0.005, 0.0199, 0.02, 0.12, 0.25, 0.4]))

        assert found.tolist() == [1, 2, 2, 3, 5, 7, 7]

    def test_the_bands(self):
        found = bands(np.array([0.0, 0.005, 0.019, 0.03, 0.05, 0.2]))

        assert found.tolist() == ["passive", "enhanced", "enhanced", "active", "unconstrained",
                                  "unconstrained"]

    def test_the_bounds_are_settings(self):
        settings = RiskRatingSettings(volatility_bounds=(0.1,), tracking_bounds=(0.01, 0.1, 0.2))

        assert ratings(0.15, settings.volatility_bounds).tolist() == [2]
        assert bands(0.05, settings.tracking_bounds).tolist() == ["enhanced"]

    @pytest.mark.parametrize("given", [{"volatility_bounds": (0.05, 0.02)},
                                       {"volatility_bounds": ()},
                                       {"tracking_bounds": (0.01, 0.02)},
                                       {"years": 0}])
    def test_settings_that_make_no_scale_are_refused(self,
                                                     given):
        with pytest.raises(ValueError):
            RiskRatingSettings(**given)


class TestWeeks:

    def test_a_week_s_days_compound(self):
        returns = pd.Series([0.01, 0.02, -0.01, 0.0, 0.03, 0.01],
                            index=pd.bdate_range("2024-06-03", periods=6))

        found = weekly(returns)

        assert found.index[0] == pd.Timestamp("2024-06-07")
        assert found.iloc[0] == pytest.approx(1.01 * 1.02 * 0.99 * 1.0 * 1.03 - 1.0)
        assert found.iloc[1] == pytest.approx(0.01)


class TestTheRating:

    def test_the_volatility_of_weekly_returns_annualised(self):
        returns = noise(days(3 * YEAR), 0.12, seed=1)

        found = risk_rating(returns, None, {}, SETTINGS, YEAR).set_index("book")

        expected = formulas.volatility(weekly(returns), 52)
        assert found.loc["fund", "volatility"] == pytest.approx(expected)
        assert found.loc["fund", "rating"] == ratings(expected)[0]
        assert found.loc["fund", "band"] is None

    def test_only_the_last_five_years_count(self):
        """Three wild years, then six calm ones (five calendar years are
        more than five years of 252 days): the rating is the calm years'."""
        index = days(9 * YEAR)
        wild = noise(index[:3 * YEAR], 0.40, seed=2)
        calm = noise(index[3 * YEAR:], 0.03, seed=3)
        returns = pd.concat([wild, calm])

        found = risk_rating(returns, None, {}, SETTINGS, YEAR).set_index("book")

        assert found.loc["fund", "rating"] == 3
        assert RiskRatingSettings(years=9) != SETTINGS
        longer = risk_rating(returns, None, {}, RiskRatingSettings(years=9), YEAR)
        assert longer.set_index("book").loc["fund", "rating"] == 6

    def test_the_band_against_the_tracked_comparator(self):
        index = days(3 * YEAR)
        theirs = noise(index, 0.12, seed=4)
        ours = theirs + noise(index, 0.01, seed=5)

        found = risk_rating(ours, levels(theirs), {"index": levels(theirs)}, SETTINGS,
                            YEAR).set_index("book")

        assert found.loc["fund", "band"] == "enhanced"
        assert 0.005 < found.loc["fund", "tracking_error"] < 0.02
        assert found.loc["index", "rating"] == 5
        assert pd.isna(found.loc["index", "band"])
        assert pd.isna(found.loc["index", "tracking_error"])

    def test_under_a_year_there_is_none(self):
        returns = noise(days(YEAR - 1), 0.1, seed=6)

        assert risk_rating(returns, None, {}, SETTINGS, YEAR) is None
        assert risk_distribution(returns, None, SETTINGS, YEAR) is None


class TestTheDistribution:

    def test_the_first_year_is_not_counted(self):
        returns = noise(days(3 * YEAR), 0.12, seed=7)

        found = risk_distribution(returns, None, SETTINGS, YEAR)

        assert found["days"].sum() == 3 * YEAR - (YEAR - 1)
        assert found["share"].sum() == pytest.approx(1.0)
        assert found["band"].isna().all()

    def test_a_run_that_calms_down_moves_down_the_scale(self):
        index = days(4 * YEAR)
        returns = pd.concat([noise(index[:2 * YEAR], 0.30, seed=8),
                             noise(index[2 * YEAR:], 0.03, seed=9)])

        found = risk_distribution(returns, None, SETTINGS, YEAR)

        assert found["rating"].tolist() == sorted(found["rating"].tolist())
        assert found["rating"].iloc[0] == 3
        assert found["rating"].iloc[-1] == 7

    def test_each_cell_visited_with_its_band(self):
        index = days(3 * YEAR)
        theirs = noise(index, 0.12, seed=10)
        ours = theirs + noise(index, 0.01, seed=11)

        found = risk_distribution(ours, levels(theirs), SETTINGS, YEAR)

        assert set(found["band"]) == {"enhanced"}
        assert found["days"].sum() == 3 * YEAR - (YEAR - 1)


class TestTheCatalogue:

    def run(self,
            count: int = 3 * YEAR) -> RunData:
        index = days(count)
        theirs = noise(index, 0.12, seed=12)
        ours = theirs + noise(index, 0.01, seed=13)

        return RunData.of_returns(ours, YEAR, comparators={"index": levels(theirs)})

    def test_the_fund_s_row(self):
        found = CATALOGUE["risk_rating"].compute(Context(self.run())).set_index("book")

        assert found.loc["fund", "rating"] == 5
        assert found.loc["fund", "band"] == "enhanced"
        assert list(found.index) == ["fund", "index"]

    def test_the_settings_are_read(self):
        options = AnalysisOptions(risk_rating=RiskRatingSettings(
            tracking_bounds=(0.001, 0.002, 0.003)))

        found = CATALOGUE["risk_rating"].compute(Context(self.run(), options))

        assert found.set_index("book").loc["fund", "band"] == "unconstrained"

    def test_labelled_columns(self):
        columns = {column.name: column for column in CATALOGUE["risk_distribution"].columns}

        assert columns["band"].labels["unconstrained"] == "Unconstrained"
        assert CATALOGUE["risk_rating"].kind == "table"

    def test_a_short_run_is_too_short(self):
        for statistic_id in ("risk_rating", "risk_distribution"):
            with pytest.raises(Unavailable) as raised:
                CATALOGUE[statistic_id].compute(Context(self.run(YEAR - 1)))

            assert raised.value.code == TOO_SHORT
