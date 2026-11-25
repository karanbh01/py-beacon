# tests/test_risk_return.py
"""BN-429/BN-430/BN-431: each portfolio's risk and return, the frontier in
hindsight, and the statistics beacon-ui's Risk view reads. Each name's is the
per-name table's at weekly returns (BN-461), in test_analytics_instrument.

Seeded weekly-scale noise on a few names: the index holds four of them over
the run and drops one before its end; one is listed too late to have a year;
one the index never holds.
"""
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from beacon.analysis.data import RunData
from beacon.analysis.panels import FUND_WEIGHTS, INDEX_WEIGHTS, NAME_RETURNS
from beacon.analysis.performance.members import (
    RiskReturnSettings,
    compounded,
    hindsight_frontier,
    portfolio_risk_return,
    risk_return,
)
from beacon.analysis.reports.catalogue import CATALOGUE
from beacon.analysis.reports.context import Context
from beacon.analysis.reports.options import AnalysisOptions, CrisesOptions
from beacon.analysis.reports.statistics import NO_INDEX, RUN_PREDATES, TOO_SHORT, Unavailable
from beacon.analysis.reports.wire import payload

YEAR = 252
DAYS = pd.bdate_range("2019-01-07", periods=3 * YEAR)
SETTINGS = RiskReturnSettings(frontier_names=3)


def names() -> pd.DataFrame:
    rng = np.random.default_rng(7)
    vols = np.array([0.10, 0.20, 0.30, 0.40, 0.25, 0.15]) / np.sqrt(YEAR)
    drifts = np.array([0.02, 0.06, 0.10, 0.14, 0.05, 0.03]) / YEAR
    found = pd.DataFrame(rng.standard_normal((len(DAYS), 6)) * vols + drifts, index=DAYS,
                         columns=["A", "B", "C", "D", "LATE", "NEVER"])
    found.loc[:DAYS[-YEAR // 2], "LATE"] = np.nan

    return found


def index_weights() -> pd.DataFrame:
    found = pd.DataFrame(0.0, index=DAYS, columns=["A", "B", "C", "D", "LATE", "NEVER"])
    found[["A", "B", "C", "D"]] = [0.4, 0.3, 0.2, 0.1]
    found.loc[DAYS[-YEAR // 2]:, "LATE"] = 0.1
    found.loc[DAYS[-10]:, "D"] = 0.0

    return found


class TestCompounding:

    def test_weeks_end_on_friday_and_months_at_their_end(self):
        returns = pd.Series(0.01, index=pd.bdate_range("2024-06-03", periods=25))

        weekly = compounded(returns)
        monthly = compounded(returns, "monthly")

        assert weekly.index[0] == pd.Timestamp("2024-06-07")
        assert weekly.iloc[0] == pytest.approx(1.01 ** 5 - 1)
        assert monthly.index[0] == pd.Timestamp("2024-06-30")
        assert monthly.iloc[0] == pytest.approx(1.01 ** 20 - 1)


class TestPortfolios:

    def test_measured_as_the_names_are(self):
        daily = names()

        found = portfolio_risk_return({"fund": daily["A"], "index": daily["B"]}, SETTINGS)
        names_ = risk_return(compounded(daily[["A", "B"]]), 52)

        np.testing.assert_allclose(found["volatility"], names_["volatility"])
        assert found["book"].tolist() == ["fund", "index"]


class TestTheFrontier:

    @pytest.fixture(scope="class")
    def found(self):
        return hindsight_frontier(names(), index_weights(), SETTINGS)

    def test_the_largest_latest_members_with_every_week(self,
                                                         found):
        assert found.names == 3
        assert found.periods == len(compounded(names()["A"]))

    def test_it_rises_from_the_minimum_variance_point(self,
                                                       found):
        table = found.table

        assert len(table) == 50
        assert table["minimum_variance"].tolist() == [True] + [False] * 49
        assert table["return"].is_monotonic_increasing
        assert table["volatility"].iloc[0] == table["volatility"].min()

    def test_its_end_is_the_best_name(self,
                                      found):
        weekly = compounded(names()[["A", "B", "C"]])

        assert found.table["return"].iloc[-1] == pytest.approx(weekly.mean().max() * 52,
                                                               rel=1e-6)

    def test_under_a_year_or_too_few_weeks_for_its_names_has_none(self):
        short = names().iloc[:YEAR // 2]
        many = RiskReturnSettings(frontier_names=3)

        assert hindsight_frontier(short, index_weights(), many) is None
        assert hindsight_frontier(names().iloc[:200], index_weights(),
                                  replace(many, frequency="monthly")) is None


class TestTheCatalogue:

    def run(self,
            tables: bool = True) -> RunData:
        daily = names()
        weights = index_weights()
        fund = (daily[["A", "B", "C", "D"]] * [0.4, 0.3, 0.2, 0.1]).sum(axis=1)
        index = 100.0 * (1.0 + (daily[["A", "B", "C", "D"]] * [0.4, 0.3, 0.2, 0.1])
                         .sum(axis=1) * 1.01).cumprod()
        run = RunData.of_returns(fund, YEAR, comparators={"index": index})

        if not tables:
            return run

        return replace(run, tables={NAME_RETURNS: daily, INDEX_WEIGHTS: weights,
                                    FUND_WEIGHTS: weights})

    def context(self,
                **options) -> Context:
        chosen = AnalysisOptions(risk_return=SETTINGS, **options)

        return Context(self.run(), chosen)

    def test_the_portfolios(self):
        found = CATALOGUE["portfolio_risk_return"].compute(self.context())

        assert found["book"].tolist() == ["fund", "index"]

    def test_the_frontier_with_its_universe(self):
        found = payload("table", CATALOGUE["efficient_frontier"].compute(self.context()))

        assert found["summary"]["names"] == 3
        assert found["summary"]["first"] == "2019-01-11"
        assert len(found["point"]["data"]) == 50

    def test_a_run_without_its_tables(self):
        context = Context(self.run(tables=False), AnalysisOptions(risk_return=SETTINGS))

        with pytest.raises(Unavailable) as raised:
            CATALOGUE["efficient_frontier"].compute(context)

        assert raised.value.code == RUN_PREDATES

    def test_a_run_without_an_index(self):
        context = Context(RunData.of_returns(names()["A"], YEAR))

        with pytest.raises(Unavailable) as raised:
            CATALOGUE["efficient_frontier"].compute(context)

        assert raised.value.code == NO_INDEX

    def test_a_frontier_too_short(self):
        run = self.run()
        tables = {**run.tables, NAME_RETURNS: names().iloc[:YEAR // 2]}
        context = Context(replace(run, tables=tables), AnalysisOptions(risk_return=SETTINGS))

        with pytest.raises(Unavailable) as raised:
            CATALOGUE["efficient_frontier"].compute(context)

        assert raised.value.code == TOO_SHORT

    def test_the_return_distribution(self):
        found = CATALOGUE["return_distribution"].compute(self.context())

        assert list(found.columns) == ["low", "high", "fund", "index", "normal"]
        assert found["index"].sum() == pytest.approx(1.0)

    def test_crisis_rows_with_their_status_and_their_paths(self):
        context = self.context(crises=CrisesOptions(episodes="named"))

        crises = CATALOGUE["crisis_returns"].compute(context)
        paths = CATALOGUE["crisis_paths"].compute(context)

        covered = crises[crises["status"].isin(["covered", "partial"])]
        assert crises["status"].iloc[0] == "before_run"
        assert covered["name"].tolist() == ["Covid crash"]
        assert set(paths["crisis"]) == {"Covid crash"}
        assert set(paths["book"]) == {"fund", "index"}
        assert set(paths["phase"]) <= {"crisis", "after"}
