# tests/test_analytics_categories.py
"""BN-435 on: each category of `result.analytics.portfolio`, its methods its
statistics, each giving what the run's analysis gives for it.

The run is test_strategy_futures' eight-name market-cap index held physically
through 2024, with its total-return version as the benchmark.
"""
import inspect

import pandas as pd
import pytest

from beacon.analysis.analytics.shape import plain
from beacon.analysis.placement import DROPPED, FOLDED, PART_OF
from beacon.analysis.reports.catalogue import CATALOGUE
from beacon.analysis.reports.context import Context
from beacon.analysis.reports.statistics import FIGURES, Unavailable
from beacon.analysis.request import AnalysisRequest, analyse
from beacon.backtest import Backtest

# The categories wired so far, each with the statistics it gives.
WIRED = ["performance", "risk"]


@pytest.fixture(scope="module")
def result():
    from beacon.index.calculation import IndexCalculator
    from test_strategy_futures import DATA, END, START, index

    benchmark = IndexCalculator(index("TOTAL_RETURN"), DATA).run(start_date=START, end_date=END)

    return Backtest(initial_capital=1e7, data_provider=DATA, cache=False,
                    transaction_cost_bps=5.0, benchmark=benchmark).run(index(), start=START,
                                                                       end=END)


def category(result,
             name: str):
    return getattr(result.analytics.portfolio, name)


def statistics_of(result) -> list[tuple[str, str]]:
    return [(name, statistic_id) for name in WIRED
            for statistic_id in category(result, name).statistics]


@pytest.fixture(scope="module")
def block(result):
    """Today's analysis of every statistic wired, as the API gives it."""
    wanted = [statistic_id for _, statistic_id in statistics_of(result)]

    return analyse(result, AnalysisRequest(reports={}, statistics=wanted))


def named(value) -> dict:
    """A method's value as named figures, its parts joined with ``/``."""
    if isinstance(value, pd.DataFrame):
        return {"/".join(str(part) for part in (column, *_parts(row))): value.loc[row, column]
                for column in value.columns for row in value.index}

    if isinstance(value, pd.Series):
        return {"/".join(str(part) for part in _parts(key)): item for key, item in value.items()}

    return {"": value}


def _parts(key) -> tuple:
    return key if isinstance(key, tuple) else (key,)


@pytest.mark.parametrize("name", WIRED)
class TestItsMethods:
    def test_its_methods_are_its_statistics(self,
                                            result,
                                            name):
        found = category(result, name)
        methods = {member for member, _ in inspect.getmembers(type(found), inspect.isfunction)
                   if not member.startswith("_")}

        assert methods == {*found.statistics, *found.extras}

    def test_it_gives_the_statistics_placed_in_it(self,
                                                  result,
                                                  name):
        placed = {statistic_id for statistic_id, part in PART_OF.items()
                  if part == name and statistic_id not in {*FOLDED, *DROPPED}}

        assert set(category(result, name).statistics) == placed

    def test_each_takes_the_keywords_its_statistic_does(self,
                                                        result,
                                                        name):
        found = category(result, name)

        for statistic_id in found.statistics:
            parameters = inspect.signature(getattr(found, statistic_id)).parameters

            assert tuple(parameters) == CATALOGUE[statistic_id].forms, statistic_id
            assert all(parameter.kind is inspect.Parameter.KEYWORD_ONLY
                       for parameter in parameters.values())


def test_no_statistic_is_in_two_categories(result):
    placed = [statistic_id for _, statistic_id in statistics_of(result)]

    assert len(placed) == len(set(placed))


def test_each_gives_what_the_run_s_analysis_gives(result,
                                                 block):
    context = Context(result.run_data, None, result._data_fetcher)

    for name, statistic_id in statistics_of(result):
        method = getattr(category(result, name), statistic_id)
        spec = CATALOGUE[statistic_id]

        if statistic_id in block["unavailable"]:
            with pytest.raises(Unavailable):
                method()
            continue

        if spec.kind != FIGURES:
            today = plain(spec.compute(context))
            found = method(of="all") if "of" in spec.forms else method()
            same = found.equals(today) if hasattr(found, "equals") else found == today
            assert same, statistic_id
            continue

        every = ({"of": "all"} if "of" in spec.forms
                 else {"against": "all"} if spec.figure_names[:1] == ("comparator",) else {})
        found = named(method(**every))
        # A figure the run cannot give is None in the block, NaN in a Series.
        point = {key: float("nan") if item is None else item
                 for key, item in block["statistics"][statistic_id]["point"].items()}

        if every:
            assert found == pytest.approx(point, nan_ok=True), statistic_id
        else:
            fund_only = {key.split("/", 1)[-1] if key.startswith("fund") else key: item
                         for key, item in point.items()}
            fund_only = {("" if key == "fund" else key): item for key, item in fund_only.items()}
            assert found == pytest.approx(fund_only, nan_ok=True), statistic_id


class TestPerformance:
    def test_returns_are_the_run_s(self,
                                   result):
        performance = result.analytics.portfolio.performance

        pd.testing.assert_series_equal(performance.returns(), result.run_data.returns,
                                       check_names=False)
        assert list(performance.returns(of="all").columns) == ["fund", "index", "benchmark"]

    def test_a_window_s_returns(self,
                                result):
        found = result.analytics.portfolio.performance.returns(start="2024-07-01")

        assert found.index[0] >= pd.Timestamp("2024-07-01")

    def test_a_portfolio_the_run_lacks(self,
                                       result):
        with pytest.raises(ValueError, match="No portfolio 'optimised' here"):
            result.analytics.portfolio.performance.returns(of="optimised")
