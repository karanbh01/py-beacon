# tests/test_analytics_forms.py
"""BN-434: the keywords that pick the form an analytic is given in.

The run is test_strategy_futures' eight-name market-cap index held physically
through 2024, with its total-return version as the benchmark.
"""
import pandas as pd
import pytest

from beacon.analysis.performance.settings import AnalysisSettings
from beacon.analysis.reports.catalogue import CATALOGUE
from beacon.analysis.reports.catalogue.forms import (
    AGAINST_TRACKED,
    BY_BOOK_TABLES,
    KEYWORDS,
    PERIODS,
    ROLLING,
    STEP,
    THROUGH_TIME,
    TRAILING_WINDOWS,
    WINDOW,
)
from beacon.analysis.reports.context import Context
from beacon.analysis.reports.options import AnalysisOptions
from beacon.analysis.reports.statistics import Unavailable
from beacon.backtest import Backtest


@pytest.fixture(scope="module")
def result():
    from beacon.index.calculation import IndexCalculator
    from test_strategy_futures import DATA, END, START, index

    benchmark = IndexCalculator(index("TOTAL_RETURN"), DATA).run(start_date=START, end_date=END)

    return Backtest(initial_capital=1e7, data_provider=DATA, cache=False,
                    transaction_cost_bps=5.0, benchmark=benchmark).run(index(), start=START,
                                                                       end=END)


@pytest.fixture(scope="module")
def get(result):
    return lambda statistic_id, **form: result.analytics._value(statistic_id, form)


@pytest.fixture(scope="module")
def figures(result):
    context = Context(result.run_data)

    return lambda statistic_id: CATALOGUE[statistic_id].compute(context)


class TestTheCatalogueSaysWhatEachTakes:
    def test_every_keyword_is_known(self):
        assert all(set(spec.forms) <= set(KEYWORDS) for spec in CATALOGUE)

    def test_a_step_goes_with_rolling(self):
        assert all((ROLLING in spec.forms) == (STEP in spec.forms) for spec in CATALOGUE)
        assert all((ROLLING in spec.forms) == (spec.rolling is not None) for spec in CATALOGUE)

    def test_the_sets_name_statistics_in_the_catalogue(self):
        named = WINDOW | PERIODS | TRAILING_WINDOWS | THROUGH_TIME | AGAINST_TRACKED
        assert named | BY_BOOK_TABLES <= {spec.id for spec in CATALOGUE}

    def test_a_portfolio_s_figure_takes_of_and_a_relative_one_against(self):
        assert CATALOGUE["total_return"].forms == ("of", "start", "end", "by", "trailing",
                                                   "rolling", "step", "grid")
        assert "of" not in CATALOGUE["tracking_error"].forms
        assert "against" in CATALOGUE["tracking_error"].forms
        assert "of" not in CATALOGUE["money_weighted_return"].forms
        assert {"of", "as_of"} <= set(CATALOGUE["holdings_count"].forms)


class TestTheShape:
    def test_bare_it_is_the_fund_s_figure_as_a_float(self,
                                                     get,
                                                     figures):
        found = get("total_return")

        assert isinstance(found, float)
        assert found == figures("total_return")["fund"]

    def test_of_all_gives_a_series_by_portfolio(self,
                                                get,
                                                figures):
        found = get("total_return", of="all")

        assert found.index.name == "book"
        assert found.to_dict() == pytest.approx(figures("total_return"))

    def test_of_picks_a_portfolio(self,
                                  get,
                                  figures):
        assert get("sharpe_ratio", of="benchmark") == figures("sharpe_ratio")["benchmark"]

    def test_a_figure_s_other_parts_are_its_index(self,
                                                  get):
        found = get("value_at_risk")

        assert list(found.index.names) == ["method", "confidence", "horizon"]
        assert get("positive_share").index.name == "frequency"

    def test_each_added_dimension_adds_one(self,
                                           get):
        assert isinstance(get("value_at_risk", of="all"), pd.DataFrame)
        assert isinstance(get("total_return", by="quarter", of="all"), pd.DataFrame)

    def test_a_count_is_a_float_too(self,
                                    get):
        assert isinstance(get("holdings_count", as_of="2024-06-28"), float)


class TestWindows:
    def test_start_and_end_measure_that_window(self,
                                               result,
                                               get):
        window = AnalysisSettings(start="2024-07-01", end="2024-09-30")
        context = Context(result.run_data, AnalysisOptions(window=window))
        own = CATALOGUE["sharpe_ratio"].compute(context)["fund"]

        assert get("sharpe_ratio", start="2024-07-01", end="2024-09-30") == own

    def test_by_quarter_compounds_to_the_year(self,
                                              get):
        quarters = get("total_return", by="quarter")

        assert list(quarters.index) == ["2024 Q1", "2024 Q2", "2024 Q3", "2024 Q4"]
        assert (1 + quarters).prod() - 1 == pytest.approx(get("total_return"), rel=1e-12)

    def test_trailing_windows_end_on_the_last_day(self,
                                                  get):
        found = get("total_return", trailing=["3M", "1Y", "YTD", "inception"])

        assert found.index.name == "window"
        assert found["YTD"] == pytest.approx(get("total_return"), rel=1e-12)
        assert found["3M"] == pytest.approx(get("total_return", start="2024-10-01"), rel=1e-12)
        assert pd.isna(found["1Y"])  # longer than the run

    def test_a_rolling_window_ends_each_day(self,
                                            result,
                                            get):
        rolled = get("volatility", rolling=63)
        days = result.run_data.returns.index

        assert rolled.index[-1] == days[-1]
        assert rolled.iloc[-1] == pytest.approx(get("volatility", start=days[-63]), rel=1e-9)

    def test_a_rolling_window_named_as_a_span(self,
                                              get):
        assert get("volatility", rolling="3M").equals(get("volatility", rolling=63))

    def test_a_rolling_window_can_step_month_end(self,
                                                 get):
        stepped = get("volatility", rolling=63, step="month_end")

        assert all(day.is_month_end or day.month != (day + pd.offsets.BDay()).month
                   for day in stepped.index)

    def test_against_picks_the_comparator(self,
                                          get,
                                          figures):
        assert get("tracking_error", against="benchmark") == figures("tracking_error")[
            "benchmark"]
        assert get("tracking_error", against="all").index.name == "comparator"


class TestThroughTime:
    def test_as_of_is_the_last_row_on_or_before_the_day(self,
                                                        get):
        daily = get("holdings_count", of="all")
        found = get("holdings_count", as_of="2024-06-15", of="all")

        assert found.name == pd.Timestamp("2024-06-14")
        assert found.to_dict() == daily.loc["2024-06-14"].to_dict()

    def test_bare_a_dated_table_is_the_fund_s_column(self,
                                                     get):
        assert get("holdings_count").name == "fund"

    def test_nothing_is_given_before_the_run(self,
                                             get):
        with pytest.raises(ValueError, match="nothing on or before"):
            get("holdings_count", as_of="2023-06-30")


class TestRefused:
    def test_a_keyword_it_does_not_take_names_those_it_does(self,
                                                            get):
        with pytest.raises(TypeError, match="takes of, start, end, by, trailing"):
            get("total_return", names=["AAA"])

    def test_one_time_dimension_at_a_time(self,
                                          get):
        with pytest.raises(ValueError, match="not by and rolling"):
            get("total_return", by="year", rolling=63)

    def test_step_without_rolling(self,
                                  get):
        with pytest.raises(ValueError, match="step goes with rolling"):
            get("volatility", step="daily")

    def test_a_portfolio_the_run_lacks(self,
                                       get):
        with pytest.raises(ValueError, match="No portfolio 'optimised' here"):
            get("total_return", of="optimised")

    def test_an_unknown_period_or_window(self,
                                         get):
        with pytest.raises(ValueError, match="by is one of month, quarter, year"):
            get("total_return", by="week")
        with pytest.raises(ValueError, match="Unknown window 2W"):
            get("total_return", trailing=["2W"])

    def test_what_the_run_cannot_give(self,
                                      get):
        with pytest.raises(Unavailable, match="money arriving or leaving"):
            get("money_weighted_return")
