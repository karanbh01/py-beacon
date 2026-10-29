# tests/test_analysis_reports.py
"""BN-398/BN-400: a run's analysis as reports.

The catalogue of statistics, the reports they are read in, the request that
picks them, and the analytics block a run gives: on test_strategy_futures'
eight-name market-cap index held physically through 2024, with a total
return benchmark, and on the share-class and dividend runs of other tests.
"""
import json
import re

import numpy as np
import pandas as pd
import pytest

from beacon.analysis.performance.relative_analytics import RelativeAnalytics
from beacon.analysis.performance.settings import AnalysisSettings
from beacon.analysis.reports.catalogue import CATALOGUE, CATEGORIES
from beacon.analysis.reports.context import Context
from beacon.analysis.reports.definitions import CORE, EXTENDED, REPORTS, VIEWS
from beacon.analysis.reports.options import (
    MONTH_END,
    SECTIONS,
    AnalysisOptions,
    RollingOptions,
    describe,
)
from beacon.analysis.reports.rolling import FAST, rolling, rolling_all, window_by_window
from beacon.analysis.reports.statistics import (
    HEAVY,
    KINDS,
    LIGHT,
    MODERATE,
    PARTS,
    REASONS,
)
from beacon.analysis.request import FAILED, MESSAGES, AnalysisRequest, analyse
from beacon.backtest import Backtest

EVERY_REPORT = AnalysisRequest(reports=dict.fromkeys(REPORTS, EXTENDED))


@pytest.fixture(scope="module")
def result():
    from beacon.index.calculation import IndexCalculator
    from test_strategy_futures import DATA, END, START, index

    benchmark = IndexCalculator(index("TOTAL_RETURN"), DATA).run(start_date=START, end_date=END)

    return Backtest(initial_capital=1e7, data_provider=DATA, cache=False,
                    transaction_cost_bps=5.0, benchmark=benchmark).run(index(), start=START,
                                                                       end=END)


@pytest.fixture(scope="module")
def block(result):
    return analyse(result, EVERY_REPORT)


def context_of(result,
               options: AnalysisOptions | None = None) -> Context:
    return Context(result.run_data, options, result._data_fetcher)


class TestTheCatalogue:

    def test_every_statistic_is_described_in_full(self):
        for statistic in CATALOGUE:
            assert re.fullmatch(r"[a-z][a-z0-9_]*", statistic.id), statistic.id
            assert statistic.category in CATEGORIES, statistic.id
            assert statistic.kind in KINDS and statistic.part in PARTS, statistic.id
            assert statistic.rolling in (None, LIGHT, MODERATE, HEAVY), statistic.id
            assert statistic.description.endswith("."), statistic.id

    def test_every_setting_a_statistic_reads_exists(self):
        for statistic in CATALOGUE:
            for path in statistic.settings:
                section, name = path.split(".")

                assert name in SECTIONS[section][1], (statistic.id, path)

    def test_labels_are_unique(self):
        labels = [statistic.label for statistic in CATALOGUE]

        assert len(labels) == len(set(labels))

    def test_settings_are_described_for_a_form(self):
        sections = describe()
        levels = next(parameter for section in sections if section["name"] == "value_at_risk"
                      for parameter in section["parameters"] if parameter["name"] == "levels")
        window = next(parameter for section in sections if section["name"] == "rolling"
                      for parameter in section["parameters"] if parameter["name"] == "window")

        assert [section["name"] for section in sections] == list(SECTIONS)
        assert levels["type"] == "array" and levels["items"]["unit"] == "fraction"
        assert levels["default"] == [0.95, 0.99]
        assert window["type"] == "integer" and 252 in window["suggestions"]
        assert window["choices"] is None


class TestTheReports:

    def test_every_statistic_named_is_in_the_catalogue(self):
        for report in REPORTS.values():
            for level in report.levels.values():
                assert all(entry.id in CATALOGUE for entry in level.entries), report.id

    def test_the_extended_view_covers_the_core(self):
        for report in REPORTS.values():
            core, extended = report.level(CORE), report.level(EXTENDED)

            assert set(core.included) <= set(extended.included), report.id

    def test_everything_extended_is_optional_for_the_core(self):
        for report in REPORTS.values():
            core, extended = report.level(CORE), report.level(EXTENDED)
            covered = set(extended.included) | set(extended.optional)

            assert set(core.optional) == covered - set(core.included), report.id

    def test_only_statistics_that_roll_have_rolling_views(self):
        for report in REPORTS.values():
            for level in report.levels.values():
                for entry in level.entries:
                    assert entry.view in VIEWS
                    assert entry.view == "point" or CATALOGUE[entry.id].rolling is not None

    def test_sections_follow_the_categories(self):
        sections = [section.category for section in REPORTS["factsheet"].level(CORE).sections]

        assert sections == [category for category in CATEGORIES if category in sections]


class TestTheRequest:

    def test_an_optional_statistic_is_added_and_an_included_one_taken_out(self):
        request = AnalysisRequest(add={"factsheet": ["sortino_ratio"]},
                                  remove={"factsheet": ["etf_premium"]})
        shown = [statistic_id for statistic_id, _ in request.selection()["factsheet"]]

        assert "sortino_ratio" in shown and "etf_premium" not in shown

    @pytest.mark.parametrize("asked, message", [
        ({"reports": {"nope": "core"}}, "Unknown report"),
        ({"reports": {"factsheet": "deep"}}, "level"),
        ({"add": {"factsheet": ["total_return"]}}, "Not optional"),
        ({"remove": {"factsheet": ["sortino_ratio"]}}, "Not included"),
        ({"time_views": {"factsheet": {"trailing_returns": "rolling"}}}, "does not roll"),
        ({"time_views": {"factsheet": {"volatility": "sideways"}}}, "time view"),
        ({"add": {"risk": ["skew"]}}, "Unknown report asked for"),
        ({"statistics": ["nonsense"]}, "Unknown statistic"),
    ])
    def test_a_wrong_choice_is_refused(self,
                                       asked,
                                       message):
        with pytest.raises(ValueError, match=message):
            AnalysisRequest(**asked)

    def test_a_statistic_in_both_views_is_worked_out_both_ways(self):
        request = AnalysisRequest(time_views={"factsheet": {"volatility": "both",
                                                            "sharpe_ratio": "rolling"}})

        assert request.wanted()["volatility"] == {"point", "rolling"}
        assert request.wanted()["sharpe_ratio"] == {"rolling"}

    def test_it_reads_back_as_plain_data(self):
        echoed = AnalysisRequest(options=AnalysisOptions(window=AnalysisSettings(
            start=pd.Timestamp("2024-03-01")))).as_data()

        assert json.loads(json.dumps(echoed)) == echoed
        assert echoed["options"]["window"]["start"] == "2024-03-01"
        assert echoed["options"]["value_at_risk"]["levels"] == [0.95, 0.99]


class TestTheBlock:

    def test_every_statistic_asked_for_is_worked_out_or_says_why(self,
                                                                 block):
        asked = set(EVERY_REPORT.wanted())

        assert set(block["statistics"]) | set(block["unavailable"]) == asked
        assert all(reason["code"] in REASONS for reason in block["unavailable"].values())
        assert not [statistic_id for statistic_id, reason in block["unavailable"].items()
                    if reason["code"] == FAILED]

    def test_it_is_plain_data(self,
                              block):
        assert json.loads(json.dumps(block)) == json.loads(json.dumps(block, default=str))

    def test_a_run_without_futures_or_flows_says_so(self,
                                                    block):
        assert block["unavailable"]["futures_trading"]["code"] == "no_futures"
        assert block["unavailable"]["net_flows"]["code"] == "no_flows"

    def test_figures_are_the_library_s(self,
                                       result,
                                       block):
        statistics = block["statistics"]
        relative = RelativeAnalytics(result.run_data, "benchmark")

        assert statistics["sharpe_ratio"]["point"]["fund"] == result.analytics.headline()[
            "sharpe_ratio"]
        assert statistics["beta"]["point"]["benchmark"] == pytest.approx(
            relative.regression().beta, rel=1e-12)
        assert statistics["total_return"]["point"]["active"] == pytest.approx(
            statistics["total_return"]["point"]["fund"]
            - statistics["total_return"]["point"]["index"], abs=1e-15)

    def test_reports_name_their_statistics_by_id_in_order(self,
                                                          block):
        factsheet = block["reports"]["factsheet"]

        assert factsheet["level"] == EXTENDED
        assert [entry["id"] for entry in factsheet["statistics"]] == REPORTS[
            "factsheet"].level(EXTENDED).included

    def test_a_table_carries_its_summary(self,
                                         block):
        episodes = block["statistics"]["drawdown_episodes"]

        assert set(episodes["point"]) == {"index", "columns", "data"}
        assert "ulcer_index" in episodes["summary"]

    def test_signed_names_are_names_the_statistic_gives(self,
                                                        block):
        """Every name a statistic marks signed is the last part of one of its
        figures, columns or summary figures, where the run gives it."""
        for statistic_id, entry in block["statistics"].items():
            signed = set(CATALOGUE[statistic_id].signed) - {"*"}
            point = entry["point"]
            names = point.get("columns", point) if isinstance(point, dict) else []
            given = {str(name).split("/")[-1] for name in [*names, *(entry.get("summary") or {})]}

            # A comparator the run lacks gives no names of its own.
            if signed and given:
                assert signed - given <= {"optimised_return"}, statistic_id

    def test_a_volatility_or_a_weight_is_never_signed(self):
        for statistic_id in ("volatility", "tracking_error", "herfindahl", "holdings_count"):
            assert CATALOGUE[statistic_id].signed == ()

    def test_it_names_the_groupings_the_run_kept(self,
                                                 block):
        assert block["groupings"] == ["CURRENCY", "EXCHANGE"]

    def test_each_part_says_so_as_it_starts(self,
                                            result):
        said = []

        analyse(result, EVERY_REPORT, said.append)

        assert said == [MESSAGES[part] for part in ("performance", "portfolio", "risk",
                                                    "attribution")]

    def test_a_factsheet_core_needs_no_risk_model(self,
                                                  result):
        found = analyse(result, AnalysisRequest(remove={"factsheet": ["factor_betas",
                                                                      "average_correlation",
                                                                      "top_weights"]}))

        assert "risk" not in found["timings"]


class TestRolling:

    def test_the_one_pass_figures_are_the_window_by_window_ones(self,
                                                                result):
        context = context_of(result, AnalysisOptions(rolling=RollingOptions(window=63)))
        statistics = [CATALOGUE[statistic_id] for statistic_id in FAST]
        fast = rolling_all(statistics, context)
        slow = window_by_window(statistics, context, "daily")

        for statistic_id in FAST:
            ours, theirs = fast[statistic_id], slow[statistic_id][fast[statistic_id].columns]

            assert ours.index.equals(theirs.index), statistic_id
            np.testing.assert_allclose(ours.to_numpy(float), theirs.to_numpy(float),
                                       rtol=1e-12, atol=1e-14, err_msg=statistic_id)

    def test_a_window_s_figure_is_its_point_figure(self,
                                                   result):
        context = context_of(result, AnalysisOptions(rolling=RollingOptions(window=63)))
        table = rolling(CATALOGUE["kappa_3"], context)
        last = table.index[-1]
        start = context.performance.returns.index[-63]
        windowed = context_of(result, AnalysisOptions(window=AnalysisSettings(start=start,
                                                                              end=last)))

        assert table.loc[last, "fund"] == CATALOGUE["kappa_3"].compute(windowed)["fund"]

    def test_month_end_steps_end_on_month_ends(self,
                                               result):
        context = context_of(result, AnalysisOptions(rolling=RollingOptions(window=63,
                                                                            step=MONTH_END)))
        table = rolling(CATALOGUE["skew"], context)

        assert all(day == group.max() for day, group in zip(
            table.index, [table.index[table.index.to_period("M") == day.to_period("M")]
                          for day in table.index], strict=True))

    def test_a_statistic_that_does_not_roll_is_refused(self,
                                                       result):
        with pytest.raises(ValueError, match="does not roll"):
            rolling(CATALOGUE["trailing_returns"], context_of(result))


class TestAWindowAfterTheStart:
    """A window starting after the run measures the comparator's first day
    from its close before, as it measures the fund's."""

    def test_relative_figures_measure_the_comparator_s_first_day(self,
                                                                 result):
        start = pd.Timestamp("2024-07-01")
        settings = AnalysisSettings(start=start)
        relative = RelativeAnalytics(result.run_data, "benchmark", settings)
        levels = result.run_data.comparators["benchmark"]
        before = levels.loc[:start].index[-2]

        assert relative.theirs.iloc[0] == pytest.approx(
            levels.loc[relative.theirs.index[0]] / levels.loc[before] - 1.0, rel=1e-12)
        assert relative.theirs.iloc[0] != 0.0

    def test_the_tracking_difference_agrees_with_the_fund_s_own(self,
                                                                result):
        settings = AnalysisSettings(start=pd.Timestamp("2024-07-01"), tracked="benchmark")
        relative = RelativeAnalytics(result.run_data, "benchmark", settings)
        own = result.run_data and Context(result.run_data, AnalysisOptions(window=settings))

        assert (float((1 + relative.ours).prod() - (1 + relative.theirs).prod())
                == pytest.approx(own.performance.tracking_difference(), rel=1e-12))

    def test_the_whole_run_is_unchanged(self,
                                        result):
        relative = RelativeAnalytics(result.run_data, "benchmark")

        assert relative.theirs.iloc[0] == 0.0


class TestShareClasses:

    @pytest.fixture(scope="class")
    def classes(self):
        from beacon.backtest import ShareClass
        from test_backtest_share_classes import classes

        return classes(ShareClass("Retail"), ShareClass("Inst", management_fee_bps=10),
                       capital={"Retail": 5e6, "Inst": 3e6})

    def test_each_class_is_a_book(self,
                                  classes):
        found = analyse(classes, AnalysisRequest(reports={}, statistics=["total_return",
                                                                         "class_costs"]))
        total = found["statistics"]["total_return"]["point"]

        assert {"class/Retail", "class/Inst"} <= set(total)
        assert "class_costs" in found["statistics"]

        for name in ("Retail", "Inst"):
            nav = classes.share_classes[name].nav_per_unit.dropna()

            assert total[f"class/{name}"] == pytest.approx(nav.iloc[-1] / nav.iloc[0] - 1.0,
                                                           rel=1e-12)
