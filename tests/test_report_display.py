# tests/test_report_display.py
"""BN-458: a report's formatted display, in Jupyter, printed and as a page.

The run is test_strategy_futures' eight-name market-cap index held physically
through 2024, with its total-return version as the benchmark.
"""
import re
from datetime import date

import numpy as np
import pandas as pd
import pytest

from beacon.analysis.analytics.display import BLANK, formatted, page_of


@pytest.fixture(scope="module")
def analytics():
    from beacon.backtest import Backtest
    from beacon.index.calculation import IndexCalculator
    from test_strategy_futures import DATA, END, START, index

    benchmark = IndexCalculator(index("TOTAL_RETURN"), DATA).run(start_date=START, end_date=END)

    return Backtest(initial_capital=1e7, data_provider=DATA, cache=False,
                    transaction_cost_bps=5.0, benchmark=benchmark).run(
        index(), start=START, end=END).analytics


@pytest.fixture(scope="module")
def factsheet(analytics):
    return analytics.reports.factsheet(level="extended")


def block(page,
          section,
          title):
    found = next(part for part in page.sections if part.title == section)

    return next(item for item in found.blocks if item.title == title)


class TestEachFigureFromItsUnit:
    def test_by_unit(self):
        assert formatted(0.171434, "fraction") == "17.14%"
        assert formatted(1.0498, "ratio") == "1.05"
        assert formatted(4.72, "bps") == "4.7 bps"
        assert formatted(1234.0, "count") == "1,234"
        assert formatted(1234.567, "money") == "1,234.57"

    def test_without_a_unit(self):
        assert formatted(5.0, None) == "5"
        assert formatted(29_061_986_005.53, None) == "29,061,986,006"
        assert formatted(0.0118, None) == "0.0118"

    def test_what_is_not_a_number(self):
        assert formatted(None, "fraction") == formatted(np.nan, "ratio") == BLANK
        assert formatted(pd.Timestamp("2024-10-01"), None) == "2024-10-01"
        assert formatted(date(2024, 10, 1), "fraction") == "2024-10-01"
        assert formatted(np.bool_(False), None) == "No"
        assert formatted("2024-W38", "fraction") == "2024-W38"


class TestThePage:
    def test_its_header(self,
                        factsheet):
        page = page_of(factsheet)

        assert page.title == "Factsheet report"
        assert dict(page.facts) == {"Level": "Extended", "Window": "2024-01-02 to 2024-12-31",
                                    "Currency": "USD",
                                    "Portfolios": "Fund, Index, Benchmark"}

    def test_a_section_a_category_in_the_report_s_order(self,
                                                         factsheet):
        titles = [section.title for section in page_of(factsheet).sections]

        assert titles[:3] == ["Performance", "Risk", "Relative"]
        assert len(titles) == len(set(titles))

    def test_single_figures_a_column_a_portfolio(self,
                                                 factsheet):
        page = page_of(factsheet)
        performance = page.sections[0].blocks[0]
        relative = next(section for section in page.sections
                        if section.title == "Relative").blocks[0]
        rows = {row[0]: row[1:] for row in performance.rows}

        assert performance.header == ["", "Fund", "Index", "Benchmark", "Active"]
        assert rows["Total return"][0] == formatted(
            factsheet["total_return"].headline, "fraction")
        # Against a comparator, under each comparator.
        assert relative.header == ["", "Index", "Benchmark"]

    def test_a_table_to_its_first_rows(self,
                                       factsheet):
        found = block(page_of(factsheet), "Performance", "Period returns")
        rows = len(factsheet["period_returns"].headline)

        assert len(found.rows) == 5
        assert found.note == f"And {rows - 5} more rows."

    def test_each_figure_of_several_from_its_own_unit(self,
                                                      factsheet):
        gains = dict(block(page_of(factsheet), "Performance", "Average gain and loss").rows)
        alpha = dict(block(page_of(factsheet), "Relative", "Alpha and its t-statistic").rows)

        assert re.fullmatch(r"\d+", gains["Gains"])
        assert gains["Ratio"] == formatted(factsheet["gains_and_losses"].headline["ratio"],
                                           "ratio")
        assert not alpha["T"].endswith("%") and alpha["P"].endswith("%")

    def test_through_time_as_its_latest(self,
                                        factsheet):
        found = block(page_of(factsheet), "Holdings", "Holdings count, latest")
        counts = factsheet["holdings_count"].headline

        assert found.rows == [[counts.index[-1].strftime("%Y-%m-%d"),
                               formatted(counts.iloc[-1], None)]]

    def test_rolling_views_as_their_latest(self,
                                           factsheet):
        found = block(page_of(factsheet), "Risk", "Rolling 252 days, latest")
        rows = {row[0]: row[1:] for row in found.rows}

        assert rows["Volatility"] == [formatted(factsheet["volatility"].rolling.iloc[-1],
                                                "fraction"), "2024-12-31"]

    def test_what_was_unavailable_and_why(self,
                                          factsheet):
        unavailable = dict(page_of(factsheet).unavailable)

        assert unavailable["Net flows"] == factsheet.unavailable["net_flows"]


class TestItsForms:
    def test_in_jupyter(self,
                        factsheet):
        found = factsheet._repr_html_()

        assert found.startswith('<div class="beacon-report">')
        assert found.count("<details open>") == len(page_of(factsheet).sections)
        assert "<summary>Relative</summary>" in found and "Not available:" in found

    def test_printed(self,
                     factsheet):
        found = str(factsheet)

        assert found.startswith("Factsheet report\n================")
        assert "\nRelative\n--------\n" in found
        assert "Net flows: This needs a run with money arriving or leaving." in found

    def test_as_a_page_to_save(self,
                               factsheet):
        found = factsheet.to_html()

        assert found.startswith("<!doctype html>")
        assert "<title>Factsheet report</title>" in found
        assert factsheet._repr_html_() in found

    def test_every_report_draws(self,
                                analytics):
        for name in ("performance", "risk", "tracking_error", "factor", "portfolio",
                     "attribution", "implementation"):
            report = getattr(analytics.reports, name)(level="extended")

            assert str(report).startswith(f"{report.label} report")
            assert report.to_html()


class TestWhatItReads:
    def test_figures_against_a_comparator_alone_side_by_side(self,
                                                             analytics):
        found = analytics.table(["tracking_error", "beta"], of="all")

        assert list(found.columns) == ["index", "benchmark"]

    def test_the_fund_s_figures_alone_side_by_side(self):
        from types import SimpleNamespace

        from beacon.analysis.analytics.reports import Report

        asked = []

        def table(ids,
                  **form):
            asked.append(form)

            return pd.DataFrame({"value": [0.1]}, index=ids)

        # Only the money-weighted return is given for the fund alone, and it
        # needs flows this run does not have.
        found = Report._side_by_side(SimpleNamespace(_analytics=SimpleNamespace(table=table)),
                                     ["money_weighted_return"])

        assert asked == [{}] and list(found.columns) == ["fund"]

    def test_factor_betas_roll_a_column_a_factor(self,
                                                 analytics):
        found = analytics.get("factor_betas", rolling=63)

        assert {"market", "size"} <= set(found.columns.get_level_values(-1))
