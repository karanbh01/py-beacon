# tests/test_reporting.py
"""The Excel reports: a path of either kind, and the valuation date kept."""
import pandas as pd
import pytest

from beacon.portfolio.base import Portfolio
from beacon.portfolio.reporting import ReportGenerator

pytest.importorskip("openpyxl")

VALUATION = pd.Timestamp("2024-06-28")


def portfolio() -> Portfolio:
    return Portfolio(portfolio_id="book", initial_cash=1_000.0)


class TestHoldingsReport:

    def test_a_pathlib_path_is_accepted(self,
                                        tmp_path):
        """BN-260: `report_path.endswith` raised AttributeError on a Path."""
        target = tmp_path / "holdings.xlsx"

        ReportGenerator().generate_holdings_report_excel(portfolio(), target,
                                                         VALUATION)

        assert target.is_file()

    def test_the_valuation_date_is_in_the_workbook(self,
                                                   tmp_path):
        """BN-260: it reached only the log."""
        target = tmp_path / "holdings.xlsx"

        ReportGenerator().generate_holdings_report_excel(portfolio(), target,
                                                         VALUATION)
        sheet = pd.read_excel(target, sheet_name="HoldingsSummary")

        assert sheet.columns[0] == "valuation_date"
        assert set(pd.to_datetime(sheet["valuation_date"])) == {VALUATION}

    def test_a_missing_extension_is_appended(self,
                                             tmp_path):
        ReportGenerator().generate_holdings_report_excel(
            portfolio(), tmp_path / "holdings", VALUATION)

        assert (tmp_path / "holdings.xlsx").is_file()

    def test_a_string_path_still_works(self,
                                       tmp_path):
        target = str(tmp_path / "holdings.xlsx")

        ReportGenerator().generate_holdings_report_excel(portfolio(), target,
                                                         VALUATION)

        assert (tmp_path / "holdings.xlsx").is_file()


class TestPerformanceReport:

    def test_a_pathlib_path_is_accepted(self,
                                        tmp_path):
        frame = pd.DataFrame({"nav": [100.0, 101.0]},
                             index=pd.to_datetime(["2024-01-02", "2024-01-03"]))

        ReportGenerator().generate_performance_report_excel(
            frame, tmp_path / "performance")

        assert (tmp_path / "performance.xlsx").is_file()
