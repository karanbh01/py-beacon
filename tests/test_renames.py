# tests/test_renames.py
"""BN-462: prose says portfolio, not book, and the public names follow. The
names before 0.11.0 keep working, warning, until 0.12.0."""
import importlib
import subprocess
import sys
import warnings

import pytest

from beacon.backtest import IndexRecords, PortfolioRecord, ShareClassRecord


class TestTheTypes:

    @pytest.mark.parametrize(("module", "old", "new"), [
        ("beacon.backtest.books", "Book", PortfolioRecord),
        ("beacon.backtest.books", "IndexBooks", IndexRecords),
        ("beacon.backtest.result", "Book", PortfolioRecord),
        ("beacon.backtest.result", "IndexBooks", IndexRecords),
        ("beacon.backtest.share_classes", "ShareClassBook", ShareClassRecord),
        ("beacon.backtest", "ShareClassBook", ShareClassRecord),
    ])
    def test_an_old_name_is_the_new_type_warning(self,
                                                 module,
                                                 old,
                                                 new):
        with pytest.warns(DeprecationWarning, match=f"{module}.{old} is deprecated and will be "
                                                    f"removed in 0.12.0; use {new.__name__}"):
            found = getattr(importlib.import_module(module), old)

        assert found is new

    def test_an_unknown_name_is_still_refused(self):
        with pytest.raises(AttributeError, match="has no attribute 'Nothing'"):
            _ = importlib.import_module("beacon.backtest.books").Nothing

    def test_importing_the_package_does_not_warn(self):
        found = subprocess.run([sys.executable, "-W", "error::DeprecationWarning", "-c",
                                "import beacon.backtest, beacon.analysis.risk.liquidity"],
                               capture_output=True, text=True, check=False)

        assert "is deprecated and will be removed" not in found.stderr


class TestTheConstants:

    @pytest.mark.parametrize("module", ["beacon.analysis.attribution.analytics",
                                        "beacon.analysis.performance.portfolio",
                                        "beacon.analysis.risk.concentration",
                                        "beacon.analysis.risk.contributions",
                                        "beacon.analysis.risk.liquidity"])
    def test_books_is_portfolios(self,
                                 module):
        found = importlib.import_module(module)

        with pytest.warns(DeprecationWarning, match="use PORTFOLIOS"):
            assert found.BOOKS is found.PORTFOLIOS


class TestTheKeyword:

    @pytest.fixture(scope="class")
    def attribution(self):
        from beacon.analysis.attribution.analytics import Attribution
        from beacon.analysis.data import RunData
        from test_backtest_flows import run

        return Attribution(RunData.from_result(run(capital=2e8)))

    def test_book_is_passed_as_of_warning(self,
                                          attribution):
        with pytest.warns(DeprecationWarning, match=r"weights\(book=\) is deprecated and will "
                                                    r"be removed in 0\.12\.0; use of="):
            found = attribution.weights(book="fund")

        assert found.equals(attribution.weights(of="fund"))

    def test_of_does_not_warn(self,
                              attribution):
        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)

            _ = attribution.weights(of="fund"), attribution.weights("fund")

    def test_not_both(self,
                      attribution):
        with pytest.raises(TypeError, match="takes of= or book=, not both"):
            attribution.weights(of="fund", book="fund")
