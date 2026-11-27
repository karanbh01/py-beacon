# tests/test_renamed_portfolio_names.py
"""BN-479: the public names BN-462 missed say portfolio, not book. Each old
name and keyword keeps working, warning, until 0.12.0; a share class's net
assets in the base currency are read under their new name from a run or a
job saved before 0.11.0."""
import importlib
import json
import subprocess
import sys
import warnings
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from beacon.analysis.attribution.currency import currency_parts
from beacon.analysis.class_panels import CLASS_DAYS
from beacon.analysis.data import RunData
from beacon.analysis.packing import StoredSource
from beacon.analysis.panels import read_panels, write_panels
from beacon.analysis.performance.crises import crisis_paths
from beacon.analysis.performance.distribution import return_distribution
from beacon.analysis.performance.fund_figures import characteristics_over_time
from beacon.analysis.performance.members import RiskReturnSettings, portfolio_risk_return
from beacon.analysis.performance.risk_rating import RiskRatingSettings, risk_rating
from beacon.analysis.performance.style import style_box, style_classification
from beacon.analysis.risk import forecast
from beacon.analysis.risk.concentration import lorenz
from beacon.analysis.risk.var import RiskOfLoss
from beacon.backtest import (
    CapacityCap,
    Implementation,
    ScreenContext,
    ShareClassRecord,
    WeightCap,
)
from beacon.backtest.class_benchmark import class_tracking
from beacon.backtest.implementation import plan
from beacon.server.store import SCHEMA_VERSION_KEY, DocumentStore

DAYS = pd.bdate_range("2024-01-01", periods=300)
RETURNS = pd.Series(np.random.default_rng(7).normal(0.0003, 0.01, len(DAYS)), index=DAYS)
LEVELS = (1.0 + RETURNS).cumprod() * 100.0


def record(base: pd.Series | None = None) -> ShareClassRecord:
    """A class's record with a NAV per unit following RETURNS."""
    per_unit = LEVELS / 100.0
    empty = pd.Series(0.0, index=DAYS)

    return ShareClassRecord(name="EUR", distribution="accumulating", management_fee_bps=0.0,
                            net_assets=per_unit * 1e6, units=pd.Series(1e6, index=DAYS),
                            nav_per_unit=per_unit, fees=empty, distributions=empty,
                            distributions_per_unit=empty, currency="EUR",
                            base_net_assets=base if base is not None else per_unit * 9e5)


def positions() -> pd.DataFrame:
    """Two placed names, held by the fund and the index."""
    return pd.DataFrame({"name": ["AAA", "BBB"], "size": [0.5, 1.5], "style": [0.5, 2.5],
                         "fund": [0.6, 0.4], "index": [0.5, 0.5]})


def silent(call):
    """*call*'s answer, failing on any deprecation warning."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)

        return call()


class TestTheShareClassRecord:

    def test_base_net_assets_is_the_base_currency_figure(self):
        found = silent(lambda: record().base_net_assets)

        assert found.iloc[-1] == pytest.approx(LEVELS.iloc[-1] / 100.0 * 9e5)

    def test_book_net_assets_warns_and_gives_it(self):
        share_class = record()

        with pytest.warns(DeprecationWarning, match=r"ShareClassRecord\.book_net_assets is "
                                                    r"deprecated and will be removed in "
                                                    r"0\.12\.0; use base_net_assets"):
            found = share_class.book_net_assets

        assert found is share_class.base_net_assets


class TestRiskOfLoss:

    def test_books_warns_and_gives_the_portfolios(self):
        loss = RiskOfLoss(RunData.of_returns(RETURNS, 252, 0.0, currency="USD"))

        with pytest.warns(DeprecationWarning, match=r"RiskOfLoss\.books\(\) is deprecated .* "
                                                    r"use portfolios\(\)"):
            found = loss.books()

        assert found == silent(loss.portfolios) == ["fund"]


class TestClassTracking:

    def test_book_is_passed_as_share_class_warning(self):
        share_class = record()

        with pytest.warns(DeprecationWarning, match=r"class_tracking\(book=\) is deprecated "
                                                    r".* use share_class="):
            found = class_tracking(book=share_class, benchmark=LEVELS)

        assert found == silent(lambda: class_tracking(share_class=share_class,
                                                      benchmark=LEVELS))

    def test_not_both(self):
        with pytest.raises(TypeError, match="takes share_class= or book=, not both"):
            class_tracking(share_class=record(), benchmark=LEVELS, book=record())


class TestThePerformanceFunctions:

    @pytest.mark.parametrize("function", [
        lambda **kw: risk_rating(RETURNS, None, settings=RiskRatingSettings(),
                                 periods_per_year=252, **kw),
        lambda **kw: portfolio_risk_return(settings=RiskReturnSettings(), **kw),
        lambda **kw: crisis_paths([], **kw),
        lambda **kw: return_distribution(bins=10, **kw),
    ], ids=["risk_rating", "portfolio_risk_return", "crisis_paths", "return_distribution"])
    def test_books_is_passed_as_portfolios_warning(self,
                                                   function):
        given = {"fund": RETURNS, "index": RETURNS * 0.9}

        with pytest.warns(DeprecationWarning, match=r"\(books=\) is deprecated .* "
                                                    r"use portfolios="):
            found = function(books=given)

        assert found.equals(silent(lambda: function(portfolios=given)))

    def test_characteristics_over_time(self):
        taken = pd.DataFrame({"IDENTIFIER": ["AAA", "BBB"], "MARKET_CAP": [1e9, 3e9]},
                             index=pd.MultiIndex.from_tuples([(DAYS[0], 0), (DAYS[0], 1)]))
        weights = {"fund": pd.DataFrame({"AAA": [0.5], "BBB": [0.5]}, index=DAYS[:1])}

        with pytest.warns(DeprecationWarning, match="use portfolios="):
            found = characteristics_over_time(taken, books=weights)

        assert found.equals(characteristics_over_time(taken, portfolios=weights))

    @pytest.mark.parametrize("function", [style_box, style_classification])
    def test_the_style_tables(self,
                              function):
        with pytest.warns(DeprecationWarning, match="use portfolios="):
            found = function(positions(), books=("fund", "index"))

        assert found.equals(function(positions(), portfolios=("fund", "index")))


class TestTheForecast:

    def test_book_returns_is_portfolio_returns_warning(self):
        with pytest.warns(DeprecationWarning, match=r"forecast\.book_returns is deprecated "
                                                    r".* use portfolio_returns"):
            found = forecast.book_returns

        assert found is forecast.portfolio_returns

    def test_importing_it_does_not_warn(self):
        found = subprocess.run([sys.executable, "-W", "error::DeprecationWarning", "-c",
                                "import beacon.analysis.risk.forecast, "
                                "beacon.backtest.class_benchmark, beacon.analysis.packing"],
                               capture_output=True, text=True, check=False)

        assert "is deprecated and will be removed" not in found.stderr

    def test_an_unknown_name_is_still_refused(self):
        with pytest.raises(AttributeError, match="has no attribute 'Nothing'"):
            _ = importlib.import_module("beacon.analysis.risk.forecast").Nothing


class TestTheCurrencyParts:

    def test_stored_source_takes_currency(self):
        rates = pd.DataFrame({"GBP": [1.25]}, index=DAYS[:1])

        with pytest.warns(DeprecationWarning, match=r"__init__\(book=\) is deprecated .* "
                                                    r"use currency="):
            source = StoredSource(pd.DataFrame(), rates, pd.DataFrame(), book="usd")

        assert source.currency == "USD"

        with pytest.warns(DeprecationWarning, match=r"StoredSource\.book is deprecated .* "
                                                    r"use currency"):
            assert source.book == "USD"

    def test_currency_parts_takes_base(self):
        lagged = pd.DataFrame({"AAA": [0.5, 0.5]}, index=DAYS[:2])
        base = pd.DataFrame({"AAA": [0.02, 0.01]}, index=DAYS[:2])
        local = pd.DataFrame({"AAA": [0.01, 0.01]}, index=DAYS[:2])

        with pytest.warns(DeprecationWarning, match=r"currency_parts\(book=\) is deprecated"):
            found = currency_parts(lagged, book=base, local=local)

        expected = currency_parts(lagged, base=base, local=local)

        assert all(found[part].equals(expected[part]) for part in expected)


class TestSavedBefore:

    def test_a_saved_class_days_table_reads_base_net_assets(self,
                                                            tmp_path):
        old = pd.DataFrame({"CLASS": ["EUR", "EUR"], "NET_ASSETS": [1e6, 1.1e6],
                            "BOOK_NET_ASSETS": [9e5, 9.9e5]},
                           index=pd.DatetimeIndex(DAYS[:2], name="DATE"))
        write_panels(tmp_path, {CLASS_DAYS: old})

        found = read_panels(tmp_path)[CLASS_DAYS]

        assert list(found.columns) == ["CLASS", "NET_ASSETS", "BASE_NET_ASSETS"]
        assert found["BASE_NET_ASSETS"].tolist() == [9e5, 9.9e5]

    def test_a_stored_job_s_share_classes_are_migrated(self,
                                                      tmp_path):
        store = DocumentStore("job_results", root=tmp_path)
        document = {"job_id": "old", "status": "succeeded", SCHEMA_VERSION_KEY: 4,
                    "result": {"share_classes": [{"name": "EUR", "net_assets": {"data": [1]},
                                                  "book_net_assets": {"data": [0.9]}}]}}
        (store.directory / "old.json").write_text(json.dumps(document), encoding="utf-8")

        found = store.read("old")

        assert found is not None
        assert found["result"]["share_classes"][0] == {"name": "EUR",
                                                       "net_assets": {"data": [1]},
                                                       "base_net_assets": {"data": [0.9]}}
        assert found[SCHEMA_VERSION_KEY] == 5

    def test_a_job_without_share_classes_is_untouched(self,
                                                      tmp_path):
        store = DocumentStore("job_results", root=tmp_path)
        document = {"job_id": "plain", "status": "succeeded", "result": {"nav": []},
                    SCHEMA_VERSION_KEY: 4}
        (store.directory / "plain.json").write_text(json.dumps(document), encoding="utf-8")

        found = store.read("plain")

        assert found is not None
        assert found["result"] == {"nav": []}


class TestTheCapacityValue:
    """BN-480: the capacity caps' ``book_value`` is ``portfolio_value``."""

    @staticmethod
    def _plan(**value):
        context = ScreenContext(MagicMock(), "USD")

        return plan(Implementation(caps=[WeightCap(0.4)]), {"A": 0.6, "B": 0.4},
                    pd.Timestamp("2024-01-02"), held=set(), stale=set(), context=context,
                    **value)

    def test_plan_s_book_value_warns_and_gives_the_same_step(self):
        with pytest.warns(DeprecationWarning, match=r"plan\(book_value=\) is deprecated"):
            old = self._plan(book_value=1e6)

        assert old.weights == self._plan(portfolio_value=1e6).weights

    def test_a_cap_that_still_names_it_book_value_works(self):
        class Fixed(CapacityCap):
            def max_value(self,
                          asset_id,
                          date,
                          book_value,
                          context):
                return 0.25 * book_value

        step = plan(Implementation(caps=[Fixed()]), {"A": 0.6, "B": 0.4},
                    pd.Timestamp("2024-01-02"), held=set(), stale=set(),
                    context=ScreenContext(MagicMock(), "USD"), portfolio_value=1e6)

        assert step.capped == pytest.approx({"A": 0.25, "B": 0.25})

    def test_the_lorenz_curve_s_column_is_share_of_portfolio(self):
        curve = lorenz(pd.Series({"A": 0.5, "B": 0.5}))

        assert list(curve.columns) == ["share_of_names", "share_of_portfolio"]
