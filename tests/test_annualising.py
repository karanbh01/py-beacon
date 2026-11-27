# tests/test_annualising.py
"""BN-465: a figure annualised over too short a window is left blank,
wherever it is asked for, unless ``annualise`` says otherwise.

The run is test_strategy_futures' eight-name market-cap index held physically
through 2024, 252 daily returns, with its total-return version as the
benchmark.
"""
import numpy as np
import pandas as pd
import pytest

from beacon import ModellingAssumptions
from beacon.analysis.annualising import blank, blanked_over, days_in, long_enough
from beacon.analysis.data import RunData
from beacon.analysis.performance.settings import AnalysisSettings
from beacon.analysis.reports.options import AnalysisOptions
from beacon.analysis.request import AnalysisRequest, analyse
from beacon.backtest import Backtest


def run(assumptions: ModellingAssumptions | None = None):
    from beacon.index.calculation import IndexCalculator
    from test_strategy_futures import DATA, END, START, index

    benchmark = IndexCalculator(index("TOTAL_RETURN"), DATA).run(start_date=START, end_date=END)

    return Backtest(initial_capital=1e7, data_provider=DATA, cache=False,
                    transaction_cost_bps=5.0, benchmark=benchmark,
                    modelling_assumptions=assumptions).run(index(), start=START, end=END)


@pytest.fixture(scope="module")
def result():
    return run()


@pytest.fixture(scope="module")
def portfolio(result):
    return result.analytics.portfolio


class TestTheRule:
    def test_a_window_in_days(self):
        assert days_in("6M", 252) == 126
        assert days_in("1Y", 252) == 252
        assert days_in("0M", 252) == 0

        with pytest.raises(ValueError, match="number of months or years"):
            days_in("26W", 252)

    def test_a_calendar_year_is_a_year(self):
        assert long_enough(250, "1Y", 252)
        assert not long_enough(246, "1Y", 252)
        assert long_enough(1, "0M", 252)

    def test_every_figure_or_the_ones_named(self):
        assert blank({"fund": 1.0, "index": 2.0}, ()) == {"fund": None, "index": None}
        assert blank({"index": {"alpha": 0.01, "t": 2.0}}, ("alpha",)) == {
            "index": {"alpha": None, "t": 2.0}}

    def test_a_spread_is_never_blank(self):
        assert blanked_over("volatility", 5, "1Y", "6M", 252) is None
        assert blanked_over("sharpe_ratio", 63, "1Y", "6M", 252) == ()
        assert blanked_over("sharpe_ratio", 126, "1Y", "6M", 252) is None


class TestTheThresholds:
    def test_the_library_s(self):
        resolved = ModellingAssumptions().with_defaults()

        assert resolved.min_annualised_rate_window == "1Y"
        assert resolved.min_annualised_ratio_window == "6M"

    def test_a_window_that_is_not_one_is_refused(self):
        with pytest.raises(ValueError, match="min_annualised_ratio_window"):
            ModellingAssumptions(min_annualised_ratio_window="six months")

        with pytest.raises(ValueError, match="min_annualised_rate_window"):
            AnalysisSettings(min_annualised_rate_window="1")

    def test_a_run_records_its_own(self,
                                   tmp_path):
        # BN-477: and the fewest days a period's risk is given over.
        shorter = run(ModellingAssumptions(min_annualised_ratio_window="3M", min_risk_days=3))
        shorter.run_data.write(tmp_path / "run")
        again = RunData.from_record(tmp_path / "run")

        assert again.min_annualised_ratio_window == "3M"
        assert again.min_annualised_rate_window == "1Y"
        assert shorter.analytics.run_data.min_risk_days == again.min_risk_days == 3
        assert shorter.analytics.portfolio.performance.sharpe_ratio(by="quarter").notna().all()

    def test_a_run_kept_before_the_fewest_risk_days_were(self,
                                                         result,
                                                         tmp_path):
        import json

        from beacon.analysis.data import RUN_FILE

        result.analytics.run_data.write(tmp_path / "run")
        path = tmp_path / "run" / RUN_FILE
        kept = json.loads(path.read_text(encoding="utf-8"))
        del kept["min_risk_days"]
        path.write_text(json.dumps(kept), encoding="utf-8")

        assert RunData.from_record(tmp_path / "run").min_risk_days == 5

    def test_fewest_risk_days_that_are_not_a_number_of_days_are_refused(self):
        with pytest.raises(ValueError, match="min_risk_days"):
            AnalysisSettings(min_risk_days=1)


class TestEveryWindow:
    def test_the_whole_run_is_a_year(self,
                                     portfolio):
        assert not np.isnan(portfolio.performance.sharpe_ratio())
        assert not np.isnan(portfolio.performance.annualised_return())

    def test_a_quarter_has_no_sharpe_ratio_but_its_volatility(self,
                                                              portfolio):
        assert portfolio.performance.sharpe_ratio(by="quarter").isna().all()
        assert portfolio.risk.volatility(by="quarter").notna().all()

    def test_trailing_windows(self,
                              portfolio):
        found = portfolio.performance.sharpe_ratio(trailing=["3M", "6M"])

        assert np.isnan(found["3M"]) and not np.isnan(found["6M"])
        assert np.isnan(portfolio.performance.annualised_return(trailing=["6M"])["6M"])

    def test_a_window_s_alpha_but_not_its_significance(self,
                                                       portfolio):
        found = portfolio.relative.alpha(start="2024-10-01", against="index")

        assert np.isnan(found["alpha"]) and not np.isnan(found["t"])

    def test_rolling_windows(self,
                             portfolio):
        assert portfolio.performance.sharpe_ratio(rolling=63).isna().all()
        assert portfolio.performance.sharpe_ratio(rolling="6M").notna().any()

    def test_the_api_s_figures_too(self,
                                   result):
        options = AnalysisOptions(window=AnalysisSettings(start=pd.Timestamp("2024-10-01")))
        block = analyse(result, AnalysisRequest(reports={}, statistics=["sharpe_ratio"],
                                                options=options))

        assert block["statistics"]["sharpe_ratio"]["point"]["fund"] is None


class TestTheOverride:
    def test_annualised_over_any_window(self,
                                        portfolio):
        annualised = portfolio.performance.sharpe_ratio(by="quarter", annualise=True)
        daily = portfolio.performance.sharpe_ratio(by="quarter", annualise=False)

        assert annualised.notna().all()
        # The run's risk-free rate is 0, so the two differ by the scaling alone.
        pd.testing.assert_series_equal(annualised, daily * np.sqrt(252))

    def test_per_day(self,
                     result,
                     portfolio):
        returns = result.run_data.returns
        total = portfolio.performance.total_return()

        assert portfolio.risk.volatility(annualise=False) == pytest.approx(
            portfolio.risk.volatility() / np.sqrt(252), rel=1e-12)
        assert portfolio.performance.annualised_return(annualise=False) == pytest.approx(
            (1 + total) ** (1 / len(returns)) - 1, rel=1e-12)

    def test_a_rolling_window_in_months_counts_the_run_s_days(self,
                                                             portfolio):
        found = portfolio.performance.sharpe_ratio(rolling="3M", annualise=False)

        assert found.notna().any()

    def test_only_true_false_or_none(self,
                                     portfolio):
        with pytest.raises(ValueError, match="annualise is True, False or None"):
            portfolio.performance.sharpe_ratio(annualise="yes")

    def test_a_statistic_not_annualised_does_not_take_it(self,
                                                         result):
        with pytest.raises(TypeError, match="total_return takes"):
            result.analytics.get("total_return", annualise=True)

    def test_a_table_gives_it_to_those_annualised(self,
                                                  result):
        found = result.analytics.table(["total_return", "sharpe_ratio"], by="quarter",
                                       annualise=True)

        assert found.notna().all().all()
