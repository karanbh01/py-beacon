# tests/test_short_window_figures.py
"""BN-466: the short-window rule (BN-465) where figures are given together:
each calendar period's, a comparison's and the run's summary. A rate (an
annualised return) is blank over less than a year, a ratio (the Sharpe and
information ratios) over less than six months; a spread (volatility,
tracking error) is always given.

The runs are synthetic daily returns on business days, with an index whose
levels drift apart from the fund's.
"""
import numpy as np
import pandas as pd
import pytest

from beacon.analysis.data import RunData
from beacon.analysis.performance.analytics import PerformanceAnalytics
from beacon.analysis.performance.settings import AnalysisSettings
from beacon.server.schemas import metrics_from


def run(days: int,
        start: str = "2023-01-02") -> RunData:
    """A run of *days* daily returns, the first from its start, and an index."""
    index = pd.bdate_range(start, periods=days)
    rng = np.random.default_rng(7)
    ours = pd.Series(rng.normal(0.0004, 0.01, days), index=index)
    ours.iloc[0] = 0.0
    theirs = 100.0 * (1.0 + pd.Series(rng.normal(0.0003, 0.009, days), index=index)).cumprod()

    return RunData.of_returns(ours, 252, 0.01, {"index": theirs})


class TestEachPeriod:

    @pytest.fixture(scope="class")
    def analytics(self):
        return PerformanceAnalytics(run(2 * 252 + 60))

    def test_a_quarter_s_sharpe_ratio_is_blank(self,
                                              analytics):
        quarters = analytics.periods("quarterly")

        assert all(period.figures["sharpe_ratio"] is None for period in quarters)
        assert all(period.figures["volatility"] is not None for period in quarters)
        assert all(period.figures["annualised_return"] is None for period in quarters)

    def test_a_whole_year_keeps_it_and_a_short_one_does_not(self,
                                                            analytics):
        years = {period.period: period for period in analytics.periods("annual")}

        assert years["2023"].figures["sharpe_ratio"] is not None
        assert years["2023"].figures["annualised_return"] is not None
        # The run's third year has about 50 days.
        assert years["2025"].figures["sharpe_ratio"] is None

    def test_a_comparator_s_and_the_relative_figures_too(self,
                                                         analytics):
        quarter = analytics.periods("quarterly")[1].against["index"]

        assert quarter["own"]["sharpe_ratio"] is None
        assert quarter["own"]["annualised_return"] is None
        assert quarter["relative"]["annualised_relative_return"] is None
        assert quarter["relative"]["information_ratio"] is None
        assert quarter["relative"]["tracking_error"] is not None

    def test_the_period_table_follows(self,
                                      analytics):
        table = analytics.period_table("quarterly")

        assert table[("fund", "sharpe_ratio")].isna().all()
        assert table[("fund", "volatility")].notna().all()

    def test_turned_off_every_period_has_its_figures(self):
        settings = AnalysisSettings(min_annualised_rate_window="0M",
                                    min_annualised_ratio_window="0M")
        quarters = PerformanceAnalytics(run(300), settings).periods("quarterly")

        assert all(period.figures["sharpe_ratio"] is not None for period in quarters)


class TestTheWholeRun:

    def test_under_six_months_both_are_blank(self):
        analytics = PerformanceAnalytics(run(100))
        summary = analytics.summary()
        comparison = analytics.comparison("index")

        assert summary["annualised_return"] is None and summary["sharpe_ratio"] is None
        assert summary["volatility"] is not None and summary["total_return"] is not None
        assert comparison.own["sharpe_ratio"] is None
        assert comparison.relative["information_ratio"] is None

    def test_between_six_months_and_a_year_only_the_rate_is(self):
        summary = PerformanceAnalytics(run(200)).summary()

        assert summary["annualised_return"] is None
        assert summary["sharpe_ratio"] is not None

    def test_over_a_year_both_are_given(self):
        summary = PerformanceAnalytics(run(300)).summary()

        assert summary["annualised_return"] is not None
        assert summary["sharpe_ratio"] is not None

    def test_the_server_s_metrics_carry_the_blanks(self):
        metrics = metrics_from(PerformanceAnalytics(run(100)).summary())

        assert metrics.annualised_return is None and metrics.sharpe_ratio is None
        assert metrics.total_return is not None


class TestRegimesAndRiskReturn:

    def test_a_regime_of_few_days_has_no_annual_figures(self):
        from beacon.analysis.performance.regimes import regime_of

        returns = run(400).returns
        theirs = returns * 0.8
        few = pd.Series(returns.index < returns.index[60], index=returns.index)
        many = pd.Series(True, index=returns.index)

        short = regime_of(returns, theirs, few, 0.0, 252, "1Y")
        whole = regime_of(returns, theirs, many, 0.0, 252, "1Y")

        assert (short.fund_annual, short.comparator_annual, short.alpha) == (None, None, None)
        assert short.fund is not None and short.beta is not None
        assert None not in (whole.fund_annual, whole.comparator_annual, whole.alpha)

    def test_a_portfolio_s_return_needs_a_year_of_periods(self):
        from beacon.analysis.performance.members import (
            RiskReturnSettings,
            portfolio_risk_return,
        )

        short = portfolio_risk_return({"fund": run(150).returns}, RiskReturnSettings(), "1Y")
        long = portfolio_risk_return({"fund": run(400).returns}, RiskReturnSettings(), "1Y")

        assert np.isnan(short["return"].iloc[0]) and not np.isnan(short["volatility"].iloc[0])
        assert not np.isnan(long["return"].iloc[0])
