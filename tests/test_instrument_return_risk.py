# tests/test_instrument_return_risk.py
"""BN-477: a name's return and risk over the window, each rebalance period,
year, quarter and month, `instrument.profile(name).return_risk`.

The run is test_instrument_profile's: test_strategy_futures' eight-name
market-cap index held through 2024, rebalanced each quarter (2024-01-02,
04-01, 07-01 and 10-01), each name capped at 30% and positions under 5.5%
dropped, so N0 is capped and N7 removed at the first rebalance; its
total-return version is the benchmark.
"""
import dataclasses

import numpy as np
import pandas as pd
import pytest

from beacon.analysis.analytics import Analytics
from beacon.analysis.analytics.profile import Profile, columns_of
from beacon.analysis.analytics.profile.columns import PERIODS
from beacon.analysis.panels import BENCHMARK_WEIGHTS, FUND_WEIGHTS, NAME_RETURNS
from beacon.analysis.performance.settings import AnalysisSettings
from beacon.analysis.reports.options import AnalysisOptions, AttributionOptions
from beacon.analysis.reports.statistics import REASONS, TOO_SHORT
from beacon.analysis.request import AnalysisRequest
from beacon.assumptions import ModellingAssumptions
from test_instrument_profile import CAP, SMALLEST

FIGURES = ("excess_return", "contribution", "beta", "correlation", "volatility_share")
RISK = ("beta", "correlation", "volatility_share")
PORTFOLIOS = ("fund", "index", "benchmark")
NAMES = [f"N{i}" for i in range(8)]
# The risk figures against each portfolio, null over a short period.
NULLED = [f"{figure}/{portfolio}" for figure in RISK for portfolio in PORTFOLIOS]
REBALANCES = pd.to_datetime(["2024-01-02", "2024-04-01", "2024-07-01", "2024-10-01"])

# A window whose first rebalance period has four daily returns (Good Friday,
# 2024-03-29, is no session) and whose March three.
SHORT = {"start": "2024-03-26", "end": "2024-06-28"}


@pytest.fixture(scope="module")
def result():
    from beacon.backtest import Implementation
    from beacon.backtest.capacity import MinimumPosition, WeightCap
    from beacon.index.calculation import IndexCalculator
    from test_strategy_futures import DATA, END, START, index, run

    benchmark = IndexCalculator(index("TOTAL_RETURN"), DATA).run(start_date=START, end_date=END)
    implementation = Implementation(caps=[WeightCap(CAP)],
                                    minimum_position=MinimumPosition(weight=SMALLEST))

    return run(index(), implementation=implementation, benchmark=benchmark,
               transaction_cost_bps=5.0)


@pytest.fixture(scope="module")
def table(result):
    return result.analytics.instrument.profile("N0").return_risk


def analysed(result,
             data=None,
             **options) -> Analytics:
    """The run's analytics under the analysis options given."""
    request = AnalysisRequest(reports={}, statistic_tables={},
                              options=AnalysisOptions(**options))

    return Analytics(data or result.analytics.run_data, request)


def cut(table: pd.DataFrame,
        period: str) -> pd.DataFrame:
    return table[table["period"] == period]


class TestItsRows:

    def test_the_cuts_in_order(self,
                               table):
        assert table["period"].tolist() == (["run"] + ["rebalance"] * 4 + ["year"]
                                            + ["quarter"] * 4 + ["month"] * 12)
        assert table.index.name == "DATE"
        assert (table.index[0], table["start"].iloc[0]) == (pd.Timestamp("2024-12-31"),
                                                            pd.Timestamp("2024-01-02"))

    def test_a_rebalance_period_runs_from_the_day_after_to_the_next_inclusive(self,
                                                                            table):
        rebalances = cut(table, "rebalance")

        # The window starts on the first rebalance, so the first runs from
        # the day after it.
        assert list(rebalances.index) == [*REBALANCES[1:], pd.Timestamp("2024-12-31")]
        assert rebalances["start"].tolist() == list(pd.to_datetime(
            ["2024-01-03", "2024-04-02", "2024-07-02", "2024-10-02"]))

    def test_the_calendar_periods_by_their_days_in_the_window(self,
                                                              table):
        quarters = cut(table, "quarter")

        assert list(quarters.index) == list(pd.to_datetime(["2024-03-28", "2024-06-28",
                                                            "2024-09-30", "2024-12-31"]))
        assert quarters["start"].tolist() == list(pd.to_datetime(
            ["2024-01-02", "2024-04-01", "2024-07-01", "2024-10-01"]))
        assert cut(table, "year").index.tolist() == [pd.Timestamp("2024-12-31")]
        assert cut(table, "month")["start"].iloc[5] == pd.Timestamp("2024-06-03")

    def test_a_window_clips_each_period(self,
                                        result):
        found = result.analytics.instrument.profile("N0", start="2024-05-15",
                                                    end="2024-08-15").return_risk
        spans = [(period, str(start.date()), str(end.date()))
                 for period, start, end in zip(found["period"], found["start"], found.index,
                                               strict=True)]

        assert spans == [("run", "2024-05-15", "2024-08-15"),
                         ("rebalance", "2024-05-15", "2024-07-01"),
                         ("rebalance", "2024-07-02", "2024-08-15"),
                         ("year", "2024-05-15", "2024-08-15"),
                         ("quarter", "2024-05-15", "2024-06-28"),
                         ("quarter", "2024-07-01", "2024-08-15"),
                         ("month", "2024-05-15", "2024-05-31"),
                         ("month", "2024-06-03", "2024-06-28"),
                         ("month", "2024-07-01", "2024-07-31"),
                         ("month", "2024-08-01", "2024-08-15")]


class TestItsFigures:

    def test_its_return_and_volatility_as_the_per_name_table_s(self,
                                                              result,
                                                              table):
        instrument = result.analytics.instrument

        for end, row in table.iterrows():
            window = {"names": ["N0"], "start": row["start"], "end": end}

            assert row["return"] == pytest.approx(
                instrument.total_return(**window).iloc[0], rel=1e-12)
            assert row["volatility"] == pytest.approx(
                instrument.volatility(**window).iloc[0], rel=1e-12)

    def test_against_each_comparator_as_the_per_name_table_s(self,
                                                             result,
                                                             table):
        instrument = result.analytics.instrument

        for end, row in cut(table, "rebalance").iterrows():
            for portfolio in ("index", "benchmark"):
                window = {"names": ["N0"], "start": row["start"], "end": end, "against": portfolio}

                assert row[f"excess_return/{portfolio}"] == pytest.approx(
                    instrument.excess_return(**window).iloc[0], rel=1e-12)
                assert row[f"beta/{portfolio}"] == pytest.approx(
                    instrument.beta(**window).iloc[0], rel=1e-12)
                assert row[f"correlation/{portfolio}"] == pytest.approx(
                    instrument.correlation(**window).iloc[0], rel=1e-12)

    def test_against_the_fund(self,
                              result,
                              table):
        context = result.analytics._context
        fund = context.performance.returns
        mine = context.run.tables[NAME_RETURNS]["N0"].reindex(fund.index)
        both = mine.notna() & fund.notna()
        run = table.iloc[0]

        assert run["excess_return/fund"] == pytest.approx(
            (1 + mine[both]).prod() - (1 + fund[both]).prod(), rel=1e-12)
        assert run["correlation/fund"] == pytest.approx(mine[both].corr(fund[both]), rel=1e-12)
        assert run["beta/fund"] == pytest.approx(mine[both].cov(fund[both])
                                                 / fund[both].var(), rel=1e-12)


class TestItsContribution:

    def test_the_fund_s_is_the_attribution_part_s(self,
                                                 result,
                                                 table):
        quarters = result.analytics.instrument.profile("N0").attribution

        assert cut(table, "quarter")["contribution/fund"].tolist() == pytest.approx(
            quarters["contribution"].tolist(), rel=1e-12)

    def test_each_cut_adds_up_to_the_whole_window_s(self,
                                                    table):
        for portfolio in PORTFOLIOS:
            sums = table.groupby("period")[f"contribution/{portfolio}"].sum()

            assert sums.to_numpy() == pytest.approx([table[f"contribution/{portfolio}"].iloc[0]]
                                                    * len(sums), rel=1e-12)

    def test_a_comparator_s_over_the_window_is_its_attribution_s(self,
                                                                 result,
                                                                 table):
        attribution = result.analytics._context.attribution

        for portfolio in ("index", "benchmark"):
            found = attribution.contribution(portfolio).to_frame().set_index("asset_id")

            assert table[f"contribution/{portfolio}"].iloc[0] == pytest.approx(
                found.loc["N0", "contribution"], rel=1e-12)

    def test_each_period_on_its_own(self,
                                    result,
                                    table):
        alone = analysed(result, attribution=AttributionOptions(linking="period"))
        found = alone.instrument.profile("N0").return_risk
        quarters = alone.instrument.profile("N0").attribution

        assert found.attrs["linking"] == "period"
        assert cut(found, "quarter")["contribution/fund"].tolist() == pytest.approx(
            quarters["contribution"].tolist(), rel=1e-12)

        # The whole window is linked the same either way; its periods then do
        # not add up to it.
        for portfolio in PORTFOLIOS:
            column = f"contribution/{portfolio}"

            assert found[column].iloc[0] == pytest.approx(table[column].iloc[0], rel=1e-12)
            assert abs(cut(found, "month")[column].sum() - found[column].iloc[0]) > 1e-6


class TestItsPortfolios:

    def test_a_column_for_each_figure_against_each_portfolio_kept(self,
                                                                  table):
        assert list(table.columns) == ["period", "start", "return", "volatility",
                                       *(f"{figure}/{portfolio}" for portfolio in PORTFOLIOS
                                         for figure in FIGURES)]
        assert not any(column.startswith("tracking_error") for column in table.columns)

    def test_none_for_a_portfolio_the_run_kept_no_weights_for(self,
                                                              result):
        data = result.analytics.run_data
        kept = {name: frame for name, frame in data.tables.items()
                if name != BENCHMARK_WEIGHTS}
        found = Analytics(dataclasses.replace(data, tables=kept)).instrument.profile("N0")

        assert not any(column.endswith("/benchmark") for column in found.return_risk.columns)
        assert "contribution/index" in found.return_risk.columns


class TestItsShareOfVolatility:

    def test_every_name_s_add_up_to_1(self,
                                      result):
        profiles = [result.analytics.instrument.profile(name).return_risk for name in NAMES]

        for portfolio in PORTFOLIOS:
            shares = sum(found[f"volatility_share/{portfolio}"] for found in profiles)

            assert shares.to_numpy() == pytest.approx(np.ones(len(shares)), rel=1e-9)

    def test_0_over_a_period_the_portfolio_did_not_hold_it(self,
                                                           result):
        found = cut(result.analytics.instrument.profile("N7").return_risk, "rebalance")
        first = found.iloc[0]
        held = result.analytics._context.run.tables[FUND_WEIGHTS]["N7"].fillna(0.0)

        # Removed at the first rebalance, held by the index throughout.
        assert (held[(held.index >= REBALANCES[0]) & (held.index < REBALANCES[1])] == 0).all()
        assert (first["contribution/fund"], first["volatility_share/fund"]) == (0.0, 0.0)
        assert first["volatility_share/index"] > 0.0


@pytest.fixture(scope="module")
def short(result):
    return result.analytics.instrument.profile("N0", **SHORT).return_risk


class TestAShortPeriod:

    def test_its_risk_is_null_and_says_why(self,
                                           short):
        leading = cut(short, "rebalance").iloc[0]
        march = cut(short, "month").iloc[0]
        why = REASONS[TOO_SHORT][1]

        assert (leading["start"], leading.name) == (pd.Timestamp("2024-03-26"), REBALANCES[1])

        for row in (leading, march):
            assert np.isnan(row["volatility"])
            assert all(np.isnan(row[column]) for column in NULLED)
            assert not np.isnan(row["return"]) and not np.isnan(row["contribution/fund"])
            assert all(not np.isnan(row[f"excess_return/{portfolio}"])
                       for portfolio in PORTFOLIOS)

        assert short.attrs["min_risk_days"] == 5
        assert short.attrs["unavailable"] == {"volatility": why,
                                              **dict.fromkeys(NULLED, why)}

    def test_fewer_days_through_the_analysis_settings(self,
                                                      result):
        fewer = analysed(result, window=AnalysisSettings(min_risk_days=2))
        found = fewer.instrument.profile("N0", **SHORT).return_risk

        assert found.attrs["min_risk_days"] == 2
        assert found.attrs["unavailable"] == {}
        assert not found.drop(columns=["period", "start"]).isna().any().any()

    def test_fewer_days_through_the_modelling_assumptions(self,
                                                          result):
        # The run's own, which the analysis settings override.
        data = dataclasses.replace(result.analytics.run_data, min_risk_days=4)
        found = Analytics(data).instrument.profile("N0", **SHORT).return_risk
        more = analysed(result, data, window=AnalysisSettings(min_risk_days=6))
        leading = cut(found, "rebalance").iloc[0]

        # Four returns of the name, three days attributed: the first day's
        # weight at the close before is not in the window.
        assert found.attrs["min_risk_days"] == 4
        assert not np.isnan(leading["volatility"]) and not np.isnan(leading["beta/index"])
        assert np.isnan(leading["volatility_share/index"])
        assert more.instrument.profile("N0", **SHORT).return_risk.attrs["min_risk_days"] == 6

    def test_a_backtest_s_modelling_assumptions(self):
        from beacon.backtest import Backtest
        from test_strategy_futures import DATA, END, START, index

        run = Backtest(initial_capital=1e7, data_provider=DATA, cache=False,
                       modelling_assumptions=ModellingAssumptions(min_risk_days=2)).run(
            index(), start=START, end=END)
        found = run.analytics.instrument.profile("N0", **SHORT).return_risk

        assert run.analytics.run_data.min_risk_days == 2
        assert found.attrs["min_risk_days"] == 2
        assert not np.isnan(cut(found, "month")["volatility"].iloc[0])


class TestItsColumns:

    def test_each_column_s_label_and_unit(self,
                                          table):
        columns = {column.name: column for column in columns_of(table)}

        assert columns["period"].labels == PERIODS
        assert PERIODS["run"] == "Whole window" and PERIODS["rebalance"] == "Rebalance period"
        assert (columns["start"].label, columns["start"].unit) == ("From", None)
        assert (columns["return"].unit, columns["volatility"].unit) == ("fraction", "fraction")
        assert (columns["contribution/index"].label,
                columns["contribution/index"].unit) == ("Contribution to the index", "fraction")
        assert (columns["beta/benchmark"].label,
                columns["beta/benchmark"].unit) == ("Beta to the benchmark", "ratio")
        assert columns["volatility_share/fund"].label == "Share of the fund's volatility"
        assert columns["excess_return/fund"].label == "Excess return over the fund"

    def test_a_factor_s_beta_is_labelled_as_before(self):
        found = columns_of(pd.DataFrame({"beta/market": [1.0]}))

        assert (found[0].label, found[0].unit) == ("Market beta", "ratio")

    def test_it_is_a_part_after_the_attribution(self):
        assert Profile.parts[2:4] == ("attribution", "return_risk")


class TestThroughTheServer:

    def test_its_rows_and_why_a_column_is_null(self,
                                               result):
        from beacon.server.analysis_schemas import InstrumentProfile
        from beacon.server.instrument import instrument_profile

        found = instrument_profile(result.analytics, "N0", SHORT["start"], SHORT["end"], None)
        part = found["parts"]["return_risk"]
        names = [column["name"] for column in part["columns"]]
        kept = result.analytics.instrument.profile("N0", **SHORT).return_risk

        assert part["dates"] == [date.strftime("%Y-%m-%d") for date in kept.index]
        assert part["data"][1][names.index("start")] == "2024-03-26"
        assert part["data"][1][names.index("volatility")] is None
        assert part["unavailable"] == kept.attrs["unavailable"]
        assert found["parts"]["daily_weights"]["unavailable"] is None
        assert part["columns"][0]["labels"] == PERIODS
        InstrumentProfile.model_validate(found)
