# tests/test_cash_management.py
"""BN-407: how a fund managed its cash, recorded by the engine and measured.

Runs reuse test_backtest_flows' two-name book of 1,000,000, with a cash buffer
of 5% held between 4% and 6% and trades costing 10 basis points. When GROW
falls 3% a day the cash's share rises through the band and is spent down;
when it rises 3% a day the share falls through it and cash is raised.
"""
import pandas as pd
import pytest

from beacon.analysis.data import RunData
from beacon.analysis.performance import cash_management
from beacon.analysis.performance.portfolio import PortfolioAnalytics
from beacon.backtest import Implementation
from beacon.portfolio.reasons import CASH_RAISE, CASH_SPEND_DOWN, REDEMPTION
from test_backtest_flows import fetcher, run

BAND = Implementation(cash_buffer=0.05, cash_band=(0.04, 0.06))


def banded(rate: float):
    return run(data=fetcher(rate), implementation=BAND, transaction_cost_bps=10.0)


class TestTheEngineRecordsEachEvent:

    @pytest.mark.parametrize(("rate", "kind", "reason"),
                             [(-0.03, "spend-down", CASH_SPEND_DOWN),
                              (0.03, "raise", CASH_RAISE)])
    def test_events_reconcile_with_the_cash_trades(self,
                                                   rate,
                                                   kind,
                                                   reason):
        result = banded(rate)
        events = result.cash_band_events
        made = [trade for trade in result.portfolio.transactions if trade.reason == reason]
        days = pd.DatetimeIndex([trade.transaction_date for trade in made])

        assert len(events) and set(events["kind"]) == {kind}
        assert set(days) == set(events.index)
        assert events["trades"].sum() == len(made)
        assert events["traded"].sum() == pytest.approx(
            sum(trade.quantity * trade.price for trade in made))
        assert events["cost"].sum() == pytest.approx(
            sum(trade.transaction_cost for trade in made))

    def test_each_brings_the_cash_back_from_outside_the_band(self):
        events = banded(-0.03).cash_band_events

        assert (events["cash_before"] > events["upper"]).all()
        assert events["cash_after"].to_numpy() == pytest.approx(0.05, abs=1e-4)
        assert (events["target"] == 0.05).all()

    def test_a_run_without_a_band_records_none(self):
        result = run(data=fetcher(-0.03))

        assert result.cash_band_events.empty


EVENTS = pd.DataFrame({"kind": ["raise", "spend-down", "raise"],
                       "cash_before": [0.03, 0.07, 0.035], "cash_after": [0.05, 0.052, 0.049],
                       "target": 0.05, "lower": 0.04, "upper": 0.06,
                       "traded": [2000.0, 1800.0, 1500.0], "cost": [2.0, 0.9, 1.5],
                       "trades": [4, 3, 2]},
                      index=pd.DatetimeIndex(["2024-01-15", "2024-01-29", "2024-03-12"],
                                             name="date"))


class TestTheFigures:

    def test_each_event_s_gap_and_cost(self):
        table = cash_management.band_events(EVENTS)

        assert table["gap_before"].tolist() == pytest.approx([0.02, 0.02, 0.015])
        assert table["gap_closed"].tolist() == pytest.approx([1.0, 0.9, 14 / 15])
        assert table["cost_bps"].tolist() == pytest.approx([10.0, 5.0, 10.0])

    def test_the_summary_by_kind(self):
        summary = cash_management.band_summary(EVENTS)

        assert summary.loc["raise", "events"] == 2
        assert summary.loc["raise", "mean_cost"] == pytest.approx(1.75)
        assert summary.loc["spend-down", "cost_bps"] == pytest.approx(5.0)
        assert summary.loc["all", "cost_bps"] == pytest.approx(4.4 / 5300 * 1e4)
        assert summary.loc["all", "mean_gap_closed"] == pytest.approx((1 + 0.9 + 14 / 15) / 3)

    def test_when_they_happened(self):
        months = cash_management.band_calendar(EVENTS)
        years = cash_management.band_calendar(EVENTS, "Y")

        assert months.loc["2024-01"].tolist() == pytest.approx([1, 1, 2.9])
        assert months.loc["2024-03", "raise"] == 1
        assert years.loc["2024", "raise"] == 2

    def test_a_run_without_events_has_an_empty_calendar(self):
        empty = EVENTS.iloc[:0]

        assert cash_management.band_calendar(empty).empty
        assert cash_management.band_summary(empty).loc["all", "events"] == 0


class TestThroughTheAnalytics:

    def test_a_stored_run_keeps_its_events(self,
                                           tmp_path):
        data = RunData.from_result(banded(0.03))
        data.write(tmp_path / "run")
        portfolio = PortfolioAnalytics(RunData.from_record(tmp_path / "run"))

        pd.testing.assert_frame_equal(portfolio.cash_band_events(),
                                      PortfolioAnalytics(data).cash_band_events(),
                                      check_freq=False)
        assert portfolio.cash_band_summary().loc["raise", "cost_bps"] == pytest.approx(10.0)

    def test_every_trade_made_to_raise_or_spend_cash(self):
        from beacon.backtest.flows import DatedFlows
        from test_backtest_flows import DAYS

        result = run(data=fetcher(0.03), implementation=BAND,
                     flows=DatedFlows({DAYS[30]: -2e5}), transaction_cost_bps=10.0)
        table = PortfolioAnalytics(RunData.from_result(result)).cash_trading()

        assert (CASH_RAISE, CASH_RAISE) in table.index
        assert (CASH_RAISE, REDEMPTION) in table.index
        assert table.loc[(CASH_RAISE, CASH_RAISE), "trades"] == sum(
            trade.reason == CASH_RAISE for trade in result.portfolio.transactions)

    def test_a_run_without_a_band_has_no_events(self):
        portfolio = PortfolioAnalytics(RunData.from_result(run()))

        assert portfolio.cash_band_events().empty
        assert portfolio.cash_band_summary().loc["all", "events"] == 0
