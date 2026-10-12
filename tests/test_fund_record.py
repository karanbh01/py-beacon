# tests/test_fund_record.py
"""BN-268: Fund, the product record.

A fund over test_backtest_flows' two names, as an index definition so
`fund.backtest` runs the whole Backtest path: GROW rises and FLAT does not,
equal weighted, monthly.
"""
from unittest.mock import patch

import pytest

from beacon.backtest import Backtest, PeriodicFlows, uk_oeic, us_etf
from beacon.data.base import MarketData, ReferenceData
from beacon.data.fetcher import DataFetcher
from beacon.fund import Fund, ShareClass
from beacon.index.constructor import IndexDefinition
from beacon.index.methodology import EqualWeighted
from test_backtest_flows import DAYS, fetcher

END = str(DAYS[-1].date())
CAPITAL = 1_000_000.0
CLASS_CAPITAL = {"Acc": 600_000.0, "I Acc": 300_000.0, "Dist": 100_000.0}


def definition() -> IndexDefinition:
    return IndexDefinition(index_id="TWO", index_name="Two names",
                           base_date=str(DAYS[0].date()), base_value=100.0,
                           currency="USD", eligibility_rules=[],
                           weighting_scheme=EqualWeighted(),
                           rebalancing_frequency="MONTHLY", calendar="XNYS",
                           universe_identifiers=["GROW", "FLAT"])


def with_shares():
    """test_backtest_flows' data, with the share counts an index needs."""
    data = fetcher()
    market = data.fetch_market_data(["GROW", "FLAT"]).reset_index()
    market["SHARES_OUTSTANDING"] = 1e6
    reference = data.fetch_reference_data(["GROW", "FLAT"]).reset_index()
    reference["DATE_FROM"] = "2020-01-01"

    return DataFetcher(MarketData.from_dataframe(market),
                       ReferenceData.from_dataframe(reference))


def fund(**settings) -> Fund:
    return Fund(name="Sample", strategy=definition(),
                vehicle=uk_oeic(management_fee_bps=50),
                share_classes=[ShareClass("Acc"),
                               ShareClass("I Acc", management_fee_bps=10),
                               ShareClass("Dist", distribution="distributing")],
                **settings)


class TestTheRecord:

    def test_it_holds_what_a_prospectus_does(self):
        record = fund(currency="usd", documents={"kiid": "kiid.pdf"})

        assert record.currency == "USD"
        assert record.documents == {"kiid": "kiid.pdf"}
        assert [c.name for c in record.share_classes] == ["Acc", "I Acc", "Dist"]

    def test_with_no_classes_it_has_one(self):
        record = Fund(name="Plain", strategy=definition(), vehicle=uk_oeic())

        assert record.share_class().distribution == "accumulating"

    def test_a_class_by_name(self):
        assert fund().share_class("I Acc").management_fee_bps == 10

        with pytest.raises(KeyError, match="Acc, I Acc, Dist"):
            fund().share_class("Z")

    @pytest.mark.parametrize("build", [
        lambda: Fund(name="", strategy=definition(), vehicle=uk_oeic()),
        lambda: Fund(name="Twins", strategy=definition(), vehicle=uk_oeic(),
                     share_classes=[ShareClass("A"), ShareClass("A")]),
        lambda: ShareClass("A", distribution="sometimes"),
        lambda: ShareClass("A", management_fee_bps=-1),
    ])
    def test_an_impossible_record_is_refused(self,
                                             build):
        with pytest.raises(ValueError):
            build()


class TestBacktestingTheFund:
    """BN-279: every class together, in one book."""

    def run(self,
            **settings):
        return fund().backtest(end=END, initial_capital=CLASS_CAPITAL,
                               data_provider=with_shares(), cache=False, **settings)

    def test_every_class_runs_in_one_book(self):
        result = self.run()

        assert list(result.share_classes) == ["Acc", "I Acc", "Dist"]
        assert result.portfolio.initial_capital == sum(CLASS_CAPITAL.values())

    def test_a_cheaper_class_returns_more(self):
        classes = self.run().share_classes

        assert classes["I Acc"].total_return() > classes["Acc"].total_return()
        assert classes["I Acc"].management_fee_bps == 10
        assert classes["Acc"].management_fee_bps == 50

    def test_the_run_takes_the_flows_and_settings_given(self):
        result = self.run(flows={"Dist": PeriodicFlows(fraction=0.01)},
                          transaction_cost_bps=5.0)

        assert result.share_classes["Dist"].flows
        assert not result.share_classes["Acc"].flows
        assert result.portfolio.transactions[0].transaction_cost > 0.0

    def test_the_fund_passes_its_classes_to_the_backtest(self):
        with patch("beacon.fund.fund.Backtest", wraps=Backtest) as built:
            self.run()

        assert [c.name for c in built.call_args.kwargs["share_classes"]] == [
            "Acc", "I Acc", "Dist"]
        assert built.call_args.kwargs["vehicle"].management_fee_bps == 50

    def test_a_fund_of_one_class_takes_one_capital(self):
        record = Fund(name="Plain", strategy=definition(), vehicle=uk_oeic())
        result = record.backtest(end=END, initial_capital=CAPITAL,
                                 data_provider=with_shares(), cache=False)

        assert list(result.share_classes) == ["Default"]

    def test_an_etf_fund_is_quoted(self):
        record = Fund(name="Listed", strategy=definition(),
                      vehicle=us_etf(creation_unit=100))
        result = record.backtest(end=END, initial_capital=CAPITAL,
                                 data_provider=with_shares(), cache=False)

        assert not result.market.empty
