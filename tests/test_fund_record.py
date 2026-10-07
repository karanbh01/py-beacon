# tests/test_fund_record.py
"""BN-268: Fund, the product record, and the deprecation of IndexFund and ETF.

A fund over test_backtest_flows' two names, as an index definition so
`fund.backtest` runs the whole Backtest path: GROW rises and FLAT does not,
equal weighted, monthly.
"""
from unittest.mock import patch

import pytest

from beacon.backtest import Backtest, PeriodicFlows, uk_oeic, us_etf
from beacon.data.base import MarketData, ReferenceData
from beacon.data.fetcher import DataFetcher
from beacon.fund import ETF, Fund, IndexFund, ShareClass
from beacon.index.calculation import IndexCalculator
from beacon.index.constructor import IndexDefinition
from beacon.index.methodology import EqualWeighted
from beacon.portfolio.base import Portfolio
from test_backtest_flows import DAYS, fetcher

END = str(DAYS[-1].date())
CAPITAL = 1_000_000.0


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


class TestBacktestingAClass:

    def run(self,
            share_class: str | None = None,
            **settings):
        return fund().backtest(end=END, initial_capital=CAPITAL,
                               share_class=share_class,
                               data_provider=with_shares(), cache=False,
                               **settings)

    def test_a_cheaper_class_returns_more(self):
        retail = self.run("Acc").summary()["total_return"]
        institutional = self.run("I Acc").summary()["total_return"]

        assert institutional > retail

    def test_a_class_uses_its_own_fee_and_leaves_the_vehicle_alone(self):
        record = fund()
        cheaper = record.vehicle_for(record.share_class("I Acc"))

        assert cheaper.management_fee_bps == 10
        assert record.vehicle.management_fee_bps == 50
        assert cheaper.name == "UK OEIC"

    def test_the_run_takes_the_flows_and_settings_given(self):
        result = self.run(flows=PeriodicFlows(fraction=0.01),
                          transaction_cost_bps=5.0)

        assert result.flows
        assert result.portfolio.transactions[0].transaction_cost > 0.0

    def test_a_distributing_class_pays_its_income_out(self):
        with patch("beacon.fund.fund.Backtest", wraps=Backtest) as built:
            self.run("Dist")

        assert built.call_args.kwargs["dividends"] == "distribute"

    def test_an_etf_fund_is_quoted(self):
        record = Fund(name="Listed", strategy=definition(),
                      vehicle=us_etf(creation_unit=100))
        result = record.backtest(end=END, initial_capital=CAPITAL,
                                 data_provider=with_shares(), cache=False)

        assert not result.market.empty


class TestTheOldFundsAreDeprecated:

    def build(self,
              fund_class,
              **extra):
        data = with_shares()

        return fund_class("OLD", target_index_definition=definition(),
                          index_agent=IndexCalculator(definition(), data),
                          portfolio=Portfolio("seed", initial_cash=CAPITAL),
                          data_provider=data, **extra)

    def test_an_index_fund_warns(self):
        with pytest.warns(DeprecationWarning, match="0.6.0"):
            self.build(IndexFund)

    def test_an_etf_warns_once(self):
        with pytest.warns(DeprecationWarning) as caught:
            self.build(ETF, etf_ticker="OLD")

        assert len([w for w in caught if w.category is DeprecationWarning]) == 1
        assert "ucits_etf" in str(caught[0].message)
