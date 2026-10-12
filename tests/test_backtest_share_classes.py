# tests/test_backtest_share_classes.py
"""BN-279: share classes sharing one pool of assets.

Three names through 2024 on one market factor, each paying a quarterly
dividend inside the run, with the ex-date drop in its price, so a
distributing class has income to pay out. The classes' net assets must always
add up to the fund's, each class's own fee, flows and distributions must
reach it alone, and one class must reproduce a run with no classes exactly.
"""
import numpy as np
import pandas as pd
import pytest

from beacon.backtest import (
    Backtest,
    DatedFlows,
    PeriodicFlows,
    ShareClass,
    Vehicle,
    uk_oeic,
    us_etf,
)
from beacon.data.base import MarketData, ReferenceData
from beacon.data.corporate_actions import CorporateActions
from beacon.data.fetcher import DataFetcher
from beacon.index.constructor import IndexDefinition
from beacon.index.methodology import EqualWeighted
from beacon.index.schedule import sessions

DAYS = sessions(pd.Timestamp("2024-01-02"), pd.Timestamp("2024-12-31"), "XNYS")
NAMES = ["A", "B", "C"]
EX_DATES = [pd.Timestamp(day) for day in ("2024-03-15", "2024-06-14", "2024-09-13",
                                          "2024-12-13")]
DIVIDEND = 0.6
START, END = "2024-01-02", "2024-12-31"
FLOW_DAY = pd.Timestamp("2024-05-01")


def fetcher() -> DataFetcher:
    random = np.random.default_rng(17)
    market = random.normal(0.0004, 0.009, len(DAYS))
    rows, actions = [], []

    for name in NAMES:
        price = 50.0
        for day, move in zip(DAYS, market + random.normal(0.0, 0.004, len(DAYS)),
                             strict=True):
            price *= 1.0 + move
            if day in EX_DATES:
                price -= DIVIDEND
            rows.append({"IDENTIFIER": name, "DATE": day, "CLOSE": price,
                         "SHARES_OUTSTANDING": 1e8})

        actions += [{"IDENTIFIER": name, "EX_DATE": day, "TYPE": "DIVIDEND",
                     "VALUE": DIVIDEND} for day in EX_DATES]

    reference = pd.DataFrame([{"IDENTIFIER": name, "NAME": name, "DATE_FROM": "2020-01-01",
                               "CURRENCY": "USD", "EXCHANGE": "XNYS"} for name in NAMES])

    return DataFetcher(MarketData.from_dataframe(pd.DataFrame(rows)),
                       ReferenceData.from_dataframe(reference),
                       CorporateActions.from_dataframe(pd.DataFrame(actions)))


DATA = fetcher()
DEFINITION = IndexDefinition(index_id="EW", index_name="Equal", base_date=START,
                             base_value=100.0, currency="USD", eligibility_rules=[],
                             weighting_scheme=EqualWeighted(),
                             rebalancing_frequency="QUARTERLY", calendar="XNYS",
                             universe_identifiers=NAMES)
VEHICLE = Vehicle(management_fee_bps=50)


def run(**settings):
    settings.setdefault("vehicle", VEHICLE)
    return Backtest(data_provider=DATA, cache=False, **settings).run(
        DEFINITION, start=START, end=END)


def classes(*share_classes, capital=None, **settings):
    return run(share_classes=list(share_classes),
               initial_capital=capital or {share_class.name: 1e6
                                           for share_class in share_classes},
               **settings)


def growth(series: pd.Series) -> float:
    return float(series.iloc[-1] / series.iloc[0])


@pytest.fixture(scope="module")
def pooled():
    return classes(ShareClass("Retail"), ShareClass("Inst", management_fee_bps=10),
                   ShareClass("Dist", distribution="distributing"),
                   capital={"Retail": 5e6, "Inst": 3e6, "Dist": 2e6})


class TestOneClass:

    def test_it_is_the_run_without_classes(self):
        plain = run(initial_capital=1e6)
        one = classes(ShareClass("A"))

        assert one.trading_nav.equals(plain.trading_nav)

    def test_its_nav_per_unit_is_the_funds(self):
        one = classes(ShareClass("A"))

        assert one.share_classes["A"].nav_per_unit.iloc[1:].to_numpy() == pytest.approx(
            one.nav_per_unit.to_numpy(), rel=1e-12)


class TestThePool:

    def test_the_classes_add_up_to_the_fund(self, pooled):
        books = pooled.share_classes.values()
        total = sum(book.net_assets for book in books)
        fund = (pooled.portfolio.nav
                - pooled.fees_payable.reindex(pooled.portfolio.nav.index).ffill().fillna(0.0))

        assert total.to_numpy() == pytest.approx(fund.reindex(total.index).to_numpy(),
                                                  rel=1e-9)

    def test_the_book_trades_every_class_money(self, pooled):
        first = pooled.portfolio.transactions[0].transaction_date
        bought = sum(t.quantity * t.price for t in pooled.portfolio.transactions
                     if t.transaction_date == first)

        assert bought == pytest.approx(1e7, rel=0.01)

    def test_a_class_fee_reaches_it_alone(self, pooled):
        retail = pooled.share_classes["Retail"]
        inst = pooled.share_classes["Inst"]
        days = (DAYS[-1] - DAYS[0]).days

        # The same pool, less 40bp a year more for retail.
        gap = growth(inst.nav_per_unit) / growth(retail.nav_per_unit) - 1.0
        assert gap == pytest.approx(0.004 * days / 365, rel=0.05)
        assert inst.management_fee_bps == 10
        assert retail.management_fee_bps == 50

    def test_the_fee_is_each_class_on_its_own_assets(self, pooled):
        charged = sum(book.fees.sum() for book in pooled.share_classes.values())
        paid = -sum(flow.amount for flow in pooled.portfolio.cash_flows if flow.kind == "FEE")
        owed = pooled.fees_payable.iloc[-1]

        assert charged == pytest.approx(paid + owed, rel=1e-9)


class TestDistributions:

    def test_a_distributing_class_pays_its_share_out(self, pooled):
        dist = pooled.share_classes["Dist"]
        paid = -sum(flow.amount for flow in pooled.portfolio.cash_flows
                    if flow.kind == "DISTRIBUTION")

        assert dist.distributions.sum() > 0.0
        assert dist.distributions.sum() == pytest.approx(paid, rel=1e-9)
        assert pooled.share_classes["Retail"].distributions.empty

    def test_paying_out_costs_it_no_return(self, pooled):
        dist = pooled.share_classes["Dist"].total_return()
        retail = pooled.share_classes["Retail"].total_return()

        # Same fee: the distributions, added back, give the same total return.
        assert dist == pytest.approx(retail, abs=5e-4)

    def test_but_it_shrinks_against_the_others(self, pooled):
        dist = pooled.share_classes["Dist"].nav_per_unit
        retail = pooled.share_classes["Retail"].nav_per_unit

        assert growth(dist) < growth(retail)


class TestFlows:

    def test_a_flow_creates_units_in_its_own_class(self):
        result = classes(ShareClass("A"), ShareClass("B"),
                         flows={"B": DatedFlows({FLOW_DAY: 5e5})})
        a, b = result.share_classes["A"], result.share_classes["B"]

        assert a.units.nunique() == 1
        assert b.units.iloc[-1] > b.units.iloc[0]
        assert [record.date for record in b.flows] == [FLOW_DAY]
        assert a.flows == []

    def test_a_flow_is_sized_against_its_class(self):
        result = classes(ShareClass("A"), ShareClass("B"),
                         capital={"A": 4e6, "B": 1e6},
                         flows={"B": PeriodicFlows(fraction=0.01)})
        b = result.share_classes["B"]
        first = b.flows[0]

        assert first.amount == pytest.approx(0.01 * b.net_assets.asof(
            first.date - pd.Timedelta(days=1)), rel=0.05)

    def test_a_flow_moves_no_value_between_classes(self):
        # Dealt at its own NAV per unit, B's subscription buys B units worth
        # exactly what it paid: A and B, on the same fee, earn the same
        # return per unit every day. (The flow still changes the book, by
        # paying the fee owed from its cash, as a real fund's would, and so
        # every class's return a little.)
        flowed = classes(ShareClass("A"), ShareClass("B"),
                         flows={"B": DatedFlows({FLOW_DAY: 5e5})})
        a = flowed.share_classes["A"].get_returns()
        b = flowed.share_classes["B"].get_returns()

        assert b.to_numpy() == pytest.approx(a.to_numpy(), abs=1e-12)

    def test_a_swing_moves_the_dealing_price(self):
        result = classes(ShareClass("A"), ShareClass("B"),
                         flows={"B": DatedFlows({FLOW_DAY: 5e5})},
                         vehicle=uk_oeic(), transaction_cost_bps=20.0)
        record = result.share_classes["B"].flows[0]

        assert record.dealing_price > record.nav_per_unit
        assert record.adjustment > 0.0


class TestAnExchangeTradedFund:

    def test_creation_units_are_shared_by_who_asked(self):
        # Demand below a creation unit is carried until it fills one; each
        # class is credited in proportion to what it asked for.
        result = classes(ShareClass("A"), ShareClass("B"),
                         flows={"A": PeriodicFlows(fraction=0.02),
                                "B": PeriodicFlows(fraction=0.01)},
                         vehicle=us_etf(creation_unit=1_000))
        a = sum(record.amount for record in result.share_classes["A"].flows)
        b = sum(record.amount for record in result.share_classes["B"].flows)
        dealt = sum(record.amount for record in result.flows)

        assert a + b == pytest.approx(dealt, rel=1e-9)
        assert a / b == pytest.approx(2.0, rel=0.1)

    def test_the_classes_still_add_up(self):
        result = classes(ShareClass("A"), ShareClass("B"),
                         flows={"A": PeriodicFlows(fraction=0.02)},
                         vehicle=us_etf(creation_unit=1_000))
        total = sum(book.net_assets for book in result.share_classes.values())

        assert total.iloc[1:].to_numpy() == pytest.approx(
            result.trading_nav.reindex(total.index[1:]).to_numpy(), rel=1e-9)


class TestRefused:

    def test_capital_by_class_needs_classes(self):
        with pytest.raises(ValueError, match="share_classes"):
            run(initial_capital={"A": 1e6})

    def test_several_classes_need_capital_by_class(self):
        with pytest.raises(ValueError, match="by class name"):
            run(share_classes=[ShareClass("A"), ShareClass("B")], initial_capital=1e6)

    def test_an_unknown_class_is_named(self):
        with pytest.raises(ValueError, match="Z"):
            classes(ShareClass("A"), capital={"A": 1e6, "Z": 1.0})

    def test_classes_decide_their_own_distributions(self):
        with pytest.raises(ValueError, match="distribution"):
            classes(ShareClass("A"), dividends="distribute")

    def test_names_must_differ(self):
        with pytest.raises(ValueError, match="differ"):
            run(share_classes=[ShareClass("A"), ShareClass("A")],
                initial_capital={"A": 1e6})
