# tests/test_backtest_hedged_classes.py
"""BN-307: share classes in another currency, hedged and not.

Three names priced in dollars through 2024, and a dollar-to-euro rate that
drifts up about 10% over the year. An unhedged euro class must earn the
book's return times the currency's move, exactly; a hedged one must take the
move out, earning the book's return plus the interest rate differential
instead. With rate series in the data the euro's 3% against the dollar's 5%
costs the hedged class about 2% a year; without them both currencies take
the run's cash rate and the hedge carries nothing.
"""
import numpy as np
import pandas as pd
import pytest

from beacon.backtest import Backtest, DatedFlows, ShareClass, Vehicle
from beacon.data import pair_reference, short_rate_reference
from beacon.data.base import MarketData, ReferenceData
from beacon.data.fetcher import DataFetcher
from beacon.exceptions import CalculationError
from beacon.index.constructor import IndexDefinition
from beacon.index.methodology import EqualWeighted
from beacon.index.schedule import sessions

DAYS = sessions(pd.Timestamp("2024-01-02"), pd.Timestamp("2024-12-31"), "XNYS")
NAMES = ["A", "B", "C"]
START, END = "2024-01-02", "2024-12-31"
USD_RATE, EUR_RATE = 0.05, 0.03


def fetcher(rates: bool = True, fx: bool = True) -> DataFetcher:
    random = np.random.default_rng(23)
    market = random.normal(0.0004, 0.009, len(DAYS))
    rows = []

    for name in NAMES:
        prices = 50.0 * np.cumprod(1.0 + market + random.normal(0.0, 0.004, len(DAYS)))
        rows += [{"IDENTIFIER": name, "DATE": day, "CLOSE": price, "SHARES_OUTSTANDING": 1e8}
                 for day, price in zip(DAYS, prices, strict=True)]

    references = []

    if fx:
        drift = 0.92 * np.cumprod(1.0 + 0.0004 + random.normal(0.0, 0.003, len(DAYS)))
        rows += [{"IDENTIFIER": "USDEUR", "DATE": day, "CLOSE": rate}
                 for day, rate in zip(DAYS, drift, strict=True)]
        references.append(pair_reference("USDEUR"))

    if rates:
        rows += [{"IDENTIFIER": f"{currency} 1M", "DATE": day, "CLOSE": rate}
                 for currency, rate in (("USD", USD_RATE), ("EUR", EUR_RATE))
                 for day in DAYS]
        references += [short_rate_reference(f"{currency} 1M", currency, tenor="1M")
                       for currency in ("USD", "EUR")]

    reference = pd.DataFrame([*({"IDENTIFIER": name, "NAME": name, "DATE_FROM": "2020-01-01",
                                 "CURRENCY": "USD", "EXCHANGE": "XNYS"} for name in NAMES),
                              *references])

    return DataFetcher(MarketData.from_dataframe(pd.DataFrame(rows)),
                       ReferenceData.from_dataframe(reference))


DATA = fetcher()
NO_RATES = fetcher(rates=False)
DEFINITION = IndexDefinition(index_id="EW", index_name="Equal", base_date=START,
                             base_value=100.0, currency="USD", eligibility_rules=[],
                             weighting_scheme=EqualWeighted(),
                             rebalancing_frequency="QUARTERLY", calendar="XNYS",
                             universe_identifiers=NAMES)


def run(*share_classes, data=DATA, capital=None, **settings):
    return Backtest(initial_capital=capital or {c.name: 1e6 for c in share_classes},
                    share_classes=list(share_classes), data_provider=data, cache=False,
                    vehicle=Vehicle(), **settings).run(DEFINITION, start=START, end=END)


def growth(series: pd.Series) -> float:
    return float(series.iloc[-1] / series.iloc[0])


def fx(data: DataFetcher, day) -> float:
    return float(data.fx_rate_on("USD", "EUR", day))


@pytest.fixture(scope="module")
def three():
    return run(ShareClass("USD"), ShareClass("EUR", currency="EUR"),
               ShareClass("EUR H", currency="EUR", hedged=True), data=NO_RATES)


class TestAClassInAnotherCurrency:

    def test_it_earns_the_book_return_times_the_currency(self, three):
        book = three.share_classes["USD"].nav_per_unit
        euro = three.share_classes["EUR"].nav_per_unit
        moved = fx(NO_RATES, DAYS[-1]) / fx(NO_RATES, DAYS[0])

        assert growth(euro) == pytest.approx(growth(book) * moved, rel=1e-9)
        assert three.share_classes["EUR"].currency == "EUR"

    def test_its_capital_and_flows_are_in_its_currency(self):
        result = run(ShareClass("EUR", currency="EUR"), data=NO_RATES,
                     flows={"EUR": DatedFlows({DAYS[100]: 100_000.0})})
        euro = result.share_classes["EUR"]

        assert euro.net_assets.iloc[0] == pytest.approx(1e6)
        assert euro.book_net_assets.iloc[0] == pytest.approx(1e6 / fx(NO_RATES, DAYS[0]))
        assert euro.flows[0].amount == pytest.approx(100_000.0)

    def test_the_classes_still_add_up_in_the_book_currency(self, three):
        total = sum(book.book_net_assets for book in three.share_classes.values())

        assert total.iloc[1:].to_numpy() == pytest.approx(
            three.trading_nav.reindex(total.index[1:]).to_numpy(), rel=1e-9)


class TestAHedgedClass:

    def test_it_takes_the_currency_move_out(self, three):
        book = growth(three.share_classes["USD"].nav_per_unit)
        hedged = growth(three.share_classes["EUR H"].nav_per_unit)
        unhedged = growth(three.share_classes["EUR"].nav_per_unit)

        assert abs(unhedged / book - 1.0) > 0.05
        assert abs(hedged / book - 1.0) < 0.005

    def test_it_rolls_at_each_month_start(self, three):
        hedges = three.share_classes["EUR H"].hedges
        rolls = [trade.date for trade in hedges if trade.reason == "roll"]

        assert hedges[0].reason == "open"
        assert hedges[0].date == DAYS[0]
        assert [day.month for day in rolls] == list(range(2, 13))
        assert all(day == DAYS[DAYS.month == day.month][0] for day in rolls)

    def test_the_differential_is_its_carry(self):
        carried = run(ShareClass("USD"), ShareClass("EUR H", currency="EUR", hedged=True))
        free = run(ShareClass("USD"), ShareClass("EUR H", currency="EUR", hedged=True),
                   data=NO_RATES)
        days = (DAYS[-1] - DAYS[0]).days

        # The euro earns 3% against the dollar's 5%: the hedged class gives
        # up about the 2% a year the forward points cost it.
        cost = (growth(free.share_classes["EUR H"].nav_per_unit)
                / growth(carried.share_classes["EUR H"].nav_per_unit) - 1.0)
        assert cost == pytest.approx((USD_RATE - EUR_RATE) * days / 360, rel=0.15)

    def test_its_hedge_is_its_own(self, three):
        # The dollar class's per-unit return is the same as in a fund
        # without the hedged class, to within the trades its settlements
        # cause: a loss is paid by selling across the pool and a gain is
        # invested across it, which the classes share. A gain left in cash
        # until the next rebalance cost the dollar class 1.5e-4 in this year.
        alone = run(ShareClass("USD"), data=NO_RATES)

        assert growth(three.share_classes["USD"].nav_per_unit) == pytest.approx(
            growth(alone.share_classes["USD"].nav_per_unit), rel=1e-5)

    def test_a_settlement_is_a_hedge_cash_flow(self, three):
        settled = [flow for flow in three.portfolio.cash_flows if flow.kind == "HEDGE"]
        trades = [trade for trade in three.share_classes["EUR H"].hedges[1:]
                  if trade.settled != 0.0]

        assert len(settled) == len(trades)

    def test_a_tight_band_resizes_it_between_rolls(self):
        tight = run(ShareClass("EUR H", currency="EUR", hedged=True, hedge_band=0.001),
                    data=NO_RATES)
        loose = run(ShareClass("EUR H", currency="EUR", hedged=True, hedge_band=None),
                    data=NO_RATES)

        assert any(trade.reason == "band" for trade in tight.share_classes["EUR H"].hedges)
        assert not any(trade.reason == "band"
                       for trade in loose.share_classes["EUR H"].hedges)


class TestTheCostOfHedging:
    """BN-408: the carry each forward locks in, and a cost on each trade."""

    def test_each_trade_records_the_carry_it_locks_in(self):
        hedges = run(ShareClass("EUR H", currency="EUR", hedged=True)).share_classes[
            "EUR H"].hedges

        for trade in hedges:
            assert trade.expiry > trade.date
            assert trade.carry == pytest.approx(
                trade.notional * (trade.rate - trade.spot) / trade.spot)

        # The euro's rate is below the dollar's, so every forward costs carry.
        assert all(trade.carry < 0.0 for trade in hedges)

    def test_a_cost_on_the_notional_each_trade_deals(self):
        hedged = ShareClass("EUR H", currency="EUR", hedged=True, hedge_cost_bps=2.0)
        result = run(ShareClass("USD"), hedged, data=NO_RATES)
        hedges = result.share_classes["EUR H"].hedges
        dealt = [trade.notional if trade.reason != "band" else
                 abs(trade.notional - before.notional)
                 for before, trade in zip([hedges[0], *hedges[:-1]], hedges, strict=True)]

        assert [trade.cost for trade in hedges] == pytest.approx(
            [notional * 2.0 / 10_000 for notional in dealt])

    def test_the_cost_is_the_class_s_alone(self):
        free = run(ShareClass("USD"), ShareClass("EUR H", currency="EUR", hedged=True),
                   data=NO_RATES)
        costly = run(ShareClass("USD"), ShareClass("EUR H", currency="EUR", hedged=True,
                                                   hedge_cost_bps=5.0), data=NO_RATES)
        paid = sum(trade.cost for trade in costly.share_classes["EUR H"].hedges)

        assert paid > 0.0
        assert growth(costly.share_classes["USD"].nav_per_unit) == pytest.approx(
            growth(free.share_classes["USD"].nav_per_unit), rel=1e-6)
        assert growth(costly.share_classes["EUR H"].nav_per_unit) < growth(
            free.share_classes["EUR H"].nav_per_unit)


class TestRefused:

    def test_a_hedge_needs_a_currency(self):
        with pytest.raises(ValueError, match="currency"):
            ShareClass("H", hedged=True)

    def test_a_currency_is_a_code(self):
        with pytest.raises(ValueError, match="three-letter"):
            ShareClass("E", currency="EURO")

    @pytest.mark.parametrize("settings", [{"hedge_ratio": 0.0}, {"hedge_band": -0.1},
                                          {"hedge_cost_bps": -1.0}])
    def test_hedge_settings_are_checked(self, settings):
        with pytest.raises(ValueError, match="hedge"):
            ShareClass("H", currency="EUR", hedged=True, **settings)

    def test_a_hedge_into_the_book_currency_is_refused(self):
        with pytest.raises(ValueError, match="nothing to hedge"):
            run(ShareClass("H", currency="USD", hedged=True))

    def test_a_class_currency_needs_a_rate(self):
        with pytest.raises(CalculationError, match="USD to EUR"):
            run(ShareClass("EUR", currency="EUR"), data=fetcher(fx=False))


class TestShortRates:

    def test_the_hedge_warns_once_without_them(self,
                                               caplog):
        with caplog.at_level("WARNING", logger="beacon.backtest.hedging"):
            run(ShareClass("EUR H", currency="EUR", hedged=True), data=NO_RATES)

        assert caplog.text.count("No EUR short rate in the data") == 1

    def test_rate_series_and_pairs_are_not_instruments(self):
        assert set(DATA.instrument_identifiers) == set(NAMES)
