# tests/test_backtest_etf.py
"""BN-268: the ETF vehicle, its creations and its quotes on the exchange.

Runs reuse test_backtest_flows' two-name book of 1,000,000. Creation units
are 1,000 shares at a launch price of 100, so one unit is worth about
100,000. The quote model is tested directly on a MarketMaker, where each
driver can be isolated.
"""
import pandas as pd
import pytest

from beacon.backtest import (
    Act1940Limits,
    DatedFlows,
    EtfMarket,
    EtfVehicle,
    UcitsLimits,
    ucits_etf,
    us_etf,
)
from beacon.backtest.etf import MarketMaker
from test_backtest_flows import DAYS, QUIET, run

UNIT = 1_000
UNIT_VALUE = UNIT * 100.0
QUIET_MARKET = EtfMarket(noise_bps=0.0, return_sensitivity=0.0, flow_sensitivity=0.0)


def etf(**settings) -> EtfVehicle:
    return EtfVehicle(creation_unit=UNIT, **settings)


def maker(**market) -> MarketMaker:
    settings = {"noise_bps": 0.0, "return_sensitivity": 0.0,
                "flow_sensitivity": 0.0, **market}

    return MarketMaker(etf(ap_fee=0.0, market=EtfMarket(**settings)))


class TestCreationUnits:

    def test_demand_is_dealt_in_whole_units_and_the_rest_carries(self):
        """3.5 units today: 3 dealt. 0.6 more tomorrow: 1.1, so 1 dealt."""
        result = run(flows=DatedFlows({QUIET: 3.5 * UNIT_VALUE,
                                       DAYS[11]: 0.6 * UNIT_VALUE}),
                     vehicle=etf(in_kind=True))

        assert [record.creation_units for record in result.flows] == [3.0, 1.0]
        assert [record.date for record in result.flows] == [QUIET, DAYS[11]]

    def test_less_than_a_unit_waits(self):
        result = run(flows=DatedFlows({QUIET: 0.5 * UNIT_VALUE}),
                     vehicle=etf(in_kind=True))

        assert result.flows == []

    def test_redemptions_are_whole_units_too(self):
        result = run(flows=DatedFlows({QUIET: -2.7 * UNIT_VALUE}),
                     vehicle=etf(in_kind=True))

        assert result.flows[0].creation_units == -2.0


class TestInKind:

    def test_the_fund_does_not_pay_to_trade(self):
        result = run(flows=DatedFlows({QUIET: 3 * UNIT_VALUE,
                                       DAYS[30]: -2 * UNIT_VALUE}),
                     vehicle=etf(in_kind=True), transaction_cost_bps=20.0)
        flow_days = {QUIET, DAYS[30]}

        assert all(trade.transaction_cost == 0.0
                   for trade in result.portfolio.transactions
                   if trade.transaction_date in flow_days)

    def test_nav_per_unit_does_not_notice_a_creation(self):
        plain = run(vehicle=etf(in_kind=True))
        created = run(flows=DatedFlows({QUIET: 3 * UNIT_VALUE}),
                      vehicle=etf(in_kind=True))

        assert created.nav_per_unit[QUIET] == pytest.approx(
            plain.nav_per_unit[QUIET], rel=1e-9)


class TestCashCreations:

    def test_the_creation_fee_is_paid_into_the_fund(self):
        result = run(flows=DatedFlows({QUIET: 3.5 * UNIT_VALUE}),
                     vehicle=etf(), transaction_cost_bps=20.0)
        record = result.flows[0]

        assert record.creation_units == 3.0
        assert record.adjustment == pytest.approx(record.amount * 0.002, rel=1e-6)

    def test_a_set_fee_is_used_as_given(self):
        result = run(flows=DatedFlows({QUIET: 3.5 * UNIT_VALUE}),
                     vehicle=etf(creation_fee_bps=50.0))
        record = result.flows[0]

        assert record.adjustment == pytest.approx(record.amount * 0.005)


class TestTheQuotes:

    def test_each_day_is_quoted(self):
        result = run(vehicle=etf())
        market = result.market

        assert list(market.index) == list(DAYS)
        assert (market["bid"] < market["market_price"]).all()
        assert (market["market_price"] < market["ask"]).all()

    def test_the_price_never_leaves_the_band(self):
        result = run(vehicle=etf(market=EtfMarket(noise_bps=500.0)),
                     transaction_cost_bps=20.0)
        market = result.market

        assert (market["premium"] <= market["create_cost"] + 1e-12).all()
        assert (market["premium"] >= -market["redeem_cost"] - 1e-12).all()

    def test_the_quotes_repeat_with_their_seed(self):
        first = run(vehicle=etf(market=EtfMarket(seed=3))).market
        again = run(vehicle=etf(market=EtfMarket(seed=3))).market
        other = run(vehicle=etf(market=EtfMarket(seed=4))).market

        assert first.equals(again)
        assert not first["premium"].equals(other["premium"])

    def test_creations_push_the_price_to_a_premium(self):
        market = EtfMarket(noise_bps=0.0, return_sensitivity=0.0)
        result = run(flows=DatedFlows({QUIET: 3 * UNIT_VALUE}),
                     vehicle=etf(market=market, ap_fee=1e5))

        assert result.market["premium"][QUIET] > 0.0
        assert result.market["premium"][DAYS[9]] == 0.0

    def test_the_summary_reports_the_exchange(self):
        summary = run(vehicle=etf()).summary()

        assert {"market_return", "average_premium", "average_spread",
                "days_at_premium"} <= set(summary)

    def test_an_exchange_investor_pays_the_spread(self):
        result = run(vehicle=etf(market=QUIET_MARKET))
        nav_growth = result.nav_per_unit.iloc[-1] / result.nav_per_unit.iloc[0]

        assert 1.0 + result.market_return() < nav_growth

    def test_another_vehicle_has_no_quotes(self):
        result = run()

        assert result.market.empty
        assert result.market_return() is None


class TestTheMarketModel:

    def test_the_premium_decays_by_its_persistence(self):
        quoted = maker(flow_sensitivity=1.0, persistence=0.5)
        day = pd.Timestamp("2024-01-02")

        first = quoted.quote(day, 100.0, 100.0, 0.001, basket_cost=0.01)
        second = quoted.quote(day, 100.0, 100.0, 0.0, basket_cost=0.01)

        assert second.premium == pytest.approx(first.premium * 0.5)

    def test_a_sharp_fall_opens_a_discount(self):
        quoted = maker(return_sensitivity=0.05)
        quote = quoted.quote(pd.Timestamp("2024-01-02"), 96.0, 100.0, 0.0,
                             basket_cost=0.01)

        assert quote.premium == pytest.approx(-0.002)

    def test_the_spread_follows_the_basket_and_volatility(self):
        day = pd.Timestamp("2024-01-02")
        calm, wild = maker(), maker()

        for move in (1.0, 1.0, 1.0):
            calm_quote = calm.quote(day, 100.0 * move, 100.0, 0.0, basket_cost=0.001)

        for move in (1.03, 0.97, 1.03):
            wild_quote = wild.quote(day, 100.0 * move, 100.0, 0.0, basket_cost=0.001)

        assert calm_quote.spread == pytest.approx(0.0002 + 0.001)
        assert wild_quote.spread > calm_quote.spread

    def test_the_spread_is_never_below_a_tick(self):
        quote = maker(spread_floor_bps=0.0).quote(pd.Timestamp("2024-01-02"),
                                                  1.0, 1.0, 0.0, basket_cost=0.0)

        assert quote.spread == pytest.approx(0.01)

    def test_cash_and_in_kind_bands_differ_by_what_the_ap_pays(self):
        in_kind = MarketMaker(etf(in_kind=True, ap_fee=0.0, market=QUIET_MARKET))
        cash = MarketMaker(etf(creation_fee_bps=50.0, ap_fee=0.0,
                               market=QUIET_MARKET))
        day = pd.Timestamp("2024-01-02")

        assert in_kind.quote(day, 100.0, 100.0, 0.0, 0.002).create_cost == 0.002
        assert cash.quote(day, 100.0, 100.0, 0.0, 0.002).create_cost == 0.005


class TestPresets:

    def test_a_ucits_etf_creates_in_cash(self):
        vehicle = ucits_etf()

        assert not vehicle.in_kind
        assert [type(limit) for limit in vehicle.limits] == [UcitsLimits]

    def test_a_us_etf_creates_in_kind(self):
        vehicle = us_etf(creation_unit=25_000)

        assert vehicle.in_kind
        assert vehicle.creation_unit == 25_000
        assert [type(limit) for limit in vehicle.limits] == [Act1940Limits]


class TestTheSettingsAreChecked:

    @pytest.mark.parametrize("build", [
        lambda: EtfVehicle(creation_unit=0),
        lambda: EtfVehicle(ap_fee=-1.0),
        lambda: EtfMarket(persistence=1.0),
        lambda: EtfMarket(tick=-0.01),
        lambda: EtfMarket(volatility_days=1),
    ])
    def test_an_impossible_setting_is_refused(self,
                                              build):
        with pytest.raises(ValueError):
            build()
