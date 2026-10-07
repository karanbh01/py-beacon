# tests/test_backtest_dealing.py
"""BN-268: pricing methods, diversification limits and the open-ended presets.

The runs reuse test_backtest_flows' two-name book. Trading costs 20 basis
points, so a large flow dilutes the fund unless its pricing makes the
dealing investors pay.
"""
import pytest

from beacon.backtest import (
    Act1940Limits,
    DatedFlows,
    DilutionLevy,
    DualPricing,
    ScreenContext,
    SinglePricing,
    SwingPricing,
    UcitsLimits,
    Vehicle,
    irish_icav,
    luxembourg_sicav,
    preset,
    uk_oeic,
    us_mutual_fund,
)
from beacon.backtest.implementation import Implementation, plan
from test_backtest_flows import DAYS, QUIET, fetcher, run

COST_BPS = 20.0
NAV = 2.0
FUND = 1_000.0


def protected(pricing,
              flow: float = 500_000.0):
    """NAV per unit at the end of a run with one large flow."""
    return run(flows=DatedFlows({QUIET: flow}),
               vehicle=Vehicle(pricing=pricing),
               transaction_cost_bps=COST_BPS)


class TestTheDealingArithmetic:

    def test_single_pricing_deals_at_nav(self):
        deal = SinglePricing().deal(100.0, NAV, FUND, 0.01)

        assert (deal.units, deal.cash, deal.price, deal.adjustment) == (
            50.0, 100.0, NAV, 0.0)

    def test_a_swing_up_gives_buyers_fewer_units(self):
        deal = SwingPricing(factor_bps=100).deal(101.0, NAV, FUND, 0.0)

        assert deal.price == pytest.approx(NAV * 1.01)
        assert deal.units == pytest.approx(50.0)
        assert deal.adjustment == pytest.approx(1.0)

    def test_a_swing_down_pays_sellers_less(self):
        deal = SwingPricing(factor_bps=100).deal(-100.0, NAV, FUND, 0.0)

        assert deal.units == pytest.approx(-50.0)
        assert deal.cash == pytest.approx(-99.0)
        assert deal.adjustment == pytest.approx(1.0)

    def test_a_partial_swing_ignores_a_small_flow(self):
        pricing = SwingPricing(factor_bps=100, threshold=0.2)

        assert pricing.deal(100.0, NAV, FUND, 0.0).adjustment == 0.0
        assert pricing.deal(300.0, NAV, FUND, 0.0).adjustment > 0.0

    def test_dual_pricing_charges_each_side_its_own_spread(self):
        pricing = DualPricing(offer_bps=50, bid_bps=30)

        assert pricing.deal(100.0, NAV, FUND, 0.0).price == pytest.approx(NAV * 1.005)
        assert pricing.deal(-100.0, NAV, FUND, 0.0).price == pytest.approx(NAV * 0.997)

    def test_a_levy_is_paid_into_the_fund_at_nav(self):
        deal = DilutionLevy(rate_bps=100).deal(100.0, NAV, FUND, 0.0)

        assert (deal.price, deal.units, deal.adjustment) == pytest.approx(
            (NAV, 49.5, 1.0))

    def test_a_redemption_levy_spares_subscriptions(self):
        levy = DilutionLevy(rate_bps=100, on="redemptions")

        assert levy.deal(100.0, NAV, FUND, 0.0).adjustment == 0.0
        assert levy.deal(-100.0, NAV, FUND, 0.0).cash == pytest.approx(-99.0)

    def test_an_unset_rate_is_the_estimate(self):
        deal = SwingPricing().deal(-100.0, NAV, FUND, 0.004)

        assert deal.adjustment == pytest.approx(0.4)


class TestTheFundIsProtected:

    @pytest.mark.parametrize("pricing", [SwingPricing(), DualPricing(),
                                         DilutionLevy()])
    def test_the_investors_who_stay_keep_more(self,
                                               pricing):
        diluted = protected(SinglePricing())
        kept = protected(pricing)

        assert kept.nav_per_unit.iloc[-1] > diluted.nav_per_unit.iloc[-1]

    def test_without_impact_the_estimate_is_the_fixed_cost(self):
        result = protected(SwingPricing())
        record = result.flows[0]

        assert record.dealing_price == pytest.approx(
            record.nav_per_unit * (1 + COST_BPS / 10_000))
        assert record.adjustment > 0.0

    def test_single_pricing_records_no_adjustment(self):
        record = protected(SinglePricing()).flows[0]

        assert record.adjustment == 0.0
        assert record.dealing_price == record.nav_per_unit

    def test_a_redemption_pays_its_own_way_too(self):
        diluted = protected(SinglePricing(), flow=-500_000.0)
        kept = protected(SwingPricing(), flow=-500_000.0)

        assert kept.flows[0].amount > -500_000.0
        assert kept.nav_per_unit.iloc[-1] > diluted.nav_per_unit.iloc[-1]


def limited(limit,
            target: dict[str, float],
            rule: str = "pro_rata") -> dict[str, float]:
    """*target* through one diversification limit, with no data needed."""
    step = plan(Implementation(redistribution=rule), target, DAYS[0],
                held=set(), stale=set(),
                context=ScreenContext(fetcher(), "USD"),
                book_value=1e6, limits=[limit])

    return step.weights


class TestDiversificationLimits:

    def test_an_index_tracking_ucits_fund_holds_35_and_20(self):
        weights = limited(UcitsLimits(),
                          {"A": 0.5, "B": 0.25, **{f"N{i}": 0.025 for i in range(10)}})

        assert weights["A"] == pytest.approx(0.35)
        assert weights["B"] == pytest.approx(0.20)
        assert sum(weights.values()) == pytest.approx(1.0)

    def test_the_standard_ucits_rule_is_5_10_40(self):
        target = {"A": 0.15, "B": 0.12, "C": 0.10, "D": 0.09, "E": 0.08,
                  **{f"N{i}": 0.46 / 23 for i in range(23)}}
        weights = limited(UcitsLimits(index_tracking=False), target)
        large = [weight for weight in weights.values() if weight > 0.05 + 1e-12]

        assert max(weights.values()) <= 0.10 + 1e-12
        assert sum(large) <= 0.40 + 1e-12
        assert sum(weights.values()) == pytest.approx(1.0)

    def test_the_1940_act_keeps_large_holdings_to_a_quarter(self):
        target = {"A": 0.2, "B": 0.15, "C": 0.1, **{f"N{i}": 0.55 / 25 for i in range(25)}}
        weights = limited(Act1940Limits(), target)
        large = [weight for weight in weights.values() if weight > 0.05 + 1e-12]

        assert sum(large) <= 0.25 + 1e-12
        assert sum(weights.values()) == pytest.approx(1.0)

    def test_under_the_cash_rule_the_excess_is_cash(self):
        weights = limited(UcitsLimits(), {"A": 0.6, "B": 0.4}, rule="cash")

        assert weights == pytest.approx({"A": 0.35, "B": 0.20})

    def test_a_limit_shapes_a_run(self):
        result = run(vehicle=Vehicle(limits=[UcitsLimits(index_tracking=False)]))

        assert result.rebalance_steps[0].capped == pytest.approx(
            {"GROW": 0.10, "FLAT": 0.10})


class TestPresets:

    @pytest.mark.parametrize("build,pricing,limits", [
        (uk_oeic, SwingPricing, UcitsLimits),
        (luxembourg_sicav, SwingPricing, UcitsLimits),
        (irish_icav, DilutionLevy, UcitsLimits),
        (us_mutual_fund, SinglePricing, Act1940Limits),
    ])
    def test_each_has_its_structures_settings(self,
                                               build,
                                               pricing,
                                               limits):
        vehicle = build()

        assert isinstance(vehicle.pricing, pricing)
        assert [type(limit) for limit in vehicle.limits] == [limits]

    def test_a_luxembourg_sicav_swings_only_on_large_flows(self):
        assert luxembourg_sicav().pricing.threshold == 0.02

    def test_any_setting_can_be_changed(self):
        vehicle = uk_oeic(management_fee_bps=15, pricing=DilutionLevy())

        assert vehicle.management_fee_bps == 15
        assert isinstance(vehicle.pricing, DilutionLevy)
        assert vehicle.name == "UK OEIC"

    def test_a_us_mutual_fund_can_charge_a_redemption_fee(self):
        pricing = us_mutual_fund(redemption_fee_bps=200).pricing

        assert isinstance(pricing, DilutionLevy)
        assert pricing.on == "redemptions"

    def test_a_preset_by_name(self):
        assert preset("irish_icav", management_fee_bps=10).name == "Irish ICAV"

        with pytest.raises(KeyError, match="uk_oeic"):
            preset("cayman_spc")


class TestTheSettingsAreChecked:

    @pytest.mark.parametrize("build", [
        lambda: SwingPricing(factor_bps=-1),
        lambda: SwingPricing(threshold=-0.1),
        lambda: DualPricing(bid_bps=-5),
        lambda: DilutionLevy(on="sometimes"),
    ])
    def test_an_impossible_setting_is_refused(self,
                                              build):
        with pytest.raises(ValueError):
            build()
