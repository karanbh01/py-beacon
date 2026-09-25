# tests/test_fx_routes.py
"""BN-235: a rate is found from the stored pair, its inverse, or a cross.

The lookup used to find only the exact pair. With GBPUSD stored, USD to GBP
found nothing, although it is one over GBPUSD, and GBP to EUR found nothing,
although it is GBPUSD divided by EURUSD. So a GBP index holding US shares
refused for want of a rate it effectively had.
"""
import pandas as pd
import pytest

from beacon.data.base import MarketData
from beacon.data.fetcher import FX_EXACT_DAY, DataFetcher
from beacon.index.calculation import IndexCalculator
from beacon.index.constructor import IndexDefinition
from beacon.index.methodology import MarketCapWeighted
from beacon.testing import dataset

DAYS = pd.bdate_range("2024-01-01", "2024-01-31")
GBPUSD = 1.25
EURUSD = 1.10
# EURUSD prints only from here, so a cross before it has one leg missing.
EUR_FROM = pd.Timestamp("2024-01-10")
# A day EURUSD skips, to tell the two policies apart.
EUR_GAP = pd.Timestamp("2024-01-17")


def fetcher(policy: str = "CARRY_FORWARD",
            extra: list[dict] | None = None) -> DataFetcher:
    rows = [{"IDENTIFIER": "GBPUSD", "DATE": day, "RATE": GBPUSD} for day in DAYS]
    rows += [{"IDENTIFIER": "EURUSD", "DATE": day, "RATE": EURUSD}
             for day in DAYS if day >= EUR_FROM and day != EUR_GAP]
    rows += extra or []

    return DataFetcher(MarketData.from_dataframe(pd.DataFrame(rows)),
                       fx_policy=policy)


class TestTheRoute:

    def test_a_stored_pair_is_used_as_it_is(self):
        data = fetcher()

        assert data.fx_rate_on("GBP", "USD", "2024-01-15") == GBPUSD
        assert data.fx_route("GBP", "USD") == "direct"

    def test_the_reverse_is_one_over_the_stored_pair(self):
        data = fetcher()

        assert data.fx_rate_on("USD", "GBP", "2024-01-15") == pytest.approx(1 / GBPUSD)
        assert data.fx_route("USD", "GBP") == "inverse"

    def test_two_foreign_currencies_cross_through_the_dollar(self):
        data = fetcher()

        assert data.fx_rate_on("GBP", "EUR", "2024-01-15") == pytest.approx(
            GBPUSD / EURUSD)
        assert data.fx_route("GBP", "EUR") == "cross via USD"

    def test_a_stored_pair_wins_over_a_derived_one(self):
        """A quoted rate is the better number when one exists."""
        quoted = [{"IDENTIFIER": "USDGBP", "DATE": day, "RATE": 0.79}
                  for day in DAYS]
        data = fetcher(extra=quoted)

        assert data.fx_rate_on("USD", "GBP", "2024-01-15") == 0.79
        assert data.fx_route("USD", "GBP") == "direct"

    def test_no_route_is_still_no_rate(self):
        """Nothing is invented: no CHF pair means no CHF rate."""
        data = fetcher()

        assert data.fx_rate_on("CHF", "USD", "2024-01-15") is None
        assert data.fx_rate_on("CHF", "GBP", "2024-01-15") is None
        assert data.fx_route("CHF", "GBP") is None

    def test_the_same_currency_needs_no_route(self):
        assert fetcher().fx_route("gbp", "GBP") == "same currency"


class TestTheSettingsKeepTheirMeaning:

    def test_a_cross_never_uses_a_rate_from_after_the_day(self):
        """Before EURUSD's first rate there is no cross, rather than one
        built with a later EUR rate."""
        assert fetcher().fx_rate_on("GBP", "EUR", "2024-01-05") is None

    def test_carry_forward_covers_one_legs_gap(self):
        assert fetcher().fx_rate_on("GBP", "EUR", EUR_GAP) == pytest.approx(
            GBPUSD / EURUSD)

    def test_exact_day_refuses_when_one_leg_did_not_print(self):
        data = fetcher(FX_EXACT_DAY)

        assert data.fx_rate_on("GBP", "EUR", EUR_GAP) is None
        assert data.fx_rate_on("GBP", "EUR", "2024-01-16") == pytest.approx(
            GBPUSD / EURUSD)

    def test_the_inverse_carries_like_the_pair(self):
        data = fetcher(FX_EXACT_DAY)

        assert data.fx_rate_on("USD", "EUR", EUR_GAP) is None
        assert fetcher().fx_rate_on("USD", "EUR", EUR_GAP) == pytest.approx(1 / EURUSD)

    def test_a_rate_of_zero_has_no_inverse(self):
        zero = [{"IDENTIFIER": "JPYUSD", "DATE": DAYS[0], "RATE": 0.0}]

        assert fetcher(extra=zero).fx_rate_on("USD", "JPY", DAYS[0]) is None


class TestOneRuleForBothLookups:

    @pytest.mark.parametrize(("source", "target"),
                             [("USD", "GBP"), ("GBP", "EUR"), ("EUR", "GBP")])
    @pytest.mark.parametrize("policy", ["CARRY_FORWARD", FX_EXACT_DAY])
    def test_the_batched_lookup_agrees_day_by_day(self,
                                                  source,
                                                  target,
                                                  policy):
        data = fetcher(policy)
        batched = data.fx_rates_on(source, target, DAYS)

        for day in DAYS:
            single = data.fx_rate_on(source, target, day)
            one = batched.loc[day]

            if single is None:
                assert pd.isna(one), day
            else:
                assert one == pytest.approx(single), day

    def test_a_merge_finds_the_route_again(self):
        """Routes are cached per pair, so a merge that adds a quoted pair
        has to clear them."""
        data = fetcher()
        assert data.fx_route("USD", "GBP") == "inverse"

        data.merge_market_data(pd.DataFrame(
            [{"IDENTIFIER": "USDGBP", "DATE": day, "RATE": 0.79} for day in DAYS]))

        assert data.fx_route("USD", "GBP") == "direct"


class TestTheCaseThatRefusedBefore:

    def test_a_sterling_index_of_dollar_shares_now_runs(self):
        """The sample data stores GBPUSD only. A GBP index of its US names
        needs USD to GBP, which is the inverse, and used to refuse."""
        fetcher = dataset.data_fetcher()
        definition = IndexDefinition(index_id="GBPIDX",
                                     index_name="Sterling view",
                                     base_date=dataset.START,
                                     base_value=1000.0,
                                     currency="GBP",
                                     eligibility_rules=[],
                                     weighting_scheme=MarketCapWeighted(),
                                     rebalancing_frequency="QUARTERLY",
                                     calendar="XNYS",
                                     universe_identifiers=list(dataset.UNIVERSE))

        levels = IndexCalculator(definition, fetcher).run(
            start_date=dataset.START, end_date="2023-06-30").index_levels

        assert levels.notna().all()
        assert fetcher.fx_route("USD", "GBP") == "inverse"
