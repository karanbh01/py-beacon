# tests/test_backtest_withholding.py
"""BN-271: withholding tax on dividends by fund domicile.

Four names, each paying one dividend in February, between the first two
rebalances, so every run holds the same shares when it is paid. USCO lists
in New York, GBCO in London, IECO in New York but is incorporated in Ireland,
and CACO in Toronto, which the built-in table does not cover. Each run's
dividend is compared with a run that withholds nothing. The vehicles are plain
ones with a domicile: a preset's UCITS limits would cap these 25% weights and
change what is held.
"""
import numpy as np
import pandas as pd
import pytest

from beacon.assumptions import ModellingAssumptions
from beacon.backtest import Backtest, Vehicle, preset, uk_oeic
from beacon.backtest.withholding import (
    TREATY_RATES,
    source_country,
    withholding_rate,
)
from beacon.data.base import MarketData, ReferenceData
from beacon.data.corporate_actions import CorporateActions
from beacon.data.fetcher import DataFetcher
from beacon.index.constructor import IndexDefinition
from beacon.index.methodology import EqualWeighted
from beacon.index.schedule import sessions

DAYS = sessions(pd.Timestamp("2024-01-02"), pd.Timestamp("2024-06-28"), "XNYS")
LISTINGS = {"USCO": ("XNYS", None), "GBCO": ("XLON", None),
            "IECO": ("XNYS", "IE"), "CACO": ("XTSE", None)}
EX_DATE = "2024-02-15"
FLAT = 0.10


def fetcher() -> DataFetcher:
    random = np.random.default_rng(5)
    rows = []

    for name in LISTINGS:
        prices = 50.0 * np.cumprod(1.0 + random.normal(0.0003, 0.008, len(DAYS)))
        rows += [{"IDENTIFIER": name, "DATE": day, "CLOSE": price, "SHARES_OUTSTANDING": 1e8}
                 for day, price in zip(DAYS, prices, strict=True)]

    reference = pd.DataFrame([
        {"IDENTIFIER": name, "NAME": name, "DATE_FROM": "2020-01-01", "CURRENCY": "USD",
         "EXCHANGE": exchange, "COUNTRY_DOMICILE": country}
        for name, (exchange, country) in LISTINGS.items()])
    actions = pd.DataFrame([{"IDENTIFIER": name, "EX_DATE": EX_DATE, "TYPE": "DIVIDEND",
                             "VALUE": 1.0} for name in LISTINGS])

    return DataFetcher(MarketData.from_dataframe(pd.DataFrame(rows)),
                       ReferenceData.from_dataframe(reference),
                       CorporateActions.from_dataframe(actions))


DATA = fetcher()
DEFINITION = IndexDefinition(index_id="EW", index_name="Equal", base_date="2024-01-02",
                             base_value=100.0, currency="USD", eligibility_rules=[],
                             weighting_scheme=EqualWeighted(),
                             rebalancing_frequency="QUARTERLY", calendar="XNYS",
                             universe_identifiers=list(LISTINGS))


def dividends(vehicle=None) -> dict[str, float]:
    result = Backtest(initial_capital=1e6, data_provider=DATA, cache=False, dividends="cash",
                      modelling_assumptions=ModellingAssumptions(withholding_tax_rate=FLAT),
                      vehicle=vehicle).run(DEFINITION, start="2024-01-02", end="2024-03-28")

    return {flow.asset_id: flow.amount for flow in result.portfolio.cash_flows
            if flow.kind == "DIVIDEND"}


# Withholding nothing on any of the four, through the vehicle's own rates.
GROSS = dividends(Vehicle(withholding_rates={"US": 0.0, "GB": 0.0, "IE": 0.0, "CA": 0.0}))


def withheld(vehicle=None) -> dict[str, float]:
    paid = dividends(vehicle)

    return {name: round(1.0 - paid[name] / GROSS[name], 6) for name in LISTINGS}


class TestTheRates:

    def test_without_a_domicile_the_flat_rate_applies(self):
        assert withheld() == dict.fromkeys(LISTINGS, FLAT)

    def test_an_irish_fund(self):
        assert withheld(Vehicle(domicile="IE")) == {"USCO": 0.15, "GBCO": 0.0, "IECO": 0.0,
                                          "CACO": FLAT}

    def test_a_luxembourg_fund_pays_the_full_us_rate(self):
        assert withheld(Vehicle(domicile="LU")) == {"USCO": 0.30, "GBCO": 0.0, "IECO": FLAT,
                                                "CACO": FLAT}

    def test_a_uk_fund(self):
        assert withheld(Vehicle(domicile="GB"))["USCO"] == 0.15

    def test_a_us_fund_suffers_none_at_home(self):
        assert withheld(Vehicle(domicile="US"))["USCO"] == 0.0

    def test_the_vehicle_can_set_its_own_rates(self):
        overridden = withheld(Vehicle(domicile="IE", withholding_rates={"CA": 0.15,
                                                                         "US": 0.20}))

        assert overridden["CACO"] == 0.15
        assert overridden["USCO"] == 0.20


class TestTheSource:

    def test_the_country_of_incorporation_comes_before_the_listing(self):
        assert source_country(DATA, "IECO", EX_DATE) == "IE"

    def test_otherwise_the_listing_exchange_says(self):
        assert source_country(DATA, "USCO", EX_DATE) == "US"
        assert source_country(DATA, "GBCO", EX_DATE) == "GB"

    def test_an_unknown_name_has_no_source(self):
        assert source_country(DATA, "NOPE", EX_DATE) is None

    def test_an_unknown_pair_falls_back(self):
        assert withholding_rate("IE", "JP", None, 0.07) == 0.07
        assert withholding_rate(None, "US", None, 0.07) == 0.07

    def test_the_table_is_shares(self):
        assert all(0.0 <= rate < 1.0 for rate in TREATY_RATES.values())


class TestTheVehicle:

    @pytest.mark.parametrize(("key", "domicile"), [
        ("uk_oeic", "GB"), ("luxembourg_sicav", "LU"), ("irish_icav", "IE"),
        ("us_mutual_fund", "US"), ("ucits_etf", "IE"), ("us_etf", "US")])
    def test_each_preset_has_its_domicile(self, key, domicile):
        assert preset(key).domicile == domicile

    def test_a_preset_domicile_can_be_changed(self):
        assert uk_oeic(domicile="ie").domicile == "IE"

    @pytest.mark.parametrize("domicile", ["IRL", "1E", ""])
    def test_a_domicile_is_a_two_letter_code(self, domicile):
        with pytest.raises(ValueError, match="two-letter"):
            Vehicle(domicile=domicile)

    def test_a_rate_is_a_share(self):
        with pytest.raises(ValueError, match="share"):
            Vehicle(withholding_rates={"US": 1.2})

    def test_rates_are_keyed_by_country_code(self):
        with pytest.raises(ValueError, match="two-letter"):
            Vehicle(withholding_rates={"USA": 0.15})

    def test_a_plain_vehicle_has_no_domicile(self):
        assert Vehicle().domicile is None
        assert Vehicle().withholding_rates is None
