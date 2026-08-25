# tests/test_reference_profile.py
"""BN-148/149: the reference fields a profile view needs, and reaching them.

Audited against beacon-ui's reference view, which rendered 25 of its 30
non-membership rows as a dash. Most of these tests are coherence checks rather
than presence checks: a generated fact that contradicts another generated fact
is worse than an absent one, because the contradiction is invisible until
somebody looks.
"""
import logging
import tempfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from beacon.server import ServerConfig, create_app
from beacon.synthetic import SyntheticConfig, generate, profiles

TOKEN = "profile-token"
HEADERS = {"Authorization": f"Bearer {TOKEN}"}

# Every non-membership row the view claims, and the keys it looks for. Copied
# from `views/reference/reference.ts` rather than derived, so a change on
# either side shows up here as a failure instead of as a silent dash.
VIEW_ROWS = [
    ("Ticker", ["ticker", "symbol", "identifier"]),
    ("Name", ["name", "long_name", "longname", "short_name"]),
    ("ISIN", ["isin"]),
    ("CUSIP", ["cusip"]),
    ("FIGI", ["figi", "composite_figi"]),
    ("SEDOL", ["sedol"]),
    ("Exchange", ["exchange", "exchange_name", "full_exchange_name", "mic"]),
    ("Currency", ["currency"]),
    ("Security Type", ["security_type", "instrument_type", "quote_type",
                       "type"]),
    ("GICS Sector", ["gics_sector", "sector"]),
    ("GICS Industry Group", ["gics_industry_group", "industry_group"]),
    ("GICS Industry", ["gics_industry", "industry"]),
    ("GICS Sub-Industry", ["gics_sub_industry", "sub_industry"]),
    ("Beacon Asset Class", ["asset_class", "beacon_asset_class"]),
    ("Instrument Subtype", ["instrument_subtype", "subtype"]),
    ("Trading Status", ["trading_status", "status"]),
    ("Primary Listing", ["primary_listing", "primary_exchange"]),
    ("ADR / GDR", ["adr", "depositary_receipt"]),
    ("Options Available", ["options_available", "has_options"]),
    ("Employees", ["employees", "full_time_employees"]),
    ("Headquarters", ["headquarters", "city"]),
    ("Founded", ["founded"]),
    ("IPO Date", ["ipo_date", "first_trade_date"]),
    ("Fiscal Year End", ["fiscal_year_end"]),
    ("Dividend Frequency", ["dividend_frequency"]),
    ("Next Earnings", ["next_earnings", "earnings_date"]),
]


@pytest.fixture(scope="module")
def panel():
    logging.disable(logging.ERROR)

    try:
        return generate(SyntheticConfig(assets=300, start="2020-01-02",
                                        end="2024-12-31", seed=7))
    finally:
        logging.disable(logging.NOTSET)


@pytest.fixture(scope="module")
def reference(panel):
    return panel.reference.data


@pytest.fixture
def client(panel):
    return TestClient(create_app(ServerConfig(
        auth_token=TOKEN, data_fetcher=panel.fetcher(),
        storage_root=Path(tempfile.mkdtemp()))))


def isin_check(code: str) -> int:
    """The ISIN check digit, from the standard.

    Written out rather than imported from the generator: a test that calls the
    same function it is checking asserts only that the function is
    deterministic. Verified against real ISINs (US0378331005, GB0002634946,
    JP3633400001) before being trusted here.
    """
    digits = "".join(str(int(c) if c.isdigit() else ord(c) - 55)
                     for c in code[:-1])
    total = 0

    for position, character in enumerate(reversed(digits)):
        value = int(character)

        if position % 2 == 0:
            value *= 2

            if value > 9:
                value -= 9

        total += value

    return (10 - total % 10) % 10


class TestTheViewCanFillEveryRow:
    """The acceptance case."""

    def test_every_row_finds_a_key(self,
                                   client):
        served = {key.lower() for key
                  in client.get("/data/reference/CMPA",
                                headers=HEADERS).json()["fields"]}
        missing = [label for label, keys in VIEW_ROWS
                   if not any(key in served for key in keys)]

        assert missing == [], f"still renders a dash: {missing}"

    def test_no_row_is_served_as_null(self,
                                      client):
        """Present but null is the same dash to a reader."""
        fields = client.get("/data/reference/CMPA",
                            headers=HEADERS).json()["fields"]
        served = {key.lower(): value for key, value in fields.items()}

        empty = [label for label, keys in VIEW_ROWS
                 for key in keys
                 if key in served and served[key] in (None, "")]

        assert empty == []

    def test_country_is_the_one_the_client_must_change(self,
                                                       client):
        """Documented rather than papered over.

        The view's Country row claims only `country`; the engine carries
        `country_listing` and `country_domicile` deliberately, because the
        distinction was asked for. A third merged column would reintroduce the
        ambiguity that split them, so the fix belongs in beacon-ui.
        """
        served = {key.lower() for key
                  in client.get("/data/reference/CMPA",
                                headers=HEADERS).json()["fields"]}

        assert "country" not in served
        assert {"country_listing", "country_domicile"} <= served


class TestTheIdentifiersAreValid:
    """A wrong check digit works until the day something checks it."""

    def test_every_isin_passes_its_check_digit(self,
                                               reference):
        wrong = [code for code in reference["ISIN"]
                 if isin_check(code) != int(code[-1])]

        assert wrong == []

    def test_an_isin_is_twelve_characters(self,
                                          reference):
        assert set(reference["ISIN"].str.len()) == {12}

    def test_an_isin_starts_with_the_listing_country(self,
                                                     reference):
        prefixes = reference.apply(
            lambda row: row["ISIN"][:2] == row["COUNTRY_LISTING"], axis=1)

        assert prefixes.all()

    def test_a_cusip_is_nine_characters(self,
                                        reference):
        assert set(reference["CUSIP"].str.len()) == {9}

    def test_a_sedol_is_seven_and_has_no_vowels(self,
                                                reference):
        """SEDOL omits vowels so a code cannot spell a word."""
        assert set(reference["SEDOL"].str.len()) == {7}
        assert not any(vowel in code for code in reference["SEDOL"]
                       for vowel in "AEIOU")

    def test_a_figi_has_the_right_shape(self,
                                        reference):
        assert set(reference["FIGI"].str.len()) == {12}
        assert reference["FIGI"].str.startswith("BBG").all()

    @pytest.mark.parametrize("column", ["ISIN", "CUSIP", "SEDOL", "FIGI"])
    def test_identifiers_are_unique(self,
                                    reference,
                                    column):
        """Two instruments sharing an ISIN would break any join keyed on it,
        which is the main reason to carry one."""
        assert reference[column].nunique() == len(reference)


class TestCoherence:
    """Generated facts that must agree with each other."""

    def test_trading_status_agrees_with_the_listing_dates(self,
                                                          reference):
        """A name delisted in 2021 is not Active, and a screen on status has
        to select what a screen on dates selects."""
        delisted = reference[reference["DATE_TO"].notna()]
        live = reference[reference["DATE_TO"].isna()]

        assert len(delisted) > 0, "no delisted names, so this proves nothing"
        assert (delisted["TRADING_STATUS"] == profiles.DELISTED).all()
        assert (live["TRADING_STATUS"] == profiles.ACTIVE).all()

    def test_dividend_frequency_agrees_with_the_dividends(self,
                                                          panel,
                                                          reference):
        """A non-payer reporting "Quarterly" would make a frequency screen
        select names with no dividends."""
        joined = reference.join(panel.universe["dividend_yield"])
        payers = joined[joined["dividend_yield"] > 0]
        non_payers = joined[joined["dividend_yield"] <= 0]

        assert len(non_payers) > 0
        assert (non_payers["DIVIDEND_FREQUENCY"] == profiles.NO_DIVIDEND).all()
        assert (payers["DIVIDEND_FREQUENCY"] != profiles.NO_DIVIDEND).all()

    def test_ipo_date_is_the_listing_date(self,
                                          reference):
        """One fact, not two drawn beside each other."""
        assert (reference["IPO_DATE"] == reference["DATE_FROM"]).all()

    def test_a_company_is_founded_before_it_lists(self,
                                                  reference):
        assert (reference["FOUNDED"] < reference["IPO_DATE"].dt.year).all()

    def test_the_gics_levels_nest(self,
                                  reference):
        """A four-level hierarchy whose levels do not contain each other is
        decoration, and any roll-up computed from it would be wrong."""
        nested = reference.apply(
            lambda row: (row["SECTOR"] in row["GICS_INDUSTRY_GROUP"]
                         and row["GICS_INDUSTRY_GROUP"] in row["GICS_INDUSTRY"]
                         and row["SECTOR"] in row["SUB_INDUSTRY"]), axis=1)

        assert nested.all()

    def test_headquarters_sits_in_the_listing_country(self,
                                                      reference):
        located = reference.apply(
            lambda row: row["HEADQUARTERS"].endswith(row["COUNTRY_LISTING"]),
            axis=1)

        assert located.all()

    def test_employees_scale_with_size(self,
                                       panel,
                                       reference):
        """A mega-cap with forty staff is what makes a demo dataset obviously
        fake at the moment somebody looks closely."""
        import numpy as np

        joined = reference.join(panel.universe["market_cap"])
        correlation = np.corrcoef(np.log(joined["market_cap"]),
                                  np.log(joined["EMPLOYEES"]))[0, 1]

        assert correlation > 0.5

    def test_next_earnings_is_in_the_future(self,
                                            panel,
                                            reference):
        assert (reference["NEXT_EARNINGS"] > panel.market.date_range[1]).all()

    def test_an_adr_is_never_a_domestic_us_listing(self,
                                                   reference):
        """A depositary receipt is how a foreign company lists elsewhere."""
        american = reference[reference["COUNTRY_LISTING"] == "US"]

        assert not american["ADR"].any()

    def test_it_is_reproducible(self):
        logging.disable(logging.ERROR)

        try:
            frames = [generate(SyntheticConfig(assets=30, start="2022-01-03",
                                               end="2023-12-29", seed=4)
                               ).reference.data for _ in range(2)]
        finally:
            logging.disable(logging.NOTSET)

        assert frames[0]["ISIN"].tolist() == frames[1]["ISIN"].tolist()


class TestReachingDerivedFields:
    """BN-149: the two reference endpoints must answer the same question."""

    def test_the_single_endpoint_serves_derived_fields(self,
                                                       client):
        """It accepted `fields` and ignored it, so `market_cap` came back
        empty from here and populated from the batch form — a parameter that
        looks supported while doing nothing."""
        fields = client.get("/data/reference/CMPA", headers=HEADERS,
                            params={"fields": "NAME,market_cap"}).json()["fields"]

        assert fields["market_cap"] > 0

    def test_both_endpoints_agree(self,
                                  client):
        """So they cannot drift again."""
        params = {"fields": "NAME,market_cap,adv_3m"}
        single = client.get("/data/reference/CMPA", headers=HEADERS,
                            params=params).json()["fields"]
        batch = client.get("/data/reference", headers=HEADERS,
                           params={**params, "identifiers": "CMPA"}
                           ).json()["entries"][0]["fields"]

        assert single == batch

    def test_a_money_field_states_its_currency(self,
                                               client):
        fields = client.get("/data/reference/CMPA", headers=HEADERS,
                            params={"fields": "market_cap"}).json()["fields"]

        assert fields["market_cap_currency"] == "USD"

    def test_no_fields_still_returns_the_whole_row(self,
                                                   client):
        """The behaviour every existing caller depends on."""
        fields = client.get("/data/reference/CMPA",
                            headers=HEADERS).json()["fields"]

        assert len(fields) > 20
        assert "market_cap" not in fields

    def test_an_unknown_field_is_reported(self,
                                          client):
        """A silently absent column shows as an empty row and reads as missing
        data rather than as a misspelled request."""
        assert client.get("/data/reference/CMPA", headers=HEADERS,
                          params={"fields": "nonsense"}).status_code == 422

    def test_memberships_survive_a_field_request(self,
                                                 client):
        """They are answered only by this endpoint, so the derived path must
        not drop them."""
        body = client.get("/data/reference/CMPA", headers=HEADERS,
                          params={"fields": "NAME"}).json()

        assert "universes" in body
