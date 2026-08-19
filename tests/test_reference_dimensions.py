# tests/test_reference_dimensions.py
"""BN-133: country, domicile, and market capitalisation.

Asked for by beacon-ui (BU-85), whose Universe Set view derives its filters
from whatever columns `GET /data/reference` returns — so a column published
here becomes a control with no client change.

**Listing and domicile are separate questions.** "Listed in Germany" and
"incorporated in Ireland" are both real screens, and a single COUNTRY would
silently answer one while appearing to answer both. A company incorporated in
Ireland and listed in New York is not a US company for tax, and a screen that
treats it as one is wrong in a way nobody notices until somebody reconciles a
withholding number.

**A market cap is money.** Since BN-128 the members of one universe are quoted
in seven currencies, so a raw local value is unsortable and every comparison
across it is silently wrong. The tests below assert the conversion happens
rather than that a currency label is present.
"""
import logging
import tempfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from beacon.server import ServerConfig, create_app
from beacon.server.reference import DERIVED_CURRENCY
from beacon.synthetic import SyntheticConfig, generate
from beacon.synthetic import regions as regions_module

TOKEN = "dimension-token"
HEADERS = {"Authorization": f"Bearer {TOKEN}"}


@pytest.fixture(scope="module")
def panel():
    """Enough names that every region and a foreign domicile appear."""
    logging.disable(logging.ERROR)

    try:
        return generate(SyntheticConfig(assets=300, start="2024-01-02",
                                        end="2024-06-28", seed=1))
    finally:
        logging.disable(logging.NOTSET)


@pytest.fixture(scope="module")
def client(panel):
    app = create_app(ServerConfig(auth_token=TOKEN,
                                  data_fetcher=panel.fetcher(),
                                  storage_root=Path(tempfile.mkdtemp())))

    return TestClient(app, raise_server_exceptions=False)


def fields_for(client,
               identifier,
               names):
    """The reference fields for one identifier."""
    response = client.get("/data/reference", headers=HEADERS,
                          params={"identifiers": identifier, "fields": names})

    return response.json()["entries"][0]["fields"]


class TestCountryIsSeparateFromRegion:
    """`EXCHANGE` answered neither question: XNAS and XNYS are both the US."""

    def test_both_columns_are_published(self,
                                        panel):
        columns = set(panel.reference.data.columns)

        assert {"COUNTRY_LISTING", "COUNTRY_DOMICILE"} <= columns

    def test_the_codes_are_iso_alpha_2(self,
                                       panel):
        reference = panel.reference.data

        for column in ("COUNTRY_LISTING", "COUNTRY_DOMICILE"):
            values = reference[column].unique()

            assert all(len(value) == 2 and value.isupper() for value in values), (
                f"{column} holds something that is not an alpha-2 code: "
                f"{sorted(values)}")

    def test_the_listing_country_matches_the_venue(self,
                                                   panel):
        """A venue cannot disagree with its country: both come from the same
        row of the region table."""
        expected = {region.exchange: region.country
                    for region in regions_module.REGIONS}
        reference = panel.reference.data

        for exchange, country in zip(reference["EXCHANGE"],
                                     reference["COUNTRY_LISTING"],
                                     strict=True):
            assert country == expected[exchange]

    def test_region_and_country_are_not_the_same_question(self,
                                                          panel):
        """Europe is one region covering more than one country, which is why
        collapsing them would be wrong."""
        reference = panel.reference.data

        assert reference["REGION"].nunique() > 1
        assert reference["COUNTRY_LISTING"].nunique() > 1


class TestDomicileDivergesFromListing:
    """The distinction only earns its place if it is ever exercised."""

    def test_some_names_are_domiciled_elsewhere(self,
                                                panel):
        reference = panel.reference.data
        foreign = reference["COUNTRY_LISTING"] != reference["COUNTRY_DOMICILE"]

        assert foreign.any(), "domicile never differs, so the column says nothing"

    def test_most_names_are_domiciled_where_they_list(self,
                                                      panel):
        """Guards the test above: making every name foreign would pass it and
        describe no real market."""
        reference = panel.reference.data
        foreign = reference["COUNTRY_LISTING"] != reference["COUNTRY_DOMICILE"]

        assert 0.05 < foreign.mean() < 0.35, (
            f"{foreign.mean():.0%} of names are foreign-domiciled")

    def test_the_destinations_are_plausible_for_the_venue(self,
                                                          panel):
        """Drawn from the destinations a venue actually uses, not uniformly:
        a uniform draw produces Japanese companies incorporated in Jersey,
        which is not a thing."""
        reference = panel.reference.data

        for listing, domicile in zip(reference["COUNTRY_LISTING"],
                                     reference["COUNTRY_DOMICILE"],
                                     strict=True):
            if listing == domicile:
                continue

            allowed = regions_module.FOREIGN_DOMICILES.get(listing, ())

            assert domicile in allowed, (
                f"a {listing} listing is domiciled in {domicile}, which is "
                f"not among {allowed}")

    def test_it_is_reproducible_from_the_seed(self):
        """Two runs of one seed give the same domiciles, so a screen built on
        them is stable."""
        logging.disable(logging.ERROR)

        try:
            frames = [generate(SyntheticConfig(assets=80, start="2024-01-02",
                                               end="2024-03-28", seed=5)
                               ).reference.data["COUNTRY_DOMICILE"].tolist()
                      for _ in range(2)]
        finally:
            logging.disable(logging.NOTSET)

        assert frames[0] == frames[1]


class TestMarketCapIsDerivedAndConverted:
    """The column beacon-ui has drawn since it was built, showing a dash."""

    def test_it_is_returned_when_named(self,
                                       client,
                                       panel):
        identifier = panel.reference.data.index[0]
        fields = fields_for(client, identifier, "market_cap")

        assert fields["market_cap"] > 0

    def test_it_is_absent_unless_named(self,
                                       client,
                                       panel):
        """Derived for the same reason `adv_3m` is: a market-data join that
        changes daily, which a client should have to ask to pay for."""
        identifier = panel.reference.data.index[0]
        fields = fields_for(client, identifier, "NAME")

        assert "market_cap" not in fields

    def test_a_foreign_cap_is_actually_converted(self,
                                                 client,
                                                 panel):
        """The assertion that matters, and the one a currency label would pass
        without doing anything.

        An unconverted yen capitalisation is roughly 166x its dollar value, so
        it sorts above every US name in the universe on magnitude alone. Same
        class of defect as the dividend conversion BN-128 found, and it
        announces itself just as little.
        """
        reference = panel.reference.data
        japanese = reference.index[reference["CURRENCY"] == "JPY"]

        if not len(japanese):
            pytest.skip("no JPY name in this panel")

        identifier = japanese[0]
        fetcher = panel.fetcher()

        row = fetcher.fetch_market_data(identifier, "2024-06-01",
                                        "2024-06-28").iloc[-1]
        local = float(row["CLOSE"]) * float(row["SHARES_OUTSTANDING"])
        rate = float(fetcher.fetch_fx_rates("JPY", "USD").iloc[-1])

        reported = fields_for(client, identifier, "market_cap")["market_cap"]

        assert reported == pytest.approx(local * rate, rel=1e-6)
        assert reported < local / 10, "the value was not converted"

    def test_the_currency_is_stated(self,
                                    client,
                                    panel):
        """A number whose unit a client has to assume is one it will
        eventually assume wrongly."""
        identifier = panel.reference.data.index[0]
        fields = fields_for(client, identifier, "market_cap")

        assert fields["market_cap_currency"] == DERIVED_CURRENCY

    def test_free_float_is_the_smaller_number(self,
                                              client,
                                              panel):
        identifier = panel.reference.data.index[0]
        fields = fields_for(client, identifier,
                            "market_cap,free_float_market_cap")

        assert 0 < fields["free_float_market_cap"] <= fields["market_cap"]

    def test_caps_are_comparable_across_currencies(self,
                                                   client,
                                                   panel):
        """The whole point of converting: a ranking that mixes currencies has
        to mean something.

        Unconverted, the ordering is decided by the size of the currency unit
        rather than the size of the company, and every yen-quoted name
        outranks every dollar-quoted one.
        """
        reference = panel.reference.data
        sample = reference.index[:60].tolist()

        response = client.get("/data/reference", headers=HEADERS,
                              params={"identifiers": ",".join(sample),
                                      "fields": "market_cap"})
        entries = response.json()["entries"]

        caps = {entry["identifier"]: entry["fields"]["market_cap"]
                for entry in entries if entry["fields"]["market_cap"]}

        currencies = {identifier: reference.loc[identifier, "CURRENCY"]
                      for identifier in caps}

        assert len({*currencies.values()}) > 1, "one currency proves nothing"

        top_ten = sorted(caps, key=lambda key: caps[key], reverse=True)[:10]

        assert len({currencies[identifier] for identifier in top_ten}) > 1, (
            "the ten largest are all quoted in one currency, which is a "
            "currency artefact rather than a size ranking")

    def test_an_unknown_derived_field_still_names_the_valid_ones(self,
                                                                 client):
        response = client.get("/data/reference", headers=HEADERS,
                              params={"identifiers": "CMPA",
                                      "fields": "market_capitalisation"})

        assert response.status_code in (400, 404, 422)
        assert "market_cap" in str(response.json())


class TestTheSurface:
    """What beacon-ui generates its filters from."""

    def test_the_derived_fields_are_documented(self):
        from beacon.server.reference import (
            DERIVED_FIELDS,
            FREE_FLOAT_MARKET_CAP,
            MARKET_CAP,
        )

        assert MARKET_CAP in DERIVED_FIELDS
        assert FREE_FLOAT_MARKET_CAP in DERIVED_FIELDS

    def test_the_country_columns_reach_the_endpoint(self,
                                                    client,
                                                    panel):
        """Naming no stored column returns them all, which is how the client
        discovers what it can filter on."""
        identifier = panel.reference.data.index[0]
        response = client.get(f"/data/reference/{identifier}", headers=HEADERS)
        fields = response.json()["fields"]

        assert "COUNTRY_LISTING" in fields
        assert "COUNTRY_DOMICILE" in fields
