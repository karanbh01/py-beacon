# tests/test_reference_currency.py
"""BN-189: the reference table publishes both money figures, not one.

BN-188 made the weighting convert market caps into the **index's** currency
and left the display converting into a hard-coded USD, so a EUR index showed
dollar caps beside euro weights. The ratios agreed -- that was the bug BN-188
fixed -- but the magnitudes did not, and only one of the two numbers carried a
currency code. A reader who noticed something odd could not tell a bug from a
unit mismatch, which is the affordance that made BN-188's defect visible in
the first place.

So each money field comes back twice: the local figure, which is a fact about
the company and needs no rate to be true, and the converted one, which is what
compares to an index weight. Both are labelled, and `currency` names what the
converted half is converted into -- defaulting to USD, so no existing caller
moves.

The universe is BN-188's: one dollar name and one yen name at 150, far enough
apart that no assertion here passes by accident.
"""
import tempfile
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from beacon.data.base import MarketData, ReferenceData
from beacon.data.fetcher import DataFetcher
from beacon.exceptions import InvalidRuleError
from beacon.expressions.namespaces import DERIVED_COLUMNS
from beacon.server import ServerConfig, create_app
from beacon.server.reference import (
    COMPANION_FIELDS,
    DEFAULT_CURRENCY,
    DERIVED_FIELDS,
    MONEY_FIELDS,
    build_entries,
)

START = "2024-01-02"
END = "2024-01-31"
DATES = pd.bdate_range(START, END)

JPY_PER_USD = 150.0
SHARES = 1e9
FREE_FLOAT = 0.5

PRICE = {"USBIG": 100.0, "JPSML": 1000.0}
CURRENCY = {"USBIG": "USD", "JPSML": "JPY"}

# What the exchange reports, before anyone converts anything.
LOCAL_CAP = {name: price * SHARES for name, price in PRICE.items()}

TOKEN = "currency-token"
HEADERS = {"Authorization": f"Bearer {TOKEN}"}
MONEY = "market_cap,free_float_market_cap"


def build_fetcher(pairs: tuple[str, ...] = ("JPYUSD",)) -> DataFetcher:
    """The two-name, two-currency universe with the named FX pairs loaded."""
    rates = {"JPYUSD": 1.0 / JPY_PER_USD, "USDJPY": JPY_PER_USD}
    rows: list[dict[str, object]] = []

    for date in DATES:
        for name, price in PRICE.items():
            rows.append({"IDENTIFIER": name, "DATE": date, "CLOSE": price,
                         "SHARES_OUTSTANDING": SHARES,
                         "FREE_FLOAT": FREE_FLOAT})

        rows.extend({"IDENTIFIER": pair, "DATE": date, "RATE": rates[pair]}
                    for pair in pairs)

    reference = pd.DataFrame([
        {"IDENTIFIER": name, "DATE_FROM": "2020-01-01", "NAME": name,
         "CURRENCY": CURRENCY[name], "EXCHANGE": "XXXX"}
        for name in PRICE])

    return DataFetcher(MarketData.from_dataframe(pd.DataFrame(rows)),
                       ReferenceData.from_dataframe(reference))


def fields_for(currency: str | None = None,
               pairs: tuple[str, ...] = ("JPYUSD",),
               fields: str = MONEY) -> dict[str, dict[str, object]]:
    """The money fields for both names, keyed by identifier."""
    arguments = {} if currency is None else {"currency": currency}
    entries = build_entries(build_fetcher(pairs), list(PRICE), END,
                            [fields], **arguments)

    return {entry.identifier: entry.fields for entry in entries}


def derived_keys(fields: dict[str, object]) -> set[str]:
    """The computed half of an entry.

    Naming no stored column returns every stored column, so a request for
    `market_cap` alone still comes back with NAME and CURRENCY beside it.
    """
    stored = set(build_fetcher().reference_columns or ())

    return set(fields) - stored


@pytest.fixture(scope="module")
def client():
    app = create_app(ServerConfig(auth_token=TOKEN,
                                  data_fetcher=build_fetcher(("JPYUSD",
                                                              "USDJPY")),
                                  storage_root=Path(tempfile.mkdtemp())))

    return TestClient(app, raise_server_exceptions=False)


class TestBothFiguresArePublished:
    """The local number is a fact about the company; publishing only the
    converted one throws it away."""

    def test_the_local_figure_is_the_unconverted_one(self):
        fields = fields_for()["JPSML"]

        assert fields["market_cap_local"] == pytest.approx(LOCAL_CAP["JPSML"])
        assert fields["local_currency"] == "JPY"

    def test_the_converted_figure_sits_beside_it(self):
        fields = fields_for()["JPSML"]

        assert fields["market_cap"] == pytest.approx(
            LOCAL_CAP["JPSML"] / JPY_PER_USD)
        assert fields["market_cap_currency"] == DEFAULT_CURRENCY

    def test_the_two_differ_by_exactly_the_rate(self):
        """The point of publishing both: the reader can see *why* they differ
        rather than having to infer it."""
        fields = fields_for()["JPSML"]
        ratio = fields["market_cap_local"] / fields["market_cap"]

        assert ratio == pytest.approx(JPY_PER_USD)

    def test_a_domestic_name_reports_the_same_number_twice(self):
        """Not a special case worth suppressing: a client reading one column
        must not have to know whether the other one exists for this row."""
        fields = fields_for()["USBIG"]

        assert fields["market_cap"] == fields["market_cap_local"]
        assert fields["local_currency"] == fields["market_cap_currency"]

    def test_free_float_gets_the_same_pair(self):
        fields = fields_for()["JPSML"]

        assert fields["free_float_market_cap_local"] == pytest.approx(
            LOCAL_CAP["JPSML"] * FREE_FLOAT)
        assert fields["free_float_market_cap"] == pytest.approx(
            LOCAL_CAP["JPSML"] * FREE_FLOAT / JPY_PER_USD)

    def test_the_currency_codes_cover_both_halves(self):
        """Two numbers and one label is the ambiguity this closes."""
        fields = fields_for()["JPSML"]

        assert {fields["local_currency"], fields["market_cap_currency"]} == {
            "JPY", "USD"}


class TestTheRequestedCurrency:
    """It names what the converted half is in. It is never inferred."""

    def test_a_non_usd_currency_moves_only_the_converted_column(self):
        fields = fields_for("JPY", pairs=("JPYUSD", "USDJPY"))["USBIG"]

        assert fields["market_cap"] == pytest.approx(
            LOCAL_CAP["USBIG"] * JPY_PER_USD)
        assert fields["market_cap_local"] == pytest.approx(LOCAL_CAP["USBIG"])
        assert fields["local_currency"] == "USD"
        assert fields["market_cap_currency"] == "JPY"

    def test_the_local_figure_is_the_same_under_every_currency(self):
        """It is a property of the company, so nothing the request says may
        move it."""
        pairs = ("JPYUSD", "USDJPY")
        locals_by_currency = {
            currency: fields_for(currency, pairs)["JPSML"]["market_cap_local"]
            for currency in (None, "USD", "JPY")}

        assert len(set(locals_by_currency.values())) == 1

    def test_naming_the_names_own_currency_needs_no_rate(self):
        """A yen cap in yen is the local number, not a conversion that has to
        be looked up."""
        fields = fields_for("JPY", pairs=())["JPSML"]

        assert fields["market_cap"] == pytest.approx(LOCAL_CAP["JPSML"])
        assert fields["market_cap_currency"] == "JPY"

    def test_it_is_case_insensitive(self,
                                    client):
        lower = client.get("/data/reference", headers=HEADERS,
                           params={"identifiers": "JPSML", "fields": MONEY,
                                   "currency": "jpy"}).json()
        upper = client.get("/data/reference", headers=HEADERS,
                           params={"identifiers": "JPSML", "fields": MONEY,
                                   "currency": "JPY"}).json()

        assert lower == upper

    def test_something_that_is_not_a_currency_is_refused(self,
                                                         client):
        """Nulling every converted figure for a typo would look like missing
        FX data rather than a bad request."""
        response = client.get("/data/reference", headers=HEADERS,
                              params={"identifiers": "JPSML", "fields": MONEY,
                                      "currency": "dollars"})

        assert response.status_code in (400, 422)
        assert "currency" in str(response.json())


class TestAMissingRate:
    """The converted half goes null; the local half still reports."""

    def test_the_converted_figure_is_null(self):
        fields = fields_for(pairs=())["JPSML"]

        assert fields["market_cap"] is None
        assert fields["free_float_market_cap"] is None

    def test_the_local_figure_still_reports(self):
        """The server holds the price and the share count. Nulling their
        product because an FX pair is absent hides a number it has."""
        fields = fields_for(pairs=())["JPSML"]

        assert fields["market_cap_local"] == pytest.approx(LOCAL_CAP["JPSML"])
        assert fields["free_float_market_cap_local"] == pytest.approx(
            LOCAL_CAP["JPSML"] * FREE_FLOAT)
        assert fields["local_currency"] == "JPY"

    def test_the_convertible_name_is_untouched(self):
        fields = fields_for(pairs=())["USBIG"]

        assert fields["market_cap"] == pytest.approx(LOCAL_CAP["USBIG"])

    def test_a_name_with_no_prices_still_names_the_request_currency(self):
        """The converted currency is a property of the request, so it is
        answerable for a row that has no numbers at all."""
        fetcher = build_fetcher()
        entries = build_entries(fetcher, ["USBIG", "NOSUCH"], END, [MONEY],
                                "JPY")
        fields = entries[1].fields

        assert fields["market_cap"] is None
        assert fields["market_cap_local"] is None
        assert fields["local_currency"] is None
        assert fields["market_cap_currency"] == "JPY"


class TestExistingCallersDoNotMove:
    """Additive, checked rather than asserted."""

    def test_a_request_naming_no_currency_still_reports_usd(self):
        fields = fields_for()["JPSML"]

        assert fields["market_cap_currency"] == "USD"
        assert fields["market_cap"] == pytest.approx(
            LOCAL_CAP["JPSML"] / JPY_PER_USD)

    def test_the_added_keys_are_the_documented_companions(self):
        """A client reading the payload by key sees exactly what it did, plus
        companions it can ignore.

        Five now rather than three: BN-210 added `priced_from` and
        `price_is_stale`, which say how old the money fields are. Additive in
        the same way as the currency pair — nothing a caller already read
        moved, and the pin is kept literal so the next addition is a decision
        rather than a surprise.
        """
        before = {"market_cap", "free_float_market_cap", "market_cap_currency"}

        assert derived_keys(fields_for()["JPSML"]) - before == {
            "market_cap_local", "free_float_market_cap_local",
            "local_currency", "priced_from", "price_is_stale"}

    def test_both_endpoints_still_agree(self,
                                        client):
        """A parameter one form honours and the other ignores is the drift
        BN-149 closed here once already."""
        params = {"fields": MONEY, "currency": "JPY"}
        single = client.get("/data/reference/JPSML", headers=HEADERS,
                            params=params).json()["fields"]
        batch = client.get("/data/reference", headers=HEADERS,
                           params={**params, "identifiers": "JPSML"}
                           ).json()["entries"][0]["fields"]

        assert single == batch


class TestTheSurfacesAgree:
    """A field one surface knows about and another does not is the drift this
    codebase keeps finding."""

    def test_the_requestable_derived_fields_are_unchanged(self):
        """The companions are published, not requested, so the field picker,
        the expression namespace and this endpoint still list one set."""
        assert set(DERIVED_COLUMNS) == set(DERIVED_FIELDS)

    def test_every_published_key_is_documented(self):
        published = derived_keys(fields_for()["JPSML"])

        assert published <= set(DERIVED_FIELDS) | set(COMPANION_FIELDS)

    def test_each_money_description_names_its_counterpart(self):
        """BN-181's rule: a figure says which currency it is in and what the
        other one is."""
        for converted, local in MONEY_FIELDS.items():
            assert local in DERIVED_FIELDS[converted]
            assert "local_currency" in DERIVED_FIELDS[converted]
            assert converted in COMPANION_FIELDS[local]

    def test_no_description_states_a_fixed_currency(self):
        """The unit is per-request now, so a description that interpolated one
        would be wrong for every caller who names another."""
        for converted in MONEY_FIELDS:
            description = DERIVED_FIELDS[converted]

            assert "currency" in description
            assert f"{DEFAULT_CURRENCY} by default" in description

    def test_a_companion_cannot_be_requested_on_its_own(self):
        """It would be a second way to ask the same question, and a name
        every other surface would then have to learn."""
        with pytest.raises(InvalidRuleError):
            build_entries(build_fetcher(), ["JPSML"], END,
                          ["market_cap_local"])
