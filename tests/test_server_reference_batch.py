# tests/test_server_reference_batch.py
"""BN-115: the batch reference endpoint and the ADV it derives.

The single-name endpoint made a 512-member universe table cost 512 requests,
so the client truncated detail at sixty rows and dropped ADV entirely. What is
under test here is mostly the *shape* of the answer — order, misses, and which
fields come back — because those are what decide whether the client can delete
its truncation code, and none of them are visible from a single-name response.
"""
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from beacon.analysis.liquidity import average_daily_volume
from beacon.data.base import MarketData, ReferenceData
from beacon.data.fetcher import DataFetcher
from beacon.server import ServerConfig, create_app
from beacon.server.reference import MAX_BATCH, parse_identifiers, parse_list

TOKEN = "test-token-value"
ASSETS = ["AAA", "BBB", "CCC"]
DATES = pd.bdate_range("2025-01-02", periods=120)

# AAA trades ten times BBB, so an ADV that silently returned the wrong name's
# volume would be obvious rather than plausible.
VOLUMES = {"AAA": 10_000.0, "BBB": 1_000.0, "CCC": 5_000.0}


def build_fetcher(with_reference: bool = True) -> DataFetcher:
    """Three assets, flat volumes, and reference data for two of them."""
    market = pd.DataFrame([
        {"IDENTIFIER": asset, "DATE": date,
         "CLOSE": 100.0 + index, "VOLUME": VOLUMES[asset]}
        for asset in ASSETS
        for index, date in enumerate(DATES)])

    reference = None
    if with_reference:
        # CCC is deliberately absent: a name the market data prices but the
        # reference data has never heard of.
        reference = ReferenceData.from_dataframe(pd.DataFrame([
            {"IDENTIFIER": "AAA", "DATE_FROM": "2020-01-01",
             "NAME": "Alpha Corp", "SECTOR": "Technology", "CURRENCY": "USD"},
            {"IDENTIFIER": "BBB", "DATE_FROM": "2020-01-01",
             "NAME": "Beta Ltd", "SECTOR": "Utilities", "CURRENCY": "GBP"},
        ]))

    return DataFetcher(MarketData.from_dataframe(market), reference)


def auth() -> dict[str, str]:
    """Valid Authorization header."""
    return {"Authorization": f"Bearer {TOKEN}"}


@pytest.fixture
def client() -> TestClient:
    """Client over the three-asset fixture."""
    return TestClient(create_app(
        ServerConfig(auth_token=TOKEN, data_fetcher=build_fetcher())))


def fetch(client: TestClient,
          **params) -> dict:
    """GET the batch endpoint and return the decoded body."""
    response = client.get("/data/reference", params=params, headers=auth())

    assert response.status_code == 200, response.text

    return response.json()


class TestParsing:
    """Both spellings of a list parameter."""

    def test_comma_separated(self):
        assert parse_list(["A,B,C"]) == ["A", "B", "C"]

    def test_repeated(self):
        assert parse_list(["A", "B"]) == ["A", "B"]

    def test_mixed_and_padded(self):
        assert parse_list(["A, B", "C"]) == ["A", "B", "C"]

    def test_duplicates_collapse_keeping_first_position(self):
        """A repeat would otherwise produce two rows the client reconciles."""
        assert parse_list(["B,A,B"]) == ["B", "A"]

    def test_empty_entries_are_dropped(self):
        assert parse_list(["A,,B", ""]) == ["A", "B"]

    def test_nothing_is_an_empty_list(self):
        assert parse_list(None) == []


class TestRequestValidation:
    """What the endpoint refuses."""

    def test_no_identifiers_is_rejected(self, client):
        response = client.get("/data/reference", headers=auth())

        assert response.status_code == 422
        assert "at least one identifier" in response.text

    def test_too_many_identifiers_is_rejected(self, client):
        response = client.get("/data/reference",
                              params={"identifiers": ",".join(
                                  f"X{n}" for n in range(MAX_BATCH + 1))},
                              headers=auth())

        assert response.status_code == 422
        assert str(MAX_BATCH) in response.text

    def test_the_limit_itself_is_allowed(self):
        """An off-by-one here would reject the exact page size a client sizes
        its requests to."""
        assert len(parse_identifiers([",".join(f"X{n}"
                                               for n in range(MAX_BATCH))])) == MAX_BATCH

    def test_an_unknown_column_is_rejected_rather_than_silently_empty(self, client):
        """An absent column renders as an empty table row, which reads as
        missing data rather than as a misspelled request."""
        response = client.get("/data/reference",
                              params={"identifiers": "AAA", "fields": "SEKTOR"},
                              headers=auth())

        assert response.status_code == 422
        assert "SEKTOR" in response.text
        assert "SECTOR" in response.text, "the error should list what is available"

    def test_a_server_without_data_says_so(self):
        client = TestClient(create_app(ServerConfig(auth_token=TOKEN)))
        response = client.get("/data/reference",
                              params={"identifiers": "AAA"}, headers=auth())

        assert response.status_code == 500
        assert response.json()["error"]["code"] == "CONFIGURATION_ERROR"


class TestBatchShape:
    """Order, coverage and misses."""

    def test_one_entry_per_requested_identifier(self, client):
        body = fetch(client, identifiers="AAA,BBB")

        assert len(body["entries"]) == 2

    def test_entries_follow_the_request_order(self, client):
        """Not the store's order: a table renders straight down the list it
        asked for, and re-sorting is the client's problem to not have."""
        body = fetch(client, identifiers="BBB,AAA")

        assert [entry["identifier"] for entry in body["entries"]] == ["BBB", "AAA"]

    def test_an_unknown_identifier_does_not_fail_the_batch(self, client):
        body = fetch(client, identifiers="AAA,NOPE,BBB")
        entries = {entry["identifier"]: entry for entry in body["entries"]}

        assert len(body["entries"]) == 3
        assert entries["NOPE"]["found"] is False
        assert entries["NOPE"]["fields"] == {}
        assert entries["AAA"]["found"] is True
        assert entries["BBB"]["found"] is True

    def test_a_priced_name_with_no_reference_row_is_not_found(self, client):
        """CCC has market data and no reference data. Reporting it as found
        would put a row with no name into the table."""
        body = fetch(client, identifiers="CCC")

        assert body["entries"][0]["found"] is False

    def test_a_batch_that_matches_nothing_is_still_a_200(self, client):
        """A successful answer to a question about names this dataset does not
        carry — the per-entry flag already says so for each."""
        body = fetch(client, identifiers="NOPE,ALSONOPE")

        assert all(not entry["found"] for entry in body["entries"])

    def test_duplicates_collapse_to_one_entry(self, client):
        body = fetch(client, identifiers="AAA,AAA")

        assert len(body["entries"]) == 1

    def test_it_serves_a_full_universe_in_one_request(self):
        """The acceptance criterion: the 512-member table fills in one call,
        so DETAIL_LIMIT and its footnote can be deleted."""
        from beacon.synthetic import SyntheticConfig, generate

        dataset = generate(SyntheticConfig(assets=512, start="2023-01-02",
                                           end="2024-06-28", seed=4))
        client = TestClient(create_app(
            ServerConfig(auth_token=TOKEN, data_fetcher=dataset.fetcher())))

        names = list(dataset.universe.index)
        body = fetch(client, identifiers=",".join(names),
                     fields="NAME,SECTOR,adv_3m")

        assert [entry["identifier"] for entry in body["entries"]] == names
        assert all(entry["found"] for entry in body["entries"])

        # Present for every name, and positive for the ones that were trading
        # in the window. Since BN-130 a universe contains names that delisted
        # partway through, and those have no volume left to average -- so the
        # field is null for them rather than missing.
        volumes = [entry["fields"]["adv_3m"] for entry in body["entries"]]

        assert all("adv_3m" in entry["fields"] for entry in body["entries"])
        assert all(value is None or value > 0 for value in volumes)
        assert sum(value is not None for value in volumes) > len(names) // 2


class TestFields:
    """Which columns come back."""

    def test_all_stored_columns_by_default(self, client):
        body = fetch(client, identifiers="AAA")
        fields = body["entries"][0]["fields"]

        assert {"NAME", "SECTOR", "CURRENCY"} <= set(fields)

    def test_no_derived_field_by_default(self, client):
        """Computing ADV means slicing price history for every identifier in
        the batch — the endpoint's whole cost, and nobody should pay it
        without asking."""
        body = fetch(client, identifiers="AAA")

        assert "adv_3m" not in body["entries"][0]["fields"]

    def test_a_field_subset_is_honoured(self, client):
        body = fetch(client, identifiers="AAA", fields="NAME")

        assert set(body["entries"][0]["fields"]) == {"NAME"}

    def test_derived_fields_sit_beside_stored_ones(self, client):
        """One mapping, so a client reads what it asked to display in one
        place rather than branching on where a value came from."""
        body = fetch(client, identifiers="AAA", fields="NAME,adv_3m")

        assert set(body["entries"][0]["fields"]) == {"NAME", "adv_3m"}

    def test_timestamps_serialise_as_strings(self, client):
        body = fetch(client, identifiers="AAA")

        assert isinstance(body["entries"][0]["fields"]["DATE_FROM"], str)

    def test_an_open_validity_window_is_null_not_nan(self, client):
        """NaN is not JSON, and a client should never have to recognise a
        float that means absent."""
        body = fetch(client, identifiers="AAA")

        assert body["entries"][0]["fields"]["DATE_TO"] is None

    def test_the_as_of_date_is_echoed(self, client):
        body = fetch(client, identifiers="AAA", date="2025-03-03")

        assert body["as_of"] == "2025-03-03"

    def test_a_date_before_validity_finds_nothing(self, client):
        body = fetch(client, identifiers="AAA", date="2019-01-01")

        assert body["entries"][0]["found"] is False


class TestAverageDailyVolume:
    """The derived field, and the window it is taken over."""

    def test_it_averages_the_trailing_window(self, client):
        body = fetch(client, identifiers="AAA,BBB", fields="adv_3m")
        values = {entry["identifier"]: entry["fields"]["adv_3m"]
                  for entry in body["entries"]}

        assert values["AAA"] == pytest.approx(VOLUMES["AAA"])
        assert values["BBB"] == pytest.approx(VOLUMES["BBB"])

    def test_the_window_is_three_calendar_months(self):
        """Not 63 trading days. The two differ by several days across a
        quarter, and quoting one as the other makes two vendors disagree for
        no reason anybody can see."""
        dates = pd.bdate_range("2025-01-01", periods=200)
        frame = pd.DataFrame({
            "IDENTIFIER": "AAA", "DATE": dates,
            # 1.0 inside the last three months, 0.0 before.
            "VOLUME": [1.0 if date > pd.Timestamp("2025-06-30") else 0.0
                       for date in dates]}).set_index(["IDENTIFIER", "DATE"])

        result = average_daily_volume(frame, pd.Timestamp("2025-09-30"))

        assert result["AAA"] == pytest.approx(1.0)

    def test_the_window_is_half_open(self):
        """A day exactly three months old has rolled out, matching the
        trailing-twelve-month convention corporate actions already uses."""
        boundary = pd.Timestamp("2025-06-30")
        frame = pd.DataFrame({
            "IDENTIFIER": ["AAA", "AAA"],
            "DATE": [boundary, pd.Timestamp("2025-08-01")],
            "VOLUME": [500.0, 100.0]}).set_index(["IDENTIFIER", "DATE"])

        result = average_daily_volume(frame, pd.Timestamp("2025-09-30"))

        assert result["AAA"] == pytest.approx(100.0)

    def test_an_empty_frame_yields_nothing(self):
        assert average_daily_volume(pd.DataFrame(),
                                    pd.Timestamp("2025-09-30")).empty

    def test_data_entirely_outside_the_window_yields_nothing(self, caplog):
        """Distinct from an empty frame: there is volume, just none recent
        enough to answer the question that was asked."""
        import logging

        frame = pd.DataFrame({
            "IDENTIFIER": ["AAA", "AAA"],
            "DATE": [pd.Timestamp("2024-01-02"), pd.Timestamp("2024-02-01")],
            "VOLUME": [900.0, 800.0]}).set_index(["IDENTIFIER", "DATE"])

        with caplog.at_level(logging.WARNING):
            result = average_daily_volume(frame, pd.Timestamp("2025-09-30"))

        assert result.empty
        assert "no adv could be computed" in caplog.text.lower()

    def test_a_dataset_without_volume_yields_nothing(self):
        """An absent column is a property of the dataset, not a failed
        request, so it produces no answer rather than an error."""
        frame = pd.DataFrame({
            "IDENTIFIER": ["AAA"], "DATE": [pd.Timestamp("2025-09-01")],
            "CLOSE": [100.0]}).set_index(["IDENTIFIER", "DATE"])

        assert average_daily_volume(frame, pd.Timestamp("2025-09-30")).empty

    def test_a_name_with_no_volume_in_the_window_reports_null(self):
        """Null rather than zero, and present rather than absent.

        Zero would be a claim that it traded and nobody bought, and a
        liquidity screen would exclude it for the wrong reason. Omitting the
        key -- which this asserted until BN-130 -- was no better: the entry
        reported `found: true` and then silently lacked a field the caller had
        named, so a client had to defend with a membership test instead of a
        null check. Delistings made that case ordinary rather than exotic.
        """
        client = TestClient(create_app(ServerConfig(
            auth_token=TOKEN, data_fetcher=build_fetcher())))

        body = fetch(client, identifiers="AAA", fields="adv_3m",
                     date="2020-01-01")
        fields = body["entries"][0]["fields"]

        assert "adv_3m" in fields
        assert fields["adv_3m"] is None

    def test_it_is_computed_without_a_dataset_wide_scan(self, client):
        """The endpoint exists so the client stops fanning out; fanning out
        inside the server would only move the cost. One slice covers the
        batch, so asking for two names costs the same as asking for one."""
        both = fetch(client, identifiers="AAA,BBB", fields="adv_3m")
        single = fetch(client, identifiers="AAA", fields="adv_3m")

        assert (both["entries"][0]["fields"]["adv_3m"]
                == single["entries"][0]["fields"]["adv_3m"])


class TestSingleNameEndpointIsUnchanged:
    """The batch route must not have shadowed the one it complements."""

    def test_the_single_name_route_still_answers(self, client):
        response = client.get("/data/reference/AAA", headers=auth())

        assert response.status_code == 200
        assert response.json()["fields"]["NAME"] == "Alpha Corp"

    def test_it_still_404s_on_an_unknown_name(self, client):
        """Unlike the batch form, which reports the miss inline — the two
        answer different questions and should keep behaving differently."""
        response = client.get("/data/reference/NOPE", headers=auth())

        assert response.status_code == 404
