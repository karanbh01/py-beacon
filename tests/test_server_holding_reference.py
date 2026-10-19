# tests/test_server_holding_reference.py
"""BN-325: reference columns on the rows that name a holding, as of each
row's date.

Two names through a quarterly-rebalanced backtest from October 2023 to March
2024. AAA moves from Industrials to Technology on 2024-01-02, so a row dated
before then must say Industrials and one after Technology.
"""
import copy

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from beacon.data.base import MarketData, ReferenceData
from beacon.data.fetcher import DataFetcher
from beacon.server import ServerConfig, create_app
from beacon.server.jobs import SUCCEEDED
from conftest import submit_backtest

TOKEN = "test-token-value"
START, END = "2023-10-02", "2024-03-28"
DAYS = pd.bdate_range(START, END)
MOVED = "2024-01-02"
GROWTH = {"AAA": 1.30, "BBB": 0.85}

INDEX = {
    "id": "BT",
    "name": "Backtest Index",
    "base_date": START,
    "base_value": 1000.0,
    "currency": "USD",
    "calendar": "XNYS",
    "rebalancing_frequency": "QUARTERLY",
    "universe": {"universe_id": None, "identifiers": list(GROWTH)},
    "pipeline": {
        "selection": [],
        "weighting": {"id": "weighting", "scheme": "EqualWeighted", "params": {},
                      "max_weight": None},
        "treatment": {"corporate_actions": "ADJUST_DIVISOR"},
    },
}


def build_fetcher() -> DataFetcher:
    """Two names on geometric paths, AAA changing sector on MOVED."""
    span = len(DAYS) - 1
    rows = [{"IDENTIFIER": name, "DATE": day,
             "CLOSE": 100.0 * growth ** (position / span) * (1 + 0.01 * (position % 3)),
             "VOLUME": 1_000_000, "SHARES_OUTSTANDING": 1_000}
            for name, growth in GROWTH.items() for position, day in enumerate(DAYS)]
    reference = pd.DataFrame([
        {"IDENTIFIER": "AAA", "NAME": "Alpha", "CURRENCY": "USD", "EXCHANGE": "NYSE",
         "DATE_FROM": "2020-01-01", "DATE_TO": "2024-01-01", "SECTOR": "Industrials",
         "COUNTRY_DOMICILE": "US"},
        {"IDENTIFIER": "AAA", "NAME": "Alpha", "CURRENCY": "USD", "EXCHANGE": "NYSE",
         "DATE_FROM": MOVED, "DATE_TO": None, "SECTOR": "Technology",
         "COUNTRY_DOMICILE": "US"},
        {"IDENTIFIER": "BBB", "NAME": "Beta", "CURRENCY": "USD", "EXCHANGE": "NYSE",
         "DATE_FROM": "2020-01-01", "DATE_TO": None, "SECTOR": "Energy",
         "COUNTRY_DOMICILE": "IE"}])

    return DataFetcher(MarketData.from_dataframe(pd.DataFrame(rows)),
                       ReferenceData.from_dataframe(reference))


def auth() -> dict[str, str]:
    """Valid Authorization header."""
    return {"Authorization": f"Bearer {TOKEN}"}


def finished(client,
             response) -> dict:
    """The job a submission started, once it has run."""
    assert response.status_code == 202, response.text
    client.portal.call(client.app.state.jobs.drain)
    job = client.get(f"/jobs/{response.json()['job_id']}", headers=auth()).json()
    assert job["status"] == SUCCEEDED, job

    return job


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    """The index, a backtest of it, and a solved optimisation against it."""
    config = ServerConfig(auth_token=TOKEN, data_fetcher=build_fetcher(),
                          storage_root=tmp_path_factory.mktemp("reference"))

    with TestClient(create_app(config), raise_server_exceptions=False) as entered:
        assert entered.post("/indices", json=copy.deepcopy(INDEX),
                            headers=auth()).status_code == 200
        finished(entered, submit_backtest(entered, "BT", {"start": START, "end": END}, auth()))
        assert entered.put("/optimise/constraint-sets/loose", json={
            "id": "loose", "name": "Loose", "constraints": [
                {"id": "invested", "type": "FullInvestment", "params": {"target": 1.0}},
                {"id": "cap", "type": "PositionBounds",
                 "params": {"minimum": 0.0, "maximum": 0.8}}]},
            headers=auth()).status_code == 200
        yield entered


@pytest.fixture(scope="module")
def run_id(client) -> str:
    """A finished optimisation against the index, as it stood at the end."""
    job = finished(client, client.post("/optimise/runs",
                                       json={"index_id": "BT", "constraint_set_id": "loose"},
                                       headers=auth()))

    return job["result"]["run_id"]


def get(client,
        path: str,
        **params) -> dict:
    """A successful GET's body."""
    response = client.get(path, params=params, headers=auth())
    assert response.status_code == 200, response.text

    return response.json()


class TestTheWeights:

    def test_a_row_carries_the_default_columns(self,
                                               client):
        rows = {row["identifier"]: row for row in get(client, "/strategies/BT/weights",
                                                      asof="2023-11-01")["rows"]}

        assert rows["BBB"]["reference"] == {"name": "Beta", "asset_class": "Equity",
                                            "sector": "Energy", "country_domicile": "IE"}

    def test_a_row_is_read_as_of_the_date_asked_about(self,
                                                      client):
        before = get(client, "/strategies/BT/weights", asof="2023-11-01",
                     reference_fields="sector")
        after = get(client, "/strategies/BT/weights", asof="2024-02-01",
                    reference_fields="sector")

        sector = {row["identifier"]: row["reference"]["sector"] for row in before["rows"]}
        assert sector["AAA"] == "Industrials"
        sector = {row["identifier"]: row["reference"]["sector"] for row in after["rows"]}
        assert sector["AAA"] == "Technology"

    def test_an_empty_choice_puts_none_on_the_rows(self,
                                                   client):
        rows = get(client, "/strategies/BT/weights", reference_fields="")["rows"]

        assert all(row["reference"] is None for row in rows)

    def test_an_unknown_column_is_refused(self,
                                          client):
        response = client.get("/strategies/BT/weights", params={"reference_fields": "colour"},
                              headers=auth())

        assert response.status_code == 422
        assert "colour" in response.json()["error"]["message"]


class TestTheRun:

    def test_each_rebalance_is_read_as_of_its_date(self,
                                                   client):
        """A sector-grouped weight history uses the classification of its time."""
        snapshots = get(client, "/strategies/BT/result", reference_fields="sector")["rebalances"]
        by_date = {snapshot["date"]: snapshot["reference"]["AAA"]["sector"]
                   for snapshot in snapshots}

        assert by_date[min(by_date)] == "Industrials"
        assert by_date[max(by_date)] == "Technology"

    def test_attribution_is_read_as_of_the_window_s_end(self,
                                                        client):
        rows = get(client, "/strategies/BT/attribution", reference_fields="sector,name")[
            "contributions"]

        assert {row["asset_id"]: row["reference"]["sector"] for row in rows} == {
            "AAA": "Technology", "BBB": "Energy"}


class TestTheRecord:

    def test_a_weights_panel_carries_each_name_s_windows(self,
                                                        client):
        """A daily panel reads any of its days from the windows over its span."""
        record = get(client, "/strategies/BT/record", reference_fields="sector")
        windows = record["index"]["target"]["reference_windows"]["AAA"]

        assert [window["reference"]["sector"] for window in windows] == ["Industrials",
                                                                        "Technology"]
        assert windows[0]["date_to"] == "2024-01-01" and windows[1]["date_to"] is None

    def test_its_rebalances_are_read_as_of_their_dates(self,
                                                       client):
        snapshots = get(client, "/strategies/BT/record", reference_fields="sector")["index"][
            "target"]["rebalances"]

        assert snapshots[0]["reference"]["AAA"]["sector"] == "Industrials"


class TestTheOptimiser:

    def test_a_run_s_rows_carry_their_columns(self,
                                              client,
                                              run_id):
        rows = get(client, f"/optimise/runs/{run_id}", reference_fields="sector")["weights"]

        assert {row["asset_id"]: row["reference"]["sector"] for row in rows} == {
            "AAA": "Technology", "BBB": "Energy"}

    def test_the_frontier_carries_a_map_by_name(self,
                                                client,
                                                run_id):
        frontier = get(client, f"/optimise/runs/{run_id}/frontier", reference_fields="name")

        assert frontier["reference"] == {"AAA": {"name": "Alpha"}, "BBB": {"name": "Beta"}}


class TestThePreview:

    def test_each_asset_is_read_as_of_the_preview_date(self,
                                                       client):
        response = client.post("/indices/BT/preview", params={"reference_fields": "sector"},
                               json={"as_of": "2023-11-01"}, headers=auth())
        assert response.status_code == 200, response.text

        assets = {asset["identifier"]: asset["reference"]["sector"]
                  for asset in response.json()["assets"]}
        assert assets["AAA"] == "Industrials"


class TestTheCatalogue:

    def test_a_reference_field_carries_a_label(self,
                                               client):
        fields = {entry["path"]: entry for entry in get(client, "/data/fields")["fields"]}

        assert fields["reference.country_domicile"]["label"] == "Country of domicile"
        assert fields["reference.sector"]["label"] == "Sector"


class TestLabels:
    """BN-327: acronyms stay in capitals wherever they fall in a name."""

    @pytest.mark.parametrize(("name", "label"), [
        ("gics_industry", "GICS industry"), ("isin", "ISIN"), ("ipo_date", "IPO date"),
        ("cusip", "CUSIP"), ("sector", "Sector"), ("market_cap_usd", "Market cap USD"),
        ("country_domicile", "Country of domicile")])
    def test_a_name_reads_as_a_label(self,
                                     name,
                                     label):
        from beacon.expressions.catalogue import label_of

        assert label_of(name) == label
