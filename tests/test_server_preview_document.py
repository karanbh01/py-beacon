# tests/test_server_preview_document.py
"""BN-120: previewing a document that has not been saved.

`POST /indices/{id}/preview` reads the *stored* definition, so an editor
holding unsaved changes shows figures for the old rules with nothing on screen
to say they are stale. The test that matters is the one that proves the two
routes now disagree: same index, edited rule, different result — and the by-id
route still describing what is on disk.
"""
import copy

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from beacon.data.base import MarketData, ReferenceData
from beacon.data.fetcher import DataFetcher
from beacon.server import ServerConfig, create_app

TOKEN = "test-token-value"
AS_OF = "2025-01-02"
DATES = pd.bdate_range("2024-11-01", "2025-01-31")

# Caps are price x 1_000 shares: AAA 500k, BBB 300k, CCC 100k, DDD 60k,
# EEE 30k, FFF 10k. A threshold between any two of those cuts a known set.
PRICES = {"AAA": 500.0, "BBB": 300.0, "CCC": 100.0,
          "DDD": 60.0, "EEE": 30.0, "FFF": 10.0}
SHARES = 1_000


def auth() -> dict[str, str]:
    return {"Authorization": f"Bearer {TOKEN}"}


def build_fetcher() -> DataFetcher:
    """Six names with spread market caps."""
    market = pd.DataFrame([
        {"IDENTIFIER": name, "DATE": date, "CLOSE": price,
         "VOLUME": 500_000, "SHARES_OUTSTANDING": SHARES}
        for name, price in PRICES.items()
        for date in DATES])
    reference = pd.DataFrame([
        {"IDENTIFIER": name, "DATE_FROM": "2020-01-01", "NAME": name,
         "CURRENCY": "USD", "EXCHANGE": "NYSE"}
        for name in PRICES])

    return DataFetcher(MarketData.from_dataframe(market),
                       ReferenceData.from_dataframe(reference))


def document(minimum: float = 50_000.0,
             **overrides) -> dict:
    """An index whose market-cap floor is the thing under edit."""
    payload = {
        "id": "DRAFT",
        "name": "Draft Index",
        "base_date": AS_OF,
        "base_value": 1000.0,
        "currency": "USD",
        "rebalancing_frequency": "QUARTERLY",
        "description": None,
        "universe": {"universe_id": None, "identifiers": list(PRICES)},
        "pipeline": {
            "selection": [
                {"id": "min-mcap", "type": "MarketCapRule",
                 "params": {"min_market_cap": minimum}},
            ],
            "weighting": {"id": "weighting", "scheme": "MarketCapWeighted",
                          "params": {}, "max_weight": None},
            "treatment": {"corporate_actions": "ADJUST_DIVISOR"},
        },
    }
    payload.update(overrides)

    return copy.deepcopy(payload)


@pytest.fixture
def client(tmp_path) -> TestClient:
    """A server holding the saved index, with a data source."""
    config = ServerConfig(auth_token=TOKEN,
                          data_fetcher=build_fetcher(),
                          storage_root=tmp_path)
    started = TestClient(create_app(config), raise_server_exceptions=False)

    created = started.post("/indices", json=document(), headers=auth())
    assert created.status_code == 200, created.text

    return started


def preview_document(client: TestClient,
                     payload: dict,
                     as_of: str | None = None):
    """POST the draft route."""
    body: dict = {"document": payload}
    if as_of is not None:
        body["as_of"] = as_of

    return client.post("/indices/preview", json=body, headers=auth())


def constituents(response) -> set[str]:
    """The names that survived, off a preview response."""
    payload = response.json()

    return {asset["identifier"] for asset in payload["assets"]
            if asset["included"]}


class TestDraftPreview:
    """Editing without saving."""

    def test_it_previews_what_was_sent(self, client):
        response = preview_document(client, document())

        assert response.status_code == 200, response.text
        assert constituents(response) == {"AAA", "BBB", "CCC", "DDD"}

    def test_an_unsaved_edit_changes_the_result(self, client):
        """The whole point. Raising the floor drops two more names, and the
        editor sees that without a save."""
        edited = preview_document(client, document(minimum=250_000.0))

        assert constituents(edited) == {"AAA", "BBB"}

    def test_the_stored_index_is_untouched_by_a_preview(self, client):
        """A preview must not be a write. If it were, an editor exploring a
        threshold would silently redefine the index."""
        preview_document(client, document(minimum=250_000.0))

        stored = client.get("/indices/DRAFT", headers=auth()).json()
        rule = stored["pipeline"]["selection"][0]

        assert rule["params"]["min_market_cap"] == 50_000.0

    def test_the_by_id_route_still_describes_the_saved_definition(self, client):
        """Both routes exist because they answer different questions."""
        preview_document(client, document(minimum=250_000.0))

        saved = client.post("/indices/DRAFT/preview", json={}, headers=auth())

        assert constituents(saved) == {"AAA", "BBB", "CCC", "DDD"}

    def test_the_two_routes_agree_on_an_unedited_document(self, client):
        """Same definition, same derivation path — otherwise a client could not
        trust that switching routes changed nothing."""
        draft = preview_document(client, document())
        saved = client.post("/indices/DRAFT/preview", json={}, headers=auth())

        assert draft.json()["assets"] == saved.json()["assets"]
        assert draft.json()["weights"] == saved.json()["weights"]

    def test_it_needs_no_saved_index_at_all(self, client):
        """A definition being drafted for the first time has no id in the
        store, and previewing it must not require one."""
        response = preview_document(client, document(id="NEVER-SAVED"))

        assert response.status_code == 200
        assert client.get("/indices/NEVER-SAVED",
                          headers=auth()).status_code == 404

    def test_as_of_is_honoured(self, client):
        response = preview_document(client, document(), as_of="2025-01-15")

        assert response.status_code == 200
        assert response.json()["as_of"] == "2025-01-15"

    def test_it_defaults_to_the_documents_base_date(self, client):
        response = preview_document(client, document())

        assert response.json()["as_of"] == AS_OF


class TestUniverseResolution:
    """A draft referencing a stored universe."""

    def test_a_referenced_universe_is_resolved(self, client):
        """The by-id route never needed this: saving resolves the reference, so
        a stored document already carries its identifiers. A draft has not been
        saved, so without resolving here it would preview as an empty index."""
        created = client.put("/universes/big",
                             json={"universe_id": "big",
                                   "name": "Big names",
                                   "identifiers": ["AAA", "BBB", "CCC"]},
                             headers=auth())
        assert created.status_code == 200, created.text

        response = preview_document(client, document(
            universe={"universe_id": "big", "identifiers": []}))

        assert response.status_code == 200, response.text
        assert constituents(response) == {"AAA", "BBB", "CCC"}

    def test_an_unknown_universe_is_reported(self, client):
        response = preview_document(client, document(
            universe={"universe_id": "nope", "identifiers": []}))

        assert response.status_code == 404


class TestValidation:
    """A draft is not known-good, unlike anything in the store."""

    def test_an_unknown_rule_type_comes_back_as_findings(self, client):
        """Not as a 500 from the derivation. The editor should be able to point
        at the offending row."""
        broken = document()
        broken["pipeline"]["selection"][0]["type"] = "NoSuchRule"

        response = preview_document(client, broken)

        assert response.status_code == 422

        detail = response.json()["error"]["detail"]
        assert any(finding["code"] == "UNKNOWN_RULE_TYPE"
                   for finding in detail["findings"])

    def test_an_unknown_parameter_is_reported(self, client):
        broken = document()
        broken["pipeline"]["selection"][0]["params"] = {"min_mcap": 1.0}

        response = preview_document(client, broken)

        assert response.status_code == 422

    def test_an_infeasible_cap_is_reported(self, client):
        """A cap of 5% over six names can distribute at most 30%, and finding
        that out mid-derivation is worse than being told."""
        broken = document()
        broken["pipeline"]["weighting"]["max_weight"] = 0.05

        response = preview_document(client, broken)

        assert response.status_code == 422

    def test_a_malformed_body_is_a_422(self, client):
        response = client.post("/indices/preview",
                               json={"as_of": AS_OF}, headers=auth())

        assert response.status_code == 422


class TestRouting:
    """The literal path must not be read as an index id."""

    def test_preview_is_not_treated_as_an_index_id(self, client):
        """`/indices/{index_id}` would happily match "preview"."""
        response = preview_document(client, document())

        assert response.status_code == 200

    def test_it_requires_authentication(self, client):
        response = client.post("/indices/preview",
                               json={"document": document()})

        assert response.status_code == 401

    def test_it_needs_a_data_source(self, tmp_path):
        """Preview evaluates real rules against real prices."""
        started = TestClient(create_app(ServerConfig(
            auth_token=TOKEN, storage_root=tmp_path)),
            raise_server_exceptions=False)

        response = preview_document(started, document())

        assert response.status_code == 500
        assert response.json()["error"]["code"] == "CONFIGURATION_ERROR"
