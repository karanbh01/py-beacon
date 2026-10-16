# tests/test_server_strategies.py
"""Contract tests for the /strategies router (BN-312)."""
import copy

import pytest
from fastapi.testclient import TestClient

from beacon.server import ServerConfig, create_app

TOKEN = "test-token-value"

INDEX_ID = "GROWTH2"
CHILD_ID = "GROWTH2_OPT"

# A market-cap-weighted index over two names: a strategy needs one stored.
INDEX = {
    "id": INDEX_ID,
    "name": "Growth two",
    "base_date": "2024-01-02",
    "base_value": 1000.0,
    "currency": "USD",
    "calendar": "XNYS",
    "rebalancing_frequency": "QUARTERLY",
    "universe": {"universe_id": None, "identifiers": ["AAA", "BBB"]},
    "pipeline": {
        "selection": [],
        "weighting": {"id": "weighting", "scheme": "MarketCapWeighted",
                      "params": {}, "max_weight": None},
        "treatment": {"corporate_actions": "ADJUST_DIVISOR"},
    },
}

# Index tracking through a sampled replication, with a cap and a cash buffer.
TRACKER = {
    "id": "tracker",
    "name": "Sampled tracker",
    "description": "Holds the index through its largest names.",
    "index_id": INDEX_ID,
    "strategy": {"type": "index_tracking",
                 "replication": {"type": "SampledReplication", "params": {"holdings": 2}}},
    "implementation": {"cash_buffer": 0.01,
                       "caps": [{"type": "WeightCap", "params": {"max_weight": 0.5}}]},
}


def auth() -> dict[str, str]:
    """Valid Authorization header."""
    return {"Authorization": f"Bearer {TOKEN}"}


def tracker(**overrides) -> dict:
    """A fresh copy of the tracker strategy, with any fields replaced."""
    document = copy.deepcopy(TRACKER)
    document.update(overrides)

    return document


def findings_from(response) -> list[dict]:
    """The findings a refused save carries."""
    return response.json()["error"]["detail"]["findings"]


@pytest.fixture
def client(tmp_path) -> TestClient:
    """A client holding the index, with isolated storage and no data."""
    config = ServerConfig(auth_token=TOKEN, storage_root=tmp_path)
    entered = TestClient(create_app(config), raise_server_exceptions=False)
    created = entered.post("/indices", json=copy.deepcopy(INDEX), headers=auth())
    assert created.status_code == 200, created.text

    return entered


class TestSavingAndReading:

    def test_a_created_strategy_reads_back_as_saved(self,
                                                    client):
        created = client.post("/strategies", json=tracker(), headers=auth())

        assert created.status_code == 200, created.text
        assert created.json()["findings"] == []

        read = client.get("/strategies/tracker", headers=auth()).json()

        assert read["name"] == "Sampled tracker"
        assert read["index_id"] == INDEX_ID
        assert read["strategy"]["replication"]["params"] == {"holdings": 2}
        assert read["implementation"]["cash_buffer"] == 0.01
        assert read["problem"] is None

    def test_the_plain_index_strategy_needs_no_strategy_field(self,
                                                              client):
        """Holding the index as defined is a strategy too, and the default."""
        body = {"id": "plain", "name": "The index", "index_id": INDEX_ID}

        assert client.post("/strategies", json=body, headers=auth()).status_code == 200

        read = client.get("/strategies/plain", headers=auth()).json()

        assert read["strategy"]["type"] == "index"
        assert read["implementation"] is None

    def test_the_listing_carries_every_strategy_and_the_skip_counts(self,
                                                                     client):
        client.post("/strategies", json=tracker(), headers=auth())
        client.post("/strategies", json=tracker(id="second", name="Second"), headers=auth())

        listing = client.get("/strategies", headers=auth()).json()

        assert sorted(row["id"] for row in listing["strategies"]) == ["second", "tracker"]
        assert listing["skipped"] == 0

    def test_creating_over_a_taken_id_is_refused(self,
                                                 client):
        client.post("/strategies", json=tracker(), headers=auth())

        again = client.post("/strategies", json=tracker(name="Other"), headers=auth())

        assert again.status_code == 409
        assert "PUT /strategies/tracker" in again.json()["error"]["message"]
        assert client.get("/strategies/tracker", headers=auth()).json()["name"] == \
            "Sampled tracker"

    def test_put_replaces_a_strategy(self,
                                     client):
        client.post("/strategies", json=tracker(), headers=auth())

        replaced = client.put("/strategies/tracker", json=tracker(name="Renamed"),
                              headers=auth())

        assert replaced.status_code == 200
        assert client.get("/strategies/tracker", headers=auth()).json()["name"] == "Renamed"

    def test_put_creates_a_strategy_not_yet_stored(self,
                                                   client):
        assert client.put("/strategies/tracker", json=tracker(),
                          headers=auth()).status_code == 200

    def test_the_url_owns_the_id(self,
                                 client):
        response = client.put("/strategies/other", json=tracker(), headers=auth())

        assert response.status_code == 422
        assert client.get("/strategies/other", headers=auth()).status_code == 404

    def test_a_bad_id_is_refused(self,
                                 client):
        assert client.post("/strategies", json=tracker(id="has space"),
                           headers=auth()).status_code == 422
        assert client.get("/strategies/has%20space", headers=auth()).status_code == 422

    def test_an_unknown_field_is_refused(self,
                                         client):
        """Vehicle, flows and the period belong to the backtest, not the
        strategy; a client sending one finds out at once."""
        response = client.post("/strategies", json=tracker(vehicle={"preset": "ucits_etf"}),
                               headers=auth())

        assert response.status_code == 422


class TestDeleting:

    def test_a_delete_says_what_it_removed(self,
                                           client):
        client.post("/strategies", json=tracker(), headers=auth())

        deleted = client.delete("/strategies/tracker", headers=auth())

        assert deleted.status_code == 200
        assert deleted.json() == {"strategy_id": "tracker"}
        assert client.get("/strategies/tracker", headers=auth()).status_code == 404

    def test_deleting_what_is_not_stored_is_not_found(self,
                                                      client):
        assert client.delete("/strategies/tracker", headers=auth()).status_code == 404

    def test_an_unreadable_strategy_can_still_be_deleted(self,
                                                         client):
        client.app.state.strategy_store.write("broken", {"id": "broken"})

        assert client.get("/strategies/broken", headers=auth()).status_code == 404
        assert client.delete("/strategies/broken", headers=auth()).status_code == 200


class TestValidation:

    def test_a_save_is_checked_as_a_backtest_checks_it(self,
                                                       client):
        """The findings sit at the paths a backtest request would give them."""
        body = tracker(strategy={"type": "index_tracking",
                                 "replication": {"type": "NoSuchReplication"}},
                       implementation={"caps": [{"type": "NoSuchCap"}]})

        response = client.post("/strategies", json=body, headers=auth())

        assert response.status_code == 422
        assert response.json()["error"]["code"] == "INVALID_RULE"
        paths = {finding["path"] for finding in findings_from(response)}
        assert {"strategy.replication.type", "implementation.caps.0.type"} <= paths
        assert client.get("/strategies/tracker", headers=auth()).status_code == 404

    def test_an_unknown_strategy_type_is_refused_at_its_path(self,
                                                             client):
        response = client.post("/strategies", json=tracker(strategy={"type": "momentum"}),
                               headers=auth())

        assert [finding["path"] for finding in findings_from(response)] == ["strategy.type"]

    def test_an_active_strategy_needs_a_signal_for_its_construction(self,
                                                                    client):
        response = client.post("/strategies", json=tracker(strategy={"type": "active"}),
                               headers=auth())

        assert "strategy.signal" in {finding["path"] for finding in findings_from(response)}

    def test_its_index_must_be_stored(self,
                                      client):
        response = client.post("/strategies", json=tracker(index_id="NOPE"), headers=auth())

        assert response.status_code == 422
        (finding,) = findings_from(response)
        assert finding["path"] == "index_id"
        assert finding["code"] == "index_not_found"

    def test_validate_reports_without_saving(self,
                                             client):
        good = client.post("/strategies/validate", json=tracker(), headers=auth()).json()
        bad = client.post("/strategies/validate", json=tracker(index_id="NOPE"),
                          headers=auth()).json()

        assert good == {"valid": True, "findings": []}
        assert bad["valid"] is False
        assert bad["findings"][0]["path"] == "index_id"
        assert client.get("/strategies/tracker", headers=auth()).status_code == 404


class TestTheIndexItHolds:

    def test_an_index_a_strategy_holds_cannot_be_deleted(self,
                                                         client):
        client.post("/strategies", json=tracker(), headers=auth())

        response = client.delete(f"/indices/{INDEX_ID}", headers=auth())

        assert response.status_code == 409
        error = response.json()["error"]
        assert error["code"] == "IN_USE"
        assert error["detail"]["used_by"] == ["tracker"]
        assert client.get(f"/indices/{INDEX_ID}", headers=auth()).status_code == 200

    def test_nor_can_one_whose_optimised_child_a_strategy_holds(self,
                                                                client):
        """The cascade would take the child, so a strategy holding the child
        refuses the parent's delete too."""
        derived = client.post(f"/indices/{INDEX_ID}/optimise",
                              json={"id": CHILD_ID, "name": "Optimised"}, headers=auth())
        assert derived.status_code == 200, derived.text
        client.post("/strategies", json=tracker(index_id=CHILD_ID), headers=auth())

        response = client.delete(f"/indices/{INDEX_ID}", headers=auth())

        assert response.status_code == 409
        assert response.json()["error"]["detail"]["used_by"] == ["tracker"]

    def test_an_index_no_strategy_holds_deletes_as_before(self,
                                                          client):
        client.post("/strategies", json=tracker(), headers=auth())
        client.delete("/strategies/tracker", headers=auth())

        assert client.delete(f"/indices/{INDEX_ID}", headers=auth()).status_code == 200

    def test_a_strategy_whose_index_has_gone_still_lists_with_a_problem(self,
                                                                        client):
        """Removed through a store the refusal never saw, or not copied to
        this machine."""
        client.post("/strategies", json=tracker(), headers=auth())
        client.app.state.index_store.delete(INDEX_ID)

        (listed,) = client.get("/strategies", headers=auth()).json()["strategies"]
        read = client.get("/strategies/tracker", headers=auth()).json()

        assert listed["problem"]["code"] == "INDEX_NOT_FOUND"
        assert read["problem"] == listed["problem"]
        assert INDEX_ID in read["problem"]["message"]

    def test_a_strategy_whose_index_cannot_be_read_says_so(self,
                                                           client):
        client.post("/strategies", json=tracker(), headers=auth())
        client.app.state.index_store.write(INDEX_ID, {"id": INDEX_ID})

        read = client.get("/strategies/tracker", headers=auth()).json()

        assert read["problem"]["code"] == "INDEX_UNREADABLE"

    def test_an_unreadable_strategy_is_skipped_and_counted(self,
                                                           client):
        client.post("/strategies", json=tracker(), headers=auth())
        client.app.state.strategy_store.write("broken", {"id": "broken"})

        listing = client.get("/strategies", headers=auth()).json()

        assert [row["id"] for row in listing["strategies"]] == ["tracker"]
        assert listing["skipped"] == 1
