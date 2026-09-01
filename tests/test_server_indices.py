# tests/test_server_indices.py
"""Contract tests for the /indices and /universes routers."""
import copy

import pytest
from fastapi.testclient import TestClient

from beacon.index.constructor import IndexDefinition
from beacon.server import ServerConfig, create_app
from beacon.server.definitions import build_index_definition, validate_document
from beacon.server.schemas import IndexDocument

TOKEN = "test-token-value"

# A TECH10-shaped definition: ten names, a market-cap floor, a liquidity
# screen, market-cap weighting, quarterly rebalancing. Stands in for the
# mock named in the issue, which lives in the UI project rather than here.
TECH10 = {
    "id": "TECH10",
    "name": "Beacon Tech 10",
    "base_date": "2024-01-02",
    "base_value": 1000.0,
    "currency": "USD",
    "rebalancing_frequency": "QUARTERLY",
    "description": "Ten large-cap technology names, market-cap weighted.",
    "universe": {
        "universe_id": None,
        "identifiers": ["AAA", "BBB", "CCC", "DDD", "EEE",
                        "FFF", "GGG", "HHH", "III", "JJJ"],
    },
    "pipeline": {
        "selection": [
            {"id": "min-mcap",
             "type": "MarketCapRule",
             "params": {"min_market_cap": 1_000_000_000.0}},
            {"id": "liquidity",
             "type": "LiquidityRule",
             "params": {"min_avg_daily_volume": 100_000, "lookback_days": 60}},
        ],
        "weighting": {
            "id": "weighting",
            "scheme": "MarketCapWeighted",
            "params": {"use_free_float": True},
            "max_weight": 0.2,
        },
        "treatment": {"corporate_actions": "ADJUST_DIVISOR"},
    },
}


def auth() -> dict[str, str]:
    """Valid Authorization header."""
    return {"Authorization": f"Bearer {TOKEN}"}


def tech10() -> dict:
    """A fresh copy of the TECH10 definition."""
    return copy.deepcopy(TECH10)


@pytest.fixture
def client(tmp_path) -> TestClient:
    """Client with isolated document storage."""
    config = ServerConfig(auth_token=TOKEN, storage_root=tmp_path)
    return TestClient(create_app(config), raise_server_exceptions=False)


def findings_from(response) -> list[dict]:
    """Pull the findings out of a rejected save."""
    return response.json()["error"]["detail"]["findings"]


def as_stored(document: dict) -> dict:
    """A document as the schema defines it, defaults filled in.

    Round-trips are compared against this rather than against the literal that
    was posted. A definition that omits an optional field gets the default back,
    which is correct behaviour and not a difference worth failing on — and
    comparing to the literal would break on every additive field anyone ever
    adds, as BN-121's five did.
    """
    from beacon.server.schemas import IndexDocument

    return IndexDocument.model_validate(document).model_dump()


class TestRoundTrip:
    """The acceptance criterion: a TECH10 definition round-trips."""

    def test_post_then_get_returns_an_equal_document(self,
                                                     client):
        created = client.post("/indices", json=tech10(), headers=auth())

        assert created.status_code == 200, created.json()

        fetched = client.get("/indices/TECH10", headers=auth()).json()

        assert fetched == as_stored(tech10())

    def test_saved_definition_reports_no_findings(self,
                                                  client):
        body = client.post("/indices", json=tech10(), headers=auth()).json()

        assert body["findings"] == []
        assert body["index"]["id"] == "TECH10"

    def test_survives_a_process_restart(self,
                                        tmp_path):
        first = TestClient(create_app(ServerConfig(auth_token=TOKEN,
                                                   storage_root=tmp_path)))
        first.post("/indices", json=tech10(), headers=auth())

        second = TestClient(create_app(ServerConfig(auth_token=TOKEN,
                                                    storage_root=tmp_path)))

        assert (second.get("/indices/TECH10", headers=auth()).json()
                == as_stored(tech10()))

    def test_listing_includes_it(self,
                                 client):
        client.post("/indices", json=tech10(), headers=auth())

        body = client.get("/indices", headers=auth()).json()

        assert [index["id"] for index in body["indices"]] == ["TECH10"]

    def test_put_updates_in_place(self,
                                  client):
        client.post("/indices", json=tech10(), headers=auth())
        updated = tech10()
        updated["name"] = "Renamed"

        client.put("/indices/TECH10", json=updated, headers=auth())

        assert client.get("/indices/TECH10", headers=auth()).json()["name"] == "Renamed"

    def test_put_rejects_an_id_mismatch(self,
                                        client):
        response = client.put("/indices/OTHER", json=tech10(), headers=auth())

        assert response.status_code == 422
        assert "does not match" in response.json()["error"]["message"]

    def test_unknown_index_is_404(self,
                                  client):
        assert client.get("/indices/ABSENT", headers=auth()).status_code == 404


class TestValidationFindings:
    """Findings are addressable per rule, not a single blanket 422."""

    def test_unknown_rule_type_names_the_rule(self,
                                              client):
        body = tech10()
        body["pipeline"]["selection"][0]["type"] = "MomentumRule"

        response = client.post("/indices", json=body, headers=auth())

        assert response.status_code == 422
        finding = next(f for f in findings_from(response)
                       if f["code"] == "UNKNOWN_RULE_TYPE")
        assert finding["rule_id"] == "min-mcap"
        assert finding["path"] == "pipeline.selection[0]"

    def test_unknown_parameter_names_the_field(self,
                                               client):
        body = tech10()
        body["pipeline"]["selection"][0]["params"]["min_price"] = 5.0

        response = client.post("/indices", json=body, headers=auth())
        finding = next(f for f in findings_from(response)
                       if f["code"] == "UNKNOWN_PARAMETER")

        assert finding["rule_id"] == "min-mcap"
        assert finding["path"] == "pipeline.selection[0].params.min_price"

    def test_inverted_market_cap_range_is_reported(self,
                                                   client):
        body = tech10()
        body["pipeline"]["selection"][0]["params"] = {
            "min_market_cap": 10.0, "max_market_cap": 1.0}

        response = client.post("/indices", json=body, headers=auth())
        codes = {f["code"] for f in findings_from(response)}

        assert "INVALID_RANGE" in codes

    def test_non_positive_lookback_is_reported(self,
                                               client):
        body = tech10()
        body["pipeline"]["selection"][1]["params"]["lookback_days"] = 0

        response = client.post("/indices", json=body, headers=auth())
        finding = next(f for f in findings_from(response)
                       if f["code"] == "INVALID_VALUE")

        assert finding["rule_id"] == "liquidity"

    def test_every_problem_is_reported_at_once(self,
                                               client):
        """A user editing a pipeline needs all the errors, not the first."""
        body = tech10()
        body["pipeline"]["selection"][0]["type"] = "NopeRule"
        body["pipeline"]["selection"][1]["params"]["lookback_days"] = -1
        body["pipeline"]["weighting"]["scheme"] = "AlsoNope"
        body["base_value"] = 0.0

        response = client.post("/indices", json=body, headers=auth())
        findings = findings_from(response)

        assert {"UNKNOWN_RULE_TYPE", "UNKNOWN_SCHEME", "INVALID_VALUE"} <= {
            f["code"] for f in findings}

        # INVALID_VALUE arises twice here, from different places — the rule's
        # lookback and the index's base_value — so check the paths, not just
        # the codes, or the set would collapse them into one.
        paths = {f["path"] for f in findings}
        assert "base_value" in paths
        assert "pipeline.selection[1].params.lookback_days" in paths

    def test_duplicate_rule_ids_are_rejected(self,
                                             client):
        """Two rules sharing an id would make findings unaddressable."""
        body = tech10()
        body["pipeline"]["selection"][1]["id"] = "min-mcap"

        response = client.post("/indices", json=body, headers=auth())
        codes = {f["code"] for f in findings_from(response)}

        assert "DUPLICATE_RULE_ID" in codes

    def test_out_of_range_cap_is_rejected(self,
                                          client):
        body = tech10()
        body["pipeline"]["weighting"]["max_weight"] = 1.5

        response = client.post("/indices", json=body, headers=auth())
        finding = next(f for f in findings_from(response)
                       if f["code"] == "INVALID_CAP")

        assert finding["path"] == "pipeline.weighting.max_weight"

    def test_infeasible_cap_names_the_minimum(self,
                                              client):
        """A 5% cap across 10 names can reach only 50%; say so while editing."""
        body = tech10()
        body["pipeline"]["weighting"]["max_weight"] = 0.05

        response = client.post("/indices", json=body, headers=auth())
        finding = next(f for f in findings_from(response)
                       if f["code"] == "INFEASIBLE_CAP")

        assert "10.0000%" in finding["message"]

    def test_exactly_feasible_cap_warns_but_saves(self,
                                                  client):
        """1/n is feasible only if every member survives selection."""
        body = tech10()
        body["pipeline"]["weighting"]["max_weight"] = 0.1

        response = client.post("/indices", json=body, headers=auth())

        assert response.status_code == 200
        assert any(f["code"] == "TIGHT_CAP" for f in response.json()["findings"])

    def test_unsupported_frequency_is_reported(self,
                                               client):
        body = tech10()
        body["rebalancing_frequency"] = "DAILY"

        response = client.post("/indices", json=body, headers=auth())
        codes = {f["code"] for f in findings_from(response)}

        assert "UNSUPPORTED_FREQUENCY" in codes

    def test_empty_universe_is_reported(self,
                                        client):
        body = tech10()
        body["universe"]["identifiers"] = []

        response = client.post("/indices", json=body, headers=auth())
        codes = {f["code"] for f in findings_from(response)}

        assert "EMPTY_UNIVERSE" in codes

    def test_no_selection_rules_is_a_warning_not_an_error(self,
                                                          client):
        """A pass-all universe is legitimate, so it saves with a warning."""
        body = tech10()
        body["pipeline"]["selection"] = []

        response = client.post("/indices", json=body, headers=auth())

        assert response.status_code == 200
        assert [f["code"] for f in response.json()["findings"]] == ["NO_SELECTION_RULES"]

    def test_rejected_definition_is_not_stored(self,
                                               client):
        body = tech10()
        body["pipeline"]["weighting"]["scheme"] = "Nope"

        client.post("/indices", json=body, headers=auth())

        assert client.get("/indices/TECH10", headers=auth()).status_code == 404


class TestValidateEndpoint:

    def test_valid_definition_reports_valid(self,
                                            client):
        body = client.post("/indices/validate", json=tech10(), headers=auth()).json()

        assert body == {"valid": True, "findings": []}

    def test_invalid_definition_reports_findings_without_saving(self,
                                                                client):
        body = tech10()
        body["pipeline"]["weighting"]["scheme"] = "Nope"

        report = client.post("/indices/validate", json=body, headers=auth()).json()

        assert report["valid"] is False
        assert any(f["code"] == "UNKNOWN_SCHEME" for f in report["findings"])
        assert client.get("/indices/TECH10", headers=auth()).status_code == 404


class TestUniverses:

    def test_put_then_members(self,
                              client):
        client.put("/universes/tech",
                   json={"name": "Tech", "identifiers": ["AAA", "BBB"]},
                   headers=auth())

        body = client.get("/universes/tech/members", headers=auth()).json()

        assert body == {"universe_id": "tech", "identifiers": ["AAA", "BBB"]}

    def test_listing(self,
                     client):
        client.put("/universes/tech", json={"name": "Tech", "identifiers": []},
                   headers=auth())

        body = client.get("/universes", headers=auth()).json()

        assert [u["id"] for u in body["universes"]] == ["tech"]

    def test_unknown_universe_is_404(self,
                                     client):
        assert client.get("/universes/absent/members", headers=auth()).status_code == 404

    def test_index_resolves_a_referenced_universe(self,
                                                  client):
        """A definition may reference a universe instead of listing members."""
        members = ["AAA", "BBB", "CCC", "DDD", "EEE", "FFF"]
        client.put("/universes/tech",
                   json={"name": "Tech", "identifiers": members},
                   headers=auth())
        body = tech10()
        body["universe"] = {"universe_id": "tech", "identifiers": []}

        saved = client.post("/indices", json=body, headers=auth()).json()

        assert saved["index"]["universe"]["identifiers"] == members

    def test_cap_feasibility_is_checked_against_the_resolved_universe(self,
                                                                      client):
        """The cap must be judged on the members it will actually apply to."""
        client.put("/universes/tiny",
                   json={"name": "Tiny", "identifiers": ["AAA", "BBB"]},
                   headers=auth())
        body = tech10()   # max_weight 0.2, impossible across two names
        body["universe"] = {"universe_id": "tiny", "identifiers": []}

        response = client.post("/indices", json=body, headers=auth())
        codes = {f["code"] for f in findings_from(response)}

        assert "INFEASIBLE_CAP" in codes

    def test_reference_to_a_missing_universe_fails_on_save(self,
                                                           client):
        """Better to fail now than at calculation time."""
        body = tech10()
        body["universe"] = {"universe_id": "absent", "identifiers": []}

        response = client.post("/indices", json=body, headers=auth())

        assert response.status_code == 404

    def test_requires_authentication(self,
                                     client):
        assert client.get("/universes").status_code == 401


class TestMaterialisation:
    """A valid document must build the library object it describes."""

    def test_builds_an_index_definition(self):
        definition = build_index_definition(IndexDocument.model_validate(tech10()))

        assert isinstance(definition, IndexDefinition)
        assert definition.index_id == "TECH10"
        assert definition.rebalancing_frequency == "QUARTERLY"
        assert len(definition.eligibility_rules) == 2
        assert definition.weighting_scheme.scheme_name == "MarketCapWeighted"
        assert definition.universe_identifiers == TECH10["universe"]["identifiers"]

    def test_built_definition_produces_rebalance_dates(self):
        """The materialised object is usable, not merely constructible."""
        definition = build_index_definition(IndexDocument.model_validate(tech10()))

        dates = definition.get_rebalance_dates("2024-01-02", "2024-12-31")

        assert len(dates) == 4

    def test_validation_passes_for_the_document_we_materialise(self):
        assert validate_document(IndexDocument.model_validate(tech10())) == []


def _persist_result(registry,
                    kind,
                    result):
    """Write a succeeded result the way a finished job would.

    Straight into the persisted store, because that is the layer the cascade
    must clean: it is the one that survives restarts, and `latest_result`
    reads it rather than memory.
    """
    import uuid
    from datetime import UTC, datetime

    job_id = str(uuid.uuid4())
    registry._results.write(job_id, {
        "job_id": job_id,
        "kind": kind,
        "status": "succeeded",
        "progress": 1.0,
        "message": "done",
        "result": result,
        "error": None,
        "completed_at": datetime.now(UTC).isoformat(),
    })


class TestDelete:
    """BN-157: `DELETE /indices/{index_id}`, cascading its backtest results.

    Universes could be deleted; indices could not — same kind of document in
    the same kind of store, so the asymmetry was the whole request.
    """

    def _create(self,
                client,
                index_id="beacon-tech-10"):
        body = {**TECH10, "id": index_id, "name": index_id}
        response = client.post("/indices", headers=auth(), json=body)
        assert response.status_code in (200, 201), response.text
        return index_id

    def test_it_deletes(self,
                        client):
        index_id = self._create(client)

        assert client.delete(f"/indices/{index_id}",
                             headers=auth()).status_code == 204
        assert client.get(f"/indices/{index_id}",
                          headers=auth()).status_code == 404

    def test_a_missing_index_is_404(self,
                                    client):
        """The same answer the GET gives, not a silent 204 — deleting nothing
        should not read as having deleted something."""
        assert client.delete("/indices/never-existed",
                             headers=auth()).status_code == 404

    def test_it_requires_authentication(self,
                                        client):
        assert client.delete("/indices/anything").status_code == 401

    def test_backtest_results_cascade(self,
                                      client):
        """The decision beacon-ui filed upward, answered: results go with
        the definition. Orphaned results would be addressable by an id that
        no longer resolves — the overview route 404s on the definition load
        before it ever reaches them."""
        index_id = self._create(client)
        registry = client.app.state.jobs

        _persist_result(registry, f"backtest:{index_id}",
                        {"level": [100.0, 101.0]})

        assert registry.latest_result(f"backtest:{index_id}") is not None

        client.delete(f"/indices/{index_id}", headers=auth())

        assert registry.latest_result(f"backtest:{index_id}") is None

    def test_another_index_s_results_survive(self,
                                             client):
        """The forget is exact-kind, not prefix: deleting `core` must not
        take `core-hedged`'s results with it."""
        keep = self._create(client, "core-hedged")
        drop = self._create(client, "core")
        registry = client.app.state.jobs

        for index_id in (keep, drop):
            _persist_result(registry, f"backtest:{index_id}",
                            {"level": [100.0]})

        client.delete(f"/indices/{drop}", headers=auth())

        assert registry.latest_result(f"backtest:{drop}") is None
        assert registry.latest_result(f"backtest:{keep}") is not None

    def test_an_index_without_results_deletes_cleanly(self,
                                                      client):
        index_id = self._create(client)

        assert client.delete(f"/indices/{index_id}",
                             headers=auth()).status_code == 204

    def test_the_route_is_in_the_spec(self,
                                      client):
        spec = client.get("/openapi.json").json()

        assert "delete" in spec["paths"]["/indices/{index_id}"]
