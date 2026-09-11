# tests/test_server_optimised_indices.py
"""Contract tests for optimised indices: derivation, backtest, cascade.

Kept out of `test_server_indices.py`, which is already at the file-size limit,
and coherent on its own: everything here is about the *second* face of an index
document — the one whose methodology is a derivation rather than a pipeline.
"""
import copy
import uuid
from datetime import UTC, datetime

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from beacon.data.base import MarketData, ReferenceData
from beacon.data.fetcher import DataFetcher
from beacon.exceptions import DataNotFoundError, InvalidRuleError
from beacon.index.derived import OBJECTIVES, OptimisedIndexDefinition
from beacon.optimise.config import MIN_TRACKING_ERROR
from beacon.server import ServerConfig, create_app
from beacon.server import definitions as definitions_module
from beacon.server.definitions import build_definition
from beacon.server.schemas import IndexDocument

TOKEN = "test-token-value"

# Short window, two names, deterministic geometric paths — enough for three
# quarterly rebalances without letting this file dominate the suite.
START = "2023-10-02"
END = "2024-03-29"
DATES = pd.bdate_range(START, END)

GROWTH = {"AAA": 1.30, "BBB": 0.85}
BASE_PRICE = {"AAA": 100.0, "BBB": 50.0}
SHARES = 1_000

# Market-cap weighting puts the parent at roughly 2/3 - 1/3, so a 60% position
# cap actually binds and the solved book differs from the target one. An
# equal-weighted parent would solve to itself and prove nothing.
POSITION_CAP = 0.6

PARENT_ID = "BTP"
CHILD_ID = "BTC"


def auth() -> dict[str, str]:
    """Valid Authorization header."""
    return {"Authorization": f"Bearer {TOKEN}"}


def build_fetcher() -> DataFetcher:
    """Synthetic market and reference data with no randomness."""
    span = len(DATES) - 1
    rows = []
    for name, total_growth in GROWTH.items():
        for position, date in enumerate(DATES):
            rows.append({"IDENTIFIER": name,
                         "DATE": date,
                         "CLOSE": BASE_PRICE[name] * (total_growth ** (position / span)),
                         "VOLUME": 1_000_000,
                         "SHARES_OUTSTANDING": SHARES})

    reference = pd.DataFrame([
        {"IDENTIFIER": name,
         "DATE_FROM": "2020-01-01",
         "NAME": name,
         "CURRENCY": "USD",
         "EXCHANGE": "NYSE"}
        for name in GROWTH
    ])

    return DataFetcher(MarketData.from_dataframe(pd.DataFrame(rows)),
                       ReferenceData.from_dataframe(reference))


def parent_document(index_id: str = PARENT_ID) -> dict:
    """A market-cap-weighted, quarterly-rebalanced index over the two names."""
    return copy.deepcopy({
        "id": index_id,
        "name": f"Parent {index_id}",
        "base_date": START,
        "base_value": 1000.0,
        "currency": "USD",
        "rebalancing_frequency": "QUARTERLY",
        "description": None,
        "universe": {"universe_id": None, "identifiers": list(GROWTH)},
        "pipeline": {
            "selection": [],
            "weighting": {"id": "weighting",
                          "scheme": "MarketCapWeighted",
                          "params": {},
                          "max_weight": None},
            "treatment": {"corporate_actions": "ADJUST_DIVISOR"},
        },
    })


def optimise_body(index_id: str = CHILD_ID,
                  **overrides) -> dict:
    """A well-formed `POST /indices/{id}/optimise` body."""
    body = {
        "id": index_id,
        "name": f"Optimised {index_id}",
        "objective": MIN_TRACKING_ERROR,
        "constraints": [
            {"id": "invested", "type": "FullInvestment", "params": {"target": 1.0}},
            {"id": "bounds",
             "type": "PositionBounds",
             "params": {"minimum": 0.0, "maximum": POSITION_CAP}},
        ],
    }
    body.update(overrides)

    return body


def findings_from(response) -> list[dict]:
    """Pull the findings out of a rejected save."""
    return response.json()["error"]["detail"]["findings"]


def persist_result(registry,
                   kind,
                   result) -> None:
    """Write a succeeded result the way a finished job would."""
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


@pytest.fixture
def client(tmp_path) -> TestClient:
    """Client holding the parent definition, with no data source.

    Everything but the end-to-end backtest is document work, and a data source
    would only slow it down.
    """
    config = ServerConfig(auth_token=TOKEN, storage_root=tmp_path)
    entered = TestClient(create_app(config), raise_server_exceptions=False)
    created = entered.post("/indices", json=parent_document(), headers=auth())
    assert created.status_code == 200, created.text

    return entered


class TestDerivingAnIndex:
    """`POST /indices/{index_id}/optimise` — the UI's Optimise action."""

    def test_it_stores_a_document_carrying_the_derivation(self,
                                                          client):
        response = client.post(f"/indices/{PARENT_ID}/optimise",
                               json=optimise_body(), headers=auth())

        assert response.status_code == 200, response.text

        document = response.json()["index"]

        assert document["id"] == CHILD_ID
        assert document["derivation"]["source_index_id"] == PARENT_ID
        assert document["derivation"]["objective"] == MIN_TRACKING_ERROR
        assert [row["type"] for row in document["derivation"]["constraints"]] == [
            "FullInvestment", "PositionBounds"]

    def test_the_derived_document_has_no_pipeline(self,
                                                  client):
        """The discriminator the client branches on, from both sides."""
        client.post(f"/indices/{PARENT_ID}/optimise",
                    json=optimise_body(), headers=auth())

        document = client.get(f"/indices/{CHILD_ID}", headers=auth()).json()

        assert document["pipeline"] is None
        assert document["universe"] is None
        assert document["derivation"] is not None

    def test_it_inherits_the_parent_s_identity(self,
                                               client):
        """The child solves at the parent's snapshots, so it lives on the
        parent's calendar rather than declaring one of its own."""
        client.post(f"/indices/{PARENT_ID}/optimise",
                    json=optimise_body(), headers=auth())

        parent = client.get(f"/indices/{PARENT_ID}", headers=auth()).json()
        child = client.get(f"/indices/{CHILD_ID}", headers=auth()).json()

        for field in ("base_date", "base_value", "currency",
                      "rebalancing_frequency", "calendar",
                      "rebalance_day_rule"):
            assert child[field] == parent[field], field

    def test_it_appears_in_the_listing_like_any_index(self,
                                                      client):
        client.post(f"/indices/{PARENT_ID}/optimise",
                    json=optimise_body(), headers=auth())

        listing = client.get("/indices", headers=auth()).json()["indices"]
        by_id = {index["id"]: index for index in listing}

        assert set(by_id) == {PARENT_ID, CHILD_ID}
        assert by_id[CHILD_ID]["derivation"]["source_index_id"] == PARENT_ID
        assert by_id[PARENT_ID]["derivation"] is None

    def test_an_unknown_parent_is_404(self,
                                      client):
        response = client.post("/indices/never-existed/optimise",
                               json=optimise_body(), headers=auth())

        assert response.status_code == 404

    def test_a_taken_id_is_409(self,
                               client):
        """Provenance is server-truth, so overwriting an existing index would
        silently re-parent it."""
        response = client.post(f"/indices/{PARENT_ID}/optimise",
                               json=optimise_body(index_id=PARENT_ID),
                               headers=auth())

        assert response.status_code == 409
        assert response.json()["error"]["code"] == "CONFLICT"
        assert PARENT_ID in response.json()["error"]["message"]

    def test_a_collision_leaves_the_existing_index_alone(self,
                                                         client):
        client.post(f"/indices/{PARENT_ID}/optimise",
                    json=optimise_body(index_id=PARENT_ID), headers=auth())

        assert client.get(f"/indices/{PARENT_ID}",
                          headers=auth()).json()["pipeline"] is not None

    def test_chaining_is_allowed(self,
                                 client):
        """Optimising an optimised index falls out of the recursion."""
        client.post(f"/indices/{PARENT_ID}/optimise",
                    json=optimise_body(), headers=auth())

        response = client.post(f"/indices/{CHILD_ID}/optimise",
                               json=optimise_body(index_id="BTGC"),
                               headers=auth())

        assert response.status_code == 200, response.text
        assert (response.json()["index"]["derivation"]["source_index_id"]
                == CHILD_ID)


class TestTheDerivationIsValidated:
    """A bad derivation is refused with a finding addressed to what is wrong."""

    def test_an_unknown_objective_names_the_accepted_ones(self,
                                                          client):
        response = client.post(f"/indices/{PARENT_ID}/optimise",
                               json=optimise_body(objective="min_variance"),
                               headers=auth())

        assert response.status_code == 422

        finding = next(f for f in findings_from(response)
                       if f["code"] == "UNKNOWN_OBJECTIVE")

        assert finding["path"] == "derivation.objective"
        for objective in OBJECTIVES:
            assert objective in finding["message"]

    def test_an_unknown_constraint_type_names_the_row(self,
                                                      client):
        body = optimise_body()
        body["constraints"][1]["type"] = "NotAConstraint"

        response = client.post(f"/indices/{PARENT_ID}/optimise",
                               json=body, headers=auth())

        assert response.status_code == 422

        finding = next(f for f in findings_from(response)
                       if f["code"] == "UNKNOWN_CONSTRAINT_TYPE")

        assert finding["path"] == "derivation.constraints[1]"
        assert finding["rule_id"] == "bounds"
        assert "PositionBounds" in finding["message"]

    def test_an_unknown_constraint_parameter_names_the_field(self,
                                                             client):
        body = optimise_body()
        body["constraints"][1]["params"]["ceiling"] = 0.4

        response = client.post(f"/indices/{PARENT_ID}/optimise",
                               json=body, headers=auth())
        finding = next(f for f in findings_from(response)
                       if f["code"] == "UNKNOWN_PARAMETER")

        assert finding["path"] == "derivation.constraints[1].params.ceiling"

    def test_an_impossible_bound_is_reported_from_the_class(self,
                                                            client):
        body = optimise_body()
        body["constraints"][1]["params"] = {"minimum": 0.9, "maximum": 0.1}

        response = client.post(f"/indices/{PARENT_ID}/optimise",
                               json=body, headers=auth())
        codes = {f["code"] for f in findings_from(response)}

        assert "INVALID_PARAMETER" in codes

    def test_a_rejected_derivation_is_not_stored(self,
                                                 client):
        client.post(f"/indices/{PARENT_ID}/optimise",
                    json=optimise_body(objective="nope"), headers=auth())

        assert client.get(f"/indices/{CHILD_ID}",
                          headers=auth()).status_code == 404

    def test_the_objective_values_are_published_in_the_spec(self,
                                                            client):
        """Discoverable, not merely validated: the client renders a label
        while there is one objective and a Select when there are two."""
        spec = client.get("/openapi.json").json()
        described = spec["components"]["schemas"]["DerivationPayload"][
            "properties"]["objective"]["description"]

        for objective in OBJECTIVES:
            assert objective in described


class TestExactlyOneFace:
    """A document is a pipeline or a derivation — never both, never neither."""

    def test_both_faces_is_422(self,
                               client):
        body = parent_document(index_id="BOTH")
        body["derivation"] = {"source_index_id": PARENT_ID,
                              "objective": MIN_TRACKING_ERROR,
                              "constraints": []}

        response = client.post("/indices", json=body, headers=auth())

        assert response.status_code == 422

        message = str(response.json()["error"]["detail"]["errors"])

        assert "never both" in message
        assert "pipeline" in message and "universe" in message

    def test_neither_face_is_422(self,
                                 client):
        body = parent_document(index_id="NEITHER")
        body.pop("pipeline")
        body.pop("universe")

        response = client.post("/indices", json=body, headers=auth())

        assert response.status_code == 422
        assert "carries neither" in str(
            response.json()["error"]["detail"]["errors"])

    def test_half_a_pipeline_is_422(self,
                                    client):
        """A universe with no pipeline is not a methodology either."""
        body = parent_document(index_id="HALF")
        body.pop("pipeline")

        response = client.post("/indices", json=body, headers=auth())

        assert response.status_code == 422
        assert "pipeline is missing" in str(
            response.json()["error"]["detail"]["errors"])


class TestTheSourceIsImmutable:
    """Re-pointing a derivation is a new index, not an edit."""

    def _derived(self,
                 client) -> dict:
        client.post(f"/indices/{PARENT_ID}/optimise",
                    json=optimise_body(), headers=auth())

        return client.get(f"/indices/{CHILD_ID}", headers=auth()).json()

    def test_changing_the_source_is_422(self,
                                        client):
        client.post("/indices", json=parent_document("OTHER"), headers=auth())
        document = self._derived(client)
        document["derivation"]["source_index_id"] = "OTHER"

        response = client.put(f"/indices/{CHILD_ID}", json=document,
                              headers=auth())

        assert response.status_code == 422
        assert "cannot be changed" in response.json()["error"]["message"]

    def test_the_stored_source_is_unchanged_after_a_refusal(self,
                                                            client):
        client.post("/indices", json=parent_document("OTHER"), headers=auth())
        document = self._derived(client)
        document["derivation"]["source_index_id"] = "OTHER"

        client.put(f"/indices/{CHILD_ID}", json=document, headers=auth())
        stored = client.get(f"/indices/{CHILD_ID}", headers=auth()).json()

        assert stored["derivation"]["source_index_id"] == PARENT_ID

    def test_dropping_the_derivation_is_422(self,
                                            client):
        document = self._derived(client)
        document["derivation"] = None
        document["universe"] = {"universe_id": None, "identifiers": ["AAA"]}
        document["pipeline"] = parent_document()["pipeline"]

        response = client.put(f"/indices/{CHILD_ID}", json=document,
                              headers=auth())

        assert response.status_code == 422

    def test_the_constraints_are_editable(self,
                                          client):
        document = self._derived(client)
        document["derivation"]["constraints"] = [
            {"id": "invested", "type": "FullInvestment", "params": {"target": 1.0}}]

        response = client.put(f"/indices/{CHILD_ID}", json=document,
                              headers=auth())

        assert response.status_code == 200, response.text
        assert [row["type"] for row in client.get(
            f"/indices/{CHILD_ID}",
            headers=auth()).json()["derivation"]["constraints"]] == [
                "FullInvestment"]

    def test_the_objective_is_editable_within_the_accepted_set(self,
                                                               client):
        document = self._derived(client)
        document["derivation"]["objective"] = OBJECTIVES[0]

        response = client.put(f"/indices/{CHILD_ID}", json=document,
                              headers=auth())

        assert response.status_code == 200, response.text

    def test_an_edited_objective_outside_the_set_is_422(self,
                                                        client):
        document = self._derived(client)
        document["derivation"]["objective"] = "min_variance"

        response = client.put(f"/indices/{CHILD_ID}", json=document,
                              headers=auth())

        assert response.status_code == 422
        assert any(f["code"] == "UNKNOWN_OBJECTIVE"
                   for f in findings_from(response))


class TestMaterialisation:
    """The document builds the library object, chains and all."""

    def test_a_derived_document_builds_an_optimised_definition(self,
                                                               client):
        client.post(f"/indices/{PARENT_ID}/optimise",
                    json=optimise_body(), headers=auth())
        store = client.app.state.index_store

        definition = build_definition(
            IndexDocument.model_validate(store.read(CHILD_ID)), store)

        assert isinstance(definition, OptimisedIndexDefinition)
        assert definition.source.index_id == PARENT_ID
        assert len(definition.constraints) == 2

    def test_a_chain_builds_a_chain(self,
                                    client):
        client.post(f"/indices/{PARENT_ID}/optimise",
                    json=optimise_body(), headers=auth())
        client.post(f"/indices/{CHILD_ID}/optimise",
                    json=optimise_body(index_id="BTGC"), headers=auth())
        store = client.app.state.index_store

        definition = build_definition(
            IndexDocument.model_validate(store.read("BTGC")), store)

        assert isinstance(definition, OptimisedIndexDefinition)
        assert isinstance(definition.source, OptimisedIndexDefinition)
        assert definition.source.source.index_id == PARENT_ID

    def test_a_missing_source_is_a_404_shaped_failure(self,
                                                      client):
        """The parent went by some other route: say which id is missing rather
        than failing later on a definition nobody can explain."""
        client.post(f"/indices/{PARENT_ID}/optimise",
                    json=optimise_body(), headers=auth())
        store = client.app.state.index_store
        document = IndexDocument.model_validate(store.read(CHILD_ID))
        store.delete(PARENT_ID)

        with pytest.raises(DataNotFoundError, match=PARENT_ID):
            build_definition(document, store)


def _store_a_cycle(client) -> None:
    """Point the parent's derivation back at its own optimised child.

    Reachable through the API: the child's source is immutable, but the parent
    is a rule-driven index, and turning one into a derivation is a creation
    rather than an edit. The save resolves the chain against what is *stored*,
    where the parent is still rule-driven, so the loop closes only once the
    write lands — which is exactly why the builder carries a guard of its own.
    """
    client.post(f"/indices/{PARENT_ID}/optimise",
                json=optimise_body(), headers=auth())

    looping = client.get(f"/indices/{PARENT_ID}", headers=auth()).json()
    looping["pipeline"] = None
    looping["universe"] = None
    looping["derivation"] = {"source_index_id": CHILD_ID,
                             "objective": MIN_TRACKING_ERROR,
                             "constraints": []}

    saved = client.put(f"/indices/{PARENT_ID}", json=looping, headers=auth())
    assert saved.status_code == 200, saved.text


class TestACycleIsRefused:
    """A derivation chain that returns to itself must fail, not recurse."""

    def test_building_it_raises_naming_the_loop(self,
                                                client):
        _store_a_cycle(client)
        store = client.app.state.index_store
        document = IndexDocument.model_validate(store.read(CHILD_ID))

        with pytest.raises(InvalidRuleError, match="returns to itself"):
            build_definition(document, store)

    def test_the_message_prints_the_chain(self,
                                          client):
        _store_a_cycle(client)
        store = client.app.state.index_store

        with pytest.raises(InvalidRuleError) as raised:
            build_definition(
                IndexDocument.model_validate(store.read(PARENT_ID)), store)

        assert f"{PARENT_ID} -> {CHILD_ID} -> {PARENT_ID}" in str(raised.value)

    def test_saving_into_a_closed_cycle_is_422(self,
                                               client):
        """Once the loop is stored, the next save of either document resolves
        the chain and is refused."""
        _store_a_cycle(client)
        document = client.get(f"/indices/{PARENT_ID}", headers=auth()).json()

        response = client.put(f"/indices/{PARENT_ID}", json=document,
                              headers=auth())

        assert response.status_code == 422
        assert "returns to itself" in response.json()["error"]["message"]

    def test_a_chain_deeper_than_the_cap_is_refused(self,
                                                    client,
                                                    monkeypatch):
        """The backstop for a chain that is absurd without being a loop.

        The chain is built at the real cap and the cap is lowered afterwards,
        because a save resolves the chain too — building it under the lowered
        cap would be refused before there was anything to test.
        """
        previous = PARENT_ID
        for step in range(3):
            index_id = f"CHAIN{step}"
            derived = client.post(f"/indices/{previous}/optimise",
                                  json=optimise_body(index_id=index_id),
                                  headers=auth())
            assert derived.status_code == 200, derived.text
            previous = index_id

        monkeypatch.setattr(definitions_module, "MAX_DERIVATION_DEPTH", 2)
        store = client.app.state.index_store

        with pytest.raises(InvalidRuleError, match="deep"):
            build_definition(
                IndexDocument.model_validate(store.read(previous)), store)


class TestTheDeleteCascades:
    """`DELETE /indices/{parent}` takes the optimised children with it."""

    def test_the_child_goes_too(self,
                                client):
        client.post(f"/indices/{PARENT_ID}/optimise",
                    json=optimise_body(), headers=auth())

        response = client.delete(f"/indices/{PARENT_ID}", headers=auth())

        assert response.status_code == 200, response.text
        assert client.get(f"/indices/{CHILD_ID}",
                          headers=auth()).status_code == 404

    def test_the_response_names_everything_that_went(self,
                                                     client):
        client.post(f"/indices/{PARENT_ID}/optimise",
                    json=optimise_body(), headers=auth())

        body = client.delete(f"/indices/{PARENT_ID}", headers=auth()).json()

        assert body["index_id"] == PARENT_ID
        assert [entry["index_id"] for entry in body["deleted"]] == [
            PARENT_ID, CHILD_ID]
        assert body["deleted"][1]["derived_from"] == PARENT_ID

    def test_the_child_s_records_and_results_go_with_it(self,
                                                        client):
        client.post(f"/indices/{PARENT_ID}/optimise",
                    json=optimise_body(), headers=auth())
        registry = client.app.state.jobs
        records = client.app.state.backtest_record_store

        for index_id in (PARENT_ID, CHILD_ID):
            persist_result(registry, f"backtest:{index_id}", {"level": [100.0]})
            records.write(index_id, {"metrics": {}})

        body = client.delete(f"/indices/{PARENT_ID}", headers=auth()).json()

        assert registry.latest_result(f"backtest:{CHILD_ID}") is None
        assert records.read(CHILD_ID) is None
        assert all(entry["backtest_record_deleted"] for entry in body["deleted"])
        assert all(entry["backtest_results_deleted"] == 1
                   for entry in body["deleted"])

    def test_a_two_level_chain_goes_entirely(self,
                                             client):
        client.post(f"/indices/{PARENT_ID}/optimise",
                    json=optimise_body(), headers=auth())
        client.post(f"/indices/{CHILD_ID}/optimise",
                    json=optimise_body(index_id="BTGC"), headers=auth())

        body = client.delete(f"/indices/{PARENT_ID}", headers=auth()).json()

        assert [entry["index_id"] for entry in body["deleted"]] == [
            PARENT_ID, CHILD_ID, "BTGC"]
        assert client.get("/indices", headers=auth()).json()["indices"] == []

    def test_deleting_the_child_leaves_the_parent(self,
                                                  client):
        """The cascade runs downward only: a parent is not derived from
        anything and has its own reason to exist."""
        client.post(f"/indices/{PARENT_ID}/optimise",
                    json=optimise_body(), headers=auth())

        body = client.delete(f"/indices/{CHILD_ID}", headers=auth()).json()

        assert [entry["index_id"] for entry in body["deleted"]] == [CHILD_ID]
        assert client.get(f"/indices/{PARENT_ID}",
                          headers=auth()).status_code == 200

    def test_a_stored_cycle_does_not_hang_the_scan(self,
                                                   client):
        _store_a_cycle(client)

        body = client.delete(f"/indices/{PARENT_ID}", headers=auth()).json()

        assert sorted(entry["index_id"] for entry in body["deleted"]) == [
            CHILD_ID, PARENT_ID]
        assert client.get("/indices", headers=auth()).json()["indices"] == []


@pytest.fixture(scope="module")
def backtested(tmp_path_factory):
    """A client that has run one backtest of an optimised index.

    Module-scoped: the run is the expensive part of this file, and every
    assertion below interrogates the same completed run.
    """
    config = ServerConfig(auth_token=TOKEN,
                          data_fetcher=build_fetcher(),
                          storage_root=tmp_path_factory.mktemp("optimised"))

    with TestClient(create_app(config), raise_server_exceptions=False) as entered:
        created = entered.post("/indices", json=parent_document(), headers=auth())
        assert created.status_code == 200, created.text

        derived = entered.post(f"/indices/{PARENT_ID}/optimise",
                               json=optimise_body(), headers=auth())
        assert derived.status_code == 200, derived.text

        submitted = entered.post(f"/beacon/{CHILD_ID}/backtest",
                                 json={"start": START, "end": END},
                                 headers=auth())
        assert submitted.status_code == 202, submitted.text

        entered.portal.call(entered.app.state.jobs.drain)

        job = entered.get(f"/jobs/{submitted.json()['job_id']}",
                          headers=auth()).json()
        assert job["status"] == "succeeded", job

        yield entered


class TestBacktestingAnOptimisedIndex:
    """The acceptance criterion: it just works through the existing route."""

    def test_the_record_carries_both_books(self,
                                           backtested):
        record = backtested.get(f"/beacon/{CHILD_ID}/record",
                                headers=auth()).json()

        assert record["index"]["target"] is not None
        assert record["index"]["optimised"] is not None

    def test_the_two_books_share_a_calendar_and_differ(self,
                                                       backtested):
        """`index.target` vs `index.optimised` is "what did the constraints
        cost?" — the child solves at the parent's snapshots, so the two books
        run over the same days and must not be the same numbers."""
        record = backtested.get(f"/beacon/{CHILD_ID}/record",
                                headers=auth()).json()

        target = record["index"]["target"]["levels"]
        optimised = record["index"]["optimised"]["levels"]

        assert target["index"] == optimised["index"]
        assert target["data"], "the pre-optimisation book is empty"
        assert target["data"] != optimised["data"]

    def test_no_weight_exceeds_the_position_cap(self,
                                                backtested):
        """The solve is what the record reports, not a claim about it."""
        weights = backtested.get(f"/beacon/{CHILD_ID}/weights",
                                 headers=auth()).json()

        assert weights["weights"], "the run published no weights"

        for weight in weights["weights"].values():
            assert weight <= POSITION_CAP + 1e-6

    def test_both_books_carry_their_own_decided_weights(self,
                                                        backtested):
        """BN-173: the record publishes what each rebalance decided, per book.
        The optimised book's are the SOLVED weights and the target book's are
        the parent's own, so "what did the constraints cost?" is answerable
        from the durable record — at every rebalance, as preview answers it for
        one date."""
        record = backtested.get(f"/beacon/{CHILD_ID}/record",
                                headers=auth()).json()

        target = record["index"]["target"]["rebalances"]
        optimised = record["index"]["optimised"]["rebalances"]

        assert target, "the pre-optimisation book decided nothing"
        assert optimised, "the solved book decided nothing"
        assert [entry["date"] for entry in target] == \
            [entry["date"] for entry in optimised]
        assert [entry["weights"] for entry in target] != \
            [entry["weights"] for entry in optimised]

    def test_the_solved_decisions_are_the_run_payload_s(self,
                                                        backtested):
        """One fact, two representations: the run payload's `rebalances[]` come
        from the book the engine traded — the optimised one — and the record's
        copy of them must be the same rows."""
        jobs = backtested.get("/jobs", headers=auth()).json()["jobs"]
        child = [job for job in jobs
                 if job["kind"] == f"backtest:{CHILD_ID}"
                 and job["status"] == "succeeded"]
        assert child, "the module fixture backtests the child"

        payload = backtested.get(f"/jobs/{child[0]['job_id']}",
                                 headers=auth()).json()["result"]
        record = backtested.get(f"/beacon/{CHILD_ID}/record",
                                headers=auth()).json()

        assert record["index"]["optimised"]["rebalances"] == payload["rebalances"]

    def test_the_parent_backtests_independently(self,
                                                backtested):
        """The parent stays first-class: it is referenced, not consumed."""
        submitted = backtested.post(f"/beacon/{PARENT_ID}/backtest",
                                    json={"start": START, "end": END},
                                    headers=auth())
        assert submitted.status_code == 202, submitted.text

        backtested.portal.call(backtested.app.state.jobs.drain)

        record = backtested.get(f"/beacon/{PARENT_ID}/record",
                                headers=auth()).json()

        assert record["index"]["target"] is not None
        assert record["index"]["optimised"] is None


class TestAnOptimisedIndexAsABenchmark:
    """BN-169: a benchmark needs a level series, and an optimised index has one.

    The refusal this fixes was an accident of resolution rather than a
    judgement: benchmarks were built through the pipeline-only builder, which
    refuses a derivation, so an index that could perfectly well be compared
    against could not be named as the comparator.
    """

    def test_the_parent_can_be_measured_against_its_optimised_child(self,
                                                                    backtested):
        """The comparison the whole feature exists to make — what did the
        constraints cost? — asked the way a client asks it."""
        submitted = backtested.post(
            f"/beacon/{PARENT_ID}/backtest",
            json={"start": START, "end": END,
                  "benchmark": {"kind": "index", "id": CHILD_ID}},
            headers=auth())

        assert submitted.status_code == 202, submitted.text

        backtested.portal.call(backtested.app.state.jobs.drain)

        job = backtested.get(f"/jobs/{submitted.json()['job_id']}",
                             headers=auth()).json()

        assert job["status"] == "succeeded", job

        comparison = job["result"]["benchmark"]

        assert comparison is not None
        assert comparison["level"]["data"], "the benchmark produced no levels"

    def test_an_unknown_benchmark_index_still_404s(self,
                                                   backtested):
        """The widened resolution must not widen into accepting nothing."""
        submitted = backtested.post(
            f"/beacon/{PARENT_ID}/backtest",
            json={"start": START, "end": END,
                  "benchmark": {"kind": "index", "id": "NEVER-EXISTED"}},
            headers=auth())

        assert submitted.status_code == 202, submitted.text

        backtested.portal.call(backtested.app.state.jobs.drain)

        job = backtested.get(f"/jobs/{submitted.json()['job_id']}",
                             headers=auth()).json()

        assert job["status"] == "failed"
        assert "NEVER-EXISTED" in str(job["error"])
