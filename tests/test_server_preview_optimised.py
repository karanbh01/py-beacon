# tests/test_server_preview_optimised.py
"""BN-170: previewing an optimised index — the solve, not a waterfall.

Preview's own tests live in `test_server_preview.py` (saved) and
`test_server_preview_document.py` (draft), and both are about the rule-driven
face: a universe narrowing one rung per selection rule. This file is the
*second* face of the same endpoint, and it is kept apart for the reason the
optimised-index tests were: it needs a data source and a solved parent, where
those two need neither.

The setup is deliberately the one `test_server_optimised_indices.py` uses — a
market-cap parent over two names, and a 60% position cap that actually binds —
because a cap that does not bind proves nothing about a preview whose whole
subject is what the constraints did.
"""
import copy

import pandas as pd
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from beacon.data.base import MarketData, ReferenceData
from beacon.data.fetcher import DataFetcher
from beacon.optimise.config import MIN_TRACKING_ERROR
from beacon.server import ServerConfig, create_app
from beacon.server.schemas import PreviewResponse, PreviewSolve

TOKEN = "test-token-value"

START = "2023-10-02"
END = "2024-03-29"
DATES = pd.bdate_range(START, END)

GROWTH = {"AAA": 1.30, "BBB": 0.85}
BASE_PRICE = {"AAA": 100.0, "BBB": 50.0}
SHARES = 1_000

# At the base date the caps are 100k and 50k, so the parent publishes 2/3 and
# 1/3 and a 60% cap moves 6.67 points off AAA onto BBB. Every before/after
# assertion below is that one number seen from a different side.
POSITION_CAP = 0.6
PARENT_AAA = 2 / 3
PARENT_BBB = 1 / 3

PARENT_ID = "PVP"
CHILD_ID = "PVC"


def auth() -> dict[str, str]:
    """Valid Authorization header."""
    return {"Authorization": f"Bearer {TOKEN}"}


def build_fetcher() -> DataFetcher:
    """Synthetic market and reference data with no randomness."""
    span = len(DATES) - 1
    rows = [{"IDENTIFIER": name,
             "DATE": date,
             "CLOSE": BASE_PRICE[name] * (total_growth ** (position / span)),
             "VOLUME": 1_000_000,
             "SHARES_OUTSTANDING": SHARES}
            for name, total_growth in GROWTH.items()
            for position, date in enumerate(DATES)]

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


def constraint_rows() -> list[dict]:
    """Full investment plus the position cap that binds."""
    return [
        {"id": "invested", "type": "FullInvestment", "params": {"target": 1.0}},
        {"id": "bounds",
         "type": "PositionBounds",
         "params": {"minimum": 0.0, "maximum": POSITION_CAP}},
    ]


def derived_document(index_id: str = CHILD_ID,
                     source: str = PARENT_ID) -> dict:
    """An optimised document in the shape the store holds one."""
    return copy.deepcopy({
        "id": index_id,
        "name": f"Optimised {index_id}",
        "base_date": START,
        "base_value": 1000.0,
        "currency": "USD",
        "rebalancing_frequency": "QUARTERLY",
        "description": None,
        "derivation": {"source_index_id": source,
                       "objective": MIN_TRACKING_ERROR,
                       "constraints": constraint_rows()},
    })


@pytest.fixture(scope="module")
def client(tmp_path_factory) -> TestClient:
    """A client holding the parent and its optimised child, with market data.

    Module-scoped: every test below interrogates the same two documents, and
    nothing here writes to them.
    """
    config = ServerConfig(auth_token=TOKEN,
                          data_fetcher=build_fetcher(),
                          storage_root=tmp_path_factory.mktemp("preview"))

    with TestClient(create_app(config), raise_server_exceptions=False) as entered:
        created = entered.post("/indices", json=parent_document(), headers=auth())
        assert created.status_code == 200, created.text

        derived = entered.post(f"/indices/{PARENT_ID}/optimise",
                               json={"id": CHILD_ID,
                                     "name": f"Optimised {CHILD_ID}",
                                     "objective": MIN_TRACKING_ERROR,
                                     "constraints": constraint_rows()},
                               headers=auth())
        assert derived.status_code == 200, derived.text

        yield entered


def preview(client,
            index_id: str = CHILD_ID,
            **body) -> dict:
    """Preview a saved index, asserting the call itself worked."""
    response = client.post(f"/indices/{index_id}/preview", json=body,
                           headers=auth())
    assert response.status_code == 200, response.text

    return response.json()


def rows_by_identifier(payload: dict) -> dict[str, dict]:
    """The per-asset rows, keyed by name."""
    return {row["identifier"]: row for row in payload["assets"]}


def constraint_named(payload: dict,
                     fragment: str) -> dict:
    """The one constraint row whose label contains *fragment*."""
    matches = [row for row in payload["solve"]["constraints"]
               if fragment in row["label"]]
    assert len(matches) == 1, f"{fragment!r} matched {matches}"

    return matches[0]


class TestTheDerivedFaceOfTheResponse:
    """A derivation eliminates nothing, so it answers with the solve."""

    def test_it_carries_a_solve_and_no_steps(self,
                                             client):
        """The discriminator, from both sides. A waterfall for an index that
        narrows nothing would be an empty funnel — a worse answer than none."""
        payload = preview(client)

        assert payload["steps"] is None
        assert payload["solve"] is not None

    def test_the_solve_names_its_source_and_objective(self,
                                                      client):
        solve = preview(client)["solve"]

        assert solve["source_index_id"] == PARENT_ID
        assert solve["objective"] == MIN_TRACKING_ERROR

    def test_it_reports_the_parent_snapshot_it_solved(self,
                                                      client):
        """A derivation solves only where its parent published weights, so an
        as-of between two rebalances previews the one in force that day."""
        payload = preview(client, as_of="2023-11-15")

        assert payload["as_of"] == "2023-11-15"
        assert payload["solve"]["rebalance_date"] == START

    def test_the_rule_provenance_fields_stay_empty(self,
                                                   client):
        """`excluded_by` was not overloaded to mean something new here."""
        for row in preview(client)["assets"]:
            assert row["excluded_by"] is None
            assert row["excluded_at"] is None
            assert row["capped"] is False

    def test_an_as_of_before_the_parent_s_first_rebalance_is_refused(self,
                                                                     client):
        """There is no snapshot to solve, and inventing one would be a lie
        about a date the index did not exist on. 422 rather than 500: the date
        came from the request, so it is the client's to correct."""
        response = client.post(f"/indices/{CHILD_ID}/preview",
                               json={"as_of": "2023-09-01"}, headers=auth())

        assert response.status_code == 422, response.text
        assert PARENT_ID in response.text


class TestBindingConstraints:
    """The question the block exists to answer: what did my constraints do?"""

    def test_the_position_cap_binds(self,
                                    client):
        solve = preview(client)["solve"]

        assert any("maximum weight" in label and "AAA" in label
                   for label in solve["binding"])

    def test_the_binding_cap_carries_its_slack_and_unit(self,
                                                        client):
        """Zero slack is the definition of binding; the unit is what lets a
        client format the number without a private type-to-unit table."""
        row = constraint_named(preview(client), "maximum weight 60.0000% on AAA")

        assert row["binding"] is True
        assert row["slack"] == pytest.approx(0.0, abs=1e-6)
        assert row["unit"] == "fraction"
        assert row["kind"] == "ineq"

    def test_a_constraint_with_room_reports_how_much(self,
                                                     client):
        """The informative end. BBB solves to 40%, so its 60% cap has 20
        points of room — the reading a list of binding constraints cannot
        give, since every one of those is zero by construction."""
        row = constraint_named(preview(client), "maximum weight 60.0000% on BBB")

        assert row["binding"] is False
        assert row["slack"] == pytest.approx(POSITION_CAP - 0.4)

    def test_every_constraint_is_reported_not_only_the_binding_ones(self,
                                                                    client):
        payload = preview(client)
        rows = payload["solve"]["constraints"]

        # Full investment, plus a floor and a cap on each of the two names.
        assert len(rows) == 5
        assert any(not row["binding"] for row in rows)

    def test_the_binding_labels_are_a_subset_of_the_constraint_rows(self,
                                                                    client):
        solve = preview(client)["solve"]
        binding_rows = {row["label"] for row in solve["constraints"]
                        if row["binding"]}

        assert set(solve["binding"]) == binding_rows


class TestBeforeAndAfter:
    """The per-asset pair: what the parent published, what the solve allocated."""

    def test_the_parent_weight_is_reported_uncapped(self,
                                                    client):
        rows = rows_by_identifier(preview(client))

        assert rows["AAA"]["source_weight"] == pytest.approx(PARENT_AAA)
        assert rows["BBB"]["source_weight"] == pytest.approx(PARENT_BBB)

    def test_the_name_above_the_cap_moves_down_to_it(self,
                                                     client):
        row = rows_by_identifier(preview(client))["AAA"]

        assert row["solved_weight"] == pytest.approx(POSITION_CAP)
        assert row["weight_delta"] == pytest.approx(POSITION_CAP - PARENT_AAA)

    def test_the_freed_weight_appears_on_the_other_name(self,
                                                        client):
        row = rows_by_identifier(preview(client))["BBB"]

        assert row["solved_weight"] == pytest.approx(1.0 - POSITION_CAP)
        assert row["weight_delta"] == pytest.approx(PARENT_AAA - POSITION_CAP)

    def test_the_deltas_net_to_zero(self,
                                    client):
        """Full investment on both sides: weight is moved, never created."""
        rows = preview(client)["assets"]

        assert sum(row["weight_delta"] for row in rows) == pytest.approx(0.0)

    def test_the_final_weights_are_the_solved_ones(self,
                                                   client):
        payload = preview(client)

        assert payload["weights"]["AAA"] == pytest.approx(POSITION_CAP)
        assert payload["total_weight"] == pytest.approx(1.0)
        assert payload["cap"] is None

        for row in payload["assets"]:
            assert row["weight"] == pytest.approx(row["solved_weight"])
            assert row["included"] is True


@pytest.fixture(scope="module")
def backtested(client) -> TestClient:
    """The same client, having run one backtest of the optimised index."""
    submitted = client.post(f"/beacon/{CHILD_ID}/backtest",
                            json={"start": START, "end": END}, headers=auth())
    assert submitted.status_code == 202, submitted.text

    client.portal.call(client.app.state.jobs.drain)

    job = client.get(f"/jobs/{submitted.json()['job_id']}", headers=auth()).json()
    assert job["status"] == "succeeded", job

    return client


class TestPreviewAgreesWithTheRun:
    """The acceptance criterion, and the property the module docstring claims.

    A preview that disagrees with the run it previews is worse than no
    preview. Both go through `solve_snapshot`, so the only way they could
    differ is if the preview solved a different date's weights.
    """

    def test_the_solved_weights_match_the_run_s_own_snapshot(self,
                                                             backtested):
        payload = preview(backtested)
        rebalance = payload["solve"]["rebalance_date"]

        record = backtested.get(f"/beacon/{CHILD_ID}/record",
                                headers=auth()).json()
        book = record["index"]["optimised"]

        position = next(index for index, stamp in enumerate(book["weights"]["index"])
                        if str(stamp).startswith(rebalance))
        published = dict(zip(book["weights"]["columns"],
                             book["weights"]["data"][position], strict=True))

        assert published, "the run published no weights at the rebalance"

        for identifier, weight in published.items():
            assert payload["weights"][identifier] == pytest.approx(weight)


class TestTheRuleDrivenFaceIsUnchanged:
    """The regression: a pipeline still answers with a waterfall and no solve."""

    def test_a_pipeline_preview_still_carries_steps(self,
                                                    client):
        payload = preview(client, index_id=PARENT_ID)

        assert payload["solve"] is None
        assert payload["steps"] is not None
        assert payload["steps"][0]["position"] == 0

    def test_a_pipeline_preview_carries_no_derived_fields(self,
                                                          client):
        for row in preview(client, index_id=PARENT_ID)["assets"]:
            assert row["source_weight"] is None
            assert row["solved_weight"] is None
            assert row["weight_delta"] is None


class TestTheResponseHoldsItselfToOneFace:
    """The same discipline `IndexDocument` applies to pipeline/derivation.

    Enforced on the response rather than trusted, because the two faces are
    built by two different functions and nothing else would notice a third one
    that filled in neither.
    """

    def test_a_response_carrying_both_faces_is_refused(self):
        with pytest.raises(ValidationError, match="both"):
            PreviewResponse(index_id="X", as_of=START, steps=[],
                            solve=PreviewSolve(source_index_id=PARENT_ID,
                                               rebalance_date=START,
                                               objective=MIN_TRACKING_ERROR),
                            assets=[], weights={}, total_weight=0.0)

    def test_a_response_carrying_neither_is_refused(self):
        with pytest.raises(ValidationError, match="neither"):
            PreviewResponse(index_id="X", as_of=START, assets=[], weights={},
                            total_weight=0.0)


class TestTheDraftRoute:
    """`POST /indices/preview` — an editor holding an unsaved derivation."""

    def test_it_previews_a_derivation_that_was_never_saved(self,
                                                           client):
        """The source is resolved through the store: the document being edited
        has never been stored, but its parent has."""
        response = client.post("/indices/preview",
                               json={"document": derived_document("UNSAVED")},
                               headers=auth())

        assert response.status_code == 200, response.text

        payload = response.json()

        assert payload["index_id"] == "UNSAVED"
        assert payload["steps"] is None
        assert payload["solve"]["source_index_id"] == PARENT_ID
        assert payload["weights"]["AAA"] == pytest.approx(POSITION_CAP)

    def test_editing_the_cap_changes_the_preview_without_saving(self,
                                                                client):
        """The point of the draft route, asked of the derived face: the
        constraint is what moves, and the stored child is untouched."""
        document = derived_document("UNSAVED")
        document["derivation"]["constraints"][1]["params"]["maximum"] = 0.55

        response = client.post("/indices/preview", json={"document": document},
                               headers=auth())

        assert response.status_code == 200, response.text
        assert response.json()["weights"]["AAA"] == pytest.approx(0.55)
        assert preview(client)["weights"]["AAA"] == pytest.approx(POSITION_CAP)

    def test_an_unknown_source_is_a_404_naming_it(self,
                                                  client):
        response = client.post(
            "/indices/preview",
            json={"document": derived_document("UNSAVED", source="NEVER-EXISTED")},
            headers=auth())

        assert response.status_code == 404, response.text
        assert "NEVER-EXISTED" in response.text
