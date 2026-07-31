# tests/test_server_risk.py
"""BN-73: risk-model endpoints."""
import tempfile
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from beacon.server import ServerConfig, create_app
from beacon.testing import dataset

TOKEN = "test-token-value"
MODEL_ID = "canon-risk"

START = "2023-01-02"
END = "2024-06-28"


def auth() -> dict[str, str]:
    return {"Authorization": f"Bearer {TOKEN}"}


@pytest.fixture(scope="module")
def client():
    with tempfile.TemporaryDirectory() as storage:
        config = ServerConfig(auth_token=TOKEN,
                              data_fetcher=dataset.data_fetcher(),
                              storage_root=Path(storage))

        with TestClient(create_app(config),
                        raise_server_exceptions=False) as started:
            yield started


@pytest.fixture(scope="module")
def estimated(client):
    """A model estimated over the canonical universe."""
    response = client.post(f"/risk-models/{MODEL_ID}/estimate",
                           json={"identifiers": list(dataset.UNIVERSE),
                                 "start": START, "end": END},
                           headers=auth())
    client.portal.call(client.app.state.jobs.drain)

    job = client.get(f"/jobs/{response.json()['job_id']}", headers=auth()).json()
    assert job["status"] == "succeeded", job.get("error")

    return client.get(f"/risk-models/{MODEL_ID}", headers=auth()).json()


def matrix_of(payload: dict,
              key: str) -> np.ndarray:
    """A served matrix as a numpy array."""
    return np.array(payload[key]["data"], dtype=float)


class TestEstimationJob:

    def test_it_returns_a_job(self,
                              client):
        response = client.post("/risk-models/scratch/estimate",
                               json={"identifiers": list(dataset.UNIVERSE)},
                               headers=auth())

        assert response.status_code == 202
        assert response.json()["kind"] == "risk:scratch"

    def test_the_job_streams_progress(self,
                                      client):
        """The acceptance criterion. Progress must move before it completes,
        not jump from nothing to done."""
        registry = client.app.state.jobs
        queue = client.portal.call(_subscribe, registry)

        client.post("/risk-models/progress-check/estimate",
                    json={"identifiers": list(dataset.UNIVERSE)}, headers=auth())
        client.portal.call(registry.drain)

        events = client.portal.call(_drain_queue, queue)
        progress = [event["progress"] for event in events
                    if event.get("kind") == "risk:progress-check"]

        assert progress[-1] == pytest.approx(1.0)
        assert any(0.0 < value < 1.0 for value in progress)

    def test_the_universe_can_come_from_an_index(self,
                                                 client):
        """A client asks for "the risk model for this index" without restating
        its universe."""
        client.put("/indices/idx", json=_index_document(), headers=auth())
        client.post("/beacon/idx/backtest", json={"start": START, "end": END},
                    headers=auth())
        client.portal.call(client.app.state.jobs.drain)

        response = client.post("/risk-models/from-index/estimate",
                               json={"index_id": "idx"}, headers=auth())
        client.portal.call(client.app.state.jobs.drain)

        model = client.get("/risk-models/from-index", headers=auth()).json()

        assert set(model["asset_ids"]) == set(dataset.UNIVERSE)
        assert response.status_code == 202

    def test_neither_a_universe_nor_an_index_is_a_404(self,
                                                      client):
        response = client.post("/risk-models/nothing/estimate", json={},
                               headers=auth())

        assert response.status_code == 404

    def test_an_index_without_a_backtest_is_a_404(self,
                                                  client):
        response = client.post("/risk-models/unrun/estimate",
                               json={"index_id": "never-run"}, headers=auth())

        assert response.status_code == 404

    def test_a_single_name_is_refused(self,
                                      client):
        """A covariance over one asset is a variance, and the endpoint
        promises a matrix."""
        submitted = client.post("/risk-models/lonely/estimate",
                                json={"identifiers": ["AAA"]}, headers=auth())
        client.portal.call(client.app.state.jobs.drain)

        job = client.get(f"/jobs/{submitted.json()['job_id']}",
                         headers=auth()).json()

        assert job["status"] == "failed"
        assert "at least two" in job["error"]

    def test_it_requires_authentication(self,
                                        client):
        assert client.post("/risk-models/x/estimate").status_code == 401


class TestMatrixContract:
    """The acceptance criteria, checked on what the client actually receives."""

    def test_the_correlation_matrix_is_symmetric(self,
                                                 estimated):
        values = matrix_of(estimated, "correlation")

        assert np.allclose(values, values.T, atol=1e-12)

    def test_the_correlation_diagonal_is_unit(self,
                                              estimated):
        values = matrix_of(estimated, "correlation")

        assert np.allclose(np.diag(values), 1.0, atol=1e-12)

    def test_correlations_are_within_bounds(self,
                                            estimated):
        values = matrix_of(estimated, "correlation")

        assert values.min() >= -1.0
        assert values.max() <= 1.0

    def test_the_covariance_matrix_is_symmetric(self,
                                                estimated):
        values = matrix_of(estimated, "covariance")

        assert np.allclose(values, values.T, atol=1e-15)

    def test_the_psd_flag_matches_the_eigenvalues(self,
                                                  estimated):
        """Truthful, not asserted: the flag has to agree with the matrix that
        was actually served."""
        values = matrix_of(estimated, "covariance")
        smallest = float(np.linalg.eigvalsh(values).min())

        assert estimated["diagnostics"]["positive_semi_definite"] == (
            smallest >= -1e-10)

    def test_the_reported_smallest_eigenvalue_is_right(self,
                                                       estimated):
        values = matrix_of(estimated, "covariance")

        assert estimated["diagnostics"]["smallest_eigenvalue"] == pytest.approx(
            float(np.linalg.eigvalsh(values).min()), abs=1e-8)

    def test_volatilities_are_the_covariance_diagonal(self,
                                                      estimated):
        values = matrix_of(estimated, "covariance")
        served = [estimated["volatilities"][name]
                  for name in estimated["correlation"]["index"]]

        assert np.allclose(served, np.sqrt(np.diag(values)), atol=1e-6)

    def test_both_matrices_are_labelled_by_asset(self,
                                                 estimated):
        for key in ("correlation", "covariance"):
            assert estimated[key]["index"] == estimated["asset_ids"]
            assert estimated[key]["columns"] == estimated["asset_ids"]


class TestDiagnostics:

    def test_it_reports_how_the_estimate_was_made(self,
                                                  estimated):
        diagnostics = estimated["diagnostics"]

        assert diagnostics["target"] == "constant_correlation"
        assert 0.0 <= diagnostics["intensity"] <= 1.0
        assert diagnostics["assets"] == len(dataset.UNIVERSE)
        assert diagnostics["observations"] > 300

    def test_the_average_correlation_is_plausible(self,
                                                  estimated):
        """The sanity check a person can actually do."""
        assert 0.0 < estimated["diagnostics"]["average_correlation"] < 1.0

    def test_the_condition_number_is_reported(self,
                                              estimated):
        assert estimated["diagnostics"]["condition_number"] >= 1.0

    def test_shrinkage_improves_the_conditioning(self,
                                                 client):
        """The reason the intensity is reported at all: it is the dial between
        a noisy estimate and a stable one, and the effect should be visible."""
        for name, intensity in (("raw", 0.0), ("shrunk", 0.5)):
            client.post(f"/risk-models/{name}/estimate",
                        json={"identifiers": list(dataset.UNIVERSE),
                              "intensity": intensity}, headers=auth())
        client.portal.call(client.app.state.jobs.drain)

        raw = client.get("/risk-models/raw", headers=auth()).json()
        shrunk = client.get("/risk-models/shrunk", headers=auth()).json()

        assert (shrunk["diagnostics"]["condition_number"]
                < raw["diagnostics"]["condition_number"])

    def test_the_scaled_identity_target_is_accepted(self,
                                                    client):
        client.post("/risk-models/identity-target/estimate",
                    json={"identifiers": list(dataset.UNIVERSE),
                          "target": "scaled_identity", "intensity": 0.3},
                    headers=auth())
        client.portal.call(client.app.state.jobs.drain)

        model = client.get("/risk-models/identity-target", headers=auth()).json()

        assert model["diagnostics"]["target"] == "scaled_identity"

    def test_an_unknown_target_fails_the_job(self,
                                             client):
        submitted = client.post("/risk-models/bad-target/estimate",
                                json={"identifiers": list(dataset.UNIVERSE),
                                      "target": "wishful"}, headers=auth())
        client.portal.call(client.app.state.jobs.drain)

        job = client.get(f"/jobs/{submitted.json()['job_id']}",
                         headers=auth()).json()

        assert job["status"] == "failed"
        assert "unknown target" in job["error"]


class TestReads:

    def test_an_unestimated_model_is_a_404(self,
                                           client):
        response = client.get("/risk-models/never-estimated", headers=auth())

        assert response.status_code == 404
        assert "estimate it" in response.json()["error"]["message"]

    def test_re_estimating_supersedes_the_previous_model(self,
                                                         client):
        """The id is the caller's name for "the model I use", so a new estimate
        replaces the last one under it."""
        for window in ((START, "2023-06-30"), (START, END)):
            client.post("/risk-models/rolling/estimate",
                        json={"identifiers": list(dataset.UNIVERSE),
                              "start": window[0], "end": window[1]},
                        headers=auth())
            client.portal.call(client.app.state.jobs.drain)

        model = client.get("/risk-models/rolling", headers=auth()).json()

        assert model["end"] == END
        assert model["diagnostics"]["observations"] > 300

    def test_estimated_models_appear_in_the_listing(self,
                                                    client,
                                                    estimated):
        listing = client.get("/risk-models", headers=auth()).json()["risk_models"]

        assert MODEL_ID in {entry["model_id"] for entry in listing}

    def test_each_listing_entry_summarises_the_model(self,
                                                     client,
                                                     estimated):
        listing = client.get("/risk-models", headers=auth()).json()["risk_models"]
        entry = next(e for e in listing if e["model_id"] == MODEL_ID)

        assert entry["assets"] == len(dataset.UNIVERSE)
        assert entry["positive_semi_definite"] is True


class TestWithoutADataSource:

    def test_estimation_reports_a_missing_data_source(self):
        with tempfile.TemporaryDirectory() as storage:
            config = ServerConfig(auth_token=TOKEN, storage_root=Path(storage))

            with TestClient(create_app(config),
                            raise_server_exceptions=False) as client:
                response = client.post("/risk-models/x/estimate",
                                       json={"identifiers": ["AAA", "BBB"]},
                                       headers=auth())

        assert response.status_code == 500


class TestOpenApi:

    def test_every_endpoint_is_documented(self,
                                          client):
        paths = client.app.openapi()["paths"]

        for path in ("/risk-models",
                     "/risk-models/{model_id}",
                     "/risk-models/{model_id}/estimate"):
            assert path in paths, path


def _index_document() -> dict:
    """An equal-weighted index over the canonical universe."""
    return {
        "id": "idx",
        "name": "Index",
        "base_date": START,
        "base_value": 1000.0,
        "currency": "USD",
        "rebalancing_frequency": "QUARTERLY",
        "universe": {"universe_id": None, "identifiers": list(dataset.UNIVERSE)},
        "pipeline": {
            "selection": [],
            "weighting": {"id": "weighting", "scheme": "EqualWeighted",
                          "params": {}},
            "treatment": {"corporate_actions": "ADJUST_DIVISOR"},
        },
    }


async def _subscribe(registry):
    """Subscribe from inside the app's event loop."""
    return registry.subscribe()


async def _drain_queue(queue):
    """Read everything currently queued."""
    events = []
    while not queue.empty():
        events.append(queue.get_nowait())

    return events


class TestUniverseFromAnIncompleteRun:

    def test_a_run_without_composition_is_refused(self,
                                                  client):
        """A backtest stored before composition was recorded has a level but no
        constituents, so there is no universe to estimate over."""
        client.app.state.jobs._results.write(
            "legacy-run",
            {"job_id": "legacy-run", "kind": "backtest:legacy",
             "status": "succeeded", "completed_at": "2020-01-01T00:00:00+00:00",
             "result": {"level": {"index": [], "data": []}}})

        response = client.post("/risk-models/from-legacy/estimate",
                               json={"index_id": "legacy"}, headers=auth())

        assert response.status_code == 404
        assert "rebalance snapshots" in response.json()["error"]["message"]
