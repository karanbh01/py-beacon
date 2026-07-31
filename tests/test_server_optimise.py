# tests/test_server_optimise.py
"""BN-72: optimiser endpoints — constraint sets, runs, frontier, exposures."""
import tempfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from beacon.server import ServerConfig, create_app
from beacon.testing import dataset

TOKEN = "test-token-value"
INDEX_ID = "canon"
SET_ID = "house-limits"

START = "2023-01-02"
END = "2024-06-28"

TECHNOLOGY = dataset.sectors()["Technology"]


def auth() -> dict[str, str]:
    return {"Authorization": f"Bearer {TOKEN}"}


def index_document() -> dict:
    """An equal-weighted index over the canonical universe."""
    return {
        "id": INDEX_ID,
        "name": "Canonical Index",
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


def constraint_set(set_id: str = SET_ID) -> dict:
    """Full investment, a 25% position cap, and a sector limit that binds."""
    return {
        "id": set_id,
        "name": "House limits",
        "constraints": [
            {"id": "invested", "type": "FullInvestment", "params": {"target": 1.0}},
            {"id": "position-cap", "type": "PositionBounds",
             "params": {"minimum": 0.0, "maximum": 0.25}},
            {"id": "tech-cap", "type": "GroupBounds",
             "params": {"name": "Technology", "members": TECHNOLOGY,
                        "minimum": 0.0, "maximum": 0.20}},
        ],
    }


@pytest.fixture(scope="module")
def client():
    """A server with an index backtested and an optimisation run."""
    with tempfile.TemporaryDirectory() as storage:
        config = ServerConfig(auth_token=TOKEN,
                              data_fetcher=dataset.data_fetcher(),
                              storage_root=Path(storage))

        with TestClient(create_app(config),
                        raise_server_exceptions=False) as started:
            started.put(f"/indices/{INDEX_ID}", json=index_document(),
                        headers=auth())
            started.put(f"/optimise/constraint-sets/{SET_ID}",
                        json=constraint_set(), headers=auth())
            started.post(f"/beacon/{INDEX_ID}/backtest",
                         json={"start": START, "end": END}, headers=auth())
            started.portal.call(started.app.state.jobs.drain)

            yield started


@pytest.fixture(scope="module")
def run_id(client):
    """A completed optimisation run."""
    response = client.post("/optimise/runs",
                           json={"index_id": INDEX_ID,
                                 "constraint_set_id": SET_ID,
                                 "start": START, "end": END},
                           headers=auth())
    client.portal.call(client.app.state.jobs.drain)

    job = client.get(f"/jobs/{response.json()['job_id']}", headers=auth()).json()
    assert job["status"] == "succeeded", job.get("error")

    return job["result"]["run_id"]


@pytest.fixture(scope="module")
def result(client,
           run_id):
    """The stored run payload."""
    return client.app.state.jobs.latest_result(f"optimise:{run_id}")


class TestConstraintTypes:

    def test_it_lists_every_type_the_solver_knows(self,
                                                  client):
        """Served so a client builds its editor from the same source the
        solver reads, rather than from a copy that drifts."""
        types = client.get("/optimise/constraint-types", headers=auth()).json()["types"]

        assert {"FullInvestment", "PositionBounds", "GroupBounds",
                "TurnoverBudget", "Cardinality"} <= set(types)

    def test_each_type_names_its_parameters(self,
                                            client):
        types = client.get("/optimise/constraint-types", headers=auth()).json()["types"]

        assert types["PositionBounds"] == ["assets", "maximum", "minimum"]


class TestConstraintSetCrud:

    def test_a_set_round_trips(self,
                               client):
        stored = client.get(f"/optimise/constraint-sets/{SET_ID}",
                            headers=auth()).json()

        assert stored["id"] == SET_ID
        assert len(stored["constraints"]) == 3

    def test_it_appears_in_the_listing(self,
                                       client):
        listing = client.get("/optimise/constraint-sets",
                             headers=auth()).json()["constraint_sets"]

        assert SET_ID in {entry["id"] for entry in listing}

    def test_the_path_id_wins_over_the_body(self,
                                            client):
        """A document cannot be saved under one id while claiming another."""
        body = constraint_set(set_id="claims-something-else")

        saved = client.put("/optimise/constraint-sets/actual-id",
                           json=body, headers=auth()).json()

        assert saved["constraint_set"]["id"] == "actual-id"
        client.delete("/optimise/constraint-sets/actual-id", headers=auth())

    def test_an_unknown_set_is_a_404(self,
                                     client):
        assert client.get("/optimise/constraint-sets/nope",
                          headers=auth()).status_code == 404

    def test_deleting_removes_it(self,
                                 client):
        client.put("/optimise/constraint-sets/temporary",
                   json=constraint_set("temporary"), headers=auth())

        assert client.delete("/optimise/constraint-sets/temporary",
                             headers=auth()).status_code == 204
        assert client.get("/optimise/constraint-sets/temporary",
                          headers=auth()).status_code == 404

    def test_deleting_something_absent_is_a_404(self,
                                                client):
        assert client.delete("/optimise/constraint-sets/never-existed",
                             headers=auth()).status_code == 404

    def test_it_requires_authentication(self,
                                        client):
        assert client.get("/optimise/constraint-sets").status_code == 401


class TestConstraintSetValidation:

    def _validate(self,
                  client,
                  rows: list[dict]) -> dict:
        return client.post("/optimise/constraint-sets/validate",
                           json={"id": "x", "name": "X", "constraints": rows},
                           headers=auth()).json()

    def test_a_good_set_is_valid(self,
                                 client):
        report = self._validate(client, constraint_set()["constraints"])

        assert report["valid"] is True

    def test_an_unknown_type_is_an_error(self,
                                         client):
        report = self._validate(client, [{"type": "Vibes", "params": {}}])

        assert report["valid"] is False
        assert report["findings"][0]["code"] == "UNKNOWN_CONSTRAINT_TYPE"

    def test_an_unknown_parameter_is_an_error(self,
                                              client):
        report = self._validate(client, [
            {"type": "Cardinality", "params": {"maximum": 5, "colour": "blue"}}])

        codes = {finding["code"] for finding in report["findings"]}

        assert "UNKNOWN_PARAMETER" in codes

    def test_the_class_s_own_validation_surfaces(self,
                                                 client):
        """Not restated here: the classes already reject a minimum above a
        maximum, and duplicating the rule would let the two drift."""
        report = self._validate(client, [
            {"type": "PositionBounds", "params": {"minimum": 0.5, "maximum": 0.1}}])

        assert report["valid"] is False
        assert any(finding["code"] == "INVALID_PARAMETER"
                   for finding in report["findings"])

    def test_a_missing_parameter_is_reported(self,
                                             client):
        report = self._validate(client, [{"type": "Cardinality", "params": {}}])

        assert any(finding["code"] == "MISSING_PARAMETER"
                   for finding in report["findings"])

    def test_every_problem_is_reported_not_just_the_first(self,
                                                          client):
        """Someone fixing a constraint editor needs all the errors."""
        report = self._validate(client, [
            {"type": "Nope", "params": {}},
            {"type": "Cardinality", "params": {"maximum": 0}}])

        assert len(report["findings"]) >= 2

    def test_findings_address_the_offending_row(self,
                                                client):
        report = self._validate(client, [
            {"id": "invested", "type": "FullInvestment", "params": {}},
            {"id": "bad", "type": "Cardinality", "params": {"maximum": 0}}])

        offending = [f for f in report["findings"] if f["rule_id"] == "bad"]

        assert offending
        assert offending[0]["path"].startswith("constraints[1]")

    def test_two_full_investment_rows_are_an_error(self,
                                                   client):
        report = self._validate(client, [
            {"type": "FullInvestment", "params": {"target": 1.0}},
            {"type": "FullInvestment", "params": {"target": 0.9}}])

        assert any(finding["code"] == "DUPLICATE_FULL_INVESTMENT"
                   for finding in report["findings"])

    def test_duplicate_row_ids_are_an_error(self,
                                            client):
        """A binding constraint could not be traced back to one row."""
        report = self._validate(client, [
            {"id": "same", "type": "FullInvestment", "params": {}},
            {"id": "same", "type": "Cardinality", "params": {"maximum": 3}}])

        assert any(finding["code"] == "DUPLICATE_ROW_ID"
                   for finding in report["findings"])

    def test_no_investment_target_is_a_warning_not_an_error(self,
                                                            client):
        report = self._validate(client, [
            {"type": "PositionBounds", "params": {"minimum": 0.0, "maximum": 0.5}}])

        assert report["valid"] is True
        assert any(finding["code"] == "NO_INVESTMENT_TARGET"
                   for finding in report["findings"])

    def test_cardinality_warns_that_it_is_a_heuristic(self,
                                                      client):
        report = self._validate(client, [
            {"type": "FullInvestment", "params": {}},
            {"type": "Cardinality", "params": {"maximum": 3}}])

        assert report["valid"] is True
        assert any(finding["code"] == "NON_CONVEX_CONSTRAINT"
                   for finding in report["findings"])

    def test_an_empty_set_warns(self,
                                client):
        report = self._validate(client, [])

        assert any(finding["code"] == "NO_CONSTRAINTS"
                   for finding in report["findings"])

    def test_saving_an_invalid_set_is_refused(self,
                                              client):
        response = client.put("/optimise/constraint-sets/broken",
                              json={"id": "broken", "name": "Broken",
                                    "constraints": [{"type": "Nope", "params": {}}]},
                              headers=auth())

        assert response.status_code == 422


class TestRun:

    def test_the_job_succeeds(self,
                              result):
        assert result["index_id"] == INDEX_ID
        assert result["converged"] is True

    def test_active_weights_sum_to_zero(self,
                                        result):
        """The acceptance criterion: rearranging weight cannot create any."""
        assert result["active_sum"] == pytest.approx(0.0, abs=1e-9)

        total = sum(row["active_weight"] for row in result["weights"])
        assert total == pytest.approx(0.0, abs=1e-9)

    def test_every_name_carries_all_three_weights(self,
                                                  result):
        for row in result["weights"]:
            assert row["active_weight"] == pytest.approx(
                row["optimal_weight"] - row["index_weight"], abs=1e-12)

    def test_the_optimal_weights_are_fully_invested(self,
                                                    result):
        total = sum(row["optimal_weight"] for row in result["weights"])

        assert total == pytest.approx(1.0, abs=1e-9)

    def test_the_position_cap_is_respected(self,
                                           result):
        assert all(row["optimal_weight"] <= 0.25 + 1e-9
                   for row in result["weights"])

    def test_the_sector_cap_is_respected(self,
                                         result):
        technology = sum(row["optimal_weight"] for row in result["weights"]
                         if row["asset_id"] in TECHNOLOGY)

        assert technology <= 0.20 + 1e-9

    def test_binding_constraints_are_reported(self,
                                              result):
        """The acceptance criterion. The sector cap has to bite: two of six
        equally weighted names is 33%, well above the 20% limit."""
        labels = [entry["label"] for entry in result["binding"]]

        assert labels
        assert any("Technology" in label for label in labels)

    def test_a_binding_constraint_names_the_row_that_produced_it(self,
                                                                 result):
        """So a client can highlight the row rather than parse a label."""
        rows = {entry["row_id"] for entry in result["binding"]}

        assert "tech-cap" in rows

    def test_rows_are_ordered_by_absolute_active_weight(self,
                                                        result):
        magnitudes = [abs(row["active_weight"]) for row in result["weights"]]

        assert magnitudes == sorted(magnitudes, reverse=True)

    def test_diagnostics_are_carried(self,
                                     result):
        assert result["iterations"] >= 0
        assert result["objective"] >= 0.0
        assert result["solver_message"]

    def test_tracking_error_is_positive_when_constraints_bind(self,
                                                              result):
        assert result["tracking_error"] > 0.0

    def test_an_unknown_constraint_set_is_a_404(self,
                                                client):
        response = client.post("/optimise/runs",
                               json={"index_id": INDEX_ID,
                                     "constraint_set_id": "nope"},
                               headers=auth())

        assert response.status_code == 404

    def test_an_index_without_a_backtest_is_a_404(self,
                                                  client):
        client.put("/indices/unrun", json={**index_document(), "id": "unrun"},
                   headers=auth())

        response = client.post("/optimise/runs",
                               json={"index_id": "unrun",
                                     "constraint_set_id": SET_ID},
                               headers=auth())

        assert response.status_code == 404

    def test_an_infeasible_set_fails_the_job_with_a_reason(self,
                                                           client):
        """The optimiser refuses rather than fudging, and the message says
        what is impossible."""
        client.put("/optimise/constraint-sets/impossible",
                   json={"id": "impossible", "name": "Impossible",
                         "constraints": [
                             {"type": "FullInvestment", "params": {"target": 1.0}},
                             {"type": "PositionBounds",
                              "params": {"minimum": 0.0, "maximum": 0.05}}]},
                   headers=auth())

        submitted = client.post("/optimise/runs",
                                json={"index_id": INDEX_ID,
                                      "constraint_set_id": "impossible"},
                                headers=auth())
        client.portal.call(client.app.state.jobs.drain)

        job = client.get(f"/jobs/{submitted.json()['job_id']}",
                         headers=auth()).json()

        assert job["status"] == "failed"
        assert "cannot reach" in job["error"]


class TestFrontier:

    @pytest.fixture(scope="class")
    def payload(self,
                client,
                run_id):
        return client.get(f"/optimise/runs/{run_id}/frontier",
                          params={"risk_free_rate": 0.02}, headers=auth()).json()

    def test_it_traces_a_grid(self,
                              payload):
        assert len(payload["points"]) == 15

    def test_risk_rises_with_return(self,
                                    payload):
        assert payload["monotonic"] is True

    def test_it_names_the_minimum_variance_point(self,
                                                 payload):
        volatilities = [point["volatility"] for point in payload["points"]]

        assert payload["minimum_variance"]["volatility"] == pytest.approx(
            min(volatilities), abs=1e-6)

    def test_the_tangency_is_the_best_sharpe_found(self,
                                                   payload):
        best = max(point["sharpe_ratio"] for point in payload["points"])

        assert payload["tangency"]["sharpe_ratio"] >= best - 1e-9

    def test_every_point_respects_the_constraint_set(self,
                                                     payload):
        """The frontier solves under the same constraints as the run."""
        for point in payload["points"]:
            assert sum(point["weights"].values()) == pytest.approx(1.0, abs=1e-6)
            assert all(weight <= 0.25 + 1e-6 for weight in point["weights"].values())

    def test_the_expected_returns_are_reported(self,
                                               payload):
        """A poor forecast, so the client should be able to see exactly what
        was assumed rather than infer it."""
        assert set(payload["expected_returns"]) == set(dataset.UNIVERSE)

    def test_the_risk_free_rate_is_echoed(self,
                                          payload):
        assert payload["risk_free_rate"] == pytest.approx(0.02)

    def test_an_unknown_run_is_a_404(self,
                                     client):
        assert client.get("/optimise/runs/nope/frontier",
                          headers=auth()).status_code == 404


class TestExposures:

    @pytest.fixture(scope="class")
    def payload(self,
                client,
                run_id):
        return client.get(f"/optimise/runs/{run_id}/exposures",
                          headers=auth()).json()

    def test_it_reports_the_factors_it_could_build(self,
                                                   payload):
        """Size, momentum and volatility come from price and share count.
        Value and quality are absent rather than approximated."""
        assert payload["factors"] == ["market", "size", "momentum", "volatility"]

    def test_the_te_squared_identity_holds(self,
                                           payload):
        """The acceptance criterion.

        Exact rather than approximate: the covariance is *defined* as
        BFBᵀ + D, so the split is algebra. Pair an arbitrary covariance with
        arbitrary loadings and there is a cross term.
        """
        risk = payload["risk"]

        assert risk["reconciles"] is True
        assert abs(risk["residual"]) < 1e-15
        assert risk["factor_variance"] + risk["specific_variance"] == pytest.approx(
            risk["total_variance"], abs=1e-15)

    def test_both_parts_are_non_negative(self,
                                         payload):
        """F is PSD and D is positive, so neither can come out negative."""
        risk = payload["risk"]

        assert risk["factor_variance"] >= 0.0
        assert risk["specific_variance"] >= 0.0

    def test_tracking_error_is_the_root_of_the_total(self,
                                                     payload):
        risk = payload["risk"]

        assert risk["tracking_error"] == pytest.approx(
            risk["total_variance"] ** 0.5, rel=1e-9)

    def test_factor_contributions_sum_to_the_factor_variance(self,
                                                             payload):
        risk = payload["risk"]

        assert sum(risk["contributions"].values()) == pytest.approx(
            risk["factor_variance"], rel=1e-9)

    def test_it_reports_all_three_exposure_sets(self,
                                                payload):
        for key in ("index_exposures", "optimal_exposures", "active_exposures"):
            assert {row["factor"] for row in payload[key]} == set(payload["factors"])

    def test_active_exposure_is_the_difference(self,
                                               payload):
        index = {row["factor"]: row["exposure"] for row in payload["index_exposures"]}
        optimal = {row["factor"]: row["exposure"]
                   for row in payload["optimal_exposures"]}

        for row in payload["active_exposures"]:
            assert row["exposure"] == pytest.approx(
                optimal[row["factor"]] - index[row["factor"]], abs=1e-9)

    def test_the_active_market_exposure_nets_out(self,
                                                 payload):
        """Both sides are fully invested, so the intercept cancels."""
        active = {row["factor"]: row["exposure"]
                  for row in payload["active_exposures"]}

        assert active["market"] == pytest.approx(0.0, abs=1e-9)

    def test_r_squared_is_reported(self,
                                   payload):
        assert 0.0 <= payload["r_squared"] <= 1.0

    def test_an_unknown_run_is_a_404(self,
                                     client):
        assert client.get("/optimise/runs/nope/exposures",
                          headers=auth()).status_code == 404


class TestOpenApi:

    def test_every_endpoint_is_documented(self,
                                          client):
        paths = client.app.openapi()["paths"]

        for path in ("/optimise/constraint-types",
                     "/optimise/constraint-sets",
                     "/optimise/constraint-sets/{set_id}",
                     "/optimise/runs",
                     "/optimise/runs/{run_id}/frontier",
                     "/optimise/runs/{run_id}/exposures"):
            assert path in paths, path

    def test_validate_is_not_captured_by_the_set_route(self,
                                                       client):
        """`/constraint-sets/validate` must not read as a set called
        "validate"."""
        response = client.post("/optimise/constraint-sets/validate",
                               json={"id": "x", "name": "X", "constraints": []},
                               headers=auth())

        assert response.status_code == 200


class TestDerivationEdges:
    """Paths the happy route never reaches, driven directly."""

    def test_no_prices_for_any_name_is_refused(self):
        import pandas as pd

        from beacon.data.base import MarketData
        from beacon.data.fetcher import DataFetcher
        from beacon.exceptions import DataNotFoundError
        from beacon.server.optimisation import constituent_prices

        empty = DataFetcher(MarketData.from_dataframe(pd.DataFrame(
            {"IDENTIFIER": ["OTHER"], "DATE": ["2024-01-01"], "CLOSE": [1.0]})))

        with pytest.raises(DataNotFoundError, match="prices for any"):
            constituent_prices(empty, ["AAA", "BBB"])

    def test_momentum_is_zero_without_enough_history(self):
        """A momentum factor needs more than a month of prices to skip one."""
        import pandas as pd

        from beacon.server.optimisation import _momentum

        short = pd.DataFrame({"AAA": [1.0, 1.1, 1.2]},
                             index=pd.bdate_range("2024-01-01", periods=3))

        assert (_momentum(short) == 0.0).all()

    def test_an_as_of_before_the_first_rebalance_is_refused(self,
                                                            client):
        """Returning the first rebalance would target weights not yet in
        force."""
        from beacon.exceptions import DataNotFoundError
        from beacon.server.optimisation import target_weights_from

        backtest = client.app.state.jobs.latest_result(f"backtest:{INDEX_ID}")

        with pytest.raises(DataNotFoundError, match="a rebalance on or before"):
            target_weights_from(backtest, "2019-01-01")

    def test_an_as_of_selects_the_rebalance_in_force(self,
                                                     client):
        from beacon.server.optimisation import target_weights_from

        backtest = client.app.state.jobs.latest_result(f"backtest:{INDEX_ID}")

        weights = target_weights_from(backtest, "2023-08-15")

        assert set(weights) == set(dataset.UNIVERSE)
        assert sum(weights.values()) == pytest.approx(1.0, abs=1e-9)

    def test_a_run_without_composition_is_refused(self):
        from beacon.exceptions import DataNotFoundError
        from beacon.server.optimisation import target_weights_from

        with pytest.raises(DataNotFoundError, match="rebalance snapshots"):
            target_weights_from({"level": {}}, None)


class TestWithoutADataSource:

    def test_submitting_a_run_reports_a_missing_data_source(self):
        with tempfile.TemporaryDirectory() as storage:
            config = ServerConfig(auth_token=TOKEN, storage_root=Path(storage))

            with TestClient(create_app(config),
                            raise_server_exceptions=False) as client:
                client.put(f"/optimise/constraint-sets/{SET_ID}",
                           json=constraint_set(), headers=auth())
                response = client.post("/optimise/runs",
                                       json={"index_id": INDEX_ID,
                                             "constraint_set_id": SET_ID},
                                       headers=auth())

        # No backtest exists either, so this is a 404 before the data source is
        # reached — which is the right order: the missing run is the nearer
        # problem.
        assert response.status_code == 404


class TestSubmittingAnInvalidStoredSet:

    def test_a_set_that_became_invalid_is_refused_at_submission(self,
                                                                client):
        """Validation runs again at submission, not only at save.

        A document can reach the store by another route — a hand-edited file, a
        restored backup — and a job that fails a moment later is a worse way to
        find out.
        """
        client.app.state.constraint_store.write(
            "hand-edited",
            {"id": "hand-edited", "name": "Hand edited",
             "constraints": [{"id": "x", "type": "Nope", "params": {}}]})

        response = client.post("/optimise/runs",
                               json={"index_id": INDEX_ID,
                                     "constraint_set_id": "hand-edited"},
                               headers=auth())

        assert response.status_code == 422
