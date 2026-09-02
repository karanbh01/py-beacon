# tests/test_api_contract.py
"""BN-76: contract tests over the whole API, and the reconciliation identities.

Two kinds of test live here, and neither duplicates the per-router files.

**Contract tests** check properties that must hold across *every* endpoint —
that each is authenticated, that failures come back in one envelope, that the
spec is complete enough for a client generator. A per-router test asserts that
its own route behaves; these assert that no route is an exception. The
difference matters because the failure mode is a new router shipping without
auth or without its error responses documented, and nothing in that router's
own file would notice.

**Golden-value tests** pin the reconciliation identities the UI displays. Every
one of them is already asserted somewhere as a property — contributions sum to
the total, factor plus specific equals active risk, Δ sums to zero. Here they
are checked end to end through HTTP against recorded numbers, so a change that
preserves an identity while moving the values still shows up. An identity that
holds at the wrong number is not much use to someone reading a factsheet.
"""
import json
import tempfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from beacon.server import ServerConfig, create_app
from beacon.testing import dataset

TOKEN = "test-token-value"
INDEX_ID = "canon"
SET_ID = "limits"

START = "2023-01-02"
END = "2024-06-28"

# Paths that are deliberately unauthenticated. /health is not: every route is
# guarded, and this list existing at all is what makes an accidental addition
# to it a visible act.
PUBLIC_PATHS: set[str] = set()

# The error codes the envelope promises. A response outside this set means an
# exception escaped its handler.
ERROR_CODES = {"UNAUTHORIZED", "DATA_NOT_FOUND", "VALIDATION_ERROR",
               "CONFIGURATION_ERROR", "CALCULATION_ERROR", "REPORTING_ERROR",
               "MISSING_DEPENDENCY", "NOT_IMPLEMENTED", "INVALID_RULE"}


def auth() -> dict[str, str]:
    return {"Authorization": f"Bearer {TOKEN}"}


def index_document() -> dict:
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


def constraint_set() -> dict:
    return {
        "id": SET_ID,
        "name": "Limits",
        "constraints": [
            {"id": "invested", "type": "FullInvestment", "params": {"target": 1.0}},
            {"id": "cap", "type": "PositionBounds",
             "params": {"minimum": 0.0, "maximum": 0.25}},
            {"id": "tech", "type": "GroupBounds",
             "params": {"name": "Technology",
                        "members": dataset.sectors()["Technology"],
                        "minimum": 0.0, "maximum": 0.20}},
        ],
    }


@pytest.fixture(scope="module")
def client():
    """A server with a backtest and an optimisation already run."""
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
def spec(client):
    return client.app.openapi()


class TestEveryRouteIsGuarded:

    def test_no_route_answers_without_a_token(self,
                                              client,
                                              spec):
        """The failure this catches is a new router shipping unguarded, which
        nothing in that router's own tests would notice."""
        unguarded = []

        for path, operations in spec["paths"].items():
            if path in PUBLIC_PATHS:
                continue

            for method in operations:
                if method not in ("get", "post", "put", "delete"):
                    continue

                url = path.replace("{", "").replace("}", "")
                response = client.request(method.upper(), url)
                if response.status_code != 401:
                    unguarded.append(f"{method.upper()} {path} -> "
                                     f"{response.status_code}")

        assert unguarded == []

    def test_a_wrong_token_is_also_refused(self,
                                           client):
        response = client.get("/health",
                              headers={"Authorization": "Bearer wrong"})

        assert response.status_code == 401

    def test_the_websocket_rejects_a_bad_token(self,
                                               client):
        """Browsers cannot set headers on a handshake, so the token is a query
        parameter — and still has to be checked."""
        import contextlib

        from starlette.websockets import WebSocketDisconnect

        with contextlib.suppress(WebSocketDisconnect), \
                client.websocket_connect("/ws?token=wrong") as socket:
            socket.receive_json()


class TestErrorEnvelope:

    def test_every_failure_uses_the_envelope(self,
                                             client):
        """One shape for every failure, whatever produced it."""
        failures = [
            client.get("/indices/nope", headers=auth()),
            client.get("/data/prices/NOPE", headers=auth()),
            client.get("/risk-models/nope", headers=auth()),
            client.get("/optimise/constraint-sets/nope", headers=auth()),
            client.get("/reports/templates/nope", headers=auth()),
            client.get(f"/beacon/{INDEX_ID}/assets/NOPE", headers=auth()),
        ]

        for response in failures:
            payload = response.json()
            assert "error" in payload, response.url
            assert set(payload["error"]) >= {"code", "message"}, response.url

    def test_every_code_is_one_the_envelope_promises(self,
                                                     client):
        """A code outside the set means an exception escaped its handler."""
        for url in ("/indices/nope", "/risk-models/nope", "/reports/templates/nope"):
            code = client.get(url, headers=auth()).json()["error"]["code"]
            assert code in ERROR_CODES, f"{url} -> {code}"

    def test_an_unauthenticated_failure_uses_it_too(self,
                                                    client):
        payload = client.get("/indices/nope").json()

        assert payload["error"]["code"] == "UNAUTHORIZED"


class TestSpec:

    def test_it_is_serialisable(self,
                                spec):
        """A client generator consumes this file, so it has to be plain JSON —
        no numpy scalars or timestamps that survived into a schema."""
        assert json.loads(json.dumps(spec))

    def test_every_router_is_represented(self,
                                         spec):
        tags = {tag for operations in spec["paths"].values()
                for operation in operations.values()
                if isinstance(operation, dict)
                for tag in operation.get("tags", [])}

        assert {"data", "coverage", "indices", "jobs", "beacon", "optimise",
                "risk", "derivatives", "reports"} <= tags

    def test_every_operation_documents_the_error_envelope(self,
                                                          spec):
        """So a generated client knows failures have a shape, rather than
        discovering it."""
        missing = []

        for path, operations in spec["paths"].items():
            for method, operation in operations.items():
                if method not in ("get", "post", "put", "delete"):
                    continue
                if "401" not in operation.get("responses", {}):
                    missing.append(f"{method.upper()} {path}")

        assert missing == []

    def test_every_operation_has_a_description(self,
                                               spec):
        """The docstrings are the client's documentation."""
        undocumented = [
            f"{method.upper()} {path}"
            for path, operations in spec["paths"].items()
            for method, operation in operations.items()
            if method in ("get", "post", "put", "delete")
            and not (operation.get("description") or operation.get("summary"))
        ]

        assert undocumented == []

    def test_schemas_are_named_rather_than_inline(self,
                                                  spec):
        """Anonymous schemas generate unusable client types."""
        assert len(spec["components"]["schemas"]) > 30


class TestReconciliationIdentities:
    """Golden values for every identity the UI displays.

    Recorded from a run over the canonical dataset. They are pinned rather than
    only asserted as identities because an identity that holds at the wrong
    number is not much use to someone reading a factsheet — and because the
    dataset is fixed, so these cannot drift without something having changed.
    """

    def test_attribution_contributions_sum_to_the_total(self,
                                                        client):
        payload = client.get(f"/beacon/{INDEX_ID}/attribution",
                             headers=auth()).json()
        explained = sum(item["contribution"] for item in payload["contributions"])

        assert payload["reconciles"] is True
        assert explained == pytest.approx(payload["total_return"], abs=1e-12)

    def test_the_index_total_return_is_what_it_was(self,
                                                   client):
        """Golden: the canonical dataset over this window returns 43.25%.

        Recorded rather than derived, so a change that preserves the
        reconciliation identity while moving the number still shows up.
        """
        payload = client.get(f"/beacon/{INDEX_ID}/attribution",
                             headers=auth()).json()

        assert payload["total_return"] == pytest.approx(0.4325, abs=5e-4)

    def test_attribution_defaults_to_the_run_window(self,
                                                    client):
        """Not the whole price history.

        The index was calculated over the window it was run for. Drifting the
        last rebalance forward past that would attribute returns for a period
        the index does not cover, which reads as performance rather than as
        extrapolation — the canonical dataset runs to 2025 while this backtest
        stops in mid-2024, so the two differ by a lot.
        """
        payload = client.get(f"/beacon/{INDEX_ID}/attribution",
                             headers=auth()).json()

        assert payload["start"][:10] >= START
        assert payload["end"][:10] <= END

    def test_every_constituent_contributes(self,
                                           client):
        payload = client.get(f"/beacon/{INDEX_ID}/attribution",
                             headers=auth()).json()

        assert {item["asset_id"] for item in payload["contributions"]} == set(
            dataset.UNIVERSE)

    def test_the_weights_sum_to_one(self,
                                    client):
        payload = client.get(f"/beacon/{INDEX_ID}/weights", headers=auth()).json()

        assert sum(payload["weights"].values()) == pytest.approx(1.0, abs=1e-9)

    def test_equal_weighting_gives_the_expected_effective_count(self,
                                                                client):
        """Golden: six equally weighted names have an effective count of six.
        A different figure means the weighting changed, not the maths."""
        payload = client.get(f"/beacon/{INDEX_ID}/weights", headers=auth()).json()

        assert payload["concentration"]["effective_assets"] == pytest.approx(
            6.0, abs=1e-9)
        assert payload["concentration"]["herfindahl"] == pytest.approx(
            1 / 6, abs=1e-9)

    def test_the_optimiser_active_weights_sum_to_zero(self,
                                                      client):
        result = _optimisation(client)

        assert result["active_sum"] == pytest.approx(0.0, abs=1e-9)

    def test_the_te_squared_identity_holds(self,
                                           client):
        result = _optimisation(client)
        exposures = client.get(f"/optimise/runs/{result['run_id']}/exposures",
                               headers=auth()).json()["risk"]

        assert exposures["reconciles"] is True
        assert exposures["factor_variance"] + exposures["specific_variance"] == (
            pytest.approx(exposures["total_variance"], abs=1e-15))

    def test_the_sector_cap_binds_at_exactly_the_limit(self,
                                                       client):
        """Golden: two of six equal names is 33%, so a 20% cap binds and the
        answer sits on it."""
        result = _optimisation(client)
        technology = sum(row["optimal_weight"] for row in result["weights"]
                         if row["asset_id"] in dataset.sectors()["Technology"])

        assert technology == pytest.approx(0.20, abs=1e-6)

    def test_the_risk_matrix_diagonal_is_unit(self,
                                              client):
        client.post("/risk-models/contract/estimate",
                    json={"identifiers": list(dataset.UNIVERSE),
                          "start": START, "end": END}, headers=auth())
        client.portal.call(client.app.state.jobs.drain)

        model = client.get("/risk-models/contract", headers=auth()).json()
        diagonal = [row[position] for position, row
                    in enumerate(model["correlation"]["data"])]

        assert all(value == pytest.approx(1.0, abs=1e-12) for value in diagonal)

    def test_the_futures_fair_value_is_the_closed_form(self,
                                                       client):
        """Golden: 100 at 5% less 2% over half a year is 101.5113."""
        payload = client.post("/derivatives/futures/price",
                              json={"spot": 100.0, "risk_free_rate": 0.05,
                                    "dividend_yield": 0.02,
                                    "time_to_expiry": 0.5},
                              headers=auth()).json()

        assert payload["fair_value"] == pytest.approx(101.5113, abs=5e-4)

    def test_the_trs_accrual_is_exactly_act_360(self,
                                                client):
        """Golden: 10m at 4.5% over 91 days is 113,750."""
        payload = client.post("/derivatives/trs/price",
                              json={"start_date": "2024-01-01",
                                    "end_date": "2025-01-01",
                                    "notional": 10_000_000.0,
                                    "spread_bps": 50.0,
                                    "reference_rate_value": 0.04,
                                    "valuation_date": "2024-04-01",
                                    "last_reset_date": "2024-01-01",
                                    "spot": 105.0, "initial_price": 100.0},
                              headers=auth()).json()

        assert payload["financing_leg"] == pytest.approx(
            10_000_000.0 * 0.045 * 91 / 360, rel=1e-12)


def _optimisation(client) -> dict:
    """Run an optimisation once and reuse it."""
    if not hasattr(_optimisation, "cached"):
        submitted = client.post("/optimise/runs",
                                json={"index_id": INDEX_ID,
                                      "constraint_set_id": SET_ID,
                                      "start": START, "end": END},
                                headers=auth())
        client.portal.call(client.app.state.jobs.drain)

        job = client.get(f"/jobs/{submitted.json()['job_id']}",
                         headers=auth()).json()
        assert job["status"] == "succeeded", job.get("error")

        _optimisation.cached = job["result"]

    return _optimisation.cached


def _run_export(destination: Path) -> dict:
    """Load and run scripts/export_openapi.py against a destination."""
    import importlib.util

    path = Path(__file__).resolve().parent.parent / "scripts" / "export_openapi.py"
    spec_loader = importlib.util.spec_from_file_location("export_openapi", path)
    assert spec_loader is not None and spec_loader.loader is not None

    module = importlib.util.module_from_spec(spec_loader)
    spec_loader.loader.exec_module(module)

    return module.export(destination)


@pytest.fixture(scope="module")
def exported(tmp_path_factory):
    """One export, shared by every test that only reads the result.

    Each export builds the whole app and generates the spec (~1s), and five
    tests were paying it separately -- a third of this file's runtime for
    identical output (BN-159). Tests exercising the *write path* itself (the
    stability re-export, the directory creation) keep their own calls.
    """
    destination = tmp_path_factory.mktemp("spec") / "openapi.json"
    spec = _run_export(destination)

    return spec, destination


class TestSpecExport:
    """The artifact beacon-ui generates its client from."""

    def test_it_writes_a_file(self,
                              exported):
        _, destination = exported

        assert destination.exists()
        assert json.loads(destination.read_text(encoding="utf-8"))["openapi"]

    def test_it_needs_no_data_source(self,
                                     exported):
        """Routes and schemas are declared at import time, so the export runs
        as its own CI step without a fixture."""
        spec, _ = exported

        assert len(spec["paths"]) > 30

    def test_the_output_is_stable(self,
                                  exported,
                                  tmp_path):
        """The file is committed to a client repository and diffed there, so an
        unstable key order would make every regeneration look like a change.

        One fresh export compared against the shared one: two independent
        runs, which is what stability means."""
        _, first = exported
        second = tmp_path / "b.json"

        _run_export(second)

        assert first.read_bytes() == second.read_bytes()

    def test_it_matches_the_running_app(self,
                                        client,
                                        exported):
        """A spec exported without a data source must describe the same API as
        one serving with it, or the client is generated against a fiction."""
        spec, _ = exported

        assert set(spec["paths"]) == set(client.app.openapi()["paths"])

    def test_the_creating_directories_path_works(self,
                                                 tmp_path):
        """Its own export on purpose: the property under test is the write
        path creating missing directories, not the spec content."""
        destination = tmp_path / "nested" / "deep" / "openapi.json"
        _run_export(destination)

        assert destination.exists()


class TestFuzzStore:
    """The store the nightly schemathesis run serves.

    It used to be an *application*, loaded in-process by `schemathesis run
    --app`. That option was removed in schemathesis v4, so the server is now
    started for real and fuzzed over a socket -- which needs a store on disk
    rather than a fetcher in memory.
    """

    @staticmethod
    def _script():
        import importlib.util

        path = Path(__file__).resolve().parent.parent / "scripts" / "fuzz_store.py"
        loader = importlib.util.spec_from_file_location("fuzz_store", path)

        assert loader is not None and loader.loader is not None

        module = importlib.util.module_from_spec(loader)
        loader.loader.exec_module(module)

        return module

    def test_it_writes_a_store_the_server_can_load(self,
                                                  tmp_path):
        """End to end, because the workflow's next step is the server reading
        exactly this directory."""
        from beacon.data import store

        destination = tmp_path / "fuzzstore"

        assert self._script().main([str(destination)]) == 0

        fetcher = store.load(destination)

        assert len(fetcher.identifiers) > 0
        assert fetcher.reference is not None

    def test_it_refuses_to_run_without_a_destination(self,
                                                     capsys):
        """Exits rather than guessing, so a workflow that lost its argument
        fails at the step that wrote it rather than at the one that reads it."""
        assert self._script().main([]) == 2
        assert "usage" in capsys.readouterr().err
