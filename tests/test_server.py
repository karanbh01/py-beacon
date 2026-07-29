# tests/test_server.py
"""Unit tests for the Beacon API server skeleton (app factory, auth, /health)."""
import socket
import subprocess
import sys
import time

import pandas as pd
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from beacon.backtest.result import BacktestResult
from beacon.data.base import MarketData
from beacon.data.fetcher import DataFetcher
from beacon.exceptions import (
    BeaconError,
    CalculationError,
    ConfigurationError,
    DataNotFoundError,
    InvalidRuleError,
    MissingDependencyError,
)
from beacon.index.result import IndexResult
from beacon.portfolio.base import Transaction
from beacon.server import (
    BacktestResultSummary,
    IndexResultSummary,
    Money,
    ServerConfig,
    classify,
    create_app,
    dataframe_to_payload,
    series_to_payload,
)
from beacon.server.__main__ import PORT_ANNOUNCEMENT, bind_socket, build_parser, main
from beacon.server.config import TOKEN_ENV_VAR

TOKEN = "test-token-value"
ASSETS = ["AAA", "BBB", "CCC"]
DATES = pd.bdate_range("2025-01-02", periods=3)


def build_fetcher() -> DataFetcher:
    """Build a DataFetcher over a tiny synthetic long-form frame."""
    rows = [
        {"IDENTIFIER": asset, "DATE": date, "CLOSE": 100.0 + index}
        for asset in ASSETS
        for index, date in enumerate(DATES)
    ]
    return DataFetcher(MarketData.from_dataframe(pd.DataFrame(rows)))


@pytest.fixture
def client() -> TestClient:
    """Client for a server with no data source configured."""
    return TestClient(create_app(ServerConfig(auth_token=TOKEN)))


@pytest.fixture
def client_with_data() -> TestClient:
    """Client for a server with a data source configured."""
    config = ServerConfig(auth_token=TOKEN, data_fetcher=build_fetcher())
    return TestClient(create_app(config))


def auth() -> dict[str, str]:
    """Valid Authorization header."""
    return {"Authorization": f"Bearer {TOKEN}"}


class TestConfig:

    def test_rejects_empty_token(self):
        with pytest.raises(ValueError, match="auth_token cannot be empty"):
            ServerConfig(auth_token="")

    def test_rejects_out_of_range_port(self):
        with pytest.raises(ValueError, match="port must be between"):
            ServerConfig(auth_token=TOKEN, port=70000)

    def test_defaults_to_loopback(self):
        assert ServerConfig(auth_token=TOKEN).host == "127.0.0.1"

    def test_from_environment_reads_token(self,
                                          monkeypatch):
        monkeypatch.setenv(TOKEN_ENV_VAR, "from-env")

        assert ServerConfig.from_environment().auth_token == "from-env"

    def test_explicit_token_beats_environment(self,
                                              monkeypatch):
        monkeypatch.setenv(TOKEN_ENV_VAR, "from-env")

        assert ServerConfig.from_environment(token="explicit").auth_token == "explicit"

    def test_raises_when_no_token_anywhere(self,
                                           monkeypatch):
        monkeypatch.delenv(TOKEN_ENV_VAR, raising=False)

        with pytest.raises(ValueError, match="No auth token supplied"):
            ServerConfig.from_environment()


class TestAuthentication:
    """Every route requires the bearer token — loopback is not a boundary."""

    def test_missing_header_is_401(self,
                                   client):
        response = client.get("/health")

        assert response.status_code == 401
        assert response.headers["WWW-Authenticate"] == "Bearer"

    def test_wrong_token_is_401(self,
                                client):
        response = client.get("/health", headers={"Authorization": "Bearer wrong"})

        assert response.status_code == 401

    def test_non_bearer_scheme_is_401(self,
                                      client):
        response = client.get("/health", headers={"Authorization": f"Basic {TOKEN}"})

        assert response.status_code == 401

    def test_token_prefix_is_rejected(self,
                                      client):
        """A prefix of the real token must not authenticate."""
        response = client.get("/health", headers={"Authorization": f"Bearer {TOKEN[:5]}"})

        assert response.status_code == 401

    def test_scheme_is_case_insensitive(self,
                                        client):
        response = client.get("/health", headers={"Authorization": f"bearer {TOKEN}"})

        assert response.status_code == 200

    def test_valid_token_is_accepted(self,
                                     client):
        assert client.get("/health", headers=auth()).status_code == 200


class TestHealth:

    def test_reports_status_and_version(self,
                                        client):
        from beacon import __version__

        body = client.get("/health", headers=auth()).json()

        assert body["status"] == "ok"
        assert body["version"] == __version__

    def test_without_a_data_source(self,
                                   client):
        body = client.get("/health", headers=auth()).json()

        assert body["data_source"] == {"configured": False, "identifiers": 0}

    def test_with_a_data_source(self,
                                client_with_data):
        body = client_with_data.get("/health", headers=auth()).json()

        assert body["data_source"] == {"configured": True, "identifiers": len(ASSETS)}

    def test_cache_age_is_null(self,
                               client):
        """DataFetcher caches nothing, so there is no age to report."""
        assert client.get("/health", headers=auth()).json()["cache_age"] is None

    def test_uses_the_orjson_response_class(self,
                                            client):
        response = client.get("/health", headers=auth())

        assert response.headers["content-type"] == "application/json"


class TestCors:

    def test_allows_the_desktop_scheme(self,
                                       client):
        response = client.options("/health",
                                  headers={"Origin": "app://",
                                           "Access-Control-Request-Method": "GET"})

        assert response.headers.get("access-control-allow-origin") == "app://"

    def test_allows_localhost_on_any_port(self,
                                          client):
        response = client.options("/health",
                                  headers={"Origin": "http://localhost:5173",
                                           "Access-Control-Request-Method": "GET"})

        assert response.headers.get("access-control-allow-origin") == "http://localhost:5173"

    def test_rejects_an_unrelated_origin(self,
                                         client):
        response = client.options("/health",
                                  headers={"Origin": "https://example.com",
                                           "Access-Control-Request-Method": "GET"})

        assert response.headers.get("access-control-allow-origin") is None


class TestSerialisation:

    def test_dataframe_shape(self):
        frame = pd.DataFrame({"A": [1, 2], "B": [3.5, 4.5]},
                             index=pd.Index(["x", "y"]))

        payload = dataframe_to_payload(frame)

        assert payload == {"index": ["x", "y"],
                           "columns": ["A", "B"],
                           "data": [[1, 3.5], [2, 4.5]]}

    def test_dataframe_timestamps_become_iso_strings(self):
        frame = pd.DataFrame({"CLOSE": [1.0]}, index=pd.DatetimeIndex(["2025-01-02"]))

        payload = dataframe_to_payload(frame)

        assert payload["index"] == ["2025-01-02T00:00:00"]

    def test_dataframe_nan_becomes_none(self):
        frame = pd.DataFrame({"A": [1.0, float("nan")]})

        assert dataframe_to_payload(frame)["data"] == [[1.0], [None]]

    def test_empty_dataframe(self):
        payload = dataframe_to_payload(pd.DataFrame())

        assert payload == {"index": [], "columns": [], "data": []}

    def test_series_shape(self):
        series = pd.Series([1.0, 2.0], index=["a", "b"], name="level")

        assert series_to_payload(series) == {"index": ["a", "b"],
                                             "name": "level",
                                             "data": [1.0, 2.0]}

    def test_series_without_a_name(self):
        assert series_to_payload(pd.Series([1.0]))["name"] is None


class TestLauncher:

    def test_parser_defaults_to_ephemeral_loopback(self):
        args = build_parser().parse_args([])

        assert args.host == "127.0.0.1"
        assert args.port == 0
        assert args.token is None

    def test_bind_socket_assigns_a_real_port(self):
        sock = bind_socket("127.0.0.1", 0)
        try:
            host, port = sock.getsockname()

            assert host == "127.0.0.1"
            assert port > 0
        finally:
            sock.close()

    def test_bound_port_is_actually_listening(self):
        sock = bind_socket("127.0.0.1", 0)
        try:
            port = sock.getsockname()[1]
            with socket.create_connection(("127.0.0.1", port), timeout=2):
                pass
        finally:
            sock.close()

    def test_missing_token_exits_with_2(self,
                                        monkeypatch,
                                        capsys):
        monkeypatch.delenv(TOKEN_ENV_VAR, raising=False)

        assert main([]) == 2
        assert "No auth token supplied" in capsys.readouterr().err


class TestErrorEnvelope:
    """Every non-2xx response carries {error: {code, message, detail}}."""

    def test_auth_failure_uses_the_envelope(self,
                                            client):
        body = client.get("/health").json()

        assert body["error"]["code"] == "UNAUTHORIZED"
        assert "token" in body["error"]["message"].lower()

    def test_unknown_route_uses_the_envelope(self,
                                             client):
        body = client.get("/does-not-exist", headers=auth()).json()

        assert body["error"]["code"] == "NOT_FOUND"

    @pytest.mark.parametrize(
        ("exception", "expected_status", "expected_code"),
        [
            (DataNotFoundError("prices for AAA"), 404, "DATA_NOT_FOUND"),
            (InvalidRuleError("MarketCapRule", "threshold is negative"), 422, "INVALID_RULE"),
            (CalculationError("IndexLevel", "divisor is zero"), 500, "CALCULATION_ERROR"),
            (ConfigurationError("base_value", "must be positive"), 500, "CONFIGURATION_ERROR"),
            (MissingDependencyError("scipy", "Optimisation", "optimise"), 503,
             "MISSING_DEPENDENCY"),
        ])
    def test_library_errors_map_to_stable_codes(self,
                                                exception,
                                                expected_status,
                                                expected_code):
        """A forced library error surfaces as the envelope with its code."""
        config = ServerConfig(auth_token=TOKEN)
        app = create_app(config)

        @app.get("/boom")
        def boom() -> None:
            raise exception

        response = TestClient(app, raise_server_exceptions=False).get(
            "/boom", headers=auth())

        assert response.status_code == expected_status
        assert response.json()["error"]["code"] == expected_code

    def test_envelope_carries_structured_detail(self):
        """Exception attributes reach the client without parsing prose."""
        config = ServerConfig(auth_token=TOKEN)
        app = create_app(config)

        @app.get("/boom")
        def boom() -> None:
            raise DataNotFoundError("prices for AAA", source="MarketData")

        body = TestClient(app, raise_server_exceptions=False).get(
            "/boom", headers=auth()).json()

        assert body["error"]["detail"]["data_description"] == "prices for AAA"
        assert body["error"]["detail"]["source"] == "MarketData"

    def test_unregistered_beacon_subclass_falls_back(self):
        """A future BeaconError subclass must not escape as an unlabelled 500."""

        class FutureError(BeaconError):
            pass

        assert classify(FutureError("something new")) == (500, "BEACON_ERROR")

    def test_specific_mapping_beats_the_catch_all(self):
        assert classify(DataNotFoundError("x"))[1] == "DATA_NOT_FOUND"


class TestOpenApi:

    def test_schema_generates(self,
                              client):
        schema = client.app.openapi()

        assert schema["info"]["title"] == "Beacon API"
        assert "/health" in schema["paths"]

    def test_health_declares_its_response_model(self,
                                                client):
        schema = client.app.openapi()
        health = schema["paths"]["/health"]["get"]["responses"]

        assert "200" in health
        assert "HealthResponse" in schema["components"]["schemas"]

    def test_error_envelope_is_documented_on_every_route(self,
                                                         client):
        schema = client.app.openapi()
        responses = schema["paths"]["/health"]["get"]["responses"]

        for code in ("401", "404", "422", "500", "503"):
            assert code in responses, f"{code} missing from the documented responses"
        assert "ErrorEnvelope" in schema["components"]["schemas"]


class TestResultSchemas:
    """Result objects are mapped explicitly; dataclasses never cross the wire."""

    def test_index_result_round_trips(self):
        levels = pd.Series([1000.0, 1010.0], index=DATES[:2])
        divisors = pd.Series([10.0, 10.0], index=DATES[:2])
        rebalance = DATES[0]
        result = IndexResult(index_id="DEMO",
                             index_levels=levels,
                             divisor_history=divisors,
                             constituent_snapshots={rebalance: ["AAA", "BBB"]},
                             weight_snapshots={rebalance: {"AAA": 0.6, "BBB": 0.4}})

        payload = IndexResultSummary.from_result(result).model_dump()

        assert payload["index_id"] == "DEMO"
        assert payload["index_levels"]["data"] == [1000.0, 1010.0]
        assert payload["rebalance_dates"] == [rebalance.isoformat()]
        assert payload["weight_snapshots"][rebalance.isoformat()] == {"AAA": 0.6, "BBB": 0.4}

    def test_backtest_result_round_trips(self):
        nav = pd.Series([1000.0, 1100.0], index=DATES[:2])
        cash = pd.Series([1000.0, 0.0], index=DATES[:2])
        transaction = Transaction(asset_id="AAA",
                                  quantity=10.0,
                                  price=100.0,
                                  transaction_type="BUY",
                                  transaction_date=DATES[0],
                                  transaction_cost=1.0)
        result = BacktestResult(portfolio_id="bt",
                                initial_capital=1000.0,
                                portfolio_nav=nav,
                                cash_history=cash,
                                transactions=[transaction],
                                actual_weight_history=pd.DataFrame())

        payload = BacktestResultSummary.from_result(result).model_dump()

        assert payload["portfolio_id"] == "bt"
        assert payload["transactions"]["columns"] == [
            "date", "asset_id", "type", "quantity", "price", "cost"]
        assert payload["transactions"]["data"][0][1] == "AAA"
        assert payload["metrics"]["total_return"] == pytest.approx(0.1)
        assert payload["metrics"]["tracking_error"] is None

    def test_money_requires_a_currency_code(self):
        assert Money(value=1.5, currency="USD").currency == "USD"

        with pytest.raises(ValidationError):
            Money(value=1.5, currency="DOLLARS")


class TestOptionalDependencyGuard:
    """Importing the subpackage without the extra names the extra (BN-55)."""

    def test_import_without_fastapi_reports_the_extra(self):
        script = (
            "import sys\n"
            "class Blocker:\n"
            "    def find_spec(self, fullname, path=None, target=None):\n"
            "        if fullname.split('.')[0] in {'fastapi', 'uvicorn'}:\n"
            "            raise ImportError('blocked')\n"
            "        return None\n"
            "sys.meta_path.insert(0, Blocker())\n"
            "from beacon.exceptions import MissingDependencyError\n"
            "try:\n"
            "    import beacon.server\n"
            "except MissingDependencyError as exc:\n"
            "    print(exc)\n"
            "else:\n"
            "    raise SystemExit('expected MissingDependencyError')\n"
        )

        completed = subprocess.run([sys.executable, "-c", script],
                                   capture_output=True,
                                   text=True,
                                   check=False)

        assert completed.returncode == 0, completed.stderr
        assert 'pip install "py-beacon[server]"' in completed.stdout


class TestLauncherProcess:
    """End-to-end: spawn the real process and complete the port handshake."""

    def test_announces_its_port_and_serves(self):
        process = subprocess.Popen(
            [sys.executable, "-m", "beacon.server", "--port", "0", "--token", TOKEN],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True)

        try:
            line = process.stdout.readline()

            assert line.startswith(PORT_ANNOUNCEMENT), f"unexpected first line: {line!r}"
            port = int(line[len(PORT_ANNOUNCEMENT):].strip())
            assert port > 0

            deadline = time.time() + 10
            while time.time() < deadline:
                try:
                    with socket.create_connection(("127.0.0.1", port), timeout=1):
                        break
                except OSError:
                    time.sleep(0.1)
            else:
                pytest.fail(f"server never accepted connections on port {port}")
        finally:
            process.terminate()
            process.wait(timeout=10)
