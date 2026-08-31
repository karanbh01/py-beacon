# tests/test_server.py
"""Unit tests for the Beacon API server skeleton (app factory, auth, /health)."""
import importlib
import pkgutil
import socket
import subprocess
import sys
import time
from typing import ClassVar

import pandas as pd
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

import beacon
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
    ReportingError,
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
from beacon.server.errors import EXCEPTION_MAPPING

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


def all_beacon_error_subclasses() -> set[type]:
    """Every BeaconError subclass anywhere in the package.

    Imports every module first: `__subclasses__` only knows about classes that
    have actually been imported, so without the walk this would silently miss
    exceptions in modules no test happens to touch — exactly the hole this
    check exists to close.
    """
    for module in pkgutil.walk_packages(beacon.__path__, "beacon."):
        try:
            importlib.import_module(module.name)
        except MissingDependencyError:
            # A module behind an extra that is not installed cannot hide an
            # exception from the mapping, because it cannot be reached either.
            continue

    def descendants(cls: type) -> set[type]:
        found = set()
        for subclass in cls.__subclasses__():
            found.add(subclass)
            found |= descendants(subclass)
        return found

    # Library classes only. Tests define throwaway subclasses to exercise the
    # catch-all, and those are never shipped, so holding them to the mapping
    # would be noise.
    return {cls for cls in descendants(BeaconError)
            if cls.__module__.startswith("beacon.")}


class TestEnvelopeExhaustiveness:
    """No library exception may reach a client without a stable code.

    BN-86: `ReportingError` subclassed plain Exception, so it fell outside the
    mapping entirely and would have surfaced as an unlabelled 500. This check
    fails when the next exception is added without a decision being made about
    how it should be reported.
    """

    # Subclasses deliberately absent from EXCEPTION_MAPPING, with the reason.
    # Inheriting a parent's mapping is a legitimate choice; forgetting to map
    # something is not, and only this list distinguishes them.
    DELIBERATELY_INHERITED: ClassVar[dict[str, str]] = {
        "PipelineValidationError":
            "subclasses InvalidRuleError to inherit its 422 / INVALID_RULE "
            "mapping, and adds structured findings to the envelope detail",
        "FeatureImportError":
            "the same arrangement again, for an import whose rows name "
            "instruments the loaded data does not carry: 422 / INVALID_RULE "
            "from InvalidRuleError, with a finding naming each bad row",
        "UnknownDatasetError":
            "subclasses ExpressionError to inherit its 422 / "
            "INVALID_EXPRESSION mapping, and AttributeError so that "
            "`hasattr(data, 'typo')` answers False rather than raising",
        "UniverseValidationError":
            "the same arrangement as PipelineValidationError, for a universe "
            "whose members are not in the loaded data: 422 / INVALID_RULE "
            "from the parent, with a finding naming each missing identifier",
    }

    def test_every_subclass_is_mapped_or_deliberately_inherited(self):
        listed = {exception_type for exception_type, _, _ in EXCEPTION_MAPPING}

        unaccounted = [
            cls.__name__ for cls in all_beacon_error_subclasses()
            if cls not in listed and cls.__name__ not in self.DELIBERATELY_INHERITED
        ]

        assert not unaccounted, (
            f"These BeaconError subclasses are neither in EXCEPTION_MAPPING nor "
            f"listed as deliberately inheriting a parent's mapping: "
            f"{sorted(unaccounted)}. Add a mapping in beacon/server/errors.py, "
            f"or record the reason in DELIBERATELY_INHERITED.")

    def test_the_exclusion_list_has_no_stale_entries(self):
        """A name left here after the class is mapped or deleted is misleading."""
        names = {cls.__name__ for cls in all_beacon_error_subclasses()}

        stale = set(self.DELIBERATELY_INHERITED) - names

        assert not stale, f"DELIBERATELY_INHERITED lists unknown classes: {sorted(stale)}"

    def test_every_mapped_type_produces_a_distinct_code(self):
        """Two exceptions sharing a code would be indistinguishable to a client."""
        codes = [code for _, _, code in EXCEPTION_MAPPING]

        assert len(codes) == len(set(codes)), f"duplicate codes in the mapping: {codes}"

    def test_the_catch_all_is_last(self):
        """classify() walks in order, so BeaconError earlier would shadow all."""
        types = [exception_type for exception_type, _, _ in EXCEPTION_MAPPING]

        assert types[-1] is BeaconError

    def test_reporting_error_is_mapped(self):
        """The specific defect BN-86 fixed."""
        assert classify(ReportingError("disk full")) == (500, "REPORTING_ERROR")

    def test_reporting_error_is_a_beacon_error(self):
        assert issubclass(ReportingError, BeaconError)

    def test_reporting_error_reaches_the_client_in_the_envelope(self):
        config = ServerConfig(auth_token=TOKEN)
        app = create_app(config)

        @app.get("/boom")
        def boom() -> None:
            raise ReportingError("could not write the workbook")

        response = TestClient(app, raise_server_exceptions=False).get(
            "/boom", headers=auth())

        assert response.status_code == 500
        assert response.json()["error"]["code"] == "REPORTING_ERROR"
        assert "could not write the workbook" in response.json()["error"]["message"]

    def test_reporting_error_is_still_importable_from_its_old_home(self):
        """Moving it to beacon.exceptions must not break existing imports."""
        from beacon.portfolio.reporting import ReportingError as FromReporting

        assert FromReporting is ReportingError


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
        from beacon.portfolio.base import Holding, Portfolio

        eve = nav.index[0] - pd.tseries.offsets.BDay(1)
        portfolio = Portfolio(portfolio_id="bt", initial_cash=1000.0,
                              inception=eve)
        for date, value in nav.items():
            # The books enforce NAV == holdings + cash, so the holding
            # carries whatever the cash does not.
            invested = float(value) - float(cash.loc[date])
            holding = Holding(asset_id="NAV", quantity=1.0,
                              average_cost_price=invested,
                              current_price=invested,
                              market_value=invested)
            portfolio._history.record(date, {"NAV": holding},
                                      float(cash.loc[date]))
        portfolio.transactions.append(transaction)

        result = BacktestResult(portfolio=portfolio)

        payload = BacktestResultSummary.from_result(result).model_dump()

        assert payload["portfolio"]["portfolio_id"] == "bt"
        # Books the run did not have are null, not empty: a client can tell
        # "not measured" from "measured and empty".
        assert payload["index"] is None
        assert payload["benchmark"] is None
        assert payload["unfilled"] == []
        # The nav book keeps its day-zero row on the wire.
        assert payload["portfolio"]["nav"]["data"][0] == pytest.approx(1000.0)
        assert payload["portfolio"]["transactions"]["columns"] == [
            "date", "asset_id", "type", "quantity", "price", "cost"]
        assert payload["portfolio"]["transactions"]["data"][0][1] == "AAA"
        assert payload["metrics"]["total_return"] == pytest.approx(0.1)
        assert payload["metrics"]["tracking_error"] is None

    def test_books_serialise_with_their_weights(self):
        """An index with a daily panel arrives as a book: levels plus wide
        weights, with the true date count published beside the served rows."""
        import pandas as pd

        from beacon.backtest.result import BacktestResult, Book
        from beacon.index.result import IndexResult
        from beacon.portfolio.base import Portfolio

        dates = pd.bdate_range("2025-01-02", periods=3)
        idx = IndexResult(
            index_id="idx",
            index_levels=pd.Series([100.0, 101.0, 102.0], index=dates),
            divisor_history=pd.Series(1.0, index=dates),
            constituent_snapshots={},
            weight_snapshots={},
        )
        idx.daily_weights = pd.DataFrame({
            "DATE": list(dates) * 2,
            "IDENTIFIER": ["AAA"] * 3 + ["BBB"] * 3,
            "AMOUNT": [1.0] * 6,
            "WEIGHT": [0.6, 0.58, 0.61, 0.4, 0.42, 0.39],
        })

        portfolio = Portfolio(portfolio_id="bt", initial_cash=1000.0,
                              inception=dates[0] - pd.tseries.offsets.BDay(1))
        for date in dates:
            portfolio._history.record(date, {}, 1000.0)

        payload = BacktestResultSummary.from_result(
            BacktestResult(portfolio=portfolio,
                           index=Book.from_index(idx))).model_dump()

        book = payload["index"]
        assert book is not None
        assert book["levels"]["data"] == [100.0, 101.0, 102.0]
        assert book["weights_dates_total"] == 3
        assert set(book["weights"]["columns"]) == {"AAA", "BBB"}

    def test_positions_totals_are_published(self):
        """The bound keeps payloads renderable; the total keeps them honest —
        a client shows "last N of M" rather than believing it saw
        everything."""
        import pandas as pd

        from beacon.backtest.result import BacktestResult
        from beacon.portfolio.base import Holding, Portfolio

        dates = pd.bdate_range("2025-01-02", periods=4)
        portfolio = Portfolio(portfolio_id="bt", initial_cash=1000.0,
                              inception=dates[0] - pd.tseries.offsets.BDay(1))
        for date in dates:
            holding = Holding(asset_id="AAA", quantity=1.0,
                              average_cost_price=500.0, current_price=500.0,
                              market_value=500.0)
            portfolio._history.record(date, {"AAA": holding}, 500.0)

        payload = BacktestResultSummary.from_result(
            BacktestResult(portfolio=portfolio)).model_dump()

        book = payload["portfolio"]
        assert book["positions_total"] == 4
        assert len(book["positions"]["data"]) == 4
        assert book["weights_dates_total"] == 4

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
