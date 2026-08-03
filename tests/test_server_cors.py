# tests/test_server_cors.py
"""BN-122: which origins may call the server.

The middleware was already there. The default was `("app://",)` — a scheme with
no host, which no renderer sends — so the packaged app's real origin,
`beacon://app`, failed preflight and every direct call from it was blocked.
One wrong string, and the symptom appears in a browser console on the far side
of a process boundary, where nothing in this repo can see it.

The test that matters is the preflight itself, run as a browser runs it:
OPTIONS, with the method and the `Authorization` header being asked for.
Asserting on a plain GET would pass against a server that blocks every real
request, because a GET without an `Origin` never triggers CORS at all.
"""
import pytest
from fastapi.testclient import TestClient

from beacon.server import ServerConfig, create_app
from beacon.server.__main__ import build_parser
from beacon.server.config import (
    CORS_ORIGINS_ENV_VAR,
    DEFAULT_CORS_ORIGINS,
    LEGACY_APP_ORIGIN,
    PACKAGED_APP_ORIGIN,
    resolve_cors_origins,
)

TOKEN = "test-token-value"


@pytest.fixture
def client() -> TestClient:
    """A server on the default origins."""
    return TestClient(create_app(ServerConfig(auth_token=TOKEN)))


def preflight(client: TestClient,
              origin: str,
              method: str = "GET",
              headers: str = "authorization"):
    """A CORS preflight, shaped the way a browser sends one."""
    return client.options("/health",
                          headers={"Origin": origin,
                                   "Access-Control-Request-Method": method,
                                   "Access-Control-Request-Headers": headers})


class TestPackagedAppOrigin:
    """The correction this issue exists for."""

    def test_the_packaged_origin_is_allowed_by_default(self, client):
        """The acceptance criterion, as written: preflight from `beacon://app`
        asking for `Authorization` comes back 200 with the origin echoed."""
        response = preflight(client, PACKAGED_APP_ORIGIN)

        assert response.status_code == 200
        assert (response.headers["access-control-allow-origin"]
                == PACKAGED_APP_ORIGIN)

    def test_the_authorization_header_is_permitted(self, client):
        """Every route requires a bearer token, so a preflight that allowed the
        origin but not the header would still block every real request."""
        response = preflight(client, PACKAGED_APP_ORIGIN)
        allowed = response.headers["access-control-allow-headers"].lower()

        assert "authorization" in allowed or allowed == "*"

    def test_credentials_are_allowed(self, client):
        response = preflight(client, PACKAGED_APP_ORIGIN)

        assert response.headers.get("access-control-allow-credentials") == "true"

    def test_the_methods_the_api_uses_are_permitted(self, client):
        for method in ("GET", "POST", "PUT", "DELETE"):
            response = preflight(client, PACKAGED_APP_ORIGIN, method=method)

            assert response.status_code == 200, method

    def test_the_legacy_origin_still_works(self, client):
        """Kept rather than deleted: removing an origin from a default is how a
        working build stops working with no message saying why."""
        response = preflight(client, LEGACY_APP_ORIGIN)

        assert (response.headers["access-control-allow-origin"]
                == LEGACY_APP_ORIGIN)

    def test_localhost_is_allowed_on_any_port(self, client):
        """The dev build lands on an arbitrary port, hence a pattern rather
        than a list."""
        response = preflight(client, "http://localhost:5173")

        assert response.headers["access-control-allow-origin"] == (
            "http://localhost:5173")

    def test_an_unrelated_origin_is_not_echoed(self, client):
        """Otherwise the middleware is permitting everything and none of the
        assertions above mean anything."""
        response = preflight(client, "https://example.com")

        assert "access-control-allow-origin" not in response.headers


class TestResolution:
    """Where the allowed origins come from."""

    def test_the_default_carries_the_packaged_origin_first(self):
        assert DEFAULT_CORS_ORIGINS[0] == PACKAGED_APP_ORIGIN

    def test_nothing_supplied_gives_the_defaults(self, monkeypatch):
        monkeypatch.delenv(CORS_ORIGINS_ENV_VAR, raising=False)

        assert resolve_cors_origins() == DEFAULT_CORS_ORIGINS

    def test_explicit_origins_win(self, monkeypatch):
        monkeypatch.setenv(CORS_ORIGINS_ENV_VAR, "env://ignored")

        assert resolve_cors_origins(["cli://one"]) == ("cli://one",)

    def test_the_environment_is_next(self, monkeypatch):
        monkeypatch.setenv(CORS_ORIGINS_ENV_VAR, "a://one, b://two")

        assert resolve_cors_origins() == ("a://one", "b://two")

    def test_an_empty_environment_value_falls_through(self, monkeypatch):
        monkeypatch.setenv(CORS_ORIGINS_ENV_VAR, "   ")

        assert resolve_cors_origins() == DEFAULT_CORS_ORIGINS

    def test_duplicates_collapse(self, monkeypatch):
        monkeypatch.delenv(CORS_ORIGINS_ENV_VAR, raising=False)

        assert resolve_cors_origins(["x://y", "x://y"]) == ("x://y",)

    def test_explicit_origins_replace_rather_than_extend(self, monkeypatch):
        """An operator narrowing what may call the server should not find the
        defaults still permitted — that is the opposite of configuring it."""
        monkeypatch.delenv(CORS_ORIGINS_ENV_VAR, raising=False)

        resolved = resolve_cors_origins(["only://this"])

        assert PACKAGED_APP_ORIGIN not in resolved


class TestConfiguredServer:
    """A server told which origins to allow."""

    def test_a_configured_origin_is_allowed(self):
        client = TestClient(create_app(ServerConfig(
            auth_token=TOKEN, cors_origins=("custom://app",))))

        response = preflight(client, "custom://app")

        assert response.headers["access-control-allow-origin"] == "custom://app"

    def test_a_default_origin_is_not_when_others_are_configured(self):
        client = TestClient(create_app(ServerConfig(
            auth_token=TOKEN, cors_origins=("custom://app",))))

        response = preflight(client, PACKAGED_APP_ORIGIN)

        assert "access-control-allow-origin" not in response.headers

    def test_localhost_survives_a_narrowed_configuration(self):
        """The pattern is applied separately by the middleware, so restricting
        the exact origins must not lock out the dev build."""
        client = TestClient(create_app(ServerConfig(
            auth_token=TOKEN, cors_origins=("custom://app",))))

        response = preflight(client, "http://localhost:3000")

        assert response.headers["access-control-allow-origin"] == (
            "http://localhost:3000")


class TestLauncher:
    """The flag a packaged app would pass."""

    def test_the_flag_is_repeatable(self):
        args = build_parser().parse_args(
            ["--cors-origin", "a://one", "--cors-origin", "b://two"])

        assert args.cors_origins == ["a://one", "b://two"]

    def test_it_defaults_to_none_so_the_environment_is_consulted(self):
        assert build_parser().parse_args([]).cors_origins is None
