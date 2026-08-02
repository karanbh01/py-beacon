# tests/test_data_store.py
"""BN-113: the persisted data store, and how a spawned server finds one.

Two things are under test here and they fail differently. The store itself is
a round trip — what went in must come back, including the containers that were
absent. The resolution order is a precedence question, and the way it breaks is
by silently skipping a branch, so every branch is asserted against a store that
would give a *different* answer if the wrong one won.
"""
import gzip
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

import httpx
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from beacon.data import store
from beacon.data.base import MarketData, ReferenceData
from beacon.data.corporate_actions import CorporateActions
from beacon.data.fetcher import DataFetcher
from beacon.exceptions import ConfigurationError
from beacon.server import ServerConfig, create_app
from beacon.server.__main__ import PORT_ANNOUNCEMENT, build_parser, main
from beacon.server.config import resolve_data_source
from beacon.server.store import DocumentStore

TOKEN = "test-token-value"
ASSETS = ["AAA", "BBB"]
DATES = pd.bdate_range("2025-01-02", periods=4)


def build_fetcher(assets: list[str] | None = None,
                  with_reference: bool = True,
                  with_actions: bool = True) -> DataFetcher:
    """A small fetcher whose contents are easy to recognise after a round trip."""
    names = assets if assets is not None else ASSETS

    market = MarketData.from_dataframe(pd.DataFrame([
        {"IDENTIFIER": asset, "DATE": date,
         "CLOSE": 100.0 + index, "SHARES_OUTSTANDING": 1_000_000.0}
        for asset in names
        for index, date in enumerate(DATES)]))

    reference = None
    if with_reference:
        reference = ReferenceData.from_dataframe(pd.DataFrame([
            {"IDENTIFIER": asset, "DATE_FROM": "2020-01-01",
             "NAME": f"Company {asset}", "SECTOR": "Technology",
             "CURRENCY": "USD"}
            for asset in names]))

    actions = None
    if with_actions:
        actions = CorporateActions.from_dataframe(pd.DataFrame([
            {"IDENTIFIER": names[0], "EX_DATE": "2025-01-03",
             "TYPE": "DIVIDEND", "VALUE": 0.25}]))

    return DataFetcher(market, reference, actions)


@pytest.fixture
def saved(tmp_path) -> Path:
    """A complete store on disk."""
    return store.save(build_fetcher(), tmp_path / "store", source="synthetic")


class TestRoundTrip:
    """What was written comes back."""

    def test_market_data_survives(self, saved):
        loaded = store.load(saved)

        assert loaded.identifiers == ASSETS

        prices = loaded.fetch_market_data("AAA", "2025-01-01", "2025-01-31")
        assert prices["CLOSE"].tolist() == [100.0, 101.0, 102.0, 103.0]

    def test_reference_data_survives(self, saved):
        loaded = store.load(saved)

        reference = loaded.fetch_reference_data("BBB", "2025-01-03")

        assert reference["NAME"].iloc[0] == "Company BBB"
        assert reference["CURRENCY"].iloc[0] == "USD"

    def test_corporate_actions_survive(self, saved):
        loaded = store.load(saved)

        assert not loaded.corporate_actions.is_empty
        assert loaded.fetch_trailing_dividend("AAA", "2025-06-01") == pytest.approx(0.25)

    def test_shares_outstanding_survive(self, saved):
        """A market-data column that is not a price still has to come back:
        market-cap weighting reads it, and an index would silently fall back to
        equal weights without it."""
        loaded = store.load(saved)

        assert loaded.fetch_shares_outstanding("AAA", "2025-01-03") == 1_000_000.0

    def test_a_market_only_fetcher_round_trips_as_market_only(self, tmp_path):
        """Not as one carrying two empty files. An empty reference dataset and
        an absent one are different states, and `/data/coverage` reports them
        differently."""
        fetcher = build_fetcher(with_reference=False, with_actions=False)
        path = store.save(fetcher, tmp_path / "store")

        assert not (path / store.REFERENCE_FILE).exists()
        assert not (path / store.ACTIONS_FILE).exists()

        loaded = store.load(path)

        assert loaded.reference_identifiers is None
        assert loaded.corporate_actions.is_empty

    def test_an_empty_action_history_writes_no_file(self, tmp_path):
        fetcher = build_fetcher(with_actions=False)
        path = store.save(fetcher, tmp_path / "store")

        assert not (path / store.ACTIONS_FILE).exists()
        assert "corporate_actions" not in store.read_manifest(path).datasets

    def test_keys_are_not_written_twice(self, saved):
        """CorporateActions keeps its key columns alongside the index, so a
        naive reset_index writes IDENTIFIER and EX_DATE twice and the reload
        reads a duplicate-column frame."""
        with gzip.open(saved / store.ACTIONS_FILE, "rt", encoding="utf-8") as handle:
            header = handle.readline().strip().split(",")

        assert sorted(header) == ["EX_DATE", "IDENTIFIER", "TYPE", "VALUE"]


class TestReproducibility:
    """Same data in, same bytes out — BN-114's determinism rests on this."""

    def test_two_saves_are_byte_identical(self, tmp_path):
        fetcher = build_fetcher()

        first = store.save(fetcher, tmp_path / "one")
        second = store.save(fetcher, tmp_path / "two")

        for name in (store.MANIFEST_NAME, store.MARKET_FILE,
                     store.REFERENCE_FILE, store.ACTIONS_FILE):
            assert (first / name).read_bytes() == (second / name).read_bytes(), name

    def test_rows_end_in_a_bare_newline(self, saved):
        """The platform default would put CRLF here on Windows, and the same
        generator would produce a different store on every operating system."""
        raw = gzip.decompress((saved / store.MARKET_FILE).read_bytes())

        assert b"\r\n" not in raw

    def test_the_gzip_header_carries_no_timestamp(self, saved):
        """Bytes 4-8 of a gzip member are its mtime. Left alone, two writes a
        second apart differ while the content does not."""
        header = (saved / store.MARKET_FILE).read_bytes()[:8]

        assert header[4:8] == b"\x00\x00\x00\x00"


class TestManifest:
    """A store says what it is, and refuses to be misread."""

    def test_source_is_recorded(self, saved):
        assert store.read_manifest(saved).source == "synthetic"

    def test_source_defaults_to_local(self, tmp_path):
        path = store.save(build_fetcher(), tmp_path / "store")

        assert store.read_manifest(path).source == store.SOURCE_LOCAL

    def test_datasets_present_are_listed(self, saved):
        assert store.read_manifest(saved).datasets == (
            "market", "reference", "corporate_actions")

    def test_a_newer_schema_version_is_refused(self, saved):
        (saved / store.MANIFEST_NAME).write_text(
            json.dumps({"schema_version": store.STORE_SCHEMA_VERSION + 1,
                        "source": "synthetic", "datasets": ["market"]}),
            encoding="utf-8")

        with pytest.raises(ConfigurationError, match="Upgrade Beacon"):
            store.read_manifest(saved)

    def test_a_missing_manifest_says_so(self, tmp_path):
        with pytest.raises(ConfigurationError, match="not a Beacon data store"):
            store.read_manifest(tmp_path)

    def test_malformed_json_says_so(self, saved):
        (saved / store.MANIFEST_NAME).write_text("{not json", encoding="utf-8")

        with pytest.raises(ConfigurationError, match="not valid JSON"):
            store.read_manifest(saved)


class TestExists:
    """What counts as a store."""

    def test_a_written_store_exists(self, saved):
        assert store.exists(saved)

    def test_an_empty_directory_does_not(self, tmp_path):
        assert not store.exists(tmp_path)

    def test_a_manifest_without_market_data_does_not(self, tmp_path):
        """Prices are the floor. Reporting a price-less store as a data source
        moves the failure from startup to the first request."""
        tmp_path.mkdir(exist_ok=True)
        (tmp_path / store.MANIFEST_NAME).write_text("{}", encoding="utf-8")

        assert not store.exists(tmp_path)

    def test_loading_a_non_store_says_what_was_expected(self, tmp_path):
        with pytest.raises(ConfigurationError, match="No data store at"):
            store.load(tmp_path)


class TestResolutionOrder:
    """Which branch wins. Each is checked against a store the other branches
    would not have produced, so a skipped branch cannot pass by coincidence."""

    def test_an_explicit_path_wins(self, tmp_path, monkeypatch):
        chosen = store.save(build_fetcher(["XXX"]), tmp_path / "chosen")
        monkeypatch.setenv(store.DATA_PATH_ENV_VAR,
                           str(store.save(build_fetcher(["YYY"]), tmp_path / "env")))

        fetcher, origin = resolve_data_source(chosen)

        assert fetcher is not None and fetcher.identifiers == ["XXX"]
        assert "--data" in origin

    def test_the_environment_variable_is_next(self, tmp_path, monkeypatch):
        path = store.save(build_fetcher(["YYY"]), tmp_path / "env")
        monkeypatch.setenv(store.DATA_PATH_ENV_VAR, str(path))
        monkeypatch.setattr(store, "default_path", lambda: tmp_path / "absent")

        fetcher, origin = resolve_data_source(None)

        assert fetcher is not None and fetcher.identifiers == ["YYY"]
        assert store.DATA_PATH_ENV_VAR in origin

    def test_the_app_data_store_is_auto_loaded(self, tmp_path, monkeypatch):
        path = store.save(build_fetcher(["ZZZ"]), tmp_path / "app-data")
        monkeypatch.delenv(store.DATA_PATH_ENV_VAR, raising=False)
        monkeypatch.setattr(store, "default_path", lambda: path)

        fetcher, origin = resolve_data_source(None)

        assert fetcher is not None and fetcher.identifiers == ["ZZZ"]
        assert "app-data store" in origin

    def test_no_store_anywhere_leaves_the_server_data_less(self, tmp_path, monkeypatch):
        """The behaviour that existed before this issue, unchanged."""
        monkeypatch.delenv(store.DATA_PATH_ENV_VAR, raising=False)
        monkeypatch.setattr(store, "default_path", lambda: tmp_path / "absent")

        fetcher, origin = resolve_data_source(None)

        assert fetcher is None
        assert "no data source" in origin

    def test_an_empty_environment_variable_is_ignored(self, tmp_path, monkeypatch):
        """Spawning with `BEACON_DATA_PATH=` set but blank is how a shell
        passes "unset", and treating it as a path fails on the empty string."""
        monkeypatch.setenv(store.DATA_PATH_ENV_VAR, "   ")
        monkeypatch.setattr(store, "default_path", lambda: tmp_path / "absent")

        fetcher, _ = resolve_data_source(None)

        assert fetcher is None


class TestResolutionFailures:
    """Asking for a store that is not there differs from not asking."""

    def test_an_explicit_path_that_is_not_a_store_raises(self, tmp_path):
        with pytest.raises(ConfigurationError):
            resolve_data_source(tmp_path / "nowhere")

    def test_an_environment_path_that_is_not_a_store_raises(self, tmp_path, monkeypatch):
        monkeypatch.setenv(store.DATA_PATH_ENV_VAR, str(tmp_path / "nowhere"))

        with pytest.raises(ConfigurationError):
            resolve_data_source(None)

    def test_a_corrupt_app_data_store_warns_and_starts_data_less(self,
                                                                 tmp_path,
                                                                 monkeypatch,
                                                                 caplog):
        """Loudly failing here would leave the client unable to start the
        server that would let it write a replacement."""
        path = store.save(build_fetcher(), tmp_path / "app-data")
        (path / store.MANIFEST_NAME).write_text("{oops", encoding="utf-8")

        monkeypatch.delenv(store.DATA_PATH_ENV_VAR, raising=False)
        monkeypatch.setattr(store, "default_path", lambda: path)

        with caplog.at_level(logging.WARNING):
            fetcher, origin = resolve_data_source(None)

        assert fetcher is None
        assert "unreadable" in origin
        assert "could not be read" in caplog.text


class TestLauncher:
    """The command line beacon-ui actually spawns."""

    def test_the_data_flag_is_parsed(self):
        args = build_parser().parse_args(["--data", "/tmp/store"])

        assert args.data == Path("/tmp/store")

    def test_it_defaults_to_none(self):
        assert build_parser().parse_args([]).data is None

    def test_a_bad_data_path_exits_two_rather_than_serving(self, tmp_path, capsys):
        """Exit 2 before binding: a server that ignored --data and started
        empty would look identical to a working one until the first request."""
        code = main(["--token", TOKEN, "--data", str(tmp_path / "nowhere")])

        assert code == 2
        assert "error:" in capsys.readouterr().err


class TestDefaultPath:
    """The location branch 3 auto-loads."""

    def test_it_sits_under_the_beacon_app_data_directory(self):
        path = store.default_path()

        assert path.name == store.STORE_DIRECTORY
        assert store.APP_NAME in str(path)

    def test_it_shares_a_root_with_the_document_store(self):
        """One Beacon directory per machine, not two that differ by a letter."""
        documents = DocumentStore("watchlists")

        assert documents.directory.parent == store.default_path().parent


class TestSpawnedServer:
    """The acceptance criterion, run as written: write a store, spawn the exact
    command beacon-ui spawns, ask for prices.

    A subprocess rather than a TestClient because what broke was the launcher,
    not the app — `main()` built a config with no fetcher, and every in-process
    test passed one in and so could not have caught it."""

    @pytest.mark.timeout(60)
    def test_it_serves_a_store_given_on_the_command_line(self, tmp_path):
        path = store.save(build_fetcher(), tmp_path / "store", source="synthetic")

        process = subprocess.Popen(
            [sys.executable, "-m", "beacon.server", "--port", "0",
             "--token", TOKEN, "--data", str(path)],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        try:
            line = process.stdout.readline()
            assert line.startswith(PORT_ANNOUNCEMENT), f"first line: {line!r}"
            port = int(line[len(PORT_ANNOUNCEMENT):].strip())

            response = _get_when_ready(port, "/data/prices/AAA")

            assert response.status_code == 200

            body = response.json()
            assert body["identifier"] == "AAA"
            assert len(body["prices"]["index"]) == len(DATES)
        finally:
            process.terminate()
            process.wait(timeout=30)

    @pytest.mark.timeout(60)
    def test_without_a_store_it_still_starts(self):
        """Unchanged behaviour when there is nothing to find: the server comes
        up and reports no data source, rather than refusing to start."""
        environment = {key: value for key, value in os.environ.items()
                       if key != store.DATA_PATH_ENV_VAR}

        process = subprocess.Popen(
            [sys.executable, "-m", "beacon.server", "--port", "0", "--token", TOKEN],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=environment)

        try:
            line = process.stdout.readline()
            port = int(line[len(PORT_ANNOUNCEMENT):].strip())

            response = _get_when_ready(port, "/health")

            assert response.status_code == 200
        finally:
            process.terminate()
            process.wait(timeout=30)


def _get_when_ready(port: int,
                    path: str,
                    timeout: float = 30.0):
    """Poll until the spawned server answers, then return the response."""
    deadline = time.monotonic() + timeout

    while time.monotonic() < deadline:
        try:
            return httpx.get(f"http://127.0.0.1:{port}{path}",
                             headers={"Authorization": f"Bearer {TOKEN}"},
                             timeout=5.0)
        except httpx.TransportError:
            time.sleep(0.1)

    raise AssertionError(f"server never answered on port {port}")


class TestServedTruthfully:
    """`/health` and `/data/coverage` describe the store that was loaded."""

    @pytest.fixture
    def client(self, saved) -> TestClient:
        config = ServerConfig(auth_token=TOKEN, data_fetcher=store.load(saved))

        return TestClient(create_app(config))

    def test_health_reports_the_loaded_identifiers(self, client):
        body = client.get("/health",
                          headers={"Authorization": f"Bearer {TOKEN}"}).json()

        assert body["data_source"]["configured"] is True
        assert body["data_source"]["identifiers"] == len(ASSETS)

    def test_coverage_reports_both_datasets_configured(self, client):
        body = client.get("/data/coverage",
                          headers={"Authorization": f"Bearer {TOKEN}"}).json()
        datasets = {entry["dataset"]: entry for entry in body["datasets"]}

        assert datasets["market"]["configured"] is True
        assert datasets["market"]["identifiers"] == len(ASSETS)
        assert datasets["reference"]["configured"] is True

    def test_prices_are_served(self, client):
        """The acceptance criterion: 200 with real rows, no client changes."""
        response = client.get("/data/prices/AAA",
                              headers={"Authorization": f"Bearer {TOKEN}"})

        assert response.status_code == 200

        body = response.json()
        assert body["identifier"] == "AAA"
        assert len(body["prices"]["index"]) == len(DATES)
