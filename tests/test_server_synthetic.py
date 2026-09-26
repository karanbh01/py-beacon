# tests/test_server_synthetic.py
"""BN-237: the engine generates synthetic data into a new store, as a job.

It runs `python -m beacon.synthetic` in a child process, so these tests start
real child processes. The settings are kept tiny (twenty names over half a
year) so each takes a couple of seconds, most of it the child importing.
"""
import subprocess
import sys

import pytest
from fastapi.testclient import TestClient

from beacon.data import store as data_store
from beacon.server import ServerConfig, create_app
from beacon.server.routers import synthetic

TOKEN = "synthetic-token"
HEADERS = {"Authorization": f"Bearer {TOKEN}"}
SMALL = {"assets": 20, "start": "2024-01-02", "end": "2024-06-28", "seed": 7}


@pytest.fixture
def client(tmp_path):
    """An engine started with no data."""
    app = create_app(ServerConfig(auth_token=TOKEN,
                                  storage_root=tmp_path / "documents"))

    with TestClient(app, raise_server_exceptions=False) as started:
        yield started


def generate(client,
             **body) -> dict:
    """Start a generation, wait for it, and return the finished job."""
    response = client.post("/data/synthetic", headers=HEADERS, json=body)
    assert response.status_code == 202, response.json()

    client.portal.call(client.app.state.jobs.drain)

    return client.get(f"/jobs/{response.json()['job_id']}", headers=HEADERS).json()


def stores(client) -> list[dict]:
    return client.get("/data/stores", headers=HEADERS).json()["stores"]


class TestGenerating:

    def test_it_writes_a_new_store_and_serves_it(self,
                                                 client):
        job = generate(client, name="Sample", **SMALL)
        served = client.get("/health", headers=HEADERS).json()["data_source"]

        assert job["status"] == "succeeded", job
        assert job["kind"] == "generate:sample"
        assert job["result"]["activated"] is True
        assert served["store_id"] == "sample"
        assert served["configured"] is True
        assert [(row["id"], row["managed"], row["readable"], row["source"])
                for row in stores(client)] == [("sample", True, True, "synthetic")]

    def test_the_data_matches_the_command_line_with_the_same_settings(self,
                                                                      client,
                                                                      tmp_path):
        """One generator, one answer: the engine runs the same command."""
        job = generate(client, name="Sample", **SMALL)
        by_hand = tmp_path / "by-hand"
        subprocess.run([sys.executable, "-m", "beacon.synthetic",
                        "--assets", "20", "--start", "2024-01-02",
                        "--end", "2024-06-28", "--seed", "7",
                        "--out", str(by_hand)],
                       check=True, capture_output=True)

        from_engine = data_store.load(data_store.Path(job["result"]["path"]))
        from_command = data_store.load(by_hand)

        assert from_engine.market.data.equals(from_command.market.data)

    def test_it_can_leave_the_new_store_unserved(self,
                                                 client):
        job = generate(client, name="Later", activate=False, **SMALL)

        assert job["result"]["activated"] is False
        assert client.get("/health", headers=HEADERS).json()[
            "data_source"]["configured"] is False
        assert stores(client)[0]["readable"] is True

    def test_the_default_name_is_synthetic_data(self,
                                                client):
        job = generate(client, **SMALL)

        assert job["result"]["name"] == "Synthetic data"
        assert job["result"]["store_id"] == "synthetic-data"


class TestRefusingBadSettingsAtOnce:
    """Checked before the job starts, so nothing is registered."""

    @pytest.mark.parametrize("body", [
        {"start": "2024-06-28", "end": "2024-01-02"},
        {"calendar": "NOT-A-MIC"},
    ])
    def test_settings_that_describe_no_dataset_are_refused(self,
                                                           client,
                                                           body):
        response = client.post("/data/synthetic", headers=HEADERS,
                               json={"assets": 5, **body})

        assert response.status_code == 422
        assert stores(client) == []


class TestNothingHalfWrittenStays:

    def test_a_failed_run_removes_its_store(self,
                                            client,
                                            monkeypatch):
        monkeypatch.setattr(synthetic, "command", lambda config, out: [
            sys.executable, "-c", "import sys; print('out of memory'); sys.exit(3)"])

        job = generate(client, name="Broken", **SMALL)

        assert job["status"] == "failed"
        assert "out of memory" in job["error"]["message"]
        assert stores(client) == []
        assert list(client.app.state.managed_store_root.iterdir()) == []

    def test_a_cancelled_run_stops_the_generator_and_removes_its_store(self,
                                                                       client,
                                                                       monkeypatch):
        monkeypatch.setattr(synthetic, "command", lambda config, out: [
            sys.executable, "-c", "import time; print('started', flush=True); "
                                  "time.sleep(60)"])
        response = client.post("/data/synthetic", headers=HEADERS,
                               json={"name": "Slow", **SMALL})

        client.delete(f"/jobs/{response.json()['job_id']}", headers=HEADERS)
        client.portal.call(client.app.state.jobs.drain)

        assert stores(client) == []
        assert list(client.app.state.managed_store_root.iterdir()) == []


class TestForgettingAGeneratedStore:

    def test_its_files_are_deleted_with_it(self,
                                           client):
        """The engine made them and nothing else knows they are there."""
        job = generate(client, name="Temporary", activate=False, **SMALL)
        folder = data_store.Path(job["result"]["path"])

        response = client.delete("/data/stores/temporary", headers=HEADERS)

        assert response.status_code == 204
        assert not folder.exists()


class TestTheChildRunsLikeTheEngine:
    """The generator child keeps the engine's isolation, so an isolated
    engine never loads the user's own site-packages through its child."""

    @staticmethod
    def flags(isolated=0, no_user_site=0, ignore_environment=0):
        from types import SimpleNamespace

        return SimpleNamespace(isolated=isolated, no_user_site=no_user_site,
                               ignore_environment=ignore_environment)

    def test_an_isolated_engine_starts_an_isolated_child(self):
        assert synthetic.interpreter(self.flags(isolated=1)) == [
            sys.executable, "-I"]

    def test_a_plain_engine_starts_a_plain_child(self):
        assert synthetic.interpreter(self.flags()) == [sys.executable]

    def test_the_narrower_flags_carry_over(self):
        assert synthetic.interpreter(
            self.flags(no_user_site=1, ignore_environment=1)) == [
            sys.executable, "-s", "-E"]

    def test_an_isolated_child_still_finds_the_generator(self):
        """-I leaves the current folder off the path; the generator is
        imported from the installed package, so it still runs."""
        completed = subprocess.run(
            [sys.executable, "-I", "-m", "beacon.synthetic", "--help"],
            capture_output=True, text=True, timeout=120, check=False)

        assert completed.returncode == 0, completed.stderr
