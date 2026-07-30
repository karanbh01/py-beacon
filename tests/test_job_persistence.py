# tests/test_job_persistence.py
"""BN-91: completed job results must outlive the process that produced them."""
import asyncio

import pytest
from fastapi.testclient import TestClient

from beacon.server import ServerConfig, create_app
from beacon.server.jobs import (
    MAX_STORED_RESULTS,
    SUCCEEDED,
    JobRegistry,
)
from beacon.server.store import DocumentStore

TOKEN = "test-token-value"


def auth() -> dict[str, str]:
    """Valid Authorization header."""
    return {"Authorization": f"Bearer {TOKEN}"}


async def produce(report,
                  value: str = "done") -> dict[str, str]:
    """A job that finishes immediately with a result."""
    await report(1.0, "finished")

    return {"value": value}


async def explode(report) -> None:
    """A job that fails."""
    raise ValueError("deliberate failure")


@pytest.fixture
def store(tmp_path) -> DocumentStore:
    """A results store rooted in a temporary directory."""
    return DocumentStore("job_results", root=tmp_path)


class TestRegistryPersistence:

    @pytest.mark.asyncio
    async def test_a_successful_result_is_written(self,
                                                  store):
        registry = JobRegistry(result_store=store)

        job = registry.submit("demo", produce)
        await registry.drain()

        assert store.read(job.id) is not None

    @pytest.mark.asyncio
    async def test_the_stored_document_carries_the_result(self,
                                                          store):
        registry = JobRegistry(result_store=store)

        job = registry.submit("demo", produce)
        await registry.drain()

        assert store.read(job.id)["result"] == {"value": "done"}

    @pytest.mark.asyncio
    async def test_a_failure_is_persisted_with_its_error(self,
                                                         store):
        """A restart must not turn a known failure into a mystery."""
        registry = JobRegistry(result_store=store)

        job = registry.submit("demo", explode)
        await registry.drain()

        document = store.read(job.id)

        assert document["status"] == "failed"
        assert document["error"] == "deliberate failure"

    @pytest.mark.asyncio
    async def test_a_cancelled_job_is_persisted(self,
                                                store):
        async def forever(report):
            while True:
                await asyncio.sleep(0.01)

        registry = JobRegistry(result_store=store)
        job = registry.submit("demo", forever)
        await asyncio.sleep(0.02)
        registry.cancel(job.id)
        await registry.drain()

        assert store.read(job.id)["status"] == "cancelled"

    @pytest.mark.asyncio
    async def test_a_registry_without_a_store_still_works(self):
        """Persistence is a convenience, not part of the job's contract."""
        registry = JobRegistry()

        job = registry.submit("demo", produce)
        await registry.drain()

        assert job.status == SUCCEEDED
        assert registry.stored_snapshots() == []

    @pytest.mark.asyncio
    async def test_a_failing_store_does_not_fail_the_job(self,
                                                         store,
                                                         monkeypatch):
        """A full disk must not turn a successful backtest into a failed one."""
        def explode_on_write(*args, **kwargs):
            raise OSError("disk full")

        monkeypatch.setattr(store, "write", explode_on_write)
        registry = JobRegistry(result_store=store)

        job = registry.submit("demo", produce)
        await registry.drain()

        assert job.status == SUCCEEDED
        assert job.result == {"value": "done"}


class TestSnapshotLookup:

    @pytest.mark.asyncio
    async def test_memory_is_preferred_over_disk(self,
                                                 store):
        registry = JobRegistry(result_store=store)
        job = registry.submit("demo", produce)
        await registry.drain()

        assert registry.snapshot(job.id) == job.snapshot()

    @pytest.mark.asyncio
    async def test_a_fresh_registry_reads_the_stored_result(self,
                                                            store):
        """The acceptance criterion, at the registry level."""
        first = JobRegistry(result_store=store)
        job = first.submit("demo", produce)
        await first.drain()

        second = JobRegistry(result_store=store)
        snapshot = second.snapshot(job.id)

        assert snapshot is not None
        assert snapshot["status"] == SUCCEEDED
        assert snapshot["result"] == {"value": "done"}

    @pytest.mark.asyncio
    async def test_bookkeeping_fields_do_not_leak(self,
                                                  store):
        """completed_at and schema_version are storage, not API."""
        first = JobRegistry(result_store=store)
        job = first.submit("demo", produce)
        await first.drain()

        snapshot = JobRegistry(result_store=store).snapshot(job.id)

        assert set(snapshot) == {"job_id", "kind", "status", "progress",
                                 "message", "result", "error"}

    def test_an_unknown_job_is_none(self,
                                    store):
        assert JobRegistry(result_store=store).snapshot("nope") is None

    @pytest.mark.asyncio
    async def test_stored_snapshots_exclude_live_jobs(self,
                                                      store):
        """A job this process ran must not appear twice in a listing."""
        registry = JobRegistry(result_store=store)
        registry.submit("demo", produce)
        await registry.drain()

        assert registry.stored_snapshots() == []


class TestRetention:

    @pytest.mark.asyncio
    async def test_old_results_are_pruned(self,
                                          store):
        registry = JobRegistry(result_store=store)

        for _ in range(MAX_STORED_RESULTS + 5):
            registry.submit("demo", produce)
        await registry.drain()

        assert len(store.list_ids()) == MAX_STORED_RESULTS

    @pytest.mark.asyncio
    async def test_the_most_recent_results_survive(self,
                                                   store):
        registry = JobRegistry(result_store=store)
        ids = []

        for index in range(MAX_STORED_RESULTS + 3):
            async def body(report, value=str(index)):
                return {"value": value}

            ids.append(registry.submit("demo", body).id)
            await registry.drain()

        remaining = set(store.list_ids())

        assert ids[-1] in remaining
        assert ids[0] not in remaining

    @pytest.mark.asyncio
    async def test_under_the_limit_nothing_is_pruned(self,
                                                     store):
        registry = JobRegistry(result_store=store)

        for _ in range(5):
            registry.submit("demo", produce)
        await registry.drain()

        assert len(store.list_ids()) == 5


class TestThroughTheApi:

    @pytest.fixture
    def storage(self,
                tmp_path):
        return tmp_path

    def _client(self,
                storage) -> TestClient:
        config = ServerConfig(auth_token=TOKEN, storage_root=storage)
        return TestClient(create_app(config), raise_server_exceptions=False)

    def test_a_result_survives_a_restart(self,
                                         storage):
        """The acceptance criterion: restart, and the result is still there.

        Two independent apps over the same storage root stand in for a
        restart — nothing is shared between them but the directory on disk.
        """
        with self._client(storage) as first:
            registry = first.app.state.jobs
            job_id = first.portal.call(_submit, registry, "demo", produce).id
            first.portal.call(registry.drain)

            assert first.get(f"/jobs/{job_id}", headers=auth()).json()["result"] == {
                "value": "done"}

        with self._client(storage) as second:
            response = second.get(f"/jobs/{job_id}", headers=auth())

        assert response.status_code == 200
        assert response.json()["status"] == SUCCEEDED
        assert response.json()["result"] == {"value": "done"}

    def test_a_restarted_listing_includes_the_old_result(self,
                                                         storage):
        with self._client(storage) as first:
            registry = first.app.state.jobs
            job_id = first.portal.call(_submit, registry, "demo", produce).id
            first.portal.call(registry.drain)

        with self._client(storage) as second:
            listed = second.get("/jobs", headers=auth()).json()["jobs"]

        assert [job["job_id"] for job in listed] == [job_id]

    def test_an_unknown_job_is_still_404_after_restart(self,
                                                       storage):
        with self._client(storage) as client:
            response = client.get("/jobs/never-existed", headers=auth())

        assert response.status_code == 404
        assert response.json()["error"]["code"] == "DATA_NOT_FOUND"


async def _submit(registry,
                  kind: str,
                  body):
    """Submit a job from inside the app's event loop."""
    return registry.submit(kind, body)
