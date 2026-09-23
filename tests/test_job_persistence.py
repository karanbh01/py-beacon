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
        assert document["error"]["message"] == "deliberate failure"
        # The code survives the round trip too: a restart that kept the prose
        # and lost the classification would put the job path back where
        # BN-199 found it (BN-199).
        assert document["error"]["code"] == "INVALID_ARGUMENT"

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
        assert registry.results is None

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
    async def test_the_listing_excludes_live_jobs_held_on_disk(self,
                                                               store):
        """A job this process ran must not appear twice in a listing.

        Asserted through `_persisted`, which is where the exclusion moved when
        the listing's tolerance did (BN-178): the registry owns the store, the
        route owns the model, and the dedupe belongs with the rows it dedupes.
        """
        from beacon.server.routers.jobs import _persisted

        registry = JobRegistry(result_store=store)
        registry.submit("demo", produce)
        await registry.drain()

        # A breakdown rather than a bare count since BN-201; nothing was
        # skipped, so every cause is zero.
        rows, skipped = _persisted(registry)

        assert rows == []
        assert skipped.total == 0
        assert len(store.list_ids()) == 1


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


def wreck(store: DocumentStore,
          document_id: str) -> None:
    """Leave a truncated file in the collection, as an interrupted write would."""
    (store.directory / f"{document_id}.json").write_text(
        '{"job_id": "' + document_id + '", "kin', encoding="utf-8")


def result_document(job_id: str,
                    kind: str,
                    completed_at: str) -> dict:
    """A complete persisted result, as `_persist` writes one."""
    return {"job_id": job_id,
            "kind": kind,
            "status": SUCCEEDED,
            "progress": 1.0,
            "message": "",
            "result": {"value": job_id},
            "error": None,
            "completed_at": completed_at}


class TestTheRegistrysOtherReadersOfTheSameCollection:
    """BN-178: five callers read the results collection, and they do not all
    want the same answer.

    `GET /jobs` is a listing and skips what it cannot read; that half lives at
    the route, with `JobStatus`, and is tested in
    `test_server_document_faults.py`. What is audited here is the bookkeeping
    the registry keeps for itself, where the distinction BN-177 named actually
    bites: a *query* answering a read endpoint must skip a corrupt result, and
    *retention* must see it, because a file it cannot count is a file it can
    never reap.
    """

    def test_latest_result_skips_an_unreadable_result(self,
                                                      store):
        """The caller with the widest blast radius: `/beacon/{id}/overview`,
        the compare view and the factsheet render all read through it, so one
        bad file used to 500 endpoints that are not about jobs at all."""
        store.write("good", result_document("good", "backtest:tech", "2025-01-01"))
        wreck(store, "wreckage")

        assert JobRegistry(result_store=store).latest_result(
            "backtest:tech") == {"value": "good"}

    def test_latest_results_by_kind_skips_an_unreadable_result(self,
                                                               store):
        """Backs `/risk-models`, which is a listing by another name."""
        store.write("good", result_document("good", "risk:one", "2025-01-01"))
        wreck(store, "wreckage")

        by_kind = JobRegistry(result_store=store).latest_results_by_kind("risk:")

        assert by_kind == {"risk:one": {"value": "good"}}

    def test_forget_still_cascades_beside_an_unreadable_result(self,
                                                              store):
        """Deleting an index drops its backtest results. One unreadable file in
        the collection used to 500 the delete of an unrelated index."""
        store.write("mine", result_document("mine", "backtest:tech", "2025-01-01"))
        store.write("other", result_document("other", "backtest:other", "2025-01-01"))
        wreck(store, "wreckage")

        registry = JobRegistry(result_store=store)

        assert registry.forget("backtest:tech") == 1
        assert set(store.list_ids()) == {"other", "wreckage"}

    def test_forget_leaves_the_unreadable_result_alone(self,
                                                       store):
        """Deliberate, and the one place the tolerant answer is not obviously
        right: a document the server cannot read carries no `kind`, so deleting
        it would be a cascade guessing at what it had hold of."""
        wreck(store, "wreckage")

        JobRegistry(result_store=store).forget("backtest:tech")

        assert store.list_ids() == ["wreckage"]

    def test_retention_counts_results_it_cannot_read(self,
                                                     store):
        """The caller that must SEE a corrupt file rather than skip it.

        Counting only what parses would bound the *readable* results at the
        limit and let unreadable ones accumulate beside them without limit —
        exactly the files least worth keeping, since nothing can serve them.
        """
        for index in range(MAX_STORED_RESULTS):
            store.write(f"job-{index:03d}",
                        result_document(f"job-{index:03d}", "demo",
                                        f"2025-01-{index % 28 + 1:02d}"))
        wreck(store, "wreckage")

        registry = JobRegistry(result_store=store)
        registry._prune()

        assert len(store.list_ids()) == MAX_STORED_RESULTS
        assert "wreckage" not in store.list_ids()

    def test_retention_reaps_an_unreadable_result_first(self,
                                                        store):
        """It sorts oldest because it has no readable completion stamp, which
        is the right order and also what eventually clears the orphan `forget`
        had to leave behind."""
        wreck(store, "wreckage")
        for index in range(MAX_STORED_RESULTS):
            store.write(f"job-{index:03d}",
                        result_document(f"job-{index:03d}", "demo",
                                        f"2025-01-{index % 28 + 1:02d}"))

        JobRegistry(result_store=store)._prune()

        assert "wreckage" not in store.list_ids()
        assert len(store.list_ids()) == MAX_STORED_RESULTS

    def test_an_unreadable_result_is_answered_as_an_unknown_job(self,
                                                                store):
        """`snapshot` is a read, so it answers the way every other read does."""
        wreck(store, "wreckage")

        assert JobRegistry(result_store=store).snapshot("wreckage") is None


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
