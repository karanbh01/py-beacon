# src/beacon/server/jobs.py
"""
In-process job registry for long-running work.

A backtest or an optimisation takes long enough that holding an HTTP
connection open for it is the wrong shape: the client wants to submit, get an
id back, and either poll or listen. Jobs run as asyncio tasks in the server
process — there is no queue, no broker, and no persistence, which suits a
single local process owned by one desktop client. Restarting the server loses
in-flight jobs, and that is the correct trade for this deployment.

Every state change is published to subscribers, so the WebSocket feed and
polling see the same thing.
"""
import asyncio
import logging
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import suppress
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from .store import DocumentStore

logger = logging.getLogger(__name__)

# Job lifecycle. Terminal states never change again, which is what lets a
# poller stop and a subscriber unsubscribe.
PENDING = "pending"
RUNNING = "running"
SUCCEEDED = "succeeded"
FAILED = "failed"
CANCELLED = "cancelled"
TERMINAL_STATES = frozenset({SUCCEEDED, FAILED, CANCELLED})

# A subscriber that stops reading must not stall the job publishing to it, so
# each queue is bounded and the oldest event is dropped when it fills. Progress
# is a stream of snapshots rather than a ledger, so a dropped intermediate
# frame costs nothing; the terminal event still arrives.
SUBSCRIBER_QUEUE_SIZE = 100

# Completed results kept on disk. A backtest payload measures about 164 bytes
# per day, so a 30-year daily run is roughly 1.2 MB and this bound is a few tens
# of megabytes at absolute worst — small enough that plain JSON through the
# DocumentStore is the right storage, and no compact format is warranted.
MAX_STORED_RESULTS = 50

# Fields of a job snapshot that survive a restart. Anything else in the stored
# document is bookkeeping and is not part of the API's job shape.
PERSISTED_FIELDS = ("job_id", "kind", "status", "progress", "message",
                    "result", "error")

# Signature a job body must satisfy: it receives a progress reporter and
# returns whatever should become the job's result.
ProgressReporter = Callable[[float, str], Awaitable[None]]
JobBody = Callable[[ProgressReporter], Awaitable[Any]]


@dataclass
class Job:
    """A unit of background work and its observable state."""
    id: str
    kind: str
    status: str = PENDING
    progress: float = 0.0
    message: str = ""
    result: Any = None
    error: str | None = None
    _task: asyncio.Task[Any] | None = field(default=None, repr=False, compare=False)

    @property
    def is_terminal(self) -> bool:
        """Whether this job has finished, failed or been cancelled."""
        return self.status in TERMINAL_STATES

    def snapshot(self) -> dict[str, Any]:
        """The public view of this job.

        The result is only carried once the job has succeeded — sending a
        half-built result would invite a client to use it.
        """
        return {
            "job_id": self.id,
            "kind": self.kind,
            "status": self.status,
            "progress": round(self.progress, 6),
            "message": self.message,
            "result": self.result if self.status == SUCCEEDED else None,
            "error": self.error,
        }


class JobRegistry:
    """Owns running jobs and the subscribers watching them."""

    def __init__(self,
                 result_store: DocumentStore | None = None) -> None:
        """
        Args:
            result_store: Where completed results are persisted. None keeps
                everything in memory, which is what the unit tests want and
                what a process with nowhere to write falls back to.
        """
        self._jobs: dict[str, Job] = {}
        self._subscribers: set[asyncio.Queue[dict[str, Any]]] = set()
        self._results = result_store

    # -- subscriptions -------------------------------------------------------

    def subscribe(self) -> asyncio.Queue[dict[str, Any]]:
        """Register a subscriber and return its event queue."""
        queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(
            maxsize=SUBSCRIBER_QUEUE_SIZE)
        self._subscribers.add(queue)

        return queue

    def unsubscribe(self,
                    queue: asyncio.Queue[dict[str, Any]]) -> None:
        """Remove a subscriber."""
        self._subscribers.discard(queue)

    def publish(self,
                event: dict[str, Any]) -> None:
        """Send an event to every subscriber, dropping the oldest if full.

        Deliberately synchronous and non-blocking: a job reporting progress
        must never await a slow reader.
        """
        for queue in list(self._subscribers):
            if queue.full():
                # Discard the stalest frame to make room; a subscriber that
                # cannot keep up gets a gappy stream rather than stalling the
                # job or being disconnected.
                with suppress(asyncio.QueueEmpty):
                    queue.get_nowait()

            try:
                queue.put_nowait(event)
            except asyncio.QueueFull:
                logger.warning("Dropping event for a subscriber whose queue is full.")

    def publish_data_freshness(self,
                               dataset: str,
                               detail: dict[str, Any] | None = None) -> None:
        """Announce that a dataset's contents may have changed."""
        self.publish({"type": "data.freshness",
                      "dataset": dataset,
                      "detail": detail or {}})

    # -- jobs ----------------------------------------------------------------

    def get(self,
            job_id: str) -> Job | None:
        """Return an in-memory job by id, or None.

        Only jobs this process ran. Use :meth:`snapshot` to include results
        persisted by an earlier process.
        """
        return self._jobs.get(job_id)

    def snapshot(self,
                 job_id: str) -> dict[str, Any] | None:
        """Return a job's state, from memory or from disk.

        A job this process ran is authoritative; otherwise the persisted result
        of an earlier process is served, which is what lets a completed
        backtest survive a restart and still be readable.

        Args:
            job_id: Identifier of the job.

        Returns:
            dict or None: The snapshot, or None if the job is unknown to both.
        """
        job = self._jobs.get(job_id)
        if job is not None:
            return job.snapshot()

        return self._stored(job_id)

    def _stored(self,
                job_id: str) -> dict[str, Any] | None:
        """Read a persisted result, keeping only the API's job fields."""
        if self._results is None:
            return None

        document = self._results.read(job_id)
        if document is None:
            return None

        return {field: document.get(field) for field in PERSISTED_FIELDS}

    def stored_snapshots(self) -> list[dict[str, Any]]:
        """Every persisted result, for jobs this process did not run."""
        if self._results is None:
            return []

        return [
            {field: document.get(field) for field in PERSISTED_FIELDS}
            for document in self._results.read_all()
            if document.get("job_id") not in self._jobs
        ]

    def latest_result(self,
                      kind: str) -> dict[str, Any] | None:
        """The result of the most recent successful job of a kind.

        Read from the store rather than from memory. Every terminal job is
        persisted, so the store is the complete record, and it is the only one
        of the two that survives a restart — which is the case this exists to
        serve.

        Args:
            kind: The job kind, e.g. ``"backtest:my-index"``.

        Returns:
            dict or None: The stored result, or None when nothing of that kind
            has succeeded.
        """
        if self._results is None:
            return None

        matching = [document for document in self._results.read_all()
                    if document.get("kind") == kind
                    and document.get("status") == SUCCEEDED
                    and document.get("result") is not None]

        if not matching:
            return None

        # Ordered by completion time: ids are UUIDs and say nothing about age.
        newest = max(matching, key=lambda doc: str(doc.get("completed_at", "")))
        result: dict[str, Any] = newest["result"]

        return result

    def forget(self,
               kind: str) -> int:
        """Drop every job and persisted result of one kind.

        The cascade half of deleting the thing a kind is keyed to (BN-157):
        an index's backtest results go with its definition, rather than
        surviving under an id that no longer resolves.

        Args:
            kind: The exact kind, e.g. ``"backtest:my-index"``. Exact rather
                than a prefix, so ``backtest:core`` cannot take
                ``backtest:core-hedged`` with it.

        Returns:
            int: How many records went — in-memory jobs plus persisted
            results — so the caller can log what the delete cost.
        """
        forgotten = 0

        for job_id in [job_id for job_id, job in self._jobs.items()
                       if job.kind == kind]:
            del self._jobs[job_id]
            forgotten += 1

        if self._results is not None:
            for document in self._results.read_all():
                if document.get("kind") != kind:
                    continue

                document_id = document.get("job_id")
                if (document_id is not None
                        and self._results.delete(str(document_id))):
                    forgotten += 1

        if forgotten:
            logger.info(f"Forgot {forgotten} record(s) of kind '{kind}'.")

        return forgotten

    def latest_results_by_kind(self,
                               prefix: str) -> dict[str, dict[str, Any]]:
        """The newest successful result for every kind under a prefix.

        Reads the store, which holds every terminal job including the ones this
        process ran. `stored_snapshots()` deliberately excludes those so a
        listing does not show a job twice, and using it here would hide every
        model the running process had just estimated — which is most of them.

        Args:
            prefix: Kind prefix including its separator, e.g. ``"risk:"``.

        Returns:
            dict: Kind to its newest result.
        """
        if self._results is None:
            return {}

        newest: dict[str, dict[str, Any]] = {}
        stamps: dict[str, str] = {}

        for document in self._results.read_all():
            kind = str(document.get("kind", ""))
            if not kind.startswith(prefix):
                continue
            if document.get("status") != SUCCEEDED or document.get("result") is None:
                continue

            # Ordered by completion time: ids are UUIDs and say nothing about
            # age, so a later estimate under the same kind must win on its
            # timestamp rather than on where it happened to land.
            stamp = str(document.get("completed_at", ""))
            if kind not in stamps or stamp > stamps[kind]:
                newest[kind] = document["result"]
                stamps[kind] = stamp

        return newest

    def _persist(self,
                 job: Job) -> None:
        """Write a finished job's snapshot to disk and prune old ones."""
        if self._results is None or not job.is_terminal:
            return

        try:
            self._results.write(job.id,
                                {**job.snapshot(),
                                 "completed_at": datetime.now(UTC).isoformat()})
            self._prune()
        except Exception as exc:
            # Persistence is a convenience, not part of the job's contract. A
            # full disk must not turn a successful backtest into a failed one.
            logger.error(f"Could not persist result for job {job.id}: {exc}")

    def _prune(self) -> None:
        """Keep only the most recent MAX_STORED_RESULTS results."""
        if self._results is None:
            return

        documents = self._results.read_all()
        excess = len(documents) - MAX_STORED_RESULTS
        if excess <= 0:
            return

        # Oldest first by completion time. Ids are UUIDs, so name order says
        # nothing about age.
        oldest = sorted(documents, key=lambda doc: str(doc.get("completed_at", "")))

        for document in oldest[:excess]:
            job_id = document.get("job_id")
            if isinstance(job_id, str):
                self._results.delete(job_id)

        logger.info(f"Pruned {excess} stored job result(s) beyond the retention limit.")

    def list_jobs(self) -> list[Job]:
        """Every job this process knows about."""
        return list(self._jobs.values())

    def submit(self,
               kind: str,
               body: JobBody) -> Job:
        """Start a job and return it immediately.

        Args:
            kind: Label for what this job is, e.g. "backtest".
            body: Coroutine function taking a progress reporter.

        Returns:
            Job: The registered job, already scheduled.

        Raises:
            RuntimeError: If called off the event loop. FastAPI runs a *sync*
                endpoint in a worker thread, where there is no loop to attach
                a task to — so a submitting endpoint must be `async def`. The
                bare failure is an opaque "no running event loop", hence the
                explicit check.
        """
        try:
            asyncio.get_running_loop()
        except RuntimeError as exc:
            raise RuntimeError(
                "JobRegistry.submit() must be called from the event loop. The "
                "endpoint submitting this job is probably a sync `def`; make "
                "it `async def`.") from exc

        job = Job(id=str(uuid.uuid4()), kind=kind)
        self._jobs[job.id] = job

        job._task = asyncio.create_task(self._run(job, body))

        return job

    async def _run(self,
                   job: Job,
                   body: JobBody) -> None:
        """Drive one job through its lifecycle, publishing every transition."""
        job.status = RUNNING
        self._emit(job)

        async def report(progress: float,
                         message: str = "") -> None:
            # Clamped so a miscounting job cannot publish nonsense; a client
            # rendering a progress bar should never see 1.4.
            job.progress = min(max(progress, 0.0), 1.0)
            job.message = message
            self._emit(job)

        try:
            job.result = await body(report)
            job.status = SUCCEEDED
            job.progress = 1.0
        except asyncio.CancelledError:
            job.status = CANCELLED
            job.message = "Cancelled."
            self._emit(job)
            raise
        except Exception as exc:
            logger.error(f"Job {job.id} ({job.kind}) failed: {exc}")
            job.status = FAILED
            job.error = str(exc)
        finally:
            if job.status != CANCELLED:
                self._emit(job)
            self._persist(job)

    def _emit(self,
              job: Job) -> None:
        """Publish a job's current state."""
        self.publish({"type": "job", **job.snapshot()})

    def cancel(self,
               job_id: str) -> bool:
        """Request cancellation of a job.

        Args:
            job_id: Identifier of the job.

        Returns:
            bool: True if cancellation was requested, False if the job is
            unknown or already finished.
        """
        job = self._jobs.get(job_id)
        if job is None or job.is_terminal or job._task is None:
            return False

        job._task.cancel()

        return True

    async def drain(self) -> None:
        """Await every outstanding task. For shutdown and for tests."""
        tasks = [job._task for job in self._jobs.values() if job._task is not None]

        for task in tasks:
            # Each job already recorded its own outcome on itself; drain only
            # waits for the tasks to finish, so every exception is expected
            # here and none of them should propagate.
            with suppress(BaseException):
                await task


async def stream_events(queue: asyncio.Queue[dict[str, Any]]) -> AsyncIterator[dict[str, Any]]:
    """Yield events from a subscriber queue until cancelled."""
    while True:
        yield await queue.get()
