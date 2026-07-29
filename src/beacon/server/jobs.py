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
from typing import Any

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

    def __init__(self) -> None:
        self._jobs: dict[str, Job] = {}
        self._subscribers: set[asyncio.Queue[dict[str, Any]]] = set()

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
        """Return a job by id, or None."""
        return self._jobs.get(job_id)

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
        """
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
