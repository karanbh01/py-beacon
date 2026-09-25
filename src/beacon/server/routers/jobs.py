# src/beacon/server/routers/jobs.py
"""
Job polling and the WebSocket event feed.

Polling and the socket report the same state: the socket is a latency
optimisation, not a separate source of truth. A client that misses a frame can
always fall back to `GET /jobs/{id}`, and one that cannot hold a socket open
loses nothing but immediacy.

There is no endpoint here that *creates* a job. Jobs are submitted by the
endpoints that own the work (backtests, optimisation, risk, reports, data
loads and refreshes), so a bare "start a job" route would be dead weight and an easy way to spawn
work with no purpose.
"""
import asyncio
import hmac
from typing import Any

from ..._optional import require
from ...exceptions import DataNotFoundError
from ..documents import SkipCounts, read_collection
from ..jobs import PERSISTED_FIELDS, JobRegistry
from ..schemas import (
    AnyJobStatus,
    Identifier,
    JobCollection,
    JobStatus,
    TolerantCollection,
)

require("fastapi", "The Beacon API server")

from fastapi import APIRouter, Request, WebSocket, WebSocketDisconnect  # noqa: E402

# Closure code for a socket that failed authentication. 1008 is the WebSocket
# policy-violation code; there is no status-code channel before the handshake
# completes, so this is how a rejection is signalled.
POLICY_VIOLATION = 1008


def _registry(request: Request) -> JobRegistry:
    """Return the process's job registry."""
    registry: JobRegistry = request.app.state.jobs

    return registry


def _job_row(document_id: str,
             document: dict[str, Any]) -> JobStatus:
    """A persisted job result as the listing's row, or a ValidationError.

    Trimmed to `PERSISTED_FIELDS` first, because everything else in the stored
    document is the registry's bookkeeping and not part of the API's job shape.
    """
    return JobStatus.model_validate({field: document.get(field)
                                     for field in PERSISTED_FIELDS})


def _persisted(registry: JobRegistry) -> tuple[list[JobStatus], SkipCounts]:
    """Results persisted by an earlier process, skipping what cannot be read.

    Read here rather than in the registry (BN-178). The registry owns the
    collection but not the model: `JobStatus` is the wire shape, and a stored
    result that is valid JSON and cannot satisfy it — the shape a model gaining
    a required field produces — is exactly the fault a listing must skip, so
    the skip has to happen where the model is known. What the registry keeps is
    its bookkeeping over the same store, which answers a different question and
    is audited separately there.

    Jobs this process ran are excluded, so a listing does not show one twice:
    the live snapshot is authoritative and is added by the caller.
    """
    store = registry.results

    if store is None:
        return [], SkipCounts()

    rows, skipped = read_collection(store, _job_row, "job result")

    return [row for row in rows if registry.get(row.job_id) is None], skipped


def _authorise_socket(websocket: WebSocket,
                      token: str | None) -> bool:
    """Check a WebSocket's token.

    The token arrives as a query parameter rather than an Authorization
    header because browsers cannot set headers on a WebSocket handshake. That
    is a genuine constraint of the protocol, not a shortcut — the token is
    still compared in constant time, and the socket is loopback-only.
    """
    expected: str = websocket.app.state.auth_token

    return token is not None and hmac.compare_digest(token, expected)


def build_jobs_router() -> APIRouter:
    """Build the /jobs router and the /ws event feed.

    Returns:
        APIRouter: Router carrying job polling, cancellation and the socket.
    """
    router = APIRouter(tags=["jobs"])

    @router.get("/jobs", response_model=JobCollection)
    def list_jobs(request: Request) -> JobCollection:
        registry = _registry(request)
        live = [JobStatus(**job.snapshot()) for job in registry.list_jobs()]

        # Results persisted by an earlier process appear alongside this one's,
        # so a restart does not make completed work vanish from the listing —
        # and one it cannot read is skipped and counted rather than taking the
        # listing with it, which is what every other collection does (BN-178).
        stored, skipped = _persisted(registry)

        return JobCollection(jobs=live + stored, **TolerantCollection.skips(skipped))

    # The per-kind union (BN-172): the result payload is declared here, which
    # is the endpoint a client reads it from. Constructed as a plain JobStatus
    # either way — the response model is what narrows it, so a kind nothing
    # models yet still answers exactly as it did.
    @router.get("/jobs/{job_id}", response_model=AnyJobStatus)
    def get_job(request: Request,
                job_id: Identifier) -> JobStatus:
        snapshot = _registry(request).snapshot(job_id)
        if snapshot is None:
            raise DataNotFoundError(f"job '{job_id}'", source="JobRegistry")

        return JobStatus(**snapshot)

    @router.delete("/jobs/{job_id}", response_model=AnyJobStatus)
    def cancel_job(request: Request,
                   job_id: Identifier) -> JobStatus:
        registry = _registry(request)
        job = registry.get(job_id)
        if job is None:
            raise DataNotFoundError(f"job '{job_id}'", source="JobRegistry")

        # Cancelling a finished job is not an error — the client may simply
        # have raced the completion — so the current state is returned either
        # way and the client reads `status` to see what happened.
        registry.cancel(job_id)

        return JobStatus(**job.snapshot())

    return router


def build_events_router() -> APIRouter:
    """Build the /ws event feed.

    Kept separate from the HTTP router because the bearer dependency that
    guards every other route takes a Request and cannot run on a WebSocket
    handshake. This router is mounted unguarded and authorises itself.

    Returns:
        APIRouter: Router carrying the WebSocket endpoint.
    """
    router = APIRouter(tags=["events"])

    @router.websocket("/ws")
    async def events(websocket: WebSocket,
                     token: str | None = None) -> None:
        if not _authorise_socket(websocket, token):
            await websocket.close(code=POLICY_VIOLATION, reason="Invalid bearer token.")
            return

        await websocket.accept()
        registry: JobRegistry = websocket.app.state.jobs
        queue = registry.subscribe()

        try:
            while True:
                await websocket.send_json(await queue.get())
        except (WebSocketDisconnect, asyncio.CancelledError):
            pass
        finally:
            # Always unsubscribe: a leaked queue would keep receiving events
            # for the life of the process and slowly fill.
            registry.unsubscribe(queue)

    return router
