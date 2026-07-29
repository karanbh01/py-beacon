# tests/test_server_jobs.py
"""Tests for the job registry, job polling, and the WebSocket event feed."""
import asyncio

import pytest
from fastapi import WebSocketDisconnect
from fastapi.testclient import TestClient

from beacon.server import ServerConfig, create_app
from beacon.server.jobs import (
    CANCELLED,
    FAILED,
    RUNNING,
    SUBSCRIBER_QUEUE_SIZE,
    SUCCEEDED,
    JobRegistry,
)
from beacon.server.routers.jobs import POLICY_VIOLATION

TOKEN = "test-token-value"


def auth() -> dict[str, str]:
    """Valid Authorization header."""
    return {"Authorization": f"Bearer {TOKEN}"}


@pytest.fixture
def client(tmp_path):
    """Client entered as a context manager.

    Entering it starts the portal, which is how a synchronous test reaches
    into the app's event loop to submit and await jobs.
    """
    config = ServerConfig(auth_token=TOKEN, storage_root=tmp_path)
    with TestClient(create_app(config), raise_server_exceptions=False) as entered:
        yield entered


async def submit(registry,
                 kind: str,
                 body):
    """Submit a job from inside the app's event loop."""
    return registry.submit(kind, body)


async def pause(seconds: float = 0.02) -> None:
    """Let the loop run for a moment."""
    await asyncio.sleep(seconds)


async def forever(report) -> None:
    """A job that never finishes on its own."""
    while True:
        await asyncio.sleep(0.01)


async def counting_job(report,
                       steps: int = 4,
                       delay: float = 0.01) -> dict[str, int]:
    """A job that reports progress in steps and then returns a result."""
    for step in range(1, steps + 1):
        await asyncio.sleep(delay)
        await report(step / steps, f"step {step} of {steps}")

    return {"steps": steps}


class TestRegistry:
    """The registry in isolation, driven directly on an event loop."""

    @pytest.mark.asyncio
    async def test_job_runs_to_success(self):
        registry = JobRegistry()

        job = registry.submit("demo", counting_job)
        await registry.drain()

        assert job.status == SUCCEEDED
        assert job.progress == 1.0
        assert job.result == {"steps": 4}

    @pytest.mark.asyncio
    async def test_progress_is_reported_along_the_way(self):
        registry = JobRegistry()
        queue = registry.subscribe()

        registry.submit("demo", counting_job)
        await registry.drain()

        events = []
        while not queue.empty():
            events.append(queue.get_nowait())

        progresses = [e["progress"] for e in events if e["type"] == "job"]

        assert progresses[0] == 0.0                      # the running transition
        assert 1.0 in progresses                          # the final state
        assert sorted(progresses) == progresses           # never goes backwards

    @pytest.mark.asyncio
    async def test_failure_is_recorded_not_raised(self):
        async def failing(report):
            raise ValueError("deliberate")

        registry = JobRegistry()

        job = registry.submit("demo", failing)
        await registry.drain()

        assert job.status == FAILED
        assert job.error == "deliberate"

    @pytest.mark.asyncio
    async def test_result_is_withheld_until_success(self):
        async def slow(report):
            await asyncio.sleep(0.2)
            return "done"

        registry = JobRegistry()
        job = registry.submit("demo", slow)

        await asyncio.sleep(0.01)
        assert job.status == RUNNING
        assert job.snapshot()["result"] is None

        await registry.drain()
        assert job.snapshot()["result"] == "done"

    @pytest.mark.asyncio
    async def test_cancel_stops_a_running_job(self):
        async def forever(report):
            while True:
                await asyncio.sleep(0.01)

        registry = JobRegistry()
        job = registry.submit("demo", forever)
        await asyncio.sleep(0.02)

        assert registry.cancel(job.id) is True
        await registry.drain()
        assert job.status == CANCELLED

    @pytest.mark.asyncio
    async def test_cancel_is_false_for_unknown_or_finished(self):
        registry = JobRegistry()

        assert registry.cancel("nope") is False

        job = registry.submit("demo", counting_job)
        await registry.drain()

        assert registry.cancel(job.id) is False

    @pytest.mark.asyncio
    async def test_progress_is_clamped(self):
        """A miscounting job must not publish a progress of 1.4."""
        async def overshoots(report):
            await report(5.0, "too far")
            await report(-2.0, "too little")
            return None

        registry = JobRegistry()
        job = registry.submit("demo", overshoots)
        await registry.drain()

        assert 0.0 <= job.progress <= 1.0

    @pytest.mark.asyncio
    async def test_slow_subscriber_does_not_stall_the_job(self):
        """A full queue drops its oldest frame rather than blocking."""
        registry = JobRegistry()
        queue = registry.subscribe()

        async def chatty(report):
            for step in range(SUBSCRIBER_QUEUE_SIZE * 2):
                await report(step / (SUBSCRIBER_QUEUE_SIZE * 2), "")
            return "finished"

        job = registry.submit("demo", chatty)
        await registry.drain()

        assert job.status == SUCCEEDED
        assert queue.qsize() <= SUBSCRIBER_QUEUE_SIZE

    @pytest.mark.asyncio
    async def test_unsubscribe_stops_delivery(self):
        registry = JobRegistry()
        queue = registry.subscribe()
        registry.unsubscribe(queue)

        registry.publish_data_freshness("market")

        assert queue.empty()

    @pytest.mark.asyncio
    async def test_data_freshness_events_are_published(self):
        registry = JobRegistry()
        queue = registry.subscribe()

        registry.publish_data_freshness("market", {"identifiers": 3})

        event = queue.get_nowait()

        assert event == {"type": "data.freshness",
                         "dataset": "market",
                         "detail": {"identifiers": 3}}


class TestPolling:
    """The acceptance criterion: a job resolves via polling."""

    def test_unknown_job_is_404(self,
                                client):
        response = client.get("/jobs/nope", headers=auth())

        assert response.status_code == 404
        assert response.json()["error"]["code"] == "DATA_NOT_FOUND"

    def test_requires_authentication(self,
                                     client):
        assert client.get("/jobs").status_code == 401

    def test_job_appears_in_the_listing_and_resolves(self,
                                                     client):
        registry = client.app.state.jobs

        job = client.portal.call(submit, registry, "demo", counting_job)
        client.portal.call(registry.drain)

        listed = client.get("/jobs", headers=auth()).json()["jobs"]
        assert [j["job_id"] for j in listed] == [job.id]

        polled = client.get(f"/jobs/{job.id}", headers=auth()).json()
        assert polled["status"] == SUCCEEDED
        assert polled["progress"] == 1.0
        assert polled["result"] == {"steps": 4}

    def test_cancel_via_the_api(self,
                                client):
        registry = client.app.state.jobs

        job = client.portal.call(submit, registry, "demo", forever)
        client.portal.call(pause)

        assert client.delete(f"/jobs/{job.id}", headers=auth()).status_code == 200

        client.portal.call(registry.drain)

        assert client.get(f"/jobs/{job.id}", headers=auth()).json()["status"] == CANCELLED

    def test_cancelling_a_finished_job_is_not_an_error(self,
                                                       client):
        """The client may simply have raced the completion."""
        registry = client.app.state.jobs

        job = client.portal.call(submit, registry, "demo", counting_job)
        client.portal.call(registry.drain)

        response = client.delete(f"/jobs/{job.id}", headers=auth())

        assert response.status_code == 200
        assert response.json()["status"] == SUCCEEDED


class TestWebSocket:
    """The acceptance criterion: a slow job streams progress over the socket."""

    def test_rejects_a_missing_token(self,
                                     client):
        with (pytest.raises(WebSocketDisconnect) as excinfo,
              client.websocket_connect("/ws") as socket):
            socket.receive_json()

        assert excinfo.value.code == POLICY_VIOLATION

    def test_rejects_a_wrong_token(self,
                                   client):
        with (pytest.raises(WebSocketDisconnect) as excinfo,
              client.websocket_connect("/ws?token=wrong") as socket):
            socket.receive_json()

        assert excinfo.value.code == POLICY_VIOLATION

    def test_streams_job_progress_to_completion(self,
                                                client):
        registry = client.app.state.jobs

        with client.websocket_connect(f"/ws?token={TOKEN}") as socket:
            client.portal.call(submit, registry, "demo", counting_job)

            statuses = []
            progresses = []
            for _ in range(20):
                event = socket.receive_json()
                if event["type"] != "job":
                    continue
                statuses.append(event["status"])
                progresses.append(event["progress"])
                if event["status"] in {SUCCEEDED, FAILED, CANCELLED}:
                    break

        assert statuses[0] == RUNNING
        assert statuses[-1] == SUCCEEDED
        assert progresses[-1] == 1.0
        # Intermediate frames actually arrived, rather than only the endpoints.
        assert any(0.0 < p < 1.0 for p in progresses)

    def test_streams_data_freshness_events(self,
                                           client):
        registry = client.app.state.jobs

        async def announce() -> None:
            registry.publish_data_freshness("market")

        with client.websocket_connect(f"/ws?token={TOKEN}") as socket:
            client.portal.call(announce)

            event = socket.receive_json()

        assert event["type"] == "data.freshness"
        assert event["dataset"] == "market"

    def test_disconnecting_unsubscribes(self,
                                        client):
        registry = client.app.state.jobs

        with client.websocket_connect(f"/ws?token={TOKEN}"):
            pass

        # A leaked queue would keep accumulating events for the process's life.
        registry.publish_data_freshness("market")

        assert registry._subscribers == set()


class TestSocketPolicyCode:

    def test_policy_violation_code_is_the_websocket_one(self):
        """1008 is the protocol's policy-violation close code."""
        assert POLICY_VIOLATION == 1008
