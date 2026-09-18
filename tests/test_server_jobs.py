# tests/test_server_jobs.py
"""Tests for the job registry, job polling, and the WebSocket event feed."""
import asyncio
import json

import pytest
from fastapi import WebSocketDisconnect
from fastapi.testclient import TestClient

from beacon.exceptions import CalculationError, DataNotFoundError
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
from beacon.server.store import SCHEMA_VERSION_KEY

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
        assert job.error["message"] == "deliberate"

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


class TestAFailedJobPublishesACode:
    """BN-199: the job path was the one place a refusal and a crash looked alike.

    BN-194 and BN-196 spent two issues establishing that a deliberate refusal
    and a genuine fault must reach a client under different codes, because they
    ask different things of the reader: one names something to change, the
    other names something broken. Everywhere an error travels through the HTTP
    envelope that holds. A job recorded `str(exc)` and nothing else, so both
    arrived as prose and a client had nothing to branch on.

    Through the route rather than off the registry, deliberately (BN-200): the
    field existing on the dataclass and a client being able to read it are
    different claims, and only the second one is what the issue was for.
    """

    def failed_job(self,
                   client,
                   body) -> dict:
        """Submit a failing job through the app and poll its terminal state."""
        registry = client.app.state.jobs

        job = client.portal.call(submit, registry, "demo", body)
        client.portal.call(registry.drain)

        polled = client.get(f"/jobs/{job.id}", headers=auth()).json()
        assert polled["status"] == FAILED

        return polled["error"]

    def test_a_refusal_publishes_its_own_code(self,
                                              client):
        async def refusing(report):
            raise CalculationError(calculation_name="MarketCapWeighted",
                                   details="AAPL has no CLOSE; load its prices")

        error = self.failed_job(client, refusing)

        assert error["code"] == "CALCULATION_ERROR"
        assert "load its prices" in error["message"]

    def test_a_crash_publishes_the_unexpected_code(self,
                                                   client):
        async def crashing(report):
            raise ZeroDivisionError("division by zero")

        error = self.failed_job(client, crashing)

        assert error["code"] == "UNEXPECTED_CALCULATION_FAILURE"

    def test_a_crash_carries_the_original_type(self,
                                               client):
        """Wrapped as an `UnexpectedCalculationError` rather than given a code
        of its own, so it carries exactly what one raised in the library does —
        a client can name the failing class without reading a server log."""
        async def crashing(report):
            raise ZeroDivisionError("division by zero")

        error = self.failed_job(client, crashing)

        assert error["detail"]["original_type"] == "ZeroDivisionError"

    def test_the_two_are_distinguishable(self,
                                         client):
        """The whole issue in one assertion. Both used to be bare prose."""
        async def refusing(report):
            raise CalculationError(calculation_name="X", details="a decision")

        async def crashing(report):
            raise TypeError("a fault")

        assert (self.failed_job(client, refusing)["code"]
                != self.failed_job(client, crashing)["code"])

    def test_the_job_kind_names_the_calculation(self,
                                                client):
        """A job's kind is the honest answer to "what was being computed": it
        is what the work was, and the wrap needs a name."""
        async def crashing(report):
            raise TypeError("a fault")

        error = self.failed_job(client, crashing)

        assert error["detail"]["calculation_name"] == "demo"

    def test_a_successful_job_carries_no_error(self,
                                               client):
        registry = client.app.state.jobs

        job = client.portal.call(submit, registry, "demo", counting_job)
        client.portal.call(registry.drain)

        assert client.get(f"/jobs/{job.id}",
                          headers=auth()).json()["error"] is None


class TestTheJobCodeMatchesTheRequestCode:
    """One exception, one code, whichever way it travels.

    The point of routing both through `failure_envelope` rather than writing a
    second ladder: two ladders agree on the day they are written and drift
    afterwards, and a client that branches on `error.code` cannot tell which
    one produced the answer it got.
    """

    @pytest.mark.parametrize("exception,expected", [
        (CalculationError(calculation_name="X", details="a decision"),
         "CALCULATION_ERROR"),
        (DataNotFoundError("universe 'tech'", source="DocumentStore"),
         "DATA_NOT_FOUND"),
        (ValueError("end_date must be after start_date"),
         "INVALID_ARGUMENT"),
    ])
    def test_the_same_exception_gets_the_same_code(self,
                                                   exception,
                                                   expected):
        from beacon.server.errors import failure_envelope

        assert failure_envelope(exception, calculation="demo")["code"] == expected

    def test_an_unregistered_fault_is_not_given_the_refusal_catch_all(self):
        """`BEACON_ERROR` means "a refusal whose subclass nobody registered".
        Reusing it for a crash puts BN-194's conflation back at one remove."""
        from beacon.server.errors import failure_envelope

        envelope = failure_envelope(KeyError("CLOSE"), calculation="demo")

        assert envelope["code"] != "BEACON_ERROR"
        assert envelope["code"] == "UNEXPECTED_CALCULATION_FAILURE"


class TestAJobStoredBeforeCodesExisted:
    """BN-199: the migration, seen from the routes it keeps working.

    A v2 document holds `error` as a string. Without the v2 -> v3 migration the
    listing drops it (tolerantly, counted in `skipped`) and the detail route
    answers 422 INVALID_ARGUMENT quoting a pydantic error about the server's
    own model — the caller blamed for a shape the server changed underneath
    them. Both measured before this was written.

    Through the routes rather than the store, because the store test already
    proves the document is rewritten and that is not the claim at issue here:
    the claim is that a client can still read a job the previous build wrote.
    """

    @pytest.fixture
    def with_an_old_job(self,
                        tmp_path):
        """A server whose results store holds one pre-BN-199 failed job."""
        results = tmp_path / "job_results"
        results.mkdir(parents=True)
        (results / "old.json").write_text(
            json.dumps({"job_id": "old",
                        "kind": "backtest:BT",
                        "status": "failed",
                        "progress": 1.0,
                        "message": "",
                        "result": None,
                        "error": "AAPL has no CLOSE; load its prices",
                        SCHEMA_VERSION_KEY: 2}),
            encoding="utf-8")

        config = ServerConfig(auth_token=TOKEN, storage_root=tmp_path)
        with TestClient(create_app(config),
                        raise_server_exceptions=False) as entered:
            yield entered

    def test_the_detail_route_serves_it(self,
                                        with_an_old_job):
        response = with_an_old_job.get("/jobs/old", headers=auth())

        assert response.status_code == 200
        assert response.json()["status"] == FAILED

    def test_its_message_is_still_readable(self,
                                           with_an_old_job):
        error = with_an_old_job.get("/jobs/old", headers=auth()).json()["error"]

        assert error["message"] == "AAPL has no CLOSE; load its prices"

    def test_it_says_its_code_is_unknown_rather_than_guessing(self,
                                                              with_an_old_job):
        error = with_an_old_job.get("/jobs/old", headers=auth()).json()["error"]

        assert error["code"] == "UNCLASSIFIED_FAILURE"

    def test_it_does_not_vanish_from_the_listing(self,
                                                 with_an_old_job):
        """The tolerant listing would skip it, which is right for a corrupt
        file and wrong for one this build made unreadable."""
        listing = with_an_old_job.get("/jobs", headers=auth()).json()

        assert [job["job_id"] for job in listing["jobs"]] == ["old"]
        assert listing["skipped"] == 0
