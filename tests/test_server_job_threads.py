# tests/test_server_job_threads.py
"""BN-333: a job's work runs in a worker thread, so the server keeps answering.

Before it, a backtest ran on the event loop: on the default synthetic store
the submit's 202 took eight minutes and `/health` went unanswered, and the app
took the engine for dead.
"""
import asyncio
import threading
import time

import pytest
from fastapi.testclient import TestClient

from beacon.backtest.main import Backtest
from beacon.server import ServerConfig, create_app
from beacon.server.job_threads import JobCancelledError, run_in_thread
from test_server_backtest import TOKEN, auth, build_fetcher, index_document

# How long the slowed backtest takes, and how much sooner the server must answer.
SLOW_SECONDS = 1.5
PROMPT_SECONDS = 0.5


class TestRunInThread:

    def test_the_loop_runs_while_the_work_does(self):
        ticks: list[float] = []

        async def scenario() -> int:
            async def tick() -> None:
                while True:
                    ticks.append(time.monotonic())
                    await asyncio.sleep(0.01)

            async def report(fraction: float,
                             message: str) -> None:
                return None

            ticker = asyncio.create_task(tick())
            result = await run_in_thread(report, lambda _: (time.sleep(0.3), 42)[1])
            ticker.cancel()

            return result

        assert asyncio.run(scenario()) == 42
        assert len(ticks) > 10

    def test_reports_reach_the_loop_in_order(self):
        seen: list[tuple[float, str, bool]] = []

        async def scenario() -> None:
            loop_thread = threading.get_ident()

            async def report(fraction: float,
                             message: str) -> None:
                seen.append((fraction, message, threading.get_ident() == loop_thread))

            def work(publish) -> None:
                for step in range(5):
                    publish(step / 4, f"step {step}")

            await run_in_thread(report, work)
            await asyncio.sleep(0.05)

        asyncio.run(scenario())

        assert seen == [(step / 4, f"step {step}", True) for step in range(5)]

    def test_a_cancelled_job_stops_at_its_next_report(self):
        finished = threading.Event()
        stopped = threading.Event()

        async def scenario() -> None:
            async def report(fraction: float,
                             message: str) -> None:
                return None

            def work(publish) -> None:
                try:
                    time.sleep(0.2)
                    publish(0.5, "half")
                    finished.set()
                except JobCancelledError:
                    stopped.set()

            task = asyncio.create_task(run_in_thread(report, work))
            await asyncio.sleep(0.05)
            task.cancel()

            with pytest.raises(asyncio.CancelledError):
                await task

            await asyncio.to_thread(stopped.wait, 2.0)

        asyncio.run(scenario())

        assert stopped.is_set() and not finished.is_set()


@pytest.fixture
def slow_backtests(monkeypatch):
    """Every backtest takes SLOW_SECONDS longer, in the thread that runs it."""
    original = Backtest.run

    def slow(self, *args, **kwargs):
        time.sleep(SLOW_SECONDS)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(Backtest, "run", slow)


@pytest.fixture
def client(tmp_path):
    config = ServerConfig(auth_token=TOKEN, data_fetcher=build_fetcher(), storage_root=tmp_path)

    with TestClient(create_app(config), raise_server_exceptions=False) as entered:
        assert entered.post("/indices", json=index_document(), headers=auth()).status_code == 200
        assert entered.put("/strategies/BT", json={"id": "BT", "name": "BT", "index_id": "BT"},
                           headers=auth()).status_code == 200
        yield entered


class TestTheServerKeepsAnswering:

    def test_while_a_backtest_runs(self,
                                   client,
                                   slow_backtests):
        began = time.monotonic()
        submitted = client.post("/strategies/BT/backtest", json={}, headers=auth())
        accepted_after = time.monotonic() - began

        began = time.monotonic()
        health = client.get("/health", headers=auth())
        answered_after = time.monotonic() - began

        job_id = submitted.json()["job_id"]
        running = client.get(f"/jobs/{job_id}", headers=auth()).json()["status"]
        client.portal.call(client.app.state.jobs.drain)
        finished = client.get(f"/jobs/{job_id}", headers=auth()).json()["status"]

        assert submitted.status_code == 202
        assert accepted_after < PROMPT_SECONDS
        assert health.status_code == 200 and answered_after < PROMPT_SECONDS
        assert running in ("pending", "running")
        assert finished == "succeeded"
