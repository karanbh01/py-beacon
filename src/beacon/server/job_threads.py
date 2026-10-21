# src/beacon/server/job_threads.py
"""
Running a job's work in a worker thread, so the event loop stays free.

A job body is a coroutine the registry runs as a task on the event loop. A
backtest, an optimisation or a risk model is minutes of synchronous pandas and
numpy work, and run inside that coroutine it holds the loop for all of it:
no request is answered, `/health` included, and the submit's own 202 waits
too, because the task starts before the response goes out.

`run_in_thread` hands the work to a worker thread with a plain progress
function. A report from the thread is passed to the loop to publish, and once
the job is cancelled the next report stops the work, so a cancelled run does
not go on to write its results.
"""
# BN-333, found by beacon-ui: a backtest on the default synthetic store held
# the loop for eight minutes and the app took the engine for dead.
import asyncio
import threading
from collections.abc import Callable
from typing import TypeVar

from .jobs import ProgressReporter

Result = TypeVar("Result")

# What the work is handed: the fraction done and what is happening.
Report = Callable[[float, str], None]


class JobCancelledError(Exception):
    """Raised in the worker thread at its next report after a cancellation."""


async def run_in_thread(report: ProgressReporter,
                        work: Callable[[Report], Result]) -> Result:
    """Run *work* in a worker thread, publishing its reports on the loop.

    Args:
        report: The job's progress reporter.
        work: The synchronous work, called with a progress function.

    Returns:
        What *work* returns.

    Raises:
        asyncio.CancelledError: When the job is cancelled. The thread itself
            cannot be interrupted, so it stops at its next report instead.
    """
    loop = asyncio.get_running_loop()
    cancelled = threading.Event()

    def threaded(fraction: float,
                 message: str) -> None:
        if cancelled.is_set():
            raise JobCancelledError

        async def publish() -> None:
            await report(fraction, message)

        # Not waited on: the loop publishes reports in the order they are
        # handed over, and waiting could hang a thread on a loop shutting down.
        asyncio.run_coroutine_threadsafe(publish(), loop)

    try:
        return await asyncio.to_thread(work, threaded)
    except asyncio.CancelledError:
        cancelled.set()
        raise
