# src/beacon/server/routers/synthetic.py
"""Generate synthetic data into a new store, as a job.

The engine generates on request, so an app can start the engine first and
offer "Generate synthetic data" when nothing is loaded.

The job runs `python -m beacon.synthetic` in a child process rather than the
generator in this process, for three reasons:

- One path. The engine runs the same command a person would, so the same
  settings always give the same data, whichever way it was made.
- Memory. The default size peaks at about 2.5 GB. A child process gives it
  all back when it exits; a long-running engine might not.
- Failure. If generation runs out of memory or crashes, the child dies and
  the engine carries on. Cancelling the job kills the child.

The child prints a progress line per stage (`--progress`), which the job
reports on the event socket.
"""
# Added in BN-237. Before it, beacon-ui ran the generator itself before
# starting the engine. BN-240 reused `run_synthetic` to extend a store.
import asyncio
import shutil
import subprocess
import sys
from collections import deque
from pathlib import Path
from typing import Any

from ..._optional import require
from ...exceptions import CalculationError
from ...index.schedule import DEFAULT_CALENDAR
from ...synthetic.__main__ import (
    PROGRESS_PREFIX,
    resolve_assets,
    resolve_window,
)
from ...synthetic.dataset import (
    DEFAULT_EQUITY_PREMIUM,
    DEFAULT_RISK_FREE_RATE,
    DEFAULT_SEED,
    SyntheticConfig,
)
from ..data_stores import StoreRegistry
from ..jobs import JobRegistry, ProgressReporter
from ..schemas import GenerateJobStatus, GenerateResult
from ..store_schemas import GenerateSyntheticRequest
from .stores import managed_root, remove_managed_folder, serve_store

require("fastapi", "The Beacon API server")

from fastapi import APIRouter, FastAPI, Request, status  # noqa: E402

# How much of the job's progress generation takes; loading the new store
# takes the rest.
GENERATION_SHARE = 0.9

# Lines of the child's own output kept for the error message if it fails.
ERROR_TAIL = 20


def _config(body: GenerateSyntheticRequest) -> SyntheticConfig:
    """The settings to generate with, defaults filled by the command's rules.

    Built here, before the job starts, so a bad request (an end before its
    start, an unknown calendar) is refused at once rather than becoming a job
    that fails a moment later.

    Raises:
        ValueError: If the settings do not describe a dataset. Answered as
            422 INVALID_ARGUMENT.
    """
    start, end = resolve_window(body.start, body.end, body.long_history)
    config = SyntheticConfig(
        assets=resolve_assets(body.assets, body.extended_universe),
        start=start,
        end=end,
        seed=body.seed if body.seed is not None else DEFAULT_SEED,
        risk_free_rate=(body.risk_free_rate if body.risk_free_rate is not None
                        else DEFAULT_RISK_FREE_RATE),
        equity_premium=(body.equity_premium if body.equity_premium is not None
                        else DEFAULT_EQUITY_PREMIUM),
        features=body.features,
        calendar=body.calendar or DEFAULT_CALENDAR)

    return config


def command(config: SyntheticConfig,
            out: Path) -> list[str]:
    """The command line that generates *config* into *out*.

    Every setting is passed explicitly, dates included, so the child
    generates exactly what was validated, even if it runs past midnight.
    """
    arguments = [sys.executable, "-m", "beacon.synthetic", "--progress",
                 "--out", str(out),
                 "--assets", str(config.assets),
                 "--start", config.start,
                 "--end", config.end,
                 "--seed", str(config.seed),
                 "--risk-free-rate", repr(config.risk_free_rate),
                 "--equity-premium", repr(config.equity_premium),
                 "--calendar", config.calendar]

    if not config.features:
        arguments.append("--no-features")

    return arguments


def build_synthetic_router() -> APIRouter:
    """Build the /data/synthetic router."""
    router = APIRouter(prefix="/data/synthetic", tags=["data stores"])

    # async, as every job-submitting endpoint must be.
    @router.post("",
                 response_model=GenerateJobStatus,
                 status_code=status.HTTP_202_ACCEPTED)
    async def generate(request: Request,
                       body: GenerateSyntheticRequest | None = None
                       ) -> GenerateJobStatus:
        """Generate synthetic data into a new store, and serve it by default.

        The store is registered at once, so it appears in `GET /data/stores`
        while it is being written (not yet readable). If generation fails or
        is cancelled, it is removed again.
        """
        settings = body if body is not None else GenerateSyntheticRequest()
        config = _config(settings)

        registry: StoreRegistry = request.app.state.data_stores
        store_id = registry.new_id(settings.name)
        record = registry.create(settings.name,
                                 managed_root(request.app) / store_id,
                                 managed=True,
                                 store_id=store_id)

        jobs: JobRegistry = request.app.state.jobs
        job = jobs.submit(f"generate:{store_id}",
                          _generate_job(request.app, record, config,
                                        settings.activate))

        return GenerateJobStatus(**job.snapshot())

    return router


def _generate_job(app: FastAPI,
                  record: dict[str, Any],
                  config: SyntheticConfig,
                  activate: bool) -> Any:
    """The coroutine that generates a store and, if asked, serves it."""

    async def run(report: ProgressReporter) -> dict[str, Any]:
        final = Path(record["path"])
        partial = final.with_name(f".{final.name}.partial")
        registry: StoreRegistry = app.state.data_stores

        try:
            shutil.rmtree(partial, ignore_errors=True)
            await run_synthetic(command(config, partial), report)
            partial.rename(final)
        except BaseException:
            # Failed or cancelled: nothing half-written stays behind, in the
            # folder or in the registry.
            shutil.rmtree(partial, ignore_errors=True)
            remove_managed_folder(app, final)
            registry.delete(record["id"])
            raise

        activated = False

        # Served only when no other load is under way; otherwise the new
        # store is left ready to activate, rather than racing that load.
        if activate and not app.state.active_data.loading:
            app.state.active_data.loading = True
            await serve_store(app, record, report)
            activated = True

        return GenerateResult(store_id=record["id"],
                              name=record["name"],
                              path=str(final),
                              activated=activated).model_dump()

    return run


async def _report(report: ProgressReporter,
                  fraction: float,
                  stage: str) -> None:
    """A coroutine around one progress report, to hand across threads."""
    await report(fraction, stage)


def extend_command(path: Path,
                   end: str | None) -> list[str]:
    """The command line that extends the synthetic store at *path*."""
    arguments = [sys.executable, "-m", "beacon.synthetic", "--progress",
                 "--extend", str(path)]

    return [*arguments, "--end", end] if end is not None else arguments


async def run_synthetic(arguments: list[str],
                        report: ProgressReporter) -> None:
    """Run `python -m beacon.synthetic` in a child process, reporting progress.

    Generates a store or extends one, for the reasons in the module
    docstring.

    Raises:
        CalculationError: If the command exits unsuccessfully, carrying the
            end of its own output.
    """
    loop = asyncio.get_running_loop()

    process = subprocess.Popen(arguments,
                               stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT,
                               text=True,
                               encoding="utf-8",
                               errors="replace")

    def pump() -> list[str]:
        """Read the child's output until it exits. Runs in a worker thread."""
        tail: deque[str] = deque(maxlen=ERROR_TAIL)

        assert process.stdout is not None
        for line in process.stdout:
            text = line.rstrip()

            if text.startswith(PROGRESS_PREFIX):
                _, fraction, stage = text.split(" ", 2)
                asyncio.run_coroutine_threadsafe(
                    _report(report, float(fraction) * GENERATION_SHARE, stage),
                    loop)
            elif text:
                tail.append(text)

        process.wait()

        return list(tail)

    try:
        tail = await asyncio.to_thread(pump)
    except asyncio.CancelledError:
        process.kill()
        raise

    if process.returncode != 0:
        raise CalculationError(
            calculation_name="SyntheticGeneration",
            details=(f"the generator exited with code {process.returncode}: "
                     + (tail[-1] if tail else "no output")))
