# src/beacon/server/routers/refresh.py
"""Refresh a data store from its own source (BN-240).

What a refresh does depends on the store:

- **Synthetic data** is extended to today, keeping every day it holds. This
  runs `python -m beacon.synthetic --extend` in a child process, as
  generation does.
- **A folder** is read again, picking up files changed outside the engine.
- **A database** has its tables read again.
- **Imported files** have nothing to refresh; importing again makes a new
  store. The request is refused.
- **A folder set to refresh from Yahoo Finance** downloads new prices for its
  instruments and saves them into the folder. Yahoo is never the default: a
  store uses it only when the user chose it for that store.

Whatever a refresh changes is saved, so it survives a restart. If the store
is the one being served, the engine then serves the refreshed data, with a
new `data_version`.
"""
import asyncio
import logging
from collections.abc import Callable
from datetime import date, timedelta
from pathlib import Path
from typing import Any

from ..._optional import require
from ...data import store as data_store
from ...data.ingest import Downloader, ingest_market_data, yfinance_downloader
from ...synthetic import state as synthetic_state
from ..active_data import active_data
from ..data_stores import refresh_plan
from ..jobs import JobRegistry, ProgressReporter
from ..schemas import Identifier, RefreshJobStatus, RefreshResult
from ..store_schemas import RefreshAction, RefreshRequest
from .stores import CONFLICT_RESPONSE, claim_loading, record_or_404, serve_store
from .synthetic import extend_command, run_synthetic

require("fastapi", "The Beacon API server")

from fastapi import APIRouter, FastAPI, HTTPException, Request, status  # noqa: E402

logger = logging.getLogger(__name__)

# How much of the job the refresh itself takes; serving the result takes the
# rest.
REFRESH_SHARE = 0.9


def build_refresh_router() -> APIRouter:
    """Build the /data/stores/{store_id}/refresh route."""
    router = APIRouter(prefix="/data/stores", tags=["data stores"])

    # async, as every job-submitting endpoint must be.
    @router.post("/{store_id}/refresh",
                 response_model=RefreshJobStatus,
                 status_code=status.HTTP_202_ACCEPTED,
                 responses=CONFLICT_RESPONSE)
    async def refresh_store(request: Request,
                            store_id: Identifier,
                            body: RefreshRequest | None = None
                            ) -> RefreshJobStatus:
        """Bring a store up to date from its own source.

        Refused with 409 when the store has nothing to refresh (imported
        files, or synthetic data too old to extend), when a folder or
        database that is not being served is asked to re-read (it is read
        afresh whenever it is activated), or when the store is already
        refreshing or loading.
        """
        record = record_or_404(request, store_id)

        return submit_refresh(request, record,
                              body.end if body is not None else None)

    return router


def submit_refresh(request: Request,
                   record: dict[str, Any],
                   end: str | None = None) -> RefreshJobStatus:
    """Check a refresh can run, and start it as a job.

    Shared with the deprecated `POST /data/coverage/{dataset}/sync`, which
    refreshes the store being served.
    """
    action, reason = refresh_plan(record)

    if action is None:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT,
                            detail=reason)

    served = active_data(request).store_id == record["id"]

    if action == "reread" and not served:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"'{record['name']}' is not being served, so there is "
                   f"nothing to re-read: it is read afresh whenever it is "
                   f"activated.")

    refreshing: set[str] = request.app.state.refreshing

    if record["id"] in refreshing:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"'{record['name']}' is already refreshing.")

    if served:
        claim_loading(request)

    refreshing.add(record["id"])
    jobs: JobRegistry = request.app.state.jobs
    job = jobs.submit(f"refresh:{record['id']}",
                      _refresh_job(request.app, record, action, served, end))

    return RefreshJobStatus(**job.snapshot())


def _refresh_job(app: FastAPI,
                 record: dict[str, Any],
                 action: RefreshAction,
                 served: bool,
                 end: str | None) -> Any:
    """The coroutine that refreshes a store and, if it is served, serves it."""

    async def run(report: ProgressReporter) -> dict[str, Any]:
        refreshing: set[str] = app.state.refreshing
        rows_added = None
        last = None

        try:
            if action == "extend":
                await run_synthetic(extend_command(Path(record["path"]), end),
                                    report)
                last = await asyncio.to_thread(_extended_to,
                                               Path(record["path"]))
            elif action == "download":
                rows_added, last = await _download(app, record, report)
        except BaseException:
            if served:
                app.state.active_data.loading = False
            raise
        finally:
            refreshing.discard(record["id"])

        if served:
            loaded = await serve_store(app, record, report)
            last = loaded.end
            jobs: JobRegistry = app.state.jobs
            jobs.publish_data_freshness(
                "market", {"store": record["id"],
                           "rows_added": rows_added,
                           "data_version": app.state.active_data.data_version})

        return RefreshResult(store_id=record["id"],
                             name=record["name"],
                             action=action,
                             end=last,
                             rows_added=rows_added,
                             served=served).model_dump()

    return run


async def _download(app: FastAPI,
                    record: dict[str, Any],
                    report: ProgressReporter) -> tuple[int, str | None]:
    """Download new prices from Yahoo Finance into a folder store."""
    downloader: Downloader | None = app.state.config.market_downloader
    seen: list[tuple[int, int, str]] = []

    def note(done: int,
             total: int,
             identifier: str) -> None:
        # Collected rather than awaited: the download loop is synchronous.
        seen.append((done, total, identifier))

    await report(0.05, "Downloading from Yahoo Finance")
    added, last = await asyncio.to_thread(download_into, Path(record["path"]),
                                          downloader or yfinance_downloader(),
                                          note)

    for done, total, identifier in seen[-1:]:
        await report(REFRESH_SHARE * done / total,
                     f"Fetched {identifier} ({done}/{total})")

    return added, last


def download_into(path: Path,
                  downloader: Downloader,
                  on_progress: Callable[[int, int, str], None] | None = None
                  ) -> tuple[int, str | None]:
    """Download prices after a folder store's last date, and save them into it.

    The instruments are the ones its reference data lists, or, without
    reference data, every identifier it prices.

    Returns:
        tuple: Market rows added, and the last date the store now holds.
        Nothing is written when no rows were added.
    """
    fetcher = data_store.load(path)
    source = data_store.read_manifest(path).source
    _, last = fetcher.date_range

    instruments = (fetcher.reference.identifiers
                   if fetcher.reference is not None else fetcher.identifiers)
    start = (last + timedelta(days=1)).date().isoformat() if last else None

    result = ingest_market_data(list(instruments), downloader,
                                start=start,
                                end=date.today().isoformat(),
                                on_progress=on_progress)
    added = fetcher.merge_market_data(result.market)

    if added:
        data_store.save(fetcher, path, source=source)

    logger.info("Downloaded %d row(s) into %s (%d of %d instrument(s)).",
                added, path, len(result.fetched), len(instruments))

    _, now = fetcher.date_range

    return added, f"{now:%Y-%m-%d}" if now is not None else None


def _extended_to(path: Path) -> str | None:
    """The last session an extended synthetic store holds."""
    settings, _ = synthetic_state.load(path)

    return settings.extensions[-1]["to"] if settings.extensions else None
