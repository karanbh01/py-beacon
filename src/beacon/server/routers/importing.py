# src/beacon/server/routers/importing.py
"""Load a user's own data from CSV files or an Excel workbook into a new store.

The engine runs on the user's own machine, so it takes the files by path, the
way a data folder is registered: nothing is copied through the request, and
large files cost nothing extra to send.

Every row is checked before anything is saved. If any is wrong, the answer is
422 INVALID_RULE with one finding per problem, each naming its sheet, row and
column. Otherwise the data is saved as a new store the engine owns (managed),
and loaded straight away unless `activate` is false.
"""
import asyncio
from typing import Annotated, Literal

from ..._optional import require
from ...data import importing
from ...data import store as data_store
from ..active_data import active_data
from ..data_stores import StoreRegistry, describe
from ..jobs import JobRegistry
from ..schemas import ErrorEnvelope, LoadJobStatus
from ..store_schemas import ImportRequest, ImportResult
from .stores import load_store_job, managed_root

require("fastapi", "The Beacon API server")

from fastapi import APIRouter, Query, Request, Response, status  # noqa: E402

TEMPLATE_TYPES = {
    "xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    "csv": "application/zip",
}


def build_importing_router() -> APIRouter:
    """Build the /data/import router."""
    router = APIRouter(prefix="/data/import", tags=["data stores"])

    # async, because it may submit a load job; the reading and checking run
    # in a worker thread so a large file does not stall the engine.
    @router.post("",
                 response_model=ImportResult,
                 status_code=status.HTTP_201_CREATED,
                 responses={422: {"model": ErrorEnvelope,
                                  "description": "The files cannot be read, "
                                                 "or rows have problems: one "
                                                 "finding per problem in "
                                                 "detail.findings."}})
    async def import_files(request: Request,
                           body: ImportRequest) -> ImportResult:
        fetcher = await asyncio.to_thread(importing.load_files, body.paths)

        registry: StoreRegistry = request.app.state.data_stores
        store_id = registry.new_id(body.name)
        folder = managed_root(request.app) / store_id

        await asyncio.to_thread(data_store.save, fetcher, folder,
                                data_store.SOURCE_IMPORTED)
        record = registry.create(body.name, folder, managed=True,
                                 store_id=store_id)

        load_job = None
        holder = active_data(request)

        # Loaded only when no other load is running; otherwise it is saved
        # and can be activated later, rather than racing that load.
        if body.activate and not holder.loading:
            holder.loading = True
            jobs: JobRegistry = request.app.state.jobs
            job = jobs.submit(f"load:{store_id}",
                              load_store_job(request.app, record))
            load_job = LoadJobStatus(**job.snapshot())

        return ImportResult(store=describe(record, holder.store_id),
                            load_job=load_job)

    @router.get("/template",
                response_class=Response,
                responses={200: {"description": "A blank template with one "
                                                "example row per sheet.",
                                 "content": {kind: {} for kind
                                             in TEMPLATE_TYPES.values()}}})
    def template(format: Annotated[
            Literal["xlsx", "csv"],
            Query(description="'xlsx' for an Excel workbook, 'csv' for a zip "
                              "of CSV files.")] = "xlsx") -> Response:
        return Response(
            content=importing.template(format),
            media_type=TEMPLATE_TYPES[format],
            headers={"Content-Disposition":
                     f'attachment; filename="beacon-import-template.'
                     f'{"xlsx" if format == "xlsx" else "zip"}"'})

    return router
