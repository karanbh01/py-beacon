# src/beacon/server/routers/reports.py
"""
Report templates and rendering.

Templates are stored documents. Rendering is a job, because the result is
*bytes* — and bytes do not belong in a JSON job payload, so the render writes a
file and the job result carries its id for `GET /reports/renders/{id}` to
stream.

Two kinds of template. A stored one is rendered exactly as saved, because that
is what "I designed this page" means. A built-in one — `FACTSHEET-A4` — is
generated from an index's latest run. See `beacon.server.reports` for why there
is no templating language in between.
"""
from pathlib import Path
from typing import Any

from ..._optional import require
from ...exceptions import DataNotFoundError
from ...report.blocks import ReportTemplate
from ..jobs import JobRegistry
from ..reports import (
    BUILT_IN,
    build_factsheet,
    build_render_job,
    ensure_renderable,
    is_built_in,
    new_render_id,
    render_path,
)
from ..schemas import (
    JobStatus,
    RenderRequest,
    ReportTemplateCollection,
    ReportTemplateDocument,
)
from ..store import DocumentStore

require("fastapi", "The Beacon API server")

from fastapi import APIRouter, Request, status  # noqa: E402
from fastapi.responses import FileResponse  # noqa: E402

PDF_MEDIA_TYPE = "application/pdf"


def _store(request: Request) -> DocumentStore:
    """The template store."""
    store: DocumentStore = request.app.state.template_store
    return store


def _render_directory(request: Request) -> Path:
    """Where rendered documents are written."""
    directory: Path = request.app.state.render_directory
    return directory


def _stored_template(request: Request,
                     template_id: str) -> ReportTemplate:
    """Load a stored template, or fail with a mapped error."""
    document = _store(request).read(template_id)
    if document is None:
        raise DataNotFoundError(f"report template '{template_id}'",
                                source="DocumentStore")

    return ReportTemplate.from_dict(document)


def _resolve(request: Request,
             body: RenderRequest) -> ReportTemplate:
    """The template to render: generated for a built-in, stored otherwise."""
    if not is_built_in(body.template_id):
        return _stored_template(request, body.template_id)

    if body.index_id is None:
        raise DataNotFoundError(
            f"an index to build '{body.template_id}' from",
            source="a built-in template is generated from a run, so it needs "
                   "an `index_id`")

    registry: JobRegistry = request.app.state.jobs
    run = registry.latest_result(f"backtest:{body.index_id}")
    if run is None:
        raise DataNotFoundError(
            f"a completed backtest for index '{body.index_id}'",
            source="run POST /beacon/{index_id}/backtest first")

    # The display name comes from the index definition, not the template
    # store. Falling back to the id keeps a factsheet renderable for an index
    # that was backtested and then deleted.
    index_store: DocumentStore = request.app.state.index_store
    definition = index_store.read(body.index_id)
    name = str(definition["name"]) if definition else body.index_id

    return build_factsheet(name, run)


def build_reports_router() -> APIRouter:
    """Build the /reports router.

    Returns:
        APIRouter: Router carrying template CRUD, the render job, and the
        rendered-document download.
    """
    router = APIRouter(prefix="/reports", tags=["reports"])

    @router.get("/templates", response_model=ReportTemplateCollection)
    def list_templates(request: Request) -> ReportTemplateCollection:
        return ReportTemplateCollection(
            templates=[ReportTemplateDocument.model_validate(_as_document(entry))
                       for entry in _store(request).read_all()
                       if "template_id" in entry],
            built_in=list(BUILT_IN))

    @router.get("/templates/{template_id}", response_model=ReportTemplateDocument)
    def get_template(request: Request,
                     template_id: str) -> ReportTemplateDocument:
        template = _stored_template(request, template_id)

        return ReportTemplateDocument.model_validate(template.to_dict())

    @router.put("/templates/{template_id}", response_model=ReportTemplateDocument)
    def put_template(request: Request,
                     template_id: str,
                     body: ReportTemplateDocument) -> ReportTemplateDocument:
        # The path wins over the body, so a document cannot be saved under one
        # id while claiming another.
        payload = {**body.model_dump(), "template_id": template_id}

        # Round-tripped through the block model on the way in, so a malformed
        # block is refused at save rather than at render — by which point the
        # person who wrote it has moved on.
        template = ReportTemplate.from_dict(payload)
        _store(request).write(template_id, template.to_dict())

        return ReportTemplateDocument.model_validate(template.to_dict())

    @router.delete("/templates/{template_id}",
                   status_code=status.HTTP_204_NO_CONTENT)
    def delete_template(request: Request,
                        template_id: str) -> None:
        if not _store(request).exists(template_id):
            raise DataNotFoundError(f"report template '{template_id}'",
                                    source="DocumentStore")

        _store(request).delete(template_id)

    # async, and it must stay that way: FastAPI runs a sync endpoint in a
    # worker thread, where there is no running event loop for the registry to
    # attach a task to.
    @router.post("/render",
                 response_model=JobStatus,
                 status_code=status.HTTP_202_ACCEPTED)
    async def submit_render(request: Request,
                            body: RenderRequest) -> JobStatus:
        # The template is resolved and checked before the job is submitted, so
        # an unknown id or a missing run fails immediately rather than becoming
        # a job that fails a moment later for the client to discover.
        template = _resolve(request, body)
        ensure_renderable(template)

        render_id = new_render_id()
        registry: JobRegistry = request.app.state.jobs
        job = registry.submit(
            f"render:{render_id}",
            build_render_job(render_id, body, template, _render_directory(request)))

        return JobStatus(**job.snapshot())

    @router.get("/renders/{render_id}")
    def download(request: Request,
                 render_id: str) -> FileResponse:
        path = render_path(_render_directory(request), render_id)
        if not path.exists():
            raise DataNotFoundError(
                f"a rendered document '{render_id}'",
                source="the render may still be in progress, or may have failed")

        return FileResponse(path,
                            media_type=PDF_MEDIA_TYPE,
                            filename=f"{render_id}.pdf")

    return router


def _as_document(entry: dict[str, Any]) -> dict[str, Any]:
    """A stored template with only the fields the wire model carries."""
    return {key: value for key, value in entry.items()
            if key in {"template_id", "name", "page", "blocks"}}
