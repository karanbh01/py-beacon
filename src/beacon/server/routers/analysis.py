# src/beacon/server/routers/analysis.py
"""
Reading a run's analysis: the catalogue of what an analysis can be asked for,
a strategy's latest run's analytics block, and its large tables, by column in
JSON or as Arrow.
"""
# BN-398/BN-400.
from pathlib import Path

from fastapi import APIRouter, Request, Response, status

from ...analysis.data import RunData
from ...exceptions import DataNotFoundError
from ..active_data import current_data
from ..analysis import (
    ARROW,
    TABLE_COLUMNS,
    arrow,
    build_request,
    build_rerun_job,
    catalogue,
    columnar,
    delete_saved,
    read_analytics,
    read_saved,
    saved_analyses,
)
from ..analysis_schemas import (
    COMPUTED,
    NOT_COMPUTED,
    AnalysisCatalogue,
    AnalyticsBlock,
    AnalyticsRerun,
    AnalyticsView,
    ColumnarTable,
    SavedAnalyses,
    SavedAnalysis,
    SavedAnalysisView,
)
from ..jobs import JobRegistry
from ..schemas import AnalyticsJobStatus, Identifier
from ..store import DocumentStore
from .strategy_runs import RECORD_SOURCE, _strategy

NOT_COMPUTED_REASON = ("The run was made without an analysis, or before py-beacon 0.10.0; "
                       "rerun its analytics to have one.")


def build_analysis_router() -> APIRouter:
    """Build the router for reading a run's analysis."""
    router = APIRouter(tags=["analysis"])

    @router.get("/analysis/options", response_model=AnalysisCatalogue)
    def options() -> AnalysisCatalogue:
        """Everything an analysis can be asked for: each section of settings
        with its defaults, every statistic, every report at both levels,
        labels for figure names, the large tables a run gives, and every
        reason a statistic can be unavailable."""
        return AnalysisCatalogue.model_validate(catalogue())

    @router.get("/strategies/{strategy_id}/analytics", response_model=AnalyticsView)
    def analytics(request: Request,
                  strategy_id: Identifier) -> AnalyticsView:
        """The analysis of the strategy's latest run, the block its backtest
        job answered with; ``not_computed`` for a run made without one.

        Raises:
            DataNotFoundError: Until the strategy has been backtested.
        """
        folder = _run_folder(request, strategy_id)
        block = read_analytics(folder)

        if block is None:
            return AnalyticsView(status=NOT_COMPUTED, analytics=None, reason=NOT_COMPUTED_REASON)

        return AnalyticsView(status=COMPUTED, analytics=AnalyticsBlock.model_validate(block))

    @router.get("/strategies/{strategy_id}/tables/{name}", response_model=ColumnarTable,
                responses={200: {"content": {ARROW: {}},
                                 "description": "The whole table: by column in JSON, or as an "
                                                "Arrow IPC stream when asked for with "
                                                f"`Accept: {ARROW}`."}})
    def table(request: Request,
              strategy_id: Identifier,
              name: str) -> Response | ColumnarTable:
        """One of the strategy's latest run's large tables, whole: by
        column, repeated text as its distinct values and a code a row; or
        as Arrow, with ``Accept: application/vnd.apache.arrow.stream``.

        Raises:
            DataNotFoundError: Until the strategy has been backtested, or for
                a table the run does not give.
        """
        if name not in TABLE_COLUMNS:
            raise DataNotFoundError(f"the run's {name} table",
                                    source=f"a run gives {', '.join(TABLE_COLUMNS)}")

        run = RunData.from_record(_run_folder(request, strategy_id))

        if name not in run.tables:
            raise DataNotFoundError(f"the run's {name} table", source=RECORD_SOURCE.format(
                strategy_id=strategy_id))

        found = run.tables[name]

        if ARROW in request.headers.get("accept", ""):
            return Response(content=arrow(name, found), media_type=ARROW)

        return ColumnarTable.model_validate(columnar(name, found))

    # async, as the backtest's is: the registry schedules the job on the
    # running event loop.
    @router.post("/strategies/{strategy_id}/analytics", response_model=AnalyticsJobStatus,
                 status_code=status.HTTP_202_ACCEPTED)
    async def rerun_analytics(request: Request,
                              strategy_id: Identifier,
                              body: AnalyticsRerun | None = None) -> AnalyticsJobStatus:
        """Give an analysis of the strategy's latest run, as a job: with the
        choices its own analysis was made with, the run's analysis gains
        what is asked for and is kept, and only that is worked out; with
        other choices, everything asked for is, kept only under `save`. The
        job's result is the whole analysis.

        Raises:
            DataNotFoundError: Until the strategy has been backtested.
        """
        asked = body or AnalyticsRerun()
        folder = _run_folder(request, strategy_id)
        built = build_request(asked.analysis)
        records: DocumentStore = request.app.state.backtest_record_store
        registry: JobRegistry = request.app.state.jobs
        job = registry.submit(f"analytics:{strategy_id}",
                              build_rerun_job(strategy_id, folder, records.read(strategy_id), built,
                                              current_data(request), asked.save))

        return AnalyticsJobStatus(**job.snapshot())

    @router.get("/strategies/{strategy_id}/analyses", response_model=SavedAnalyses)
    def analyses(request: Request,
                 strategy_id: Identifier) -> SavedAnalyses:
        """Every analysis kept by name with the strategy's latest run, newest
        first; a new backtest of the strategy leaves none."""
        return SavedAnalyses(analyses=[SavedAnalysis(**entry) for entry in
                                       saved_analyses(_run_folder(request, strategy_id))])

    @router.get("/strategies/{strategy_id}/analyses/{name}", response_model=SavedAnalysisView)
    def saved(request: Request,
              strategy_id: Identifier,
              name: Identifier) -> SavedAnalysisView:
        """An analysis kept by name with the strategy's latest run.

        Raises:
            DataNotFoundError: If none is kept under the name.
        """
        return SavedAnalysisView.model_validate(read_saved(_run_folder(request, strategy_id),
                                                           name))

    @router.delete("/strategies/{strategy_id}/analyses/{name}",
                   status_code=status.HTTP_204_NO_CONTENT)
    def delete(request: Request,
               strategy_id: Identifier,
               name: Identifier) -> None:
        """Remove an analysis kept by name.

        Raises:
            DataNotFoundError: If none is kept under the name.
        """
        delete_saved(_run_folder(request, strategy_id), name)

    return router


def _run_folder(request: Request,
                strategy_id: str) -> Path:
    """Where the strategy's latest run keeps its tables, or not-found."""
    _strategy(request, strategy_id)
    records: DocumentStore = request.app.state.backtest_record_store

    if not records.exists(strategy_id):
        raise DataNotFoundError(f"a completed backtest of strategy '{strategy_id}'",
                                source=RECORD_SOURCE.format(strategy_id=strategy_id))

    return records.panels_directory(strategy_id)
