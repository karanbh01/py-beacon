# src/beacon/server/routers/analysis.py
"""
Reading a run's analysis: the catalogue of what an analysis can be asked for,
a strategy's latest run's analytics block, and its large tables, by column in
JSON or as Arrow.
"""
# BN-398/BN-400.
from pathlib import Path

from fastapi import APIRouter, Request, Response

from ...analysis.data import RunData
from ...exceptions import DataNotFoundError
from ..analysis import (
    ARROW,
    TABLE_COLUMNS,
    arrow,
    catalogue,
    columnar,
    read_analytics,
)
from ..analysis_schemas import (
    COMPUTED,
    NOT_COMPUTED,
    AnalysisCatalogue,
    AnalyticsBlock,
    AnalyticsView,
    ColumnarTable,
)
from ..schemas import Identifier
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
