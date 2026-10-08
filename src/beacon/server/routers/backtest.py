# src/beacon/server/routers/backtest.py
"""
The backtest's catalogue: `GET /backtest/options`.

Backtests themselves are submitted against an index, at
`POST /beacon/{index_id}/backtest`; this lists what such a request may carry.
"""
# BN-272, phase 9 of decisions/0006.
from ..._optional import require
from ..backtest_options import backtest_options
from ..schemas import BacktestOptions

require("fastapi", "The Beacon API server")

from fastapi import APIRouter  # noqa: E402


def build_backtest_router() -> APIRouter:
    """Build the /backtest router.

    Returns:
        APIRouter: Router carrying the backtest options.
    """
    router = APIRouter(prefix="/backtest", tags=["backtest"])

    @router.get("/options", response_model=BacktestOptions)
    def options() -> BacktestOptions:
        """Everything a backtest request may carry, from the engine.

        Each family's types with their fields (the shape
        `/indices/rule-types` uses), the vehicle presets with their resolved
        settings, the vehicle, market and implementation settings described
        field by field, the modelling assumptions with the engine's defaults,
        and the closed choices. Reads no data, so it answers on a server with
        none loaded.
        """
        return backtest_options()

    return router
