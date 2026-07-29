# src/beacon/server/routers/data.py
"""
Data router: prices and reference data.

Two endpoints from BN-65 are deliberately absent.

`/data/fundamentals` is dropped rather than deferred: fundamentals will be
served by a general *features* endpoint covering any per-instrument datapoint
that is neither reference data nor a corporate action and that can drive a
backtest or an index rule. That endpoint is not designed yet.

`/data/corporate-actions` is blocked. `IndexCalculator` adjusts a divisor for
an action it is handed, but nothing in the data layer stores or serves a
history, so there is no series to aggregate into TTM figures.
"""
from typing import Annotated

import pandas as pd

from ..._optional import require
from ...data.fetcher import DataFetcher
from ...exceptions import ConfigurationError, DataNotFoundError
from ..config import ServerConfig
from ..schemas import PricesResponse, ReferenceResponse

require("fastapi", "The Beacon API server")

from fastapi import APIRouter, Query, Request  # noqa: E402

# Resolutions this router can honestly serve. The stored data is the native
# frequency; the other two are derived by taking each period's last
# observation, which is the standard convention for a price series.
RESAMPLE_RULES = {"weekly": "W", "monthly": "ME"}
NATIVE_INTERVAL = "native"
SUPPORTED_INTERVALS = (NATIVE_INTERVAL, *RESAMPLE_RULES)

# Query parameters declared as Annotated aliases rather than as call defaults:
# FastAPI's `param = Query(...)` form puts a function call in a default, which
# is the pattern B008 exists to catch. Annotated is also the form FastAPI now
# recommends.
StartQuery = Annotated[str | None, Query(description="Inclusive start date, YYYY-MM-DD.")]
EndQuery = Annotated[str | None, Query(description="Inclusive end date, YYYY-MM-DD.")]
IntervalQuery = Annotated[str, Query(description="native, weekly or monthly.")]
ColumnsQuery = Annotated[
    list[str] | None,
    Query(description="Market-data columns to return; all by default.")]
AsOfQuery = Annotated[
    str | None,
    Query(description="Point-in-time date, YYYY-MM-DD. Returns only rows valid then.")]


def _data_fetcher(request: Request) -> DataFetcher:
    """Return the process's data source, or fail with a mapped error.

    Args:
        request: The incoming request.

    Returns:
        DataFetcher: The configured source.

    Raises:
        ConfigurationError: When the process was started without one. This
            maps to 500 rather than 404 — the request was fine, the server is
            not configured to answer it.
    """
    # app.state is untyped, so pin the config back to its real type before
    # reading through it.
    config: ServerConfig = request.app.state.config

    fetcher = config.data_fetcher
    if fetcher is None:
        raise ConfigurationError(
            "data_source",
            "This server was started without a data source, so data endpoints "
            "cannot be served. Restart it with one configured.")

    return fetcher


def _resample(frame: pd.DataFrame,
              interval: str) -> pd.DataFrame:
    """Reduce a date-indexed frame to a coarser interval.

    Args:
        frame: Date-indexed market data.
        interval: One of SUPPORTED_INTERVALS.

    Returns:
        pd.DataFrame: The frame at the requested interval. Returned unchanged
        for the native interval.
    """
    if interval == NATIVE_INTERVAL:
        return frame

    return frame.resample(RESAMPLE_RULES[interval]).last().dropna(how="all")


def build_data_router() -> APIRouter:
    """Build the /data router.

    Returns:
        APIRouter: Router carrying the prices and reference endpoints.
    """
    router = APIRouter(prefix="/data", tags=["data"])

    @router.get("/prices/{identifier}", response_model=PricesResponse)
    def prices(request: Request,
               identifier: str,
               start: StartQuery = None,
               end: EndQuery = None,
               interval: IntervalQuery = NATIVE_INTERVAL,
               columns: ColumnsQuery = None) -> PricesResponse:
        # No `adjusted` parameter: the library holds no price-adjustment
        # logic, so accepting one and returning the stored series regardless
        # would misrepresent the response rather than merely limit it.
        if interval not in SUPPORTED_INTERVALS:
            raise DataNotFoundError(
                f"interval '{interval}'",
                source=f"supported intervals are {', '.join(SUPPORTED_INTERVALS)}")

        frame = _data_fetcher(request).fetch_market_data(identifier, start, end, columns)

        if frame.empty:
            raise DataNotFoundError(f"market data for '{identifier}'",
                                    source="MarketData")

        return PricesResponse.from_frame(identifier, interval, _resample(frame, interval))

    @router.get("/reference/{identifier}", response_model=ReferenceResponse)
    def reference(request: Request,
                  identifier: str,
                  date: AsOfQuery = None) -> ReferenceResponse:
        # GICS, profile and universe memberships are in this endpoint's brief
        # but not in the data model, so nothing here invents them. Whatever
        # columns the loaded reference data carries are returned as-is.
        frame = _data_fetcher(request).fetch_reference_data(identifier, date)

        if frame.empty:
            raise DataNotFoundError(f"reference data for '{identifier}'",
                                    source="ReferenceData")

        return ReferenceResponse.from_row(identifier, frame.iloc[0])

    return router
