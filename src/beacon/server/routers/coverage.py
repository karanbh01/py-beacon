# src/beacon/server/routers/coverage.py
"""
Data-coverage reporting.

The issue behind this router asks for coverage that "reflects real cache
ages". There is no cache: `DataFetcher` reads from in-memory `MarketData` and
`ReferenceData` that were loaded once at startup. Rather than invent an age,
coverage reports what is genuinely knowable — which datasets are loaded, how
many identifiers they hold, and the span of dates they cover — and leaves
`cache_age` null with that stated in the schema.
"""
from ..._optional import require
from ...data.fetcher import DataFetcher
from ..config import ServerConfig
from ..schemas import CoverageResponse, DatasetCoverage

require("fastapi", "The Beacon API server")

from fastapi import APIRouter, HTTPException, Request, status  # noqa: E402

MARKET = "market"
REFERENCE = "reference"
DATASETS = (MARKET, REFERENCE)


def _market_coverage(fetcher: DataFetcher | None) -> DatasetCoverage:
    """Describe the market dataset."""
    if fetcher is None:
        return DatasetCoverage(dataset=MARKET, configured=False, identifiers=0)

    identifiers = fetcher.identifiers
    if not identifiers:
        return DatasetCoverage(dataset=MARKET, configured=True, identifiers=0)

    start, end = fetcher.date_range

    return DatasetCoverage(dataset=MARKET,
                           configured=True,
                           identifiers=len(identifiers),
                           start=start.isoformat(),
                           end=end.isoformat())


def _reference_coverage(fetcher: DataFetcher | None) -> DatasetCoverage:
    """Describe the reference dataset.

    Reference data has validity windows rather than a single date axis, so no
    start/end is reported for it.
    """
    if fetcher is None or fetcher.reference_identifiers is None:
        return DatasetCoverage(dataset=REFERENCE, configured=False, identifiers=0)

    return DatasetCoverage(dataset=REFERENCE,
                           configured=True,
                           identifiers=len(fetcher.reference_identifiers))


def build_coverage_router() -> APIRouter:
    """Build the /data/coverage router.

    Returns:
        APIRouter: Router carrying coverage reporting and the sync endpoint.
    """
    router = APIRouter(prefix="/data/coverage", tags=["coverage"])

    @router.get("", response_model=CoverageResponse)
    def coverage(request: Request) -> CoverageResponse:
        config: ServerConfig = request.app.state.config
        fetcher = config.data_fetcher

        return CoverageResponse(datasets=[_market_coverage(fetcher),
                                          _reference_coverage(fetcher)])

    @router.post("/{dataset}/sync")
    def sync(request: Request,
             dataset: str) -> None:
        # Deliberately not a stub that pretends to work. There is no ingestion
        # path in the library at all — the `data` extra (yfinance) is declared
        # but unused, and data arrives by being loaded into MarketData at
        # startup. A synchronous no-op returning 200 would tell the client the
        # sync succeeded. 501 states the truth and gives the client a stable
        # contract to code against; it becomes a job once BN-69 lands and an
        # ingestion path exists.
        if dataset not in DATASETS:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Unknown dataset '{dataset}'. Known: {', '.join(DATASETS)}.")

        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail=f"Syncing '{dataset}' is not implemented: this build has no "
                   "data ingestion path. Data is loaded into memory at startup.")

    return router
