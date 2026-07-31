# src/beacon/server/routers/derivatives.py
"""
Derivatives pricing endpoints.

Stateless by contract. The two pricing endpoints read nothing and write
nothing: a request carries every input, and the response is a pure function of
it. That is what makes them safe to call from a form on every keystroke, and a
test asserts the storage directory is untouched afterwards rather than trusting
the claim.

The term-structure and roll reads resolve a spot price from the data source.
They still write nothing.
"""
from typing import Annotated

from ..._optional import require
from ...data.fetcher import DataFetcher
from ...exceptions import ConfigurationError
from ..derivatives import (
    build_roll,
    build_term_structure,
    price_futures,
    price_trs,
)
from ..schemas import (
    FuturesPriceRequest,
    FuturesPriceResponse,
    RollResponse,
    TermStructureResponse,
    TrsPriceRequest,
    TrsPriceResponse,
)

require("fastapi", "The Beacon API server")

from fastapi import APIRouter, Query, Request  # noqa: E402

ExpiriesQuery = Annotated[
    list[str], Query(description="Contract expiries, YYYY-MM-DD.")]
AsOfQuery = Annotated[
    str | None, Query(description="Valuation date, YYYY-MM-DD. Defaults to the "
                                  "latest price held.")]
RateQuery = Annotated[
    float, Query(description="Continuously compounded financing rate.")]
YieldQuery = Annotated[
    float, Query(description="Continuous dividend yield.")]
ExpiryQuery = Annotated[str, Query(description="Contract expiry, YYYY-MM-DD.")]


def _data_fetcher(request: Request) -> DataFetcher:
    """The process's data source, or a mapped error."""
    fetcher = request.app.state.config.data_fetcher
    if fetcher is None:
        raise ConfigurationError(
            "data_source",
            "This server was started without a data source, so an index "
            "cannot be priced. Restart it with one configured.")

    return fetcher  # type: ignore[no-any-return]


def build_derivatives_router() -> APIRouter:
    """Build the /derivatives router.

    Returns:
        APIRouter: Router carrying the two pricing endpoints and the two
        index-level reads.
    """
    router = APIRouter(prefix="/derivatives", tags=["derivatives"])

    @router.post("/futures/price", response_model=FuturesPriceResponse)
    def futures_price(body: FuturesPriceRequest) -> FuturesPriceResponse:
        # No Request parameter: this endpoint has nothing to read from the
        # application at all, which is the clearest possible statement that it
        # is stateless.
        return price_futures(body)

    @router.post("/trs/price", response_model=TrsPriceResponse)
    def trs_price(body: TrsPriceRequest) -> TrsPriceResponse:
        return price_trs(body)

    @router.get("/{index_id}/term-structure", response_model=TermStructureResponse)
    def term_structure(request: Request,
                       index_id: str,
                       expiries: ExpiriesQuery,
                       as_of: AsOfQuery = None,
                       risk_free_rate: RateQuery = 0.0,
                       dividend_yield: YieldQuery = 0.0) -> TermStructureResponse:
        return build_term_structure(index_id, _data_fetcher(request),
                                    list(expiries), as_of, risk_free_rate,
                                    dividend_yield)

    @router.get("/{index_id}/roll", response_model=RollResponse)
    def roll(request: Request,
             index_id: str,
             front_expiry: ExpiryQuery,
             back_expiry: ExpiryQuery,
             as_of: AsOfQuery = None,
             risk_free_rate: RateQuery = 0.0,
             dividend_yield: YieldQuery = 0.0) -> RollResponse:
        return build_roll(index_id, _data_fetcher(request), front_expiry,
                          back_expiry, as_of, risk_free_rate, dividend_yield)

    return router
