# src/beacon/server/schemas.py
"""
Wire schemas for the Beacon API.

Library result objects are dataclasses holding pandas structures. They are
never exposed directly: their field names and types are internal and would
otherwise become an API contract by accident. Everything crossing the wire is
declared here, so OpenAPI describes it and a library refactor cannot silently
reshape a response.
"""
from typing import Annotated, Any

import pandas as pd
from pydantic import BaseModel, Field

from ..backtest.result import BacktestResult
from ..index.result import IndexResult
from .serialisation import dataframe_to_payload, series_to_payload

# A rate or proportion expressed as a fraction: 0.0523 is 5.23%. Kept as a
# bare float rather than an object because it is arithmetic, not a quantity
# with a unit — clients format it for display.
Pct = Annotated[float, Field(description="Fraction, not percent: 0.0523 means 5.23%.")]


class Money(BaseModel):
    """An amount with its denomination.

    A bare float would leave the currency implicit, which breaks as soon as a
    response mixes denominations.
    """
    value: float = Field(description="Amount, in the units of `currency`.")
    currency: str = Field(description="ISO 4217 code, e.g. 'USD'.", min_length=3,
                          max_length=3)


class SeriesPayload(BaseModel):
    """A pandas Series on the wire."""
    index: list[Any] = Field(description="Row labels; timestamps are ISO 8601 strings.")
    name: str | None = Field(default=None, description="Series name, if it has one.")
    data: list[Any] = Field(description="Values, aligned to `index`. NaN becomes null.")

    @classmethod
    def from_series(cls,
                    series: pd.Series) -> "SeriesPayload":
        """Build from a pandas Series."""
        return cls(**series_to_payload(series))


class TableFrame(BaseModel):
    """A pandas DataFrame on the wire.

    Row-oriented so column order is preserved and the payload stays compact.
    """
    index: list[Any] = Field(description="Row labels; timestamps are ISO 8601 strings.")
    columns: list[str] = Field(description="Column names, in order.")
    data: list[list[Any]] = Field(
        description="Rows, each aligned to `columns`. NaN becomes null.")

    @classmethod
    def from_dataframe(cls,
                       frame: pd.DataFrame) -> "TableFrame":
        """Build from a pandas DataFrame."""
        return cls(**dataframe_to_payload(frame))


class DataSourceStatus(BaseModel):
    """Whether this process has a data source, and how much it covers."""
    configured: bool = Field(description="True when a DataFetcher is attached.")
    identifiers: int = Field(description="Distinct identifiers in market data.")


class HealthResponse(BaseModel):
    """Response of `GET /health`."""
    status: str = Field(description="'ok' when the process is serving.")
    version: str = Field(description="Installed py-beacon version.")
    data_source: DataSourceStatus
    cache_age: float | None = Field(
        default=None,
        description="Seconds since the data cache was refreshed. Always null: "
                    "DataFetcher reads from memory and caches nothing today.")


class IndexResultSummary(BaseModel):
    """Serialised view of an `IndexResult`."""
    index_id: str
    index_levels: SeriesPayload
    divisor_history: SeriesPayload
    rebalance_dates: list[str] = Field(
        description="Dates carrying a constituent snapshot, ISO 8601.")
    constituent_snapshots: dict[str, list[str]] = Field(
        description="Rebalance date -> constituent identifiers.")
    weight_snapshots: dict[str, dict[str, float]] = Field(
        description="Rebalance date -> {identifier: weight}. Weights sum to 1.")

    @classmethod
    def from_result(cls,
                    result: IndexResult) -> "IndexResultSummary":
        """Build from a library `IndexResult`."""
        constituents = {
            date.isoformat(): members
            for date, members in result.constituent_snapshots.items()
        }
        weights = {
            date.isoformat(): dict(members)
            for date, members in result.weight_snapshots.items()
        }

        return cls(index_id=result.index_id,
                   index_levels=SeriesPayload.from_series(result.index_levels),
                   divisor_history=SeriesPayload.from_series(result.divisor_history),
                   rebalance_dates=sorted(constituents),
                   constituent_snapshots=constituents,
                   weight_snapshots=weights)


class BacktestMetrics(BaseModel):
    """Headline metrics from a backtest run.

    Tracking figures are null when the run had no target index to compare to.
    """
    total_return: Pct
    annualised_return: Pct
    volatility: Pct
    sharpe_ratio: float
    max_drawdown: Pct
    tracking_error: Pct | None = None
    tracking_difference: Pct | None = None


def _metric(summary: dict[str, float | None],
            key: str) -> float:
    """Read a core metric, which `BacktestResult.summary()` always populates.

    Its return type is `float | None` because the tracking figures are
    optional; the five headline metrics are not. Falls back to 0.0 rather
    than raising, so a summary shape change degrades a number instead of
    failing the request.
    """
    value = summary.get(key)

    return 0.0 if value is None else float(value)


class BacktestResultSummary(BaseModel):
    """Serialised view of a `BacktestResult`."""
    portfolio_id: str
    initial_capital: float
    portfolio_nav: SeriesPayload
    cash_history: SeriesPayload
    transactions: TableFrame
    metrics: BacktestMetrics

    @classmethod
    def from_result(cls,
                    result: BacktestResult) -> "BacktestResultSummary":
        """Build from a library `BacktestResult`."""
        rows = [
            {
                "date": transaction.transaction_date,
                "asset_id": transaction.asset_id,
                "type": transaction.transaction_type,
                "quantity": transaction.quantity,
                "price": transaction.price,
                "cost": transaction.transaction_cost,
            }
            for transaction in result.transactions
        ]
        frame = pd.DataFrame(
            rows,
            columns=["date", "asset_id", "type", "quantity", "price", "cost"])

        summary = result.summary()
        metrics = BacktestMetrics(
            total_return=_metric(summary, "total_return"),
            annualised_return=_metric(summary, "annualised_return"),
            volatility=_metric(summary, "volatility"),
            sharpe_ratio=_metric(summary, "sharpe_ratio"),
            max_drawdown=_metric(summary, "max_drawdown"),
            tracking_error=summary.get("tracking_error"),
            tracking_difference=summary.get("tracking_difference"))

        return cls(portfolio_id=result.portfolio_id,
                   initial_capital=result.initial_capital,
                   portfolio_nav=SeriesPayload.from_series(result.portfolio_nav),
                   cash_history=SeriesPayload.from_series(result.cash_history),
                   transactions=TableFrame.from_dataframe(frame),
                   metrics=metrics)


class PricesResponse(BaseModel):
    """Response of `GET /data/prices/{identifier}`."""
    identifier: str
    interval: str = Field(
        description="Resolution actually served: 'native', 'weekly' or 'monthly'.")
    prices: TableFrame = Field(description="Date-indexed market data.")

    @classmethod
    def from_frame(cls,
                   identifier: str,
                   interval: str,
                   frame: pd.DataFrame) -> "PricesResponse":
        """Build from a date-indexed market-data frame."""
        return cls(identifier=identifier,
                   interval=interval,
                   prices=TableFrame.from_dataframe(frame))


class ReferenceResponse(BaseModel):
    """Response of `GET /data/reference/{identifier}`.

    Fields are whatever columns the loaded reference data carries — the
    library does not impose a schema on it — so they are returned as a
    mapping rather than as named attributes.
    """
    identifier: str
    fields: dict[str, Any] = Field(
        description="Reference columns for this identifier, e.g. NAME, "
                    "CURRENCY, EXCHANGE. Timestamps are ISO 8601 strings.")

    @classmethod
    def from_row(cls,
                 identifier: str,
                 row: pd.Series) -> "ReferenceResponse":
        """Build from a single reference-data row."""
        payload = series_to_payload(row)
        fields = dict(zip(payload["index"], payload["data"], strict=True))

        return cls(identifier=identifier, fields=fields)


class Watchlist(BaseModel):
    """A named set of instrument identifiers."""
    id: str = Field(description="Stable identifier, used in the URL.",
                    min_length=1, max_length=64)
    name: str = Field(description="Display name.", min_length=1)
    identifiers: list[str] = Field(default_factory=list,
                                   description="Instrument identifiers, in user order.")


class WatchlistUpsert(BaseModel):
    """Body of `PUT /data/watchlists/{id}`.

    The id comes from the URL, so it is not repeated here — accepting it in
    both places invites the two to disagree.
    """
    name: str = Field(description="Display name.", min_length=1)
    identifiers: list[str] = Field(default_factory=list,
                                   description="Instrument identifiers, in user order.")


class WatchlistCollection(BaseModel):
    """Response of `GET /data/watchlists`."""
    watchlists: list[Watchlist]


class DatasetCoverage(BaseModel):
    """What the loaded data actually spans, for one dataset."""
    dataset: str = Field(description="'market' or 'reference'.")
    configured: bool = Field(description="Whether this dataset is loaded.")
    identifiers: int = Field(description="Distinct identifiers present.")
    start: str | None = Field(default=None,
                              description="Earliest date held, ISO 8601.")
    end: str | None = Field(default=None, description="Latest date held, ISO 8601.")
    cache_age: float | None = Field(
        default=None,
        description="Seconds since this dataset was refreshed. Always null: "
                    "data is loaded into memory once and never cached, so "
                    "there is no refresh time to measure.")


class CoverageResponse(BaseModel):
    """Response of `GET /data/coverage`."""
    datasets: list[DatasetCoverage]


class ErrorDetail(BaseModel):
    """The body of an error envelope."""
    code: str = Field(
        description="Stable machine-readable code; safe to branch on.")
    message: str = Field(description="Human-readable summary.")
    detail: dict[str, Any] | None = Field(
        default=None,
        description="Structured context, e.g. the offending field or rule.")


class ErrorEnvelope(BaseModel):
    """Every non-2xx response uses this shape."""
    error: ErrorDetail
