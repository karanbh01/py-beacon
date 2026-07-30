# src/beacon/server/schemas.py
"""
Wire schemas for the Beacon API.

Library result objects are dataclasses holding pandas structures. They are
never exposed directly: their field names and types are internal and would
otherwise become an API contract by accident. Everything crossing the wire is
declared here, so OpenAPI describes it and a library refactor cannot silently
reshape a response.
"""
from typing import Annotated, Any, Literal

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
        description="Seconds since the market data was last loaded or synced. "
                    "Null only when no data source is configured — there is "
                    "then nothing whose age could be reported.")


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


class CorporateAction(BaseModel):
    """One corporate action."""
    ex_date: str = Field(description="Ex-date, ISO 8601.")
    type: str = Field(description="Action type, e.g. DIVIDEND or SPLIT.")
    value: float = Field(
        description="Cash amount per share for cash actions; a share-count "
                    "multiplier for ratio actions. What it means depends on "
                    "the type, so the two are never summed together.")


class CorporateActionsResponse(BaseModel):
    """Response of `GET /data/corporate-actions/{identifier}`.

    Carries the raw history and the two aggregates that need the whole series
    to compute, so a client asking "what did this pay" does not have to
    reimplement the trailing window and get its boundary subtly wrong.
    """
    identifier: str
    actions: list[CorporateAction] = Field(default_factory=list)
    trailing_dividend: float = Field(
        default=0.0,
        description="Ordinary dividends per share over the twelve calendar "
                    "months ending at the as-of date.")
    trailing_dividend_yield: float | None = Field(
        default=None,
        description="Trailing dividend over the close on the as-of date. Null "
                    "when no price is available — a missing price is a reason "
                    "to say nothing rather than to guess.")
    cumulative_split_ratio: float = Field(
        default=1.0,
        description="Compounded share-count multiplier across the returned "
                    "window. 1.0 when there were no splits.")

    @classmethod
    def from_frame(cls,
                   identifier: str,
                   frame: pd.DataFrame,
                   trailing_dividend: float,
                   trailing_dividend_yield: float | None,
                   cumulative_split_ratio: float) -> "CorporateActionsResponse":
        """Build from a corporate-action history slice."""
        actions = [
            CorporateAction(ex_date=pd.Timestamp(row["EX_DATE"]).isoformat(),
                            type=str(row["TYPE"]),
                            value=float(row["VALUE"]))
            for _, row in frame.iterrows()
        ]

        return cls(identifier=identifier,
                   actions=actions,
                   trailing_dividend=trailing_dividend,
                   trailing_dividend_yield=trailing_dividend_yield,
                   cumulative_split_ratio=cumulative_split_ratio)


class SyncRequest(BaseModel):
    """Body of `POST /data/coverage/{dataset}/sync`."""
    identifiers: list[str] = Field(
        default_factory=list,
        description="What to fetch. Empty re-syncs everything already loaded, "
                    "which is the common case: refresh what I have.")
    start: str | None = Field(default=None,
                              description="Inclusive start date, YYYY-MM-DD.")
    end: str | None = Field(default=None,
                            description="Inclusive end date, YYYY-MM-DD.")


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


class RuleSpec(BaseModel):
    """One rule in the pipeline, addressable by its id."""
    id: str = Field(description="Client-assigned id, unique within the pipeline.",
                    min_length=1)
    type: str = Field(description="Rule class name, e.g. 'MarketCapRule'.",
                      min_length=1)
    params: dict[str, Any] = Field(default_factory=dict,
                                   description="Constructor arguments for the rule.")


class WeightingSpec(BaseModel):
    """The weighting group of the pipeline."""
    id: str = Field(default="weighting", description="Id used to address findings.")
    scheme: str = Field(description="Scheme class name, e.g. 'EqualWeighted'.",
                        min_length=1)
    params: dict[str, Any] = Field(default_factory=dict,
                                   description="Constructor arguments for the scheme.")
    max_weight: float | None = Field(
        default=None,
        description="Cap on any single constituent's weight, as a fraction "
                    "(0.1 is 10%). Applied after the scheme and iterated until "
                    "nothing breaches it. Null means uncapped.")


class TreatmentSpec(BaseModel):
    """The treatment group of the pipeline."""
    corporate_actions: str = Field(
        default="ADJUST_DIVISOR",
        description="How corporate actions affect the index. Only "
                    "ADJUST_DIVISOR is supported.")


class PipelineSpec(BaseModel):
    """The grouped rule pipeline: Selection, Weighting, Treatment."""
    selection: list[RuleSpec] = Field(default_factory=list)
    weighting: WeightingSpec
    treatment: TreatmentSpec = Field(default_factory=TreatmentSpec)


class UniverseRef(BaseModel):
    """Where an index's universe comes from.

    Either a reference to a stored universe or a literal list. `identifiers`
    is always populated on read, so consumers never have to resolve it.
    """
    universe_id: str | None = Field(
        default=None, description="Id of a stored universe, if referenced.")
    identifiers: list[str] = Field(default_factory=list,
                                   description="Resolved instrument identifiers.")


class IndexDocument(BaseModel):
    """A stored index definition."""
    id: str = Field(description="Stable identifier, used in the URL.",
                    min_length=1, max_length=64)
    name: str = Field(description="Display name.", min_length=1)
    base_date: str = Field(description="Base date, YYYY-MM-DD.")
    base_value: float = Field(description="Index level on the base date.")
    currency: str = Field(description="Index currency.", min_length=3, max_length=3)
    rebalancing_frequency: str = Field(
        description="MONTHLY, QUARTERLY, SEMI-ANNUAL or ANNUAL.")
    universe: UniverseRef
    pipeline: PipelineSpec
    description: str | None = None


class Finding(BaseModel):
    """One validation result, addressable to the rule that caused it."""
    path: str = Field(description="Dotted path to the offending field.")
    rule_id: str | None = Field(
        default=None, description="Id of the rule responsible, when there is one.")
    severity: str = Field(description="'error' blocks saving; 'warning' does not.")
    code: str = Field(description="Stable machine-readable code.")
    message: str = Field(description="Human-readable explanation.")


class ValidationReport(BaseModel):
    """Response of the validation endpoint, and of a rejected save."""
    valid: bool = Field(description="False when any finding has severity 'error'.")
    findings: list[Finding]


class IndexCollection(BaseModel):
    """Response of `GET /indices`."""
    indices: list[IndexDocument]


class SavedIndex(BaseModel):
    """Response of a successful save: the document plus any warnings."""
    index: IndexDocument
    findings: list[Finding] = Field(
        default_factory=list,
        description="Non-blocking warnings. Errors would have prevented the save.")


class Universe(BaseModel):
    """A named set of instrument identifiers."""
    id: str = Field(description="Stable identifier.", min_length=1, max_length=64)
    name: str = Field(description="Display name.", min_length=1)
    identifiers: list[str] = Field(default_factory=list)


class UniverseUpsert(BaseModel):
    """Body of `PUT /universes/{id}`."""
    name: str = Field(description="Display name.", min_length=1)
    identifiers: list[str] = Field(default_factory=list)


class UniverseCollection(BaseModel):
    """Response of `GET /universes`."""
    universes: list[Universe]


class UniverseMembers(BaseModel):
    """Response of `GET /universes/{id}/members`."""
    universe_id: str
    identifiers: list[str]


class PreviewRequest(BaseModel):
    """Body of `POST /indices/{id}/preview`."""
    as_of: str | None = Field(
        default=None,
        description="Date to evaluate the pipeline at, YYYY-MM-DD. Defaults "
                    "to the index's base date.")


class PreviewStep(BaseModel):
    """One rung of the derivation waterfall.

    There is one of these per selection rule, in pipeline order, plus a first
    entry for the universe itself so the funnel starts from a stated total.
    """
    position: int = Field(description="0 is the universe; rules follow in order.")
    rule_id: str | None = Field(
        default=None, description="Rule responsible, or null for the universe row.")
    rule_type: str | None = Field(default=None, description="Rule class name.")
    remaining: int = Field(description="Constituents surviving after this step.")
    excluded: list[str] = Field(
        default_factory=list,
        description="Identifiers this step removed. Empty for the universe row.")


class PreviewAsset(BaseModel):
    """Per-asset outcome of the derivation."""
    identifier: str
    included: bool = Field(description="Whether it reached the final index.")
    excluded_by: str | None = Field(
        default=None,
        description="Id of the first rule that excluded it. Null when included.")
    excluded_at: int | None = Field(
        default=None, description="Waterfall position where it dropped out.")
    weight: float | None = Field(
        default=None, description="Final weight as a fraction. Null when excluded.")
    uncapped_weight: float | None = Field(
        default=None,
        description="Weight before capping, when the cap bound this name.")
    capped: bool = Field(default=False,
                         description="Whether this name sits at the cap.")


class PreviewResponse(BaseModel):
    """Response of `POST /indices/{id}/preview`."""
    index_id: str
    as_of: str
    steps: list[PreviewStep]
    assets: list[PreviewAsset]
    weights: dict[str, float] = Field(
        description="Final weights as fractions, keyed by identifier.")
    total_weight: float = Field(
        description="Sum of the final weights; 1.0 for a non-empty index.")
    cap: float | None = Field(default=None,
                              description="Cap applied, as a fraction, if any.")
    cap_redistributed: float = Field(
        default=0.0, description="Weight moved off capped names onto the rest.")


BENCHMARK_INDEX = "index"
BENCHMARK_IDENTIFIER = "identifier"


class BenchmarkRef(BaseModel):
    """What to compare a backtest against.

    Distinct from the index being tracked. The tracked index measures
    replication accuracy; a benchmark measures relative performance against
    something the portfolio was never trying to replicate.
    """
    kind: Literal["index", "identifier"] = Field(
        description="'index' for a stored index definition, 'identifier' for a "
                    "market-data series.")
    id: str = Field(description="Index id, or market-data identifier.",
                    min_length=1)
    price_column: str = Field(
        default="CLOSE",
        description="Market-data column to read. Ignored when kind is 'index'.")


class RelativeMetricsPayload(BaseModel):
    """Performance against a benchmark, over their shared window."""
    reference: BenchmarkRef
    observations: int = Field(
        description="Aligned dates used, which may be fewer than either series "
                    "carried on its own.")
    start: str
    end: str
    total_return: Pct
    benchmark_return: Pct
    excess_return: Pct = Field(
        description="Portfolio minus benchmark; also the tracking difference.")
    tracking_error: Pct = Field(
        description="Annualised standard deviation of return differences.")
    correlation: float
    beta: float
    level: SeriesPayload = Field(
        description="The benchmark, rebased to 100 on the shared window.")


class BacktestRequest(BaseModel):
    """Body of `POST /beacon/{index_id}/backtest`."""
    start: str | None = Field(
        default=None,
        description="Start date, YYYY-MM-DD. Defaults to the index base date.")
    end: str | None = Field(default=None, description="End date, YYYY-MM-DD.")
    initial_capital: float = Field(default=1_000_000.0, gt=0)
    transaction_cost_bps: float = Field(
        default=0.0, ge=0,
        description="Cost per trade in basis points of notional.")
    benchmark: BenchmarkRef | None = Field(
        default=None,
        description="Optional external benchmark. The tracked index is always "
                    "reported separately; this adds a second comparison.")


class BacktestRunResult(BaseModel):
    """Result payload of a completed backtest job.

    Every series here derives from the same NAV, rebased to 100: `returns` is
    the level's percentage change, `drawdown` is the level against its running
    peak, and `annual_returns` compound back to the total. A client that
    recomputes any of them lands on these numbers exactly.
    """
    level: SeriesPayload = Field(description="Portfolio value, rebased to 100.")
    returns: SeriesPayload = Field(description="Period returns of `level`.")
    drawdown: SeriesPayload = Field(
        description="Level against its running peak; 0 at a new high.")
    annual_returns: dict[str, float] = Field(
        description="Calendar year -> return. Compounds to total_return.")
    benchmark_level: SeriesPayload = Field(
        description="The tracked index, rebased to 100 on the same axis.")
    metrics: BacktestMetrics
    benchmark: RelativeMetricsPayload | None = Field(
        default=None,
        description="Comparison against the requested external benchmark, if "
                    "one was given. Null otherwise; `metrics.tracking_error` "
                    "still reports replication accuracy against the tracked "
                    "index either way.")


class JobStatus(BaseModel):
    """State of one background job."""
    job_id: str
    kind: str = Field(description="What the job is, e.g. 'backtest'.")
    status: str = Field(
        description="pending, running, succeeded, failed or cancelled. The "
                    "last three are terminal.")
    progress: float = Field(description="Fraction complete, 0.0 to 1.0.")
    message: str = Field(default="", description="Latest progress message.")
    result: Any = Field(
        default=None,
        description="Present only once the job has succeeded; null otherwise.")
    error: str | None = Field(default=None,
                              description="Failure reason, when status is failed.")


class JobCollection(BaseModel):
    """Response of `GET /jobs`."""
    jobs: list[JobStatus]


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
        description="Seconds since this dataset was last loaded or synced. "
                    "Null when the dataset is not loaded at all, which is a "
                    "different statement from 'loaded and never refreshed'.")
    last_refreshed: str | None = Field(
        default=None,
        description="When this dataset was last loaded or synced, ISO 8601. "
                    "Carried alongside the age because an age is only "
                    "meaningful at the instant it was read, and a client "
                    "holding a response for a minute needs the timestamp.")


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
