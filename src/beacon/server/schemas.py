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


class ReportTemplateDocument(BaseModel):
    """A stored report template, as JSON.

    Blocks are kept as free-form mappings rather than a discriminated union so
    the wire shape stays exactly what `beacon.report.blocks` reads and writes.
    A second definition of the same thing here is a second definition to keep
    in step, and the block model already validates its own rows on the way in.
    """
    template_id: str = Field(min_length=1, max_length=64,
                             description="Stable identifier, used in the URL.")
    name: str = Field(min_length=1, description="Display name.")
    page: dict[str, Any] = Field(
        default_factory=dict,
        description="Page setup: size, orientation, margin.")
    blocks: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Content, drawn top to bottom. Each carries a `kind`.")


class ReportTemplateCollection(BaseModel):
    """Response of `GET /reports/templates`."""
    templates: list[ReportTemplateDocument]
    built_in: list[str] = Field(
        default_factory=list,
        description="Templates generated from a run rather than stored. These "
                    "can be rendered but not edited: they are code, not "
                    "documents.")


class RenderRequest(BaseModel):
    """Body of `POST /reports/render`."""
    template_id: str = Field(
        description="A stored template, or a built-in such as FACTSHEET-A4.")
    index_id: str | None = Field(
        default=None,
        description="Required for a built-in template, which is generated from "
                    "that index's latest completed backtest. Ignored for a "
                    "stored template, which is rendered exactly as saved.")


class RenderResult(BaseModel):
    """Result payload of a completed render job."""
    render_id: str = Field(
        description="Fetch the PDF from GET /reports/renders/{render_id}.")
    template_id: str
    index_id: str | None = None
    name: str
    blocks: int
    bytes: int = Field(description="Size of the rendered document.")
    rendered_at: str


class FuturesPriceRequest(BaseModel):
    """Body of `POST /derivatives/futures/price`.

    Stateless: every input the calculation needs is here, and nothing is read
    from or written to storage.
    """
    spot: float = Field(gt=0.0, description="Spot price of the underlying.")
    risk_free_rate: float = Field(
        default=0.0,
        description="Continuously compounded financing rate. Ignored when "
                    "`curve` is supplied.")
    curve: dict[str, float] | None = Field(
        default=None,
        description="Zero-rate pillars as {tenor_in_years: rate}. A flat curve "
                    "and a scalar rate give identical answers, so supplying "
                    "one changes nothing unless the curve has shape.")
    dividend_yield: float = Field(
        default=0.0, description="Continuous dividend yield.")
    borrow_cost: float = Field(
        default=0.0, description="Continuous borrow or financing spread.")
    dividends: list[tuple[float, float]] | None = Field(
        default=None,
        description="Discrete cash dividends as (years_to_ex, amount). When "
                    "present these are used instead of the continuous yield: "
                    "the two are different models of the same thing and "
                    "applying both would double-count.")
    valuation_date: str | None = Field(default=None, description="YYYY-MM-DD.")
    expiry: str | None = Field(default=None, description="YYYY-MM-DD.")
    time_to_expiry: float | None = Field(
        default=None,
        description="Years to expiry. Dates win when both are given, being the "
                    "less ambiguous statement.")
    contract_multiplier: float = Field(
        default=1.0, description="Index points per contract.")
    contracts: float = Field(default=1.0, description="Number of contracts.")
    market_price: float | None = Field(
        default=None,
        description="Quoted price, for the basis and implied repo. Both are "
                    "null without one rather than computed against the "
                    "theoretical value, which would make them identically "
                    "zero.")
    grid_tenors: list[float] | None = Field(
        default=None, description="Rows of the sensitivity grid, in years.")
    grid_rates: list[float] | None = Field(
        default=None, description="Columns of the sensitivity grid.")


class CarryDecomposition(BaseModel):
    """Carry split into the pieces a person can reason about.

    Each part is the price effect of one rate acting alone. They do not sum to
    the total exactly, because carry compounds rather than adds; the residual
    is reported rather than spread across the parts, which would make each of
    them slightly wrong in order to hide that the split is approximate.
    """
    total: float = Field(description="Fair value minus spot.")
    financing: float
    dividend: float = Field(description="Negative: dividends reduce the forward.")
    borrow: float
    residual: float = Field(
        description="Total minus the three parts — the compounding the "
                    "decomposition cannot attribute.")


class FuturesPriceResponse(BaseModel):
    """Response of `POST /derivatives/futures/price`."""
    fair_value: float
    time_to_expiry: float
    financing_rate: float = Field(description="Rate used, read off the curve.")
    carry: CarryDecomposition
    contract_value: float = Field(
        description="Fair value times multiplier times contracts.")
    market_price: float | None = None
    basis: float | None = Field(
        default=None, description="Market minus theoretical, when quoted.")
    implied_repo: float | None = Field(
        default=None,
        description="Financing rate the quoted price implies, when quoted.")
    sensitivity: TableFrame = Field(
        description="Fair value across a tenor x rate grid, centred on this "
                    "contract.")


class TrsPriceRequest(BaseModel):
    """Body of `POST /derivatives/trs/price`."""
    trade_id: str = Field(default="TRS", description="Identifier for the trade.")
    underlying_id: str = Field(default="INDEX")
    currency: str = Field(default="USD")
    start_date: str
    end_date: str
    notional: float = Field(gt=0.0)
    spread_bps: float = Field(default=0.0)
    reference_rate: str = Field(default="SOFR", description="Name of the index.")
    reference_rate_value: float = Field(
        default=0.0, description="Its current fixing, as a decimal.")
    payment_frequency: str = Field(default="QUARTERLY")
    reset_type: str = Field(
        default="UNFUNDED",
        description="UNFUNDED accrues reference + spread; FUNDED accrues only "
                    "the spread, and therefore has no rate sensitivity at all.")
    valuation_date: str
    last_reset_date: str | None = Field(
        default=None, description="Defaults to the start date.")
    spot: float = Field(gt=0.0, description="Underlying level today.")
    initial_price: float = Field(
        gt=0.0, description="Level at inception or last reset.")
    dividend_yield: float = Field(default=0.0)
    time_to_expiry: float | None = Field(
        default=None, description="Needed for the breakeven table.")
    futures_prices: list[float] | None = Field(
        default=None,
        description="Prices to compute breakeven spreads against.")
    curve: dict[str, float] | None = Field(
        default=None,
        description="Zero-rate pillars for projecting future periods. The "
                    "current period always accrues at the rate already fixed "
                    "at its reset.")


class TrsAccrual(BaseModel):
    """One financing period."""
    start: str
    end: str
    days: int
    rate: float = Field(
        description="Reference rate for the period: the fixing for the current "
                    "one, the curve's forward for later ones.")
    accrual_fraction: float = Field(description="ACT/360 day-count fraction.")
    amount: float


class TrsPriceResponse(BaseModel):
    """Response of `POST /derivatives/trs/price`."""
    trade_id: str
    valuation_date: str
    accrual_days: int
    accrual_fraction: float = Field(description="ACT/360, from the last reset.")
    total_return_leg: float
    financing_leg: float
    present_value: float = Field(
        description="Total return leg minus accrued financing, from the "
                    "receiver's side.")
    dv01: float = Field(
        description="Value change per +1bp. Negative for a receiver, who pays "
                    "financing — the sign carries the information a magnitude "
                    "would lose. Exactly zero on a funded swap, where only the "
                    "spread accrues.")
    fair_spread_bps: float | None = Field(
        default=None,
        description="Spread at which the trade would be worth nothing today. "
                    "Null when no time has accrued, where no spread could "
                    "balance it.")
    schedule: list[TrsAccrual]
    breakeven: list[dict[str, float]] = Field(
        default_factory=list,
        description="Breakeven financing spread against each supplied futures "
                    "price — what makes a swap and a future agree.")


class TermStructureEntry(BaseModel):
    """One expiry in a term structure."""
    expiry: str
    time_to_expiry: float
    financing_rate: float
    theoretical: float


class TermStructureResponse(BaseModel):
    """Response of `GET /derivatives/{index_id}/term-structure`."""
    index_id: str
    as_of: str
    spot: float
    entries: list[TermStructureEntry]


class RollResponse(BaseModel):
    """Response of `GET /derivatives/{index_id}/roll`.

    Both legs are priced theoretically off the same spot and curve, so this is
    the *carry* roll rather than a market one.
    """
    index_id: str
    as_of: str
    spot: float
    front_expiry: str
    back_expiry: str
    front_price: float
    back_price: float
    roll_cost: float = Field(description="Back minus front.")
    annualised_roll: float = Field(
        description="Positive in backwardation, negative in contango.")


class RiskModelRequest(BaseModel):
    """Body of `POST /risk-models/{model_id}/estimate`."""
    identifiers: list[str] = Field(
        default_factory=list,
        description="Names to estimate over. Empty uses the index named by "
                    "`index_id`.")
    index_id: str | None = Field(
        default=None,
        description="Take the universe from this index's latest run instead of "
                    "listing names.")
    start: str | None = Field(default=None,
                              description="Start of the estimation window.")
    end: str | None = Field(default=None, description="End of that window.")
    target: str = Field(
        default="constant_correlation",
        description="Structured target to shrink toward: 'constant_correlation' "
                    "or 'scaled_identity'.")
    intensity: float | None = Field(
        default=None,
        description="Weight on the target, in [0, 1]. Null uses the heuristic "
                    "from the panel's shape; 0 gives the raw sample covariance, "
                    "which on a short history across many names is mostly "
                    "noise.")
    repair: bool = Field(
        default=False,
        description="Clip negative eigenvalues if the result is not PSD. Off by "
                    "default because shrinkage should make it unnecessary and "
                    "clipping silently shifts the variances.")


class RiskDiagnosticsPayload(BaseModel):
    """How an estimate was produced, and how far it can be trusted."""
    observations: int = Field(
        description="Periods used, after dropping incomplete rows.")
    assets: int
    target: str
    intensity: float = Field(
        description="Weight placed on the structured target. 0 means the "
                    "estimate is the raw sample covariance.")
    average_correlation: float = Field(
        description="Mean off-diagonal correlation. The sanity check a person "
                    "can actually do: a diversified equity universe sits "
                    "around 0.3-0.6, and a figure far outside that says the "
                    "window or the universe is not what someone thought.")
    condition_number: float = Field(
        description="Largest eigenvalue over smallest. An optimiser inverts "
                    "this matrix, and a large value means the inverse "
                    "amplifies estimation error rather than reflecting it.")
    smallest_eigenvalue: float
    positive_semi_definite: bool = Field(
        description="Computed from the eigenvalues, not asserted. A matrix "
                    "that fails this can produce a negative portfolio "
                    "variance, and a caller about to invert it needs to know.")
    repaired: bool


class RiskModelView(BaseModel):
    """Response of `GET /risk-models/{model_id}`."""
    model_id: str
    asset_ids: list[str]
    start: str | None = None
    end: str | None = None
    correlation: TableFrame = Field(
        description="Symmetric with a unit diagonal, by construction.")
    covariance: TableFrame = Field(description="Annualised.")
    volatilities: dict[str, float] = Field(
        description="Annualised standard deviation per asset — the square root "
                    "of the covariance diagonal.")
    diagnostics: RiskDiagnosticsPayload


class RiskModelSummary(BaseModel):
    """One entry in `GET /risk-models`."""
    model_id: str
    assets: int
    observations: int
    average_correlation: float
    positive_semi_definite: bool


class RiskModelCollection(BaseModel):
    """Response of `GET /risk-models`."""
    risk_models: list[RiskModelSummary]


class ConstraintRow(BaseModel):
    """One constraint, in the shape a client's editor holds it.

    Maps 1:1 to a class in `beacon.optimise.constraints`: the row a user edits,
    the JSON that is stored and the object the solver receives are the same
    thing in three representations, so a rule cannot change meaning in
    translation.
    """
    id: str = Field(default="", max_length=64,
                    description="Stable row id. Carried back on any binding "
                                "constraint so a client can highlight the row "
                                "that bound.")
    type: str = Field(description="Constraint class, e.g. 'PositionBounds'.")
    params: dict[str, Any] = Field(
        default_factory=dict,
        description="Constructor arguments for that class, by name.")


class ConstraintSet(BaseModel):
    """A named list of constraints."""
    id: str = Field(description="Stable identifier, used in the URL.",
                    min_length=1, max_length=64)
    name: str = Field(description="Display name.", min_length=1)
    constraints: list[ConstraintRow] = Field(default_factory=list)


class ConstraintSetCollection(BaseModel):
    """Response of `GET /optimise/constraint-sets`."""
    constraint_sets: list[ConstraintSet]


class SavedConstraintSet(BaseModel):
    """Response of a successful save: the set plus any warnings."""
    constraint_set: ConstraintSet
    findings: list[Finding] = Field(
        default_factory=list,
        description="Non-blocking warnings. Errors would have prevented the "
                    "save.")


class ConstraintTypes(BaseModel):
    """Response of `GET /optimise/constraint-types`.

    Served so a client builds its editor from the same source the solver reads,
    rather than from a copy that drifts.
    """
    types: dict[str, list[str]] = Field(
        description="Constraint type -> the parameters it accepts.")


class OptimisationRunRequest(BaseModel):
    """Body of `POST /optimise/runs`."""
    index_id: str = Field(description="Index whose weights are the target.")
    constraint_set_id: str = Field(description="Constraint set to solve under.")
    as_of: str | None = Field(
        default=None,
        description="Which rebalance to target, YYYY-MM-DD. Defaults to the "
                    "latest.")
    start: str | None = Field(
        default=None,
        description="Start of the window the risk model is estimated over.")
    end: str | None = Field(default=None, description="End of that window.")
    risk_free_rate: float = Field(
        default=0.0,
        description="Used for the frontier's tangency point.")


class WeightRow(BaseModel):
    """One name's index, optimal and active weight."""
    asset_id: str
    index_weight: float
    optimal_weight: float
    active_weight: float = Field(
        description="Optimal minus index. Sums to zero across the portfolio "
                    "whenever both sides are fully invested.")


class OptimisationRunResult(BaseModel):
    """Result payload of a completed optimisation job."""
    run_id: str
    index_id: str
    constraint_set_id: str
    start: str
    end: str
    weights: list[WeightRow] = Field(
        description="Every name, largest active position first.")
    active_sum: float = Field(
        description="Sum of the active weights. Zero under full investment: "
                    "rearranging weight cannot create any.")
    tracking_error: float
    turnover: float = Field(description="One-way, against the index weights.")
    holdings: int
    binding: list[dict[str, str | None]] = Field(
        default_factory=list,
        description="Constraints the answer sits on, each traced back to the "
                    "row that produced it where one can be identified. These "
                    "are the rules that actually cost something.")
    heuristic: bool = Field(
        description="True when a non-convex constraint forced a restricted "
                    "re-solve, so the answer is feasible but not proven "
                    "optimal.")
    converged: bool
    iterations: int
    objective: float
    solver_message: str


class FrontierPoint(BaseModel):
    """One portfolio on the efficient frontier."""
    expected_return: float | None = None
    volatility: float
    sharpe_ratio: float | None = None
    weights: dict[str, float]
    binding: list[str] = Field(default_factory=list)
    heuristic: bool = False


class FrontierView(BaseModel):
    """Response of `GET /optimise/runs/{run_id}/frontier`."""
    run_id: str
    risk_free_rate: float
    expected_returns: dict[str, float] = Field(
        description="Annualised historical mean returns, per name. A poor "
                    "forecast, and the honest one: it is the only return "
                    "estimate derivable from the data the server holds. A "
                    "caller with a real view should supply it.")
    points: list[FrontierPoint]
    minimum_variance: FrontierPoint
    tangency: FrontierPoint
    monotonic: bool = Field(
        description="Whether risk rises with return across the grid. Always "
                    "true for a correct solve, so a false here means a point "
                    "did not reach optimality.")


class FactorExposure(BaseModel):
    """One factor loading."""
    factor: str
    exposure: float


class RiskDecomposition(BaseModel):
    """Active risk split into factor and specific parts.

    The two sum to the total exactly, because the covariance is *defined* as
    ``B F Bᵀ + D``. Pair an arbitrary covariance with arbitrary loadings and
    there is a cross term; the identity belongs to this model and not to any
    pairing of a matrix with some exposures.
    """
    total_variance: float
    factor_variance: float
    specific_variance: float
    tracking_error: float
    factor_share: float
    residual: float = Field(
        description="Total minus the two parts. Zero up to float noise, by "
                    "construction.")
    reconciles: bool
    contributions: dict[str, float] = Field(
        description="Each factor's share of the factor variance. May be "
                    "negative: a factor position that hedges another genuinely "
                    "reduces risk.")


class ExposuresView(BaseModel):
    """Response of `GET /optimise/runs/{run_id}/exposures`.

    Factors are the ones derivable from price and share count — size, momentum,
    volatility — plus a market intercept. Value and quality are absent rather
    than approximated: a momentum factor built from prices is the real thing, a
    value factor faked without book values would not be.
    """
    run_id: str
    factors: list[str]
    r_squared: float = Field(
        description="Read against a floor of roughly k/n rather than against "
                    "zero: fitting k factors to an n-asset cross-section "
                    "explains about that much by construction.")
    index_exposures: list[FactorExposure]
    optimal_exposures: list[FactorExposure]
    active_exposures: list[FactorExposure]
    risk: RiskDecomposition



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


class RebalanceSnapshot(BaseModel):
    """The index's composition at one rebalance.

    Both weight sets are carried. `weights` is what the index applied;
    `uncapped_weights` is what the weighting scheme produced before any cap.
    They are equal on an uncapped index, and the difference is the only way to
    answer what capping cost — a question that cannot be reconstructed from the
    applied weights alone.
    """
    date: str = Field(description="Rebalance date, YYYY-MM-DD.")
    weights: dict[str, float] = Field(description="Applied weights, summing to 1.")
    uncapped_weights: dict[str, float] = Field(
        default_factory=dict,
        description="Weights before capping. Equal to `weights` when no cap "
                    "was applied.")
    capped: list[str] = Field(
        default_factory=list,
        description="Constituents held at the cap on this date.")
    cap: float | None = Field(default=None,
                              description="Maximum single weight, if one applies.")
    redistributed: float = Field(
        default=0.0,
        description="Weight moved off capped names onto the rest.")


class ConcentrationPayload(BaseModel):
    """How concentrated a weight vector is."""
    herfindahl: float = Field(description="Sum of squared weights.")
    effective_assets: float = Field(
        description="1/HHI: how many equally weighted names would be as "
                    "concentrated. Lower than the raw count whenever weights "
                    "are uneven.")
    top_weights: dict[str, float] = Field(
        description="Combined weight of the largest N, keyed by N.")
    largest: float = Field(description="Largest single weight.")
    constituents: int


class DriftPayload(BaseModel):
    """How far weights moved between two rebalances."""
    total_absolute: float = Field(description="Sum of absolute weight changes.")
    maximum: float = Field(description="Largest single move.")
    worst: str = Field(description="Constituent that moved most.")
    turnover: float = Field(
        description="Half the total: the one-way trading needed to return to "
                    "target, since every overweight funds an underweight.")
    since: str = Field(description="The rebalance drifted from.")


class OverviewView(BaseModel):
    """Response of `GET /beacon/{index_id}/overview`."""
    index_id: str
    name: str
    start: str
    end: str
    observations: int
    rebalances: int
    last_rebalance: str
    metrics: BacktestMetrics
    concentration: ConcentrationPayload
    level: SeriesPayload


class WeightsView(BaseModel):
    """Response of `GET /beacon/{index_id}/weights`."""
    index_id: str
    as_of: str = Field(description="Date asked about.")
    rebalance_date: str = Field(
        description="Rebalance in force on that date. An index holds the "
                    "weights set at its last rebalance until the next one, so "
                    "this is usually earlier than `as_of`.")
    weights: dict[str, float]
    concentration: ConcentrationPayload
    drift: DriftPayload | None = Field(
        default=None,
        description="Movement since the previous rebalance. Null at the first, "
                    "where there is nothing to have drifted from.")
    capped: list[str] = Field(default_factory=list)
    cap: float | None = None
    cap_redistributed: float = 0.0


class ContributionPayload(BaseModel):
    """One constituent's share of the index return."""
    asset_id: str
    contribution: float
    average_weight: float
    total_return: float


class AttributionView(BaseModel):
    """Response of `GET /beacon/{index_id}/attribution`.

    Contributions are Carino-linked, so they sum to the compounded total return
    rather than approximately to it. `residual` is reported regardless and
    should sit at machine epsilon; anything larger means an assumption broke
    upstream, which is worth surfacing rather than rounding away.
    """
    index_id: str
    start: str
    end: str
    periods: int
    total_return: float
    contributions: list[ContributionPayload]
    residual: float
    reconciles: bool
    cap_drag: float | None = Field(
        default=None,
        description="Capped return minus uncapped. Null on an uncapped index: "
                    "reporting 0.0 would claim capping happened and made no "
                    "difference.")
    cost_drag: float | None = Field(
        default=None,
        description="Direct effect of transaction costs. Null at zero cost.")


class AssetView(BaseModel):
    """Response of `GET /beacon/{index_id}/assets/{identifier}`."""
    index_id: str
    identifier: str
    weight_history: dict[str, float] = Field(
        description="Rebalance date -> this name's weight. Only the rebalances "
                    "it was actually in.")
    rebalances_held: int
    total_return: float
    index_return: float
    excess_return: float
    tracking_error: float
    correlation: float
    beta: float
    observations: int
    price: SeriesPayload


class CompareEntry(BaseModel):
    """One index within a comparison, on the shared window."""
    index_id: str
    total_return: float
    level: SeriesPayload = Field(
        description="Rebased to 100 on the first shared date, so lines start "
                    "together and the comparison is of shape, not scale.")


class CompareView(BaseModel):
    """Response of `GET /beacon/compare`."""
    index_ids: list[str]
    start: str
    end: str
    observations: int = Field(
        description="Dates every index covers. Fewer than any one of them "
                    "carries alone whenever their spans differ.")
    entries: list[CompareEntry]


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
    rebalances: list[RebalanceSnapshot] = Field(
        default_factory=list,
        description="Composition at each rebalance. Everything the view "
                    "endpoints say about weights, attribution and individual "
                    "names is derived from these, so a run is readable without "
                    "recalculating the index. Daily weights are deliberately "
                    "absent: they are reconstructed from these and the prices, "
                    "and storing one per name per day would multiply the "
                    "payload by the number of trading days to save an "
                    "inexpensive calculation.")
    total_costs: float = Field(
        default=0.0,
        description="Transaction costs paid across the run, for the cost drag.")
    initial_capital: float = Field(
        default=0.0, description="Capital the simulation started with.")


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
