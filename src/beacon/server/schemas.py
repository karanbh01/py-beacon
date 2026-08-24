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
from pydantic import AfterValidator, BaseModel, Field, field_validator

from ..backtest.result import BacktestResult
from ..data.corporate_actions import (
    PAY_DATE_COLUMN,
    STATUS_COLUMN,
    kind_of,
    status_of,
)
from ..index.result import IndexResult
from ..report.blocks import BLOCK_TYPES
from ..universe import FROZEN, LIVE
from .serialisation import dataframe_to_payload, series_to_payload

# A rate or proportion expressed as a fraction: 0.0523 is 5.23%. Kept as a
# bare float rather than an object because it is arithmetic, not a quantity
# with a unit — clients format it for display.
Pct = Annotated[float, Field(description="Fraction, not percent: 0.0523 means 5.23%.")]

# A calendar date, constrained in the schema rather than left to the library.
#
# These were plain `str` until BN-131, which meant an empty string satisfied
# the model, reached a constructor several layers down, and came back as an
# unlabelled 500 -- the server appearing to break on input it should simply
# have refused. Declaring the shape here rejects it at the edge with a 422 and
# a field path, and puts the constraint in the OpenAPI document, so a client
# can see what a date is instead of discovering it.
#
# The pattern is the *shape*, which is what belongs in the OpenAPI document —
# a client can read `^\d{4}-\d{2}-\d{2}$` and know what to send. It cannot
# express whether a date exists, so `0000-00-00` and `2024-02-31` satisfy it
# and then fail in the parser. The validator below closes that, because a
# request that cannot be parsed is still the client's error and answering 500
# would be a lie about whose fault it was.
# What a stored document's identifier may contain.
#
# The server already refused anything else -- a document id becomes a
# filename, so `..` or a separator would let a request reach outside its
# collection directory. What was missing is that the *spec* said an id was
# any string at all, so a client had no way to know the rule existed until it
# broke one, and the fuzzer read every rejection as the server being wrong.
#
# Letters, digits, dash and underscore. No dots, which excludes `..` without
# a lookahead the generators handle badly, and nothing in this codebase has
# ever used one in an identifier.
IDENTIFIER_PATTERN = r"^[A-Za-z0-9_-]{1,64}$"

ISO_DATE_PATTERN = r"^\d{4}-\d{2}-\d{2}$"
def _real_date(value: str) -> str:
    """Reject a well-shaped string that is not a date anyone could observe."""
    try:
        pd.Timestamp(value)
    except (ValueError, TypeError) as error:
        raise ValueError(f"{value!r} is not a real calendar date") from error

    return value


IsoDate = Annotated[str, Field(pattern=ISO_DATE_PATTERN,
                               description="Calendar date, YYYY-MM-DD."),
                    AfterValidator(_real_date)]
Identifier = Annotated[str, Field(pattern=IDENTIFIER_PATTERN,
                                  description="Letters, digits, dash and "
                                              "underscore; up to 64.")]


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


class FeatureValue(BaseModel):
    """One field and what it was worth."""
    field: str
    value: float | None = Field(
        description="Null when nothing was knowable on the date: no coverage, "
                    "nothing published yet, or nothing recent enough.")
    type: str | None = Field(
        default=None, description="Which dataset it came from.")
    detail: str | None = Field(
        default=None, description="Free-form context the dataset carried.")
    date: str | None = Field(
        default=None, description="When the value became knowable — the "
                                  "announcement date, not the period it "
                                  "describes.")


class FeatureResponse(BaseModel):
    """Response of `GET /data/features/{identifier}`."""
    identifier: str
    as_of: str = Field(description="The date these were resolved at.")
    features: list[FeatureValue]


class FeatureBatchEntry(BaseModel):
    """One instrument in a batch feature response."""
    identifier: str
    features: list[FeatureValue]


class FeatureBatchResponse(BaseModel):
    """Response of `GET /data/features`."""
    as_of: str
    entries: list[FeatureBatchEntry]


class FeatureTypeCoverage(BaseModel):
    """One feature dataset, and how much of it is present."""
    type: str
    fields: list[str]
    identifiers: int = Field(description="Instruments this dataset covers.")
    rows: int


class FeatureCatalogue(BaseModel):
    """Response of `GET /data/features/catalogue`.

    What a client populates its controls from. Derived from the loaded data
    rather than a fixed vocabulary, so a dataset somebody loads tomorrow
    becomes a filter without a code change.
    """
    types: list[FeatureTypeCoverage]
    fields: list[str] = Field(
        description="Every field across every dataset. Names collide where "
                    "two datasets carry the same one, which is why the "
                    "per-type lists above exist.")


class FieldDescriptor(BaseModel):
    """One datapoint a client can offer as a filter."""
    path: str = Field(description="How it is written, e.g. 'reference.sector' "
                                  "or 'features.fundamentals.revenue'.")
    namespace: str = Field(description="market, reference, actions or features.")
    name: str
    dataset: str | None = Field(
        default=None, description="Feature TYPE, for feature fields only.")
    derived: bool = Field(
        default=False,
        description="Computed per request rather than stored. Screenable "
                    "either way — a client should not have to care.")


class FieldCatalogue(BaseModel):
    """Response of `GET /data/fields`.

    Every datapoint an expression can name, from one place, so a client builds
    one field picker rather than one per dataset. Derived from the loaded
    store, so a column or dataset nobody declared still appears.
    """
    fields: list[FieldDescriptor]
    namespaces: list[str]


class FeatureRow(BaseModel):
    """One row of an import."""
    identifier: str = Field(min_length=1)
    date: IsoDate
    type: str = Field(min_length=1)
    field: str = Field(min_length=1)
    value: float
    detail: str | None = None


class FeatureImport(BaseModel):
    """Body of `POST /data/features`."""
    rows: list[FeatureRow] = Field(
        description="Field-value rows. Merged into whatever is already "
                    "loaded; a row matching an existing identifier, date, "
                    "type and field replaces it.")


class FeatureImportResult(BaseModel):
    """What an import did."""
    accepted: int
    types: list[str] = Field(description="Datasets the import touched.")
    identifiers: int


class UniverseMembership(BaseModel):
    """One universe an instrument belongs to."""
    id: str = Field(description="Universe identifier, as used in the URL.")
    name: str = Field(description="Display name.")
    source: str = Field(
        description="'seeded' for one the server wrote from the dataset, "
                    "'user' for one somebody made.")


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
    universes: list[UniverseMembership] = Field(
        default_factory=list,
        description="Universes containing this instrument, so a client can "
                    "answer 'where is this used?' without reading every "
                    "universe and searching it.")

    @classmethod
    def from_row(cls,
                 identifier: str,
                 row: pd.Series,
                 universes: list[UniverseMembership] | None = None
                 ) -> "ReferenceResponse":
        """Build from a single reference-data row."""
        payload = series_to_payload(row)
        fields = dict(zip(payload["index"], payload["data"], strict=True))

        return cls(identifier=identifier, fields=fields,
                   universes=universes or [])


class IdentifierMatch(BaseModel):
    """One identifier a search or enumeration returned."""
    identifier: str = Field(description="The symbol itself.")
    name: str | None = Field(
        default=None,
        description="Display name, or null when reference data carries none. "
                    "A row without a name is still a useful suggestion, so it "
                    "is returned rather than dropped.")
    datasets: list[str] = Field(
        default_factory=list,
        description="Which datasets actually cover this identifier: 'market', "
                    "'reference', 'corporate_actions'. This is what lets a "
                    "client offer a reference-only name in a reference view "
                    "and mark it unavailable for prices, rather than "
                    "suggesting something the engine cannot then serve.")
    exchange: str | None = Field(
        default=None, description="Listing venue, when reference data has one.")
    currency: str | None = Field(
        default=None, description="Denomination, when reference data has one.")


class IdentifierSearchResponse(BaseModel):
    """Response of `GET /data/identifiers`.

    Search when `q` is given, enumeration when it is not.

    Ranking is decided server-side and is part of the contract: exact
    identifier, identifier prefix, name prefix, identifier substring, name
    substring, alphabetical within each. Once `limit` is applied a client
    cannot re-rank what it was not sent.
    """
    identifiers: list[IdentifierMatch] = Field(default_factory=list)
    total: int = Field(
        default=0,
        description="Matches *before* `limit`, so a client can say "
                    "'showing 20 of 340'.")
    truncated: bool = Field(
        default=False,
        description="Whether the limit hid anything. Derivable from `total`, "
                    "but explicit beats arithmetic at a call site.")
    version: str = Field(
        default="",
        description="Fingerprint of the data this was built from, also served "
                    "as the ETag. Changes only when a dataset syncs, so a "
                    "client can cache an enumeration and revalidate cheaply.")


class ReferenceEntry(BaseModel):
    """One identifier's row in a batch reference response."""
    identifier: str
    found: bool = Field(
        description="Whether the reference dataset holds this identifier. "
                    "False leaves `fields` empty rather than failing the "
                    "batch, so one unknown name does not lose the other 511.")
    fields: dict[str, Any] = Field(
        default_factory=dict,
        description="Requested reference columns and derived fields. A "
                    "column the dataset holds but this identifier has no "
                    "value for is present and null, which is a different "
                    "statement from the identifier being absent.")


class BatchReferenceResponse(BaseModel):
    """Response of `GET /data/reference`.

    Entries are in the order the request named them, one per identifier, so a
    table can render straight down the list without re-sorting against what it
    asked for.
    """
    entries: list[ReferenceEntry]
    as_of: str | None = Field(
        default=None,
        description="Point-in-time date applied, if one was requested.")


class CorporateAction(BaseModel):
    """One corporate action.

    `kind` is the authoritative answer to what `value` means, and the reason a
    client needs no list of type strings. Reading `type` and inferring cash or
    ratio from a hardcoded list works until a type the client has never seen
    arrives, at which point it renders as whichever the list defaults to —
    confidently, and wrongly.
    """
    ex_date: str = Field(description="Ex-date, ISO 8601.")
    type: str = Field(
        description="Action type, e.g. DIVIDEND or SPLIT. A closed set the "
                    "engine validates on load, but branch on `kind` rather "
                    "than on this: new types are added, and a client that "
                    "matches type strings breaks silently when one is.")
    kind: Literal["cash", "ratio", "structural"] = Field(
        description="What `value` means. 'cash' is an amount per share and "
                    "adds up; 'ratio' is a share-count multiplier and "
                    "compounds; 'structural' (rights issue, spin-off, merger) "
                    "carries no directly aggregable value and should not be "
                    "rendered as a quantity in either column.")
    value: float = Field(
        description="Cash amount per share for cash actions; a share-count "
                    "multiplier for ratio actions. What it means depends on "
                    "`kind`, so the two are never summed together.")
    pay_date: IsoDate | None = Field(
        default=None,
        description="Payment date, ISO 8601, where the source knows it. Null "
                    "means unknown — omit the field in the UI rather than "
                    "dashing it, since a dash reads as 'there is none'.")
    status: Literal["announced", "paid", "cancelled"] | None = Field(
        default=None,
        description="Lifecycle state, where the source knows it. Null means "
                    "unknown, not 'not yet announced'.")


def _optional(row: pd.Series,
              column: str) -> str | None:
    """A nullable column off an action row, as a string or None.

    Absent column and absent value are both None: a history reconstructed from
    prices carries neither a pay date nor a status, and a client should not
    have to distinguish "this dataset has no such column" from "this action has
    no such value" — in both cases it does not know.
    """
    if column not in row.index:
        return None

    value = row[column]
    if pd.isna(value):
        return None

    if isinstance(value, pd.Timestamp):
        return str(value.date())

    return str(value)


def _action_from(row: pd.Series) -> CorporateAction:
    """One history row as its API shape."""
    action_type = str(row["TYPE"])

    return CorporateAction(
        ex_date=pd.Timestamp(row["EX_DATE"]).isoformat(),
        type=action_type,
        # Derived here rather than stored, so it cannot disagree with the type
        # it describes and an older store gains the field for free.
        kind=kind_of(action_type),
        value=float(row["VALUE"]),
        pay_date=_optional(row, PAY_DATE_COLUMN),
        status=status_of(_optional(row, STATUS_COLUMN)))


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
        actions = [_action_from(row) for _, row in frame.iterrows()]

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
    # Deliberately `str` rather than a Literal of the four cadences, even
    # though that would put them in the OpenAPI document and stop the fuzz run
    # reporting a rejection here.
    #
    # `definitions.py` already refuses an unknown one, and refuses it *better*:
    # the response carries a coded finding, `UNSUPPORTED_FREQUENCY`, that
    # beacon-ui shows against the field while somebody is editing. Declaring
    # the enum moves the rejection to pydantic, which answers with a generic
    # validation blob and loses the code. Tightening the schema here would
    # make the spec more precise and the product worse.
    rebalancing_frequency: str = Field(
        description="MONTHLY, QUARTERLY, SEMI-ANNUAL or ANNUAL. The cadence; "
                    "`rebalance_day_rule` decides which day of the month.")
    universe: UniverseRef
    pipeline: PipelineSpec
    description: str | None = None

    # --- BN-121 metadata. All defaulted, so every stored document stays valid
    # and no migration is needed; the defaults are the behaviour indices were
    # defined against before these fields existed.
    return_type: Literal["PRICE", "TOTAL_RETURN", "NET_TOTAL_RETURN"] = Field(
        default="PRICE",
        description="How returns accumulate. PRICE ignores distributions; "
                    "TOTAL_RETURN reinvests them across the index by shrinking "
                    "the divisor; NET_TOTAL_RETURN does the same after "
                    "withholding tax. PRICE is the default, so an index "
                    "defined before this existed is unchanged.")
    withholding_tax_rate: float = Field(
        default=0.0,
        ge=0.0,
        lt=1.0,
        description="Fraction of each distribution withheld, for a net index. "
                    "A flat index-level rate rather than a per-country table: "
                    "a table is only as good as the country field behind it, "
                    "and an unpopulated one produces a number that looks "
                    "precise and is not. Ignored unless `return_type` is "
                    "NET_TOTAL_RETURN.")
    calendar: str | None = Field(
        default=None,
        description="Exchange MIC backing trading-day arithmetic, e.g. "
                    "'XNYS'. Null means Monday to Friday, which is what every "
                    "index defined before this field used. Naming one requires "
                    "the `calendars` extra — an index that declares a calendar "
                    "must never quietly compute against a different one.")
    rebalance_day_rule: str = Field(
        default="FIRST_BUSINESS_DAY",
        description="Which day of a scheduled month the rebalance falls on: "
                    "FIRST_BUSINESS_DAY, LAST_BUSINESS_DAY or THIRD_FRIDAY. A "
                    "date landing on a holiday rolls back to the previous "
                    "session.")
    publication_time: str | None = Field(
        default=None,
        description="When the level is published, e.g. '18:00 America/"
                    "New_York'. Display metadata: it says when a figure is "
                    "released and changes no figure, so nothing in the "
                    "calculation reads it.")
    effective_lag_sessions: int = Field(
        default=0,
        ge=0,
        description="Sessions between a rebalance being announced and its "
                    "weights taking effect. Stored now, honoured by the "
                    "calculator in BN-126; until then it is declared and not "
                    "applied, and 0 is the behaviour in force.")


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
        description="Content, drawn top to bottom. Each carries a `kind`, "
                    f"one of: {', '.join(sorted(BLOCK_TYPES))}.")

    @field_validator("blocks")
    @classmethod
    def _kinds_are_known(cls,
                         blocks: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Check the one field that decides how a block is built.

        Only `kind`, deliberately. The docstring above explains why blocks
        stay free-form, and that reasoning holds: restating every block's
        contents here would be a second definition to keep in step. But
        `kind` is what dispatches, and an unknown one used to travel all the
        way to `block_from_dict` and come back as a 500 -- the server
        reporting itself as broken because a client sent `{}`.
        """
        for position, block in enumerate(blocks):
            kind = block.get("kind")

            if kind not in BLOCK_TYPES:
                raise ValueError(
                    f"block {position} has kind {kind!r}; expected one of "
                    f"{', '.join(sorted(BLOCK_TYPES))}")

        return blocks


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
    valuation_date: IsoDate | None = Field(default=None, description="YYYY-MM-DD.")
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
    start_date: IsoDate
    end_date: IsoDate
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
    valuation_date: IsoDate
    last_reset_date: IsoDate | None = Field(
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
    valuation_date: IsoDate
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


class ParameterSpec(BaseModel):
    """One parameter of a configurable type, described well enough to render.

    Names, types, defaults and whether a parameter is required are read from
    the constructor, so they cannot drift from what the code accepts. Labels,
    ordering and choices are declared on the class, because a signature cannot
    carry them.
    """
    name: str = Field(description="Parameter name, as it must appear in `params`.")
    type: str = Field(
        description="Display type: number, integer, boolean, string or json. "
                    "What control to render, not the Python annotation.")
    required: bool = Field(
        description="Whether the constructor rejects the call without it.")
    default: Any = Field(
        default=None,
        description="Value used when omitted. Null both for 'no default' and "
                    "for a default of None; `required` distinguishes them.")
    label: str = Field(description="Human-readable field name.")
    order: int = Field(description="Position in the form, ascending.")
    choices: list[str] | None = Field(
        default=None,
        description="The accepted values, when this is a closed set. Null "
                    "means any value of `type` is allowed.")
    help: str | None = Field(default=None,
                             description="One line of guidance for the field.")


class TypeSpec(BaseModel):
    """One configurable type a client can offer."""
    name: str = Field(description="Class name, as it must appear in `type`.")
    label: str = Field(description="Human-readable name for the type.")
    summary: str = Field(default="",
                         description="One line describing what it does.")
    parameters: list[ParameterSpec] = Field(default_factory=list)


class RuleTypes(BaseModel):
    """Response of `GET /indices/rule-types`.

    Everything a methodology editor needs to render a real form: which rules
    and schemes exist, what each takes, and how to label and order the fields.
    Without it `RuleSpec.type` is a free-text box and `params` a list of
    key/value pairs, so a misspelled parameter is only discovered on submit.
    """
    selection: list[TypeSpec] = Field(
        description="Eligibility rules available for the selection stage.")
    weighting: list[TypeSpec] = Field(
        description="Weighting schemes available for the weighting stage.")


class ConstraintTypes(BaseModel):
    """Response of `GET /optimise/constraint-types`.

    Served so a client builds its editor from the same source the solver reads,
    rather than from a copy that drifts.
    """
    types: dict[str, list[str]] = Field(
        description="Constraint type -> the parameters it accepts. Kept for "
                    "clients written against the original shape; `specs` "
                    "carries the same set with everything needed to render it.")
    specs: list[TypeSpec] = Field(
        default_factory=list,
        description="The same constraint types in the richer shape "
                    "`/indices/rule-types` uses, so one client component can "
                    "render both editors.")


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


# How a universe came to exist. A client renders a seeded one read-only, so
# this has to be on the document rather than inferred from its id.
SOURCE_USER = "user"
SOURCE_SEEDED = "seeded"

# Whether a universe stores its membership or the question that produced it.
# Re-exported from the library so the document and the resolver cannot drift
# apart on the spelling.
MODE_FROZEN = FROZEN
MODE_LIVE = LIVE


class Universe(BaseModel):
    """A named set of instrument identifiers."""
    id: str = Field(description="Stable identifier.", min_length=1, max_length=64)
    name: str = Field(description="Display name.", min_length=1)
    identifiers: list[str] = Field(default_factory=list)
    description: str | None = Field(
        default=None, description="Optional free text.")
    source: str = Field(
        default=SOURCE_USER,
        description=f"'{SOURCE_USER}' for one somebody created, "
                    f"'{SOURCE_SEEDED}' for one the generator wrote. A seeded "
                    f"universe cannot be edited or deleted.")
    filter: dict[str, Any] | None = Field(
        default=None,
        description="The expression this universe was built from, when it was "
                    "built by filtering. Null for a curated list.")
    mode: str = Field(
        default=MODE_FROZEN,
        description=f"'{MODE_FROZEN}' means the stored identifiers are the "
                    f"membership; '{MODE_LIVE}' means the filter is "
                    f"re-evaluated on read, so the membership moves when the "
                    f"data does. A frozen universe records what it contained; "
                    f"a live one records how it was chosen. They are different "
                    f"objects and a caller needs to know which they have.")
    as_of: str | None = Field(
        default=None,
        description="The date a filter was last resolved at.")


class UniverseUpsert(BaseModel):
    """Body of `PUT /universes/{id}`."""
    name: str = Field(description="Display name.", min_length=1)
    identifiers: list[str] = Field(default_factory=list)
    description: str | None = Field(default=None)


class UniverseCreate(BaseModel):
    """Body of `POST /universes`.

    No id: the server derives one from the name, so a client cannot create two
    universes whose ids differ only in punctuation and expect them to be
    distinct documents.
    """
    name: str = Field(description="Display name.", min_length=1, max_length=64)
    identifiers: list[str] = Field(
        default_factory=list,
        description="Members. Every one must exist in the loaded reference "
                    "data. Required unless a filter is given.")
    description: str | None = Field(default=None)
    filter: dict[str, Any] | None = Field(
        default=None,
        description="A serialised expression to build the membership from, "
                    "instead of naming it. Mutually exclusive with a "
                    "non-empty `identifiers`.")
    mode: str = Field(
        default=MODE_FROZEN,
        description=f"'{MODE_LIVE}' re-evaluates the filter on every read; "
                    f"'{MODE_FROZEN}' keeps the membership it resolved to. "
                    f"Only meaningful with a filter.")


class UniverseCollection(BaseModel):
    """Response of `GET /universes`."""
    universes: list[Universe]


class UniverseMembers(BaseModel):
    """Response of `GET /universes/{id}/members`."""
    universe_id: str
    identifiers: list[str]


class ScheduleView(BaseModel):
    """Response of `GET /indices/{index_id}/schedule`.

    Derived, not stored: the next rebalance is a function of the schedule, the
    calendar and today, and storing it would leave a date that silently expires.
    """
    index_id: str
    rebalancing_frequency: str
    rebalance_day_rule: str
    calendar: str | None = Field(
        default=None, description="Null means business days.")
    as_of: str = Field(description="Date the answer was computed from.")
    next_rebalance: str | None = Field(
        default=None,
        description="Next rebalance date, ISO 8601. Null when none falls "
                    "within the lookahead — which happens only for a schedule "
                    "this server cannot project, not for a normal index.")
    days_until: int | None = Field(
        default=None,
        description="Calendar days from `as_of` to `next_rebalance`. Calendar "
                    "days rather than sessions, because it is displayed as "
                    "'in 57 days' and a reader counts those on a wall "
                    "calendar.")
    recent: list[str] = Field(
        default_factory=list,
        description="Rebalances already passed, most recent last.")
    upcoming: list[str] = Field(
        default_factory=list,
        description="Scheduled rebalances after `as_of`, soonest first.")


class PreviewRequest(BaseModel):
    """Body of `POST /indices/{id}/preview`, which previews the *saved* index."""
    as_of: str | None = Field(
        default=None,
        description="Date to evaluate the pipeline at, YYYY-MM-DD. Defaults "
                    "to the index's base date.")


class PreviewDocumentRequest(BaseModel):
    """Body of `POST /indices/preview`, which previews a document as supplied.

    The route for a draft. The by-id route reads what is stored, so while an
    editor holds unsaved changes its figures describe the old definition — with
    nothing on screen to say they are stale. This one previews exactly what was
    sent, so editing a rule updates the resolved figures without saving.
    """
    document: IndexDocument = Field(
        description="The definition to derive, saved or not.")
    as_of: str | None = Field(
        default=None,
        description="Date to evaluate the pipeline at, YYYY-MM-DD. Defaults "
                    "to the document's base date.")


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
    date: str = Field(
        description="Date these weights took effect, YYYY-MM-DD. Snapshots "
                    "are keyed by the effective date because that is when the "
                    "composition is in force.")
    announced: str | None = Field(
        default=None,
        description="When this composition was published, if earlier than "
                    "`date`. Null when the index has no effective-date lag "
                    "and the two coincide, so its presence is itself the "
                    "signal that a lag applies.")
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


class ConstituentRow(BaseModel):
    """One constituent's row in the weights table.

    Everything a row needs is here, so the table renders from one response
    rather than joining three. The two weights are the point: `raw_weight` is
    what the weighting scheme produced, `weight` is what survived the cap, and
    the difference is what capping moved.
    """
    identifier: str
    weight: float = Field(description="Applied weight, after any cap.")
    raw_weight: float = Field(
        description="Weight before capping. Equal to `weight` on an uncapped "
                    "index, and the only way to see what the cap cost.")
    capped: bool = Field(
        default=False,
        description="Whether this name was held at the cap on this rebalance.")
    shares_outstanding: float | None = Field(
        default=None,
        description="The company's shares outstanding on this date, from "
                    "market data. Deliberately NOT the number of shares the "
                    "index holds — that is a different figure needing a "
                    "divisor and a notional, and naming this one `shares` "
                    "would let the two be confused silently.")
    delta_since_rebalance: float | None = Field(
        default=None,
        description="Held weight minus target weight, for this name, as of "
                    "`as_of`. Null when `as_of` is the rebalance date itself: "
                    "the weights were just set, so nothing has drifted and a "
                    "zero would claim a measurement rather than its absence.")
    risk_contribution: float | None = Field(
        default=None,
        description="This name's share of the index's annualised volatility, "
                    "in the same units. Populated only when `risk=true` was "
                    "requested; null also when the risk model has no estimate "
                    "for this constituent, which `risk.uncovered` lists.")
    active_weight: float | None = Field(
        default=None,
        description="Weight minus the benchmark's, when a `benchmark` was "
                    "given. Negative is an underweight.")
    active_risk_contribution: float | None = Field(
        default=None,
        description="This name's share of tracking error. **Can be negative**: "
                    "an underweight that hedges an overweight genuinely "
                    "reduces tracking error, and hiding that behind an "
                    "absolute value would misreport what the index is doing.")


class RiskPayload(BaseModel):
    """How the index's volatility divides among its holdings.

    Contributions sum to `volatility` exactly rather than approximately — the
    decomposition is an identity, so a client can show the parts and the whole
    without them disagreeing.
    """
    volatility: float = Field(
        description="Annualised volatility of the covered holdings, at the "
                    "weights they are actually held.")
    covered_weight: float = Field(
        description="Fraction of the index the figure speaks for. Below 1.0 "
                    "when the model has no estimate for some constituent; the "
                    "covered names keep their real weights rather than being "
                    "renormalised, which would restate the portfolio.")
    uncovered: list[str] = Field(
        default_factory=list,
        description="Constituents with no estimate, usually for want of "
                    "history. Listed rather than only counted, so a reader "
                    "sees which names are missing.")
    window_start: str | None = Field(
        default=None, description="First date of the estimation window.")
    window_end: str | None = Field(
        default=None, description="Last date of the estimation window.")


class ActiveRiskPayload(BaseModel):
    """How tracking error against a benchmark divides among active positions.

    Contributions sum to `tracking_error` exactly, the same identity the total
    decomposition satisfies — on active weights rather than holdings.
    """
    benchmark: str = Field(description="Index the comparison is against.")
    tracking_error: float = Field(
        description="Annualised volatility of the active position.")
    covered_weight: float = Field(
        description="Share of *gross* active weight the model covers. Gross "
                    "because active weights sum to roughly zero, so a plain "
                    "sum would say nothing about coverage.")
    uncovered: list[str] = Field(
        default_factory=list,
        description="Names with no estimate, from either side.")
    contributions_not_held: dict[str, float] = Field(
        default_factory=dict,
        description="Contributions from benchmark constituents the index does "
                    "not hold. They have no row in the weights table but are "
                    "often the largest active positions there are, so omitting "
                    "them would hide the biggest sources of tracking error.")
    window_start: str | None = Field(default=None)
    window_end: str | None = Field(default=None)


class WeightsView(BaseModel):
    """Response of `GET /beacon/{index_id}/weights`."""
    index_id: str
    as_of: str = Field(description="Date asked about.")
    rebalance_date: str = Field(
        description="Rebalance in force on that date. An index holds the "
                    "weights set at its last rebalance until the next one, so "
                    "this is usually earlier than `as_of`.")
    announced_date: IsoDate | None = Field(
        default=None,
        description="When that composition was published, if earlier than "
                    "`rebalance_date`. Null when the index has no lag.")
    weights: dict[str, float]
    rows: list[ConstituentRow] = Field(
        default_factory=list,
        description="Per-constituent detail, heaviest first. Carries the same "
                    "applied weights as `weights`, which is kept because "
                    "charts and concentration maths want a mapping while a "
                    "table wants ordered rows.")
    concentration: ConcentrationPayload
    drift: DriftPayload | None = Field(
        default=None,
        description="Movement since the previous rebalance. Null at the first, "
                    "where there is nothing to have drifted from.")
    capped: list[str] = Field(default_factory=list)
    cap: float | None = None
    cap_redistributed: float = 0.0
    risk: RiskPayload | None = Field(
        default=None,
        description="The volatility decomposition, when `risk=true` was "
                    "requested. Null otherwise: estimating a covariance over "
                    "every constituent is the pane's whole cost, and nobody "
                    "should pay it without asking.")
    active_risk: ActiveRiskPayload | None = Field(
        default=None,
        description="The tracking-error decomposition, when a `benchmark` "
                    "index was named alongside `risk=true`.")


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
        description="Rebalance date -> this name's applied weight. Only the "
                    "rebalances it was actually in.")
    raw_weight_history: dict[str, float] = Field(
        default_factory=dict,
        description="The same dates -> the weight before capping. Added "
                    "alongside `weight_history` rather than replacing it, so "
                    "the drilldown can show what the cap did to this name over "
                    "time without breaking a client reading only the applied "
                    "series.")
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
    frequency: str = Field(
        default="static",
        description="How often this dataset is expected to change: 'daily', "
                    "'static' or 'event'. The engine's definition of what "
                    "stale means, so a client renders staleness from this "
                    "rather than from thresholds of its own.")
    stale_after_seconds: float | None = Field(
        default=None,
        description="Age beyond which this dataset should read as stale. "
                    "Published so the mapping from frequency to a duration "
                    "lives in one place; null means the question does not "
                    "apply, as for static data.")
    source: str | None = Field(
        default=None,
        description="Where the data was loaded from, e.g. 'synthetic', "
                    "'yfinance', 'local'. Null when a fetcher was assembled "
                    "in-process and nothing recorded a provenance.")
    field_count: int = Field(
        default=0,
        description="Data columns this dataset holds, excluding the "
                    "identifier and date keys.")
    cache_size_bytes: int | None = Field(
        default=None,
        description="Bytes the backing store occupies on disk. Null when the "
                    "data did not come from a store, in which case it has no "
                    "size to report rather than a size of zero.")


class CoverageResponse(BaseModel):
    """Response of `GET /data/coverage`."""
    datasets: list[DatasetCoverage]
    identifiers_union: int = Field(
        default=0,
        description="Distinct identifiers across every dataset. Not the sum "
                    "of the per-dataset counts: a name present in both market "
                    "and reference data would otherwise be counted twice, and "
                    "'assets covered' would exceed the universe.")
    cache_size_bytes: int | None = Field(
        default=None,
        description="Total bytes on disk for the whole store. Null when no "
                    "store backs this process.")


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
