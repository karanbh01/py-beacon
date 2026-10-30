# src/beacon/server/index_level.py
"""
An index on its own: `GET /indices/{index_id}/level`.

The index calculated alone, with no portfolio, costs or vehicle: its level,
the headline figures of that level, and its weights at each rebalance. Through
the same cache a backtest uses, so a backtest over the same dates reuses the
calculation, and the other way round.
"""
# BN-316. Every other view of an index's figures read a saved backtest run.
from pydantic import BaseModel, Field

from ..analysis.performance.levels import headline
from ..assumptions import DEFAULT_PERIODS_PER_YEAR, ModellingAssumptions
from ..backtest.main import Backtest
from ..data.fetcher import DataFetcher
from ..index.result import IndexResult
from .definitions import build_definition
from .schemas import (
    BacktestMetrics,
    IndexDocument,
    RebalanceSnapshot,
    SeriesPayload,
    metrics_from,
    rebalance_snapshots,
)
from .store import DocumentStore


class IndexLevelView(BaseModel):
    """Response of `GET /indices/{index_id}/level`."""
    index_id: str = Field(description="The index calculated.")
    name: str = Field(description="Its display name.")
    currency: str = Field(description="The currency its level is in.")
    start: str = Field(description="First date calculated, YYYY-MM-DD.")
    end: str = Field(description="Last date calculated, YYYY-MM-DD.")
    level: SeriesPayload = Field(
        description="The index level as calculated, from its base value.")
    metrics: BacktestMetrics = Field(
        description="The level's total and annualised return, volatility, "
                    "Sharpe ratio and maximum drawdown, defined as a "
                    "backtest's are; the tracking and fund figures are null.")
    rebalances: list[RebalanceSnapshot] = Field(
        description="The weights at each rebalance in the window, in date order.")


def calculate_index(document: IndexDocument,
                    index_store: DocumentStore,
                    fetcher: DataFetcher,
                    start: str | None = None,
                    end: str | None = None) -> IndexResult:
    """A stored index calculated alone, over the loaded data.

    Args:
        document: The stored definition; a derived one's source is resolved
            through *index_store*.
        index_store: Where stored definitions live.
        fetcher: The loaded data.
        start: First date; the definition's base date when None.
        end: Last date; the last date the data covers when None.
    """
    definition = build_definition(document, index_store)

    # No portfolio is simulated, so the capital plays no part.
    return Backtest(initial_capital=1.0, data_provider=fetcher).calculate(
        definition,
        start=start or str(definition.base_date.date()),
        end=end or str(fetcher.date_range[1].date()))


def build_index_level(document: IndexDocument,
                      result: IndexResult) -> IndexLevelView:
    """The view of a calculated index."""
    levels = result.index_levels
    assumptions = ModellingAssumptions().effective()
    # BN-401: measured by the library.
    figures = headline(levels,
                       assumptions.periods_per_year or DEFAULT_PERIODS_PER_YEAR,
                       assumptions.risk_free_rate or 0.0)
    # The cap as a backtest reads it: an optimised index has none of its own.
    cap = (document.pipeline.weighting.max_weight
           if document.pipeline is not None else None)

    return IndexLevelView(index_id=document.id,
                          name=document.name,
                          currency=document.currency,
                          start=str(levels.index[0])[:10] if len(levels) else "",
                          end=str(levels.index[-1])[:10] if len(levels) else "",
                          level=SeriesPayload.from_series(levels),
                          metrics=metrics_from(dict(figures)),
                          rebalances=rebalance_snapshots(result, cap))
