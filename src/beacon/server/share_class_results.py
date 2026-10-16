# src/beacon/server/share_class_results.py
"""
What a run's share classes add to its result.

Each class's record is in its own currency, its series on the run's days, and
its figures are measured against the index in its currency: converted for an
unhedged class, hedged the same way for a hedged one (see
`beacon.backtest.class_benchmark`).
"""
# BN-314.
import logging
from collections.abc import Sequence
from typing import Any

import pandas as pd

from ..assumptions import DEFAULT_PERIODS_PER_YEAR
from ..backtest.class_benchmark import class_tracking, index_in_class_currency
from ..backtest.flows import FlowRecord
from ..backtest.result import BacktestResult
from ..backtest.share_classes import ShareClass, ShareClassBook
from ..data.fetcher import DataFetcher
from ..exceptions import CalculationError
from .schemas import (
    FlowPayload,
    HedgePayload,
    SeriesPayload,
    ShareClassMetrics,
    ShareClassPayload,
)

logger = logging.getLogger(__name__)


def flow_payloads(flows: Sequence[FlowRecord]) -> list[FlowPayload]:
    """Flows as dealt, on the wire."""
    return [FlowPayload(date=str(flow.date.date()), amount=flow.amount,
                        nav_per_unit=flow.nav_per_unit, units=flow.units,
                        dealing_price=flow.dealing_price, adjustment=flow.adjustment,
                        creation_units=flow.creation_units)
            for flow in flows]


def share_class_payloads(result: BacktestResult,
                         share_classes: Sequence[ShareClass],
                         days: pd.Index,
                         index_levels: pd.Series,
                         index_currency: str,
                         fetcher: DataFetcher) -> dict[str, Any]:
    """The `share_classes` and `hedge_value` a run's payload carries; empty
    and null without share classes.

    Args:
        result: The finished run.
        share_classes: The classes it ran, for their hedge ratios.
        days: The run's days, which the classes' series are put on.
        index_levels: The index's levels, in *index_currency*.
        index_currency: The index's currency.
        fetcher: The data the exchange and short rates are read from.
    """
    if not result.share_classes:
        return {}

    assumptions = result.modelling_assumptions
    periods = (assumptions.periods_per_year if assumptions is not None else None)
    cash_rate = assumptions.cash_rate if assumptions is not None else None
    ratios = {share_class.name: share_class.hedge_ratio for share_class in share_classes}

    payloads = [_payload(book, days,
                         _benchmark(book, ratios.get(book.name, 1.0), index_levels,
                                    index_currency, fetcher, cash_rate),
                         periods or DEFAULT_PERIODS_PER_YEAR)
                for book in result.share_classes.values()]
    hedged = not result.hedge_value.empty

    return {"share_classes": payloads,
            "hedge_value": (_series(result.hedge_value, days) if hedged else None)}


def _benchmark(book: ShareClassBook,
               hedge_ratio: float,
               index_levels: pd.Series,
               index_currency: str,
               fetcher: DataFetcher,
               cash_rate: float | None) -> pd.Series:
    """The index in the class's currency, or nothing when no rate reaches it,
    which is said."""
    try:
        return index_in_class_currency(index_levels, index_currency, book.currency, fetcher,
                                       hedge_ratio=hedge_ratio if book.hedged else None,
                                       cash_rate=cash_rate)
    except CalculationError as error:
        logger.warning("Share class '%s' is not measured against the index: %s",
                       book.name, error)

        return pd.Series(dtype=float)


def _payload(book: ShareClassBook,
             days: pd.Index,
             benchmark: pd.Series,
             periods_per_year: int) -> ShareClassPayload:
    """One class's record, on the run's days."""
    metrics = class_tracking(book, benchmark, periods_per_year)

    return ShareClassPayload(
        name=book.name,
        currency=book.currency,
        hedged=book.hedged,
        distribution=book.distribution,
        management_fee_bps=book.management_fee_bps,
        net_assets=_series(book.net_assets, days),
        book_net_assets=_series(book.book_net_assets, days),
        units=_series(book.units, days),
        nav_per_unit=_series(book.nav_per_unit, days),
        fees=_series(book.fees, days, fill=0.0),
        fees_paid=float(book.fees.sum()),
        distributions=_series(book.distributions, days, fill=0.0),
        distributions_paid=float(book.distributions.sum()),
        flows=flow_payloads(book.flows),
        hedge_value=_series(book.hedge_value, days) if book.hedged else None,
        hedges=[HedgePayload(date=str(trade.date.date()), reason=trade.reason,
                             notional=trade.notional, rate=trade.rate, settled=trade.settled)
                for trade in book.hedges],
        metrics=ShareClassMetrics.model_validate(metrics))


def _series(series: pd.Series,
            days: pd.Index,
            fill: float | None = None) -> SeriesPayload:
    """A series on the run's days: carried over a day it has no entry for,
    or *fill* for a daily amount."""
    aligned = (series.reindex(days, fill_value=fill) if fill is not None
               else series.reindex(days, method="ffill"))

    return SeriesPayload.from_series(aligned)
