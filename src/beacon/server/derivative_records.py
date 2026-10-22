# src/beacon/server/derivative_records.py
"""
What a fund's swap, index futures and dividends owed add to a run's result.

A synthetic fund's swap: its value and the counterparty exposure it makes each
day, and each reset. A fund's index futures, whether it holds the index
through them or equitises its cash buffer: their exposure each day, in money
and as a share of NAV, and each roll. Every series is on the run's days, and
everything is empty or null for a fund without them, so a run without them
answers as before.
"""
# BN-315.
from typing import Any

import pandas as pd

from ..backtest.result import BacktestResult
from .schemas import FuturesRollPayload, SeriesPayload, SwapResetPayload


def derivative_payloads(result: BacktestResult,
                        days: pd.Index) -> dict[str, Any]:
    """The swap and futures records a run's payload carries."""
    payloads: dict[str, Any] = {}

    # BN-356: not a derivative, but like the swap a part of the NAV the
    # portfolio does not hold.
    if (result.dividends_receivable != 0.0).any():
        payloads["dividends_receivable"] = _on(result.dividends_receivable, days)

    if not result.swap_value.empty:
        payloads.update(
            swap_value=_on(result.swap_value, days),
            counterparty_exposure=_on(result.counterparty_exposure, days),
            swap_resets=[SwapResetPayload(date=str(reset.date.date()), settled=reset.settled,
                                          reason=reset.reason, notional=reset.notional)
                         for reset in result.swap_resets])

    if not result.futures_notional.empty:
        payloads.update(
            futures_notional=_on(result.futures_notional, days),
            futures_exposure=_on(result.futures_exposure, days),
            futures_rolls=[FuturesRollPayload(date=str(roll.date.date()),
                                              expiry=str(roll.expiry.date()),
                                              contracts=roll.contracts, price=roll.price,
                                              multiplier=roll.multiplier,
                                              notional=roll.notional, cost=roll.cost,
                                              dividend_yield=roll.dividend_yield)
                           for roll in result.futures_rolls])

    return payloads


def _on(series: pd.Series,
        days: pd.Index) -> SeriesPayload:
    """A series on the run's days, 0 on a day before it starts."""
    return SeriesPayload.from_series(series.reindex(days).fillna(0.0))
