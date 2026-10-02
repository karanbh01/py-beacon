# src/beacon/portfolio/cash_flows.py
"""
Cash a portfolio receives or pays that is not a trade.

Interest on cash, and dividends once a backtest receives them. A trade moves
cash and a holding together and is a `Transaction`; a cash flow moves cash
alone, so it is recorded on its own list rather than stretched into a trade.
"""
# BN-276: interest on cash is the first; dividends (BN-263) are the second.
from dataclasses import dataclass

import pandas as pd

INTEREST = "INTEREST"
DIVIDEND = "DIVIDEND"


@dataclass(frozen=True)
class CashFlow:
    """One amount of cash received (positive) or paid (negative).

    Attributes:
        date: When the cash moved.
        amount: How much, in the portfolio's currency.
        kind: What it was: ``"INTEREST"`` or ``"DIVIDEND"``.
        asset_id: The holding it came from, for a dividend. None for interest.
    """
    date: pd.Timestamp
    amount: float
    kind: str
    asset_id: str | None = None
