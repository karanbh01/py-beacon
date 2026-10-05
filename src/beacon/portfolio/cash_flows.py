# src/beacon/portfolio/cash_flows.py
"""
Cash a portfolio receives or pays that is not a trade.

Interest on cash, dividends, money investors put in or take out, and fees.
A trade moves
cash and a holding together and is a `Transaction`; a cash flow moves cash
alone, so it is recorded on its own list rather than stretched into a trade.
"""
# BN-276: interest on cash is the first; dividends (BN-263) are the second.
from dataclasses import dataclass

import pandas as pd

INTEREST = "INTEREST"
DIVIDEND = "DIVIDEND"
# Cash paid out of the book to its investors, recorded as a negative amount.
DISTRIBUTION = "DISTRIBUTION"
# Money investors put in (positive) and take out (negative), BN-267.
SUBSCRIPTION = "SUBSCRIPTION"
REDEMPTION = "REDEMPTION"
# The fund's own charges paid from cash, recorded as a negative amount.
FEE = "FEE"


@dataclass(frozen=True)
class CashFlow:
    """One amount of cash received (positive) or paid (negative).

    Attributes:
        date: When the cash moved.
        amount: How much, in the portfolio's currency.
        kind: What it was: ``"INTEREST"``, ``"DIVIDEND"`` (received from a
            holding), ``"DISTRIBUTION"`` (paid out to investors),
            ``"SUBSCRIPTION"`` or ``"REDEMPTION"`` (money investors put in
            or took out) or ``"FEE"`` (the fund's charges).
        asset_id: The holding it came from, for a dividend. None for interest.
    """
    date: pd.Timestamp
    amount: float
    kind: str
    asset_id: str | None = None
