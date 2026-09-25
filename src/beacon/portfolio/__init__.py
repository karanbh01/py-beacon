# src/beacon/portfolio/__init__.py
"""
Portfolio accounting: holdings, cash and the transactions that change them.

`Portfolio` is a ledger of `Holding`s keyed by asset identifier, with a cash
balance and a list of `Transaction` records. It is marked with a mapping of
prices, reports its value and weights, accepts trade instructions, and keeps
a dated record of its positions, cash and NAV. Its accounting needs no asset
objects or data source. `ReportGenerator` writes holdings and performance
reports to Excel (this needs the ``excel`` extra).
"""
from .base import Holding, Portfolio, Transaction
from .reporting import ReportGenerator

__all__ = [
    "Holding",
    "Portfolio",
    "ReportGenerator",
    "Transaction",
]
