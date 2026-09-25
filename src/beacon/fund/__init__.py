# src/beacon/fund/__init__.py
"""
Funds that track an index: `IndexFund` and its exchange-traded subclass `ETF`.

An `IndexFund` holds no trading logic of its own: it runs a `Backtest` of its
target index, seeded with its portfolio's cash, and reports NAV from that run
after deducting a daily-accrued management fee. An `ETF` adds a ticker, a
creation unit size, a simulated market price and tracking-performance
reporting.
"""
from .base import IndexFund
from .etf import ETF

__all__ = [
    "ETF",
    "IndexFund",
]
