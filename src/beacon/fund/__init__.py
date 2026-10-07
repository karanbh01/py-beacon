# src/beacon/fund/__init__.py
"""
Fund products, and the older funds that track an index.

`Fund` is a fund product: a name, a currency, share classes and documents,
with one strategy and one vehicle (see `beacon.backtest.presets`), whose
`backtest()` runs a `Backtest` of it.

`IndexFund` and its exchange-traded subclass `ETF` are deprecated and will be
removed in 0.6.0; use a `Fund` with a vehicle instead.

An `IndexFund` holds no trading logic of its own: it runs a `Backtest` of its
target index, seeded with its portfolio's cash, and reports NAV from that run
after deducting a daily-accrued management fee. An `ETF` adds a ticker, a
creation unit size, a simulated market price and tracking-performance
reporting.
"""
from .base import IndexFund
from .etf import ETF
from .fund import ACCUMULATING, DISTRIBUTING, Fund, ShareClass

__all__ = [
    "ACCUMULATING",
    "DISTRIBUTING",
    "ETF",
    "Fund",
    "IndexFund",
    "ShareClass",
]
