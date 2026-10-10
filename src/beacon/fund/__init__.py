# src/beacon/fund/__init__.py
"""
Fund products.

`Fund` is a fund product: a name, a currency, share classes and documents,
with one strategy and one vehicle (see `beacon.backtest.presets`), whose
`backtest()` runs a `Backtest` of it. An exchange-traded fund is a `Fund`
with an exchange-traded vehicle, such as `ucits_etf()` or `us_etf()`.
"""
# IndexFund and ETF, deprecated in 0.5.0 (BN-268), were removed in 0.6.0
# (BN-297).
from .fund import ACCUMULATING, DISTRIBUTING, Fund, ShareClass

__all__ = [
    "ACCUMULATING",
    "DISTRIBUTING",
    "Fund",
    "ShareClass",
]
