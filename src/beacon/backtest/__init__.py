# src/beacon/backtest/__init__.py
"""
Backtesting: simulate trading a portfolio towards a schedule of target weights.

`Backtest` is the high-level entry point: construct it with capital and cost
assumptions, then run it with an index definition and a date range. It
calculates the index (reusing a cached calculation when it can) and simulates
a portfolio tracking it. `BacktestEngine` is the engine underneath, which
trades towards each rebalance's target weights from an `IndexResult` (sells
before buys, costs in basis points of notional) and returns a
`BacktestResult` holding the portfolio's books, unfilled orders, price gaps
and tracking against the target index. `BacktestModifier` subclasses such as
`DriftThresholdModifier` can skip a rebalance or adjust its trades, and
`BacktestAssetView` gives one asset's story through a backtest.
"""
from ..portfolio.base import TradeInstruction
from .asset_view import BacktestAssetView
from .engine import BacktestEngine
from .main import Backtest
from .result import BacktestResult, PriceGap, RebalancePricing, UnfilledOrder
from .rules import BacktestModifier, DriftThresholdModifier

__all__ = [
    "Backtest",
    "BacktestAssetView",
    "BacktestEngine",
    "BacktestModifier",
    "BacktestResult",
    "DriftThresholdModifier",
    "PriceGap",
    "RebalancePricing",
    "TradeInstruction",
    "UnfilledOrder",
]
