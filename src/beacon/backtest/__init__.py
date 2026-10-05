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

An `Implementation` says how a strategy is carried out at a fund's size: its
screens (`MarketCapScreen`, `LiquidityScreen`, `MinimumPriceScreen`,
`ListingAgeScreen`, `ExclusionScreen`, `ExpressionScreen`) decide which names
may be held at each rebalance, its caps (`OwnershipCap`, `LiquidityCap`,
`WeightCap`) and `MinimumPosition` how much of each, and its redistribution
rule where the weight of the rest goes. Each rebalance's stages are recorded as a `RebalanceStep`.
"""
from ..index.result import PriceGap
from ..portfolio.base import TradeInstruction
from .asset_view import BacktestAssetView
from .capacity import (
    CapacityCap,
    LiquidityCap,
    MinimumPosition,
    OwnershipCap,
    WeightCap,
)
from .costs import ExecutionLimit, MarketImpact
from .engine import BacktestEngine
from .flows import (
    DatedFlows,
    FlowContext,
    FlowRecord,
    Flows,
    PerformanceChasingFlows,
    PeriodicFlows,
    RandomFlows,
)
from .implementation import Implementation, RebalanceStep
from .main import Backtest
from .result import BacktestResult, RebalancePricing, UnfilledOrder
from .rules import BacktestModifier, DriftThresholdModifier
from .screens import (
    ExclusionScreen,
    ExpressionScreen,
    LiquidityScreen,
    ListingAgeScreen,
    MarketCapScreen,
    MinimumPriceScreen,
    Screen,
    ScreenContext,
    ThresholdScreen,
)
from .vehicle import Vehicle

__all__ = [
    "Backtest",
    "BacktestAssetView",
    "BacktestEngine",
    "BacktestModifier",
    "BacktestResult",
    "CapacityCap",
    "DatedFlows",
    "DriftThresholdModifier",
    "ExclusionScreen",
    "ExecutionLimit",
    "ExpressionScreen",
    "FlowContext",
    "FlowRecord",
    "Flows",
    "Implementation",
    "LiquidityCap",
    "LiquidityScreen",
    "ListingAgeScreen",
    "MarketCapScreen",
    "MarketImpact",
    "MinimumPosition",
    "MinimumPriceScreen",
    "OwnershipCap",
    "PerformanceChasingFlows",
    "PeriodicFlows",
    "PriceGap",
    "RandomFlows",
    "RebalancePricing",
    "RebalanceStep",
    "Screen",
    "ScreenContext",
    "ThresholdScreen",
    "TradeInstruction",
    "UnfilledOrder",
    "Vehicle",
    "WeightCap",
]
