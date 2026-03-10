# beacon/backtest/__init__.py
"""
The __init__.py for the 'backtest' module.

This module provides an engine for backtesting index methodologies
and ETF tracking strategies.
"""
from .rules import BacktestModifier, DriftThresholdModifier
from .engine import BacktestEngine, TradeInstruction
from .result import BacktestResult
from .asset_view import BacktestAssetView

__all__ = [
    "BacktestModifier",
    "DriftThresholdModifier",
    "BacktestEngine",
    "TradeInstruction",
    "BacktestResult",
    "BacktestAssetView",
]