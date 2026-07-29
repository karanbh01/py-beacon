# src/beacon/backtest/__init__.py
"""
The __init__.py for the 'backtest' module.

This module provides an engine for backtesting index methodologies
and ETF tracking strategies.
"""
from .asset_view import BacktestAssetView
from .engine import BacktestEngine, TradeInstruction
from .result import BacktestResult
from .rules import BacktestModifier, DriftThresholdModifier

__all__ = [
    "BacktestAssetView",
    "BacktestEngine",
    "BacktestModifier",
    "BacktestResult",
    "DriftThresholdModifier",
    "TradeInstruction",
]
