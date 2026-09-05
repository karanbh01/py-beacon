# src/beacon/backtest/__init__.py
"""
The __init__.py for the 'backtest' module.

This module provides an engine for backtesting index methodologies
and ETF tracking strategies.
"""
from ..portfolio.base import TradeInstruction
from .asset_view import BacktestAssetView
from .engine import BacktestEngine
from .main import Backtest
from .result import BacktestResult, UnfilledOrder
from .rules import BacktestModifier, DriftThresholdModifier

__all__ = [
    "Backtest",
    "BacktestAssetView",
    "BacktestEngine",
    "BacktestModifier",
    "BacktestResult",
    "DriftThresholdModifier",
    "TradeInstruction",
    "UnfilledOrder",
]
