# beacon/backtest/__init__.py
"""
The __init__.py for the 'backtest' module.

This module provides an engine for backtesting index methodologies
and ETF tracking strategies.
"""
from .rules import BacktestRule, RebalanceRule
from .engine import BacktestEngine

__all__ = [
    "BacktestRule",
    "RebalanceRule",
    "BacktestEngine",
]