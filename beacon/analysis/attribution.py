# beacon/analysis/attribution.py
"""
Module for performance attribution analysis.
Initially, this will contain basic attribution methods.
More complex models like Brinson or factor models are future scope.
"""
import pandas as pd
from typing import Dict

class Attribution:
    """
    A class for handling performance attribution.
    Initially a stub, to be expanded with more complex models later.
    """
    def __init__(self) -> None:
        """Initializes the Attribution class."""
        pass

    def simple_performance_attribution(self, portfolio_returns: pd.Series, benchmark_returns: pd.Series) -> Dict[str, float]:
        """
        Provides a basic return difference between portfolio and benchmark.

        Args:
            portfolio_returns: A pandas Series of portfolio periodic returns.
            benchmark_returns: A pandas Series of benchmark periodic returns.
                               Must be of the same frequency and length as portfolio_returns.

        Returns:
            A dictionary containing the total portfolio return, total benchmark return,
            and the difference (active return).

        Raises:
            ValueError: If the input Series are not of the same length or are empty.
        """
        if not isinstance(portfolio_returns, pd.Series) or not isinstance(benchmark_returns, pd.Series):
            raise TypeError("portfolio_returns and benchmark_returns must be pandas Series.")
        if len(portfolio_returns) != len(benchmark_returns):
            raise ValueError("Portfolio returns and benchmark returns Series must be of the same length.")
        if portfolio_returns.empty:
            raise ValueError("Input Series cannot be empty.")

        total_portfolio_return = (1 + portfolio_returns).prod() - 1
        total_benchmark_return = (1 + benchmark_returns).prod() - 1
        active_return = total_portfolio_return - total_benchmark_return

        return {
            "total_portfolio_return": float(total_portfolio_return),
            "total_benchmark_return": float(total_benchmark_return),
            "active_return": float(active_return)
        }

# Expose function directly if preferred
def simple_performance_attribution(portfolio_returns: pd.Series, benchmark_returns: pd.Series) -> Dict[str, float]:
    """
    Provides a basic return difference between portfolio and benchmark.

    Args:
        portfolio_returns: A pandas Series of portfolio periodic returns.
        benchmark_returns: A pandas Series of benchmark periodic returns.
                           Must be of the same frequency and length as portfolio_returns.

    Returns:
        A dictionary containing the total portfolio return, total benchmark return,
        and the difference (active return).
    """
    attr = Attribution()
    return attr.simple_performance_attribution(portfolio_returns, benchmark_returns)