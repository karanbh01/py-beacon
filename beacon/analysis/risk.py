# beacon/analysis/risk.py
"""
Module for calculating various risk metrics for financial instruments.
"""
import pandas as pd
import numpy as np
from typing import Union

def calculate_volatility(price_series: pd.Series, window: int = 252) -> float:
    """
    Calculates annualized volatility from a price series.

    Args:
        price_series: A pandas Series of prices.
        window: The number of trading periods in a year (e.g., 252 for daily).

    Returns:
        The annualized volatility as a float.

    Raises:
        ValueError: If price_series is empty or contains non-numeric data.
    """
    if not isinstance(price_series, pd.Series):
        raise TypeError("price_series must be a pandas Series.")
    if price_series.empty:
        raise ValueError("Price series cannot be empty.")
    if not pd.api.types.is_numeric_dtype(price_series):
        raise ValueError("Price series must contain numeric data.")
    if window <= 0:
        raise ValueError("Window must be a positive integer.")

    returns = price_series.pct_change().dropna()
    annualized_volatility = returns.std() * np.sqrt(window)
    return float(annualized_volatility)

def calculate_sharpe_ratio(returns: pd.Series, risk_free_rate: float, periods_per_year: int = 252) -> float:
    """
    Calculates the annualized Sharpe Ratio.

    Args:
        returns: A pandas Series of periodic returns.
        risk_free_rate: The annualized risk-free rate.
        periods_per_year: The number of return periods in a year (e.g., 252 for daily, 12 for monthly).

    Returns:
        The annualized Sharpe Ratio as a float.

    Raises:
        ValueError: If inputs are invalid.
    """
    if not isinstance(returns, pd.Series):
        raise TypeError("returns must be a pandas Series.")
    if returns.empty:
        raise ValueError("Returns series cannot be empty.")
    if not pd.api.types.is_numeric_dtype(returns):
        raise ValueError("Returns series must contain numeric data.")
    if not isinstance(risk_free_rate, (int, float)):
        raise TypeError("risk_free_rate must be a number.")
    if periods_per_year <= 0:
        raise ValueError("periods_per_year must be a positive integer.")

    excess_returns = returns - (risk_free_rate / periods_per_year)
    mean_excess_return = excess_returns.mean()
    std_dev_excess_return = excess_returns.std()

    if std_dev_excess_return == 0: # Avoid division by zero
        return np.nan if mean_excess_return == 0 else np.inf * np.sign(mean_excess_return)

    sharpe_ratio = (mean_excess_return / std_dev_excess_return) * np.sqrt(periods_per_year)
    return float(sharpe_ratio)

def calculate_max_drawdown(price_series: pd.Series) -> float:
    """
    Calculates the maximum drawdown from a price series.

    Args:
        price_series: A pandas Series of prices.

    Returns:
        The maximum drawdown as a float (e.g., 0.2 for a 20% drawdown).

    Raises:
        ValueError: If price_series is empty or contains non-numeric data.
    """
    if not isinstance(price_series, pd.Series):
        raise TypeError("price_series must be a pandas Series.")
    if price_series.empty:
        raise ValueError("Price series cannot be empty.")
    if not pd.api.types.is_numeric_dtype(price_series):
        raise ValueError("Price series must contain numeric data.")

    cumulative_max = price_series.cummax()
    drawdown = (price_series - cumulative_max) / cumulative_max
    max_drawdown = drawdown.min()
    return float(max_drawdown) if not pd.isna(max_drawdown) else 0.0


class RiskMetricsCalculator:
    """
    A class to calculate various risk metrics.
    This class can be expanded to hold state or more complex configurations if needed.
    """
    def __init__(self) -> None:
        """Initializes the RiskMetricsCalculator."""
        pass

    def calculate_volatility(self, price_series: pd.Series, window: int = 252) -> float:
        """
        Calculates annualized volatility from a price series.

        Args:
            price_series: A pandas Series of prices.
            window: The number of trading periods in a year (e.g., 252 for daily).

        Returns:
            The annualized volatility as a float.
        """
        return calculate_volatility(price_series, window)

    def calculate_sharpe_ratio(self, returns: pd.Series, risk_free_rate: float, periods_per_year: int = 252) -> float:
        """
        Calculates the annualized Sharpe Ratio.

        Args:
            returns: A pandas Series of periodic returns.
            risk_free_rate: The annualized risk-free rate.
            periods_per_year: The number of return periods in a year (e.g., 252 for daily, 12 for monthly).

        Returns:
            The annualized Sharpe Ratio as a float.
        """
        return calculate_sharpe_ratio(returns, risk_free_rate, periods_per_year)

    def calculate_max_drawdown(self, price_series: pd.Series) -> float:
        """
        Calculates the maximum drawdown from a price series.

        Args:
            price_series: A pandas Series of prices.

        Returns:
            The maximum drawdown as a float (e.g., 0.2 for a 20% drawdown).
        """
        return calculate_max_drawdown(price_series)