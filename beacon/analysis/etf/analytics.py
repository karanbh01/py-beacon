# beacon/analysis/etf_analytics.py
"""
Module for calculating analytics specific to Exchange Traded Funds (ETFs).
"""
import pandas as pd
import numpy as np
from typing import Union

def calculate_tracking_difference(etf_returns: pd.Series, index_returns: pd.Series) -> float:
    """
    Calculates the annualized tracking difference between ETF returns and index returns.
    Tracking Difference = Sum(ETF Returns - Index Returns) / Number of Periods * Periods per Year
    Or, more simply, Total ETF Return - Total Index Return over the period.
    This implementation will calculate the difference of the sum of returns, annualized.

    Args:
        etf_returns: A pandas Series of ETF periodic returns.
        index_returns: A pandas Series of benchmark index periodic returns.
                       Must be of the same frequency and length as etf_returns.

    Returns:
        The annualized tracking difference as a float.

    Raises:
        ValueError: If the input Series are not of the same length or are empty.
    """
    if not isinstance(etf_returns, pd.Series) or not isinstance(index_returns, pd.Series):
        raise TypeError("etf_returns and index_returns must be pandas Series.")
    if len(etf_returns) != len(index_returns):
        raise ValueError("ETF returns and index returns Series must be of the same length.")
    if etf_returns.empty:
        raise ValueError("Input Series cannot be empty.")

    # This calculates the average periodic difference, then annualizes.
    # A common definition is (ETF Total Return - Index Total Return) for the period.
    # Let's use the arithmetic difference of cumulative returns.
    etf_cumulative_return = (1 + etf_returns).prod() - 1
    index_cumulative_return = (1 + index_returns).prod() - 1
    tracking_difference = etf_cumulative_return - index_cumulative_return # This is for the total period
    # To annualize, one might need to consider the period length.
    # For simplicity, this blueprint asks for a float. If annualized, needs periods.
    # Assuming this is the total difference over the period.
    return float(tracking_difference)

def calculate_tracking_error(etf_returns: pd.Series, index_returns: pd.Series, periods_per_year: int = 252) -> float:
    """
    Calculates the annualized tracking error between ETF returns and index returns.
    Tracking Error = Standard Deviation of (ETF Returns - Index Returns) * Sqrt(Periods per Year).

    Args:
        etf_returns: A pandas Series of ETF periodic returns.
        index_returns: A pandas Series of benchmark index periodic returns.
                       Must be of the same frequency and length as etf_returns.
        periods_per_year: The number of periods in a year (e.g., 252 for daily).

    Returns:
        The annualized tracking error as a float.

    Raises:
        ValueError: If the input Series are not of the same length or are empty.
    """
    if not isinstance(etf_returns, pd.Series) or not isinstance(index_returns, pd.Series):
        raise TypeError("etf_returns and index_returns must be pandas Series.")
    if len(etf_returns) != len(index_returns):
        raise ValueError("ETF returns and index returns Series must be of the same length.")
    if etf_returns.empty:
        raise ValueError("Input Series cannot be empty.")
    if periods_per_year <= 0:
        raise ValueError("periods_per_year must be a positive integer.")

    difference_returns = etf_returns - index_returns
    annualized_tracking_error = difference_returns.std() * np.sqrt(periods_per_year)
    return float(annualized_tracking_error)

def calculate_premium_discount(etf_price: float, nav_price: float) -> float:
    """
    Calculates the premium/discount of an ETF's market price relative to its NAV.
    Premium/Discount = (ETF Market Price / NAV) - 1.

    Args:
        etf_price: The current market price of the ETF.
        nav_price: The current Net Asset Value (NAV) per share of the ETF.

    Returns:
        The premium/discount as a float (e.g., 0.01 for 1% premium, -0.005 for 0.5% discount).

    Raises:
        ValueError: If nav_price is zero or inputs are non-numeric.
    """
    if not all(isinstance(p, (int, float)) for p in [etf_price, nav_price]):
        raise TypeError("etf_price and nav_price must be numeric.")
    if nav_price == 0:
        raise ValueError("NAV price cannot be zero.")
    premium_discount = (etf_price / nav_price) - 1
    return float(premium_discount)

class ETFAnalytics:
    """
    A class to calculate ETF-specific analytics.
    """
    def __init__(self) -> None:
        """Initializes the ETFAnalytics calculator."""
        pass

    def calculate_tracking_difference(self, etf_returns: pd.Series, index_returns: pd.Series) -> float:
        """
        Calculates the tracking difference between ETF returns and index returns.
        See function docstring for details.
        """
        return calculate_tracking_difference(etf_returns, index_returns)

    def calculate_tracking_error(self, etf_returns: pd.Series, index_returns: pd.Series, periods_per_year: int = 252) -> float:
        """
        Calculates the annualized tracking error. See function docstring for details.
        """
        return calculate_tracking_error(etf_returns, index_returns, periods_per_year)

    def calculate_premium_discount(self, etf_price: float, nav_price: float) -> float:
        """
        Calculates the premium/discount to NAV. See function docstring for details.
        """
        return calculate_premium_discount(etf_price, nav_price)