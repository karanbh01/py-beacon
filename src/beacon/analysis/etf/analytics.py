# src/beacon/analysis/etf/analytics.py
"""
Analytics specific to Exchange Traded Funds (ETFs): tracking difference,
tracking error, and the premium or discount of market price to NAV.

Each is available as a plain function and as a method of `ETFAnalytics`.
"""

import numpy as np
import pandas as pd


def calculate_tracking_difference(etf_returns: pd.Series,
                                  index_returns: pd.Series) -> float:
    """
    Calculates the tracking difference between ETF returns and index returns.

    Tracking difference is the ETF's cumulative return minus the index's
    cumulative return over the whole period, where each cumulative return
    compounds its periodic returns: ``prod(1 + r) - 1``. It is not
    annualised. Each Series is compounded on its own, so the two are not
    aligned by date; NaN returns are skipped.

    Args:
        etf_returns: A pandas Series of ETF periodic returns.
        index_returns: A pandas Series of benchmark index periodic returns.
                       Must be of the same frequency and length as etf_returns.

    Returns:
        The tracking difference over the period, as a float (e.g. -0.002
        when the ETF returned 0.2 percentage points less than the index).

    Raises:
        TypeError: If either input is not a pandas Series.
        ValueError: If the input Series are not of the same length or are empty.
    """
    if not isinstance(etf_returns, pd.Series) or not isinstance(index_returns, pd.Series):
        raise TypeError("etf_returns and index_returns must be pandas Series.")
    if len(etf_returns) != len(index_returns):
        raise ValueError("ETF returns and index returns Series must be of the same length.")
    if etf_returns.empty:
        raise ValueError("Input Series cannot be empty.")

    # The arithmetic difference of cumulative returns over the whole period.
    # Annualising it would need the period length, which is not passed in.
    etf_cumulative_return = (1 + etf_returns).prod() - 1
    index_cumulative_return = (1 + index_returns).prod() - 1
    tracking_difference = etf_cumulative_return - index_cumulative_return
    return float(tracking_difference)

def calculate_tracking_error(etf_returns: pd.Series,
                             index_returns: pd.Series,
                             periods_per_year: int = 252) -> float:
    """
    Calculates the annualized tracking error between ETF returns and index returns.

    Tracking error is the sample standard deviation of the periodic return
    differences (ETF minus index), multiplied by ``sqrt(periods_per_year)``.
    The difference is taken after aligning the two Series on their index, so
    they should share the same dates: a date present in only one Series
    produces NaN and is skipped, and fewer than two common dates gives NaN.

    Args:
        etf_returns: A pandas Series of ETF periodic returns.
        index_returns: A pandas Series of benchmark index periodic returns.
                       Must be of the same frequency and length as etf_returns.
        periods_per_year: The number of periods in a year (e.g., 252 for daily).

    Returns:
        The annualized tracking error as a float.

    Raises:
        TypeError: If either input is not a pandas Series.
        ValueError: If the input Series are not of the same length or are
            empty, or *periods_per_year* is not positive.
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

def calculate_premium_discount(etf_price: float,
                               nav_price: float) -> float:
    """
    Calculates the premium/discount of an ETF's market price relative to its NAV.

    Premium/Discount = (ETF Market Price / NAV) - 1.

    Args:
        etf_price: The current market price of the ETF.
        nav_price: The current Net Asset Value (NAV) per share of the ETF.

    Returns:
        The premium/discount as a float (e.g., 0.01 for 1% premium, -0.005 for 0.5% discount).

    Raises:
        TypeError: If either input is not an int or float.
        ValueError: If nav_price is zero.
    """
    if not all(isinstance(p, (int, float)) for p in [etf_price, nav_price]):
        raise TypeError("etf_price and nav_price must be numeric.")
    if nav_price == 0:
        raise ValueError("NAV price cannot be zero.")
    premium_discount = (etf_price / nav_price) - 1
    return float(premium_discount)

class ETFAnalytics:
    """
    ETF analytics as methods. Each method calls the module-level function of
    the same name and returns its result unchanged.
    """
    def __init__(self) -> None:
        """Initializes the ETFAnalytics calculator."""

    def calculate_tracking_difference(self,
                                      etf_returns: pd.Series,
                                      index_returns: pd.Series) -> float:
        """
        Cumulative ETF return minus cumulative index return over the period.

        See :func:`calculate_tracking_difference`.
        """
        return calculate_tracking_difference(etf_returns, index_returns)

    def calculate_tracking_error(self,
                                 etf_returns: pd.Series,
                                 index_returns: pd.Series,
                                 periods_per_year: int = 252) -> float:
        """
        Annualised standard deviation of the ETF-minus-index return differences.

        See :func:`calculate_tracking_error`.
        """
        return calculate_tracking_error(etf_returns, index_returns, periods_per_year)

    def calculate_premium_discount(self,
                                   etf_price: float,
                                   nav_price: float) -> float:
        """
        Market price divided by NAV, minus one.

        See :func:`calculate_premium_discount`.
        """
        return calculate_premium_discount(etf_price, nav_price)
