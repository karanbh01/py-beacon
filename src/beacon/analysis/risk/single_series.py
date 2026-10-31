# src/beacon/analysis/risk/single_series.py
"""
Risk metrics for a price or return series: annualised volatility, Sharpe ratio
and maximum drawdown.

Deprecated, and to be removed in 0.11.0: each figure is in
`beacon.analysis.performance.formulas` (`volatility`, `sharpe_ratio`,
`max_drawdown`), and a run's are its analytics' `headline()`.
"""
# BN-402: deprecated in 0.10.0. The functions keep working, with a warning
# naming the formula that replaces each.
import warnings

import numpy as np
import pandas as pd

from ..performance import formulas

# Where each deprecated name's replacement is.
FORMULAS = "beacon.analysis.performance.formulas"


def _deprecated(name: str,
                replacement: str) -> None:
    """Warn that *name* is deprecated in favour of *replacement*."""
    warnings.warn(f"{name} is deprecated and will be removed in 0.11.0; use "
                  f"{replacement}.", DeprecationWarning, stacklevel=3)


def calculate_volatility(price_series: pd.Series,
                         window: int = 252) -> float:
    """
    Calculates annualized volatility from a price series.

    Deprecated: use `beacon.analysis.performance.formulas.volatility` on the
    returns, ``volatility(prices.pct_change().dropna(), periods_per_year)``.

    The sample standard deviation of the simple period returns, multiplied by
    ``sqrt(window)``. Despite its name, *window* is the annualisation factor,
    not a rolling window: the whole series is used.

    Args:
        price_series: A pandas Series of prices.
        window: The number of trading periods in a year (e.g., 252 for daily).

    Returns:
        The annualized volatility as a float.

    Raises:
        TypeError: If price_series is not a pandas Series.
        ValueError: If price_series is empty or contains non-numeric data, or
            window is not positive.
    """
    _deprecated("calculate_volatility",
                f"{FORMULAS}.volatility(prices.pct_change().dropna(), periods_per_year)")

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

def calculate_sharpe_ratio(returns: pd.Series,
                           risk_free_rate: float,
                           periods_per_year: int = 252) -> float:
    """
    Calculates the annualized Sharpe Ratio.

    Deprecated: use `beacon.analysis.performance.formulas.sharpe_ratio`,
    ``sharpe_ratio(returns, periods_per_year, risk_free_rate)``.

    The mean return in excess of the risk-free rate over its sample standard
    deviation, multiplied by ``sqrt(periods_per_year)``: the Sharpe ratio a
    backtest reports, `beacon.analysis.performance.formulas.sharpe_ratio`.
    The annual risk-free rate is taken as the period rate that compounds to
    it. 0 when the returns do not vary.

    Args:
        returns: A pandas Series of periodic returns.
        risk_free_rate: The annualized risk-free rate.
        periods_per_year: The number of return periods in a year (e.g., 252 for daily,
                          12 for monthly).

    Returns:
        The annualized Sharpe Ratio as a float.

    Raises:
        TypeError: If returns is not a pandas Series or risk_free_rate is not
            a number.
        ValueError: If returns is empty or non-numeric, or periods_per_year
            is not positive.
    """
    _deprecated("calculate_sharpe_ratio",
                f"{FORMULAS}.sharpe_ratio(returns, periods_per_year, risk_free_rate)")

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

    # BN-405: one Sharpe ratio. This divided the rate by the periods and gave
    # NaN or infinity for returns that did not vary.
    return formulas.sharpe_ratio(returns, periods_per_year, float(risk_free_rate))

def calculate_max_drawdown(price_series: pd.Series) -> float:
    """
    Calculates the maximum drawdown from a price series.

    Deprecated: use `beacon.analysis.performance.formulas.max_drawdown`,
    ``max_drawdown(prices)``.

    The largest fall from a running peak, as a fraction of that peak.

    Args:
        price_series: A pandas Series of prices.

    Returns:
        The maximum drawdown as a non-positive float (e.g., -0.2 for a 20%
        drawdown, 0.0 when the price never falls below a previous peak).

    Raises:
        TypeError: If price_series is not a pandas Series.
        ValueError: If price_series is empty or contains non-numeric data.
    """
    _deprecated("calculate_max_drawdown", f"{FORMULAS}.max_drawdown(prices)")

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
    Risk metrics as methods. Each method calls the module-level function of
    the same name and returns its result unchanged.

    Deprecated: use the functions of `beacon.analysis.performance.formulas`.
    """
    def __init__(self) -> None:
        """Initializes the RiskMetricsCalculator."""
        _deprecated("RiskMetricsCalculator", f"the functions of {FORMULAS}")

    def calculate_volatility(self,
                             price_series: pd.Series,
                             window: int = 252) -> float:
        """
        Calculates annualized volatility from a price series.

        See :func:`calculate_volatility`.

        Args:
            price_series: A pandas Series of prices.
            window: The number of trading periods in a year (e.g., 252 for daily).

        Returns:
            The annualized volatility as a float.
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)

            return calculate_volatility(price_series, window)

    def calculate_sharpe_ratio(self,
                               returns: pd.Series,
                               risk_free_rate: float,
                               periods_per_year: int = 252) -> float:
        """
        Calculates the annualized Sharpe Ratio.

        Args:
            returns: A pandas Series of periodic returns.
            risk_free_rate: The annualized risk-free rate.
            periods_per_year: The number of return periods in a year (e.g., 252 for daily,
                              12 for monthly).

        See :func:`calculate_sharpe_ratio`.

        Returns:
            The annualized Sharpe Ratio as a float.
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)

            return calculate_sharpe_ratio(returns, risk_free_rate, periods_per_year)

    def calculate_max_drawdown(self,
                               price_series: pd.Series) -> float:
        """
        Calculates the maximum drawdown from a price series.

        See :func:`calculate_max_drawdown`.

        Args:
            price_series: A pandas Series of prices.

        Returns:
            The maximum drawdown as a non-positive float (e.g., -0.2 for a
            20% drawdown).
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)

            return calculate_max_drawdown(price_series)
