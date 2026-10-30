# src/beacon/optimise/inputs.py
"""
What an optimisation solves with, from the names' prices alone: their
expected returns and their factor exposures.

* **Expected returns**: the historical mean, annualised. A poor forecast, and
  the honest one: it is what the data supports. A caller with a real forecast
  should supply it.
* **Factor exposures**: the factors derivable from price and share count,
  z-scored across the names: size (log market capitalisation), momentum
  (trailing return, excluding the most recent month) and volatility (the
  standard deviation of returns, annualised). Value and quality are absent
  rather than approximated without book values.
"""
# BN-401: moved from the server's optimisation views, which now call these.
import numpy as np
import pandas as pd

from ..data.fetcher import DataFetcher
from ..risk.model import z_scores

# Trading days per year, for annualising the estimates.
PERIODS_PER_YEAR = 252

# Momentum skips the most recent month, the conventional construction: the last
# few weeks of a price series carry short-term reversal, which is a different
# effect pointing the other way.
MOMENTUM_LOOKBACK = 252
MOMENTUM_SKIP = 21


def expected_returns_from(prices: pd.DataFrame) -> dict[str, float]:
    """Annualised historical mean returns.

    A poor forecast, and deliberately the honest one: it is what the data
    supports. Anything more sophisticated invented here would look like a view
    the server does not have.
    """
    daily = prices.pct_change().dropna(how="all").mean()

    return {str(name): float(value) * PERIODS_PER_YEAR
            for name, value in daily.items()}


def factor_exposures(prices: pd.DataFrame,
                     fetcher: DataFetcher,
                     as_of: pd.Timestamp) -> pd.DataFrame:
    """Size, momentum and volatility loadings, standardised.

    Args:
        prices: Constituent prices.
        fetcher: For share counts.
        as_of: Date the loadings are measured at.

    Returns:
        pd.DataFrame: z-scored exposures, names on the index.
    """
    returns = prices.pct_change()

    raw = pd.DataFrame({
        "size": _size(prices, fetcher, as_of),
        "momentum": _momentum(prices),
        "volatility": returns.std() * np.sqrt(PERIODS_PER_YEAR),
    })

    return z_scores(raw.dropna(how="all").fillna(0.0))


def _size(prices: pd.DataFrame,
          fetcher: DataFetcher,
          as_of: pd.Timestamp) -> pd.Series:
    """Log market capitalisation, falling back to log price.

    The log matters: market caps span orders of magnitude, and a raw one would
    make the largest name dominate a z-score by construction rather than by
    being unusual.
    """
    values = {}

    for name in prices.columns:
        shares = fetcher.fetch_shares_outstanding(str(name), as_of)
        last = float(prices[name].dropna().iloc[-1])
        values[name] = np.log(last * shares) if shares else np.log(last)

    return pd.Series(values)


def _momentum(prices: pd.DataFrame) -> pd.Series:
    """Trailing return, skipping the most recent month.

    The skip is the conventional construction: the last few weeks carry
    short-term reversal, which is a different effect pointing the other way,
    and folding it in muddies both.
    """
    if len(prices) <= MOMENTUM_SKIP + 1:
        return pd.Series(0.0, index=prices.columns)

    recent = prices.iloc[-(MOMENTUM_SKIP + 1)]
    lookback = min(MOMENTUM_LOOKBACK, len(prices) - 1)
    earlier = prices.iloc[-lookback]

    return (recent / earlier - 1.0).astype(float)
