"""
Data loader that reads file paths from an Environment and returns a DataFetcher.
"""

from beacon.environment.config import Environment
from .base import MarketData, ReferenceData
from .fetcher import DataFetcher


def load_data(env: Environment) -> DataFetcher:
    """Read data files from the environment config and return a DataFetcher.

    Raises ValueError if market_data_path is not set.
    """
    if env.data.market_data_path is None:
        raise ValueError(
            "market_data_path is not set. "
            "Use env.set_environment(market_data_path=...) first."
        )

    market = MarketData(
        env.data.market_data_path,
        date_format=env.data.date_format,
    )

    reference = None
    if env.data.reference_data_path is not None:
        reference = ReferenceData(env.data.reference_data_path)

    return DataFetcher(market, reference)
