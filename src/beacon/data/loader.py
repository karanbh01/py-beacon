"""
Data loader that reads file paths from an Environment and returns a DataFetcher.
"""

from beacon.environment.config import Environment
from .base import MarketData, ReferenceData
from .fetcher import DataFetcher


def load_data(env: Environment) -> DataFetcher:
    """Read data files from the environment config and return a DataFetcher.

    For each dataset, a DataFrame is checked first; if not provided, the
    file path is used instead. Raises ValueError if no market data is
    available from either source.
    """
    if env.data_source.MARKET_DATA is not None:
        market = MarketData.from_dataframe(env.data_source.MARKET_DATA,
                                           date_format=env.data.DATE_FORMAT)
    elif env.data_source.MARKET_DATA_PATH is not None:
        market = MarketData(env.data_source.MARKET_DATA_PATH,
                            date_format=env.data.DATE_FORMAT)
    else:
        raise ValueError(
            "No market data provided. Set MARKET_DATA or "
            "MARKET_DATA_PATH on env.data_source."
        )

    reference = None
    if env.data_source.REFERENCE_DATA is not None:
        reference = ReferenceData.from_dataframe(env.data_source.REFERENCE_DATA)
    elif env.data_source.REFERENCE_DATA_PATH is not None:
        reference = ReferenceData(env.data_source.REFERENCE_DATA_PATH)

    return DataFetcher(market, reference)
