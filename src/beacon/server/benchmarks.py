# src/beacon/server/benchmarks.py
"""
Resolving a benchmark reference to a level series.

A benchmark arrives as a reference, not data: either the id of a stored index
definition — which has to be calculated before it can be compared against — or
a market-data identifier whose price series is the benchmark directly.
"""
import logging

import pandas as pd

from ..data.fetcher import DataFetcher
from ..exceptions import DataNotFoundError
from ..index.calculation import IndexCalculator
from .definitions import build_index_definition
from .schemas import BENCHMARK_INDEX, BenchmarkRef, IndexDocument
from .store import DocumentStore

logger = logging.getLogger(__name__)


def resolve_benchmark(reference: BenchmarkRef,
                      fetcher: DataFetcher,
                      index_store: DocumentStore,
                      start: str,
                      end: str) -> pd.Series:
    """Turn a benchmark reference into a date-indexed level series.

    Args:
        reference: What to compare against.
        fetcher: Data source.
        index_store: Where stored index definitions live.
        start: Start date, YYYY-MM-DD.
        end: End date, YYYY-MM-DD.

    Returns:
        pd.Series: Levels indexed by date. Not rebased — the caller decides,
        and returns are scale-invariant anyway.

    Raises:
        DataNotFoundError: If the referenced index or identifier does not
            exist, or carries no data over the window.
    """
    if reference.kind == BENCHMARK_INDEX:
        return _index_levels(reference, fetcher, index_store, start, end)

    return _identifier_levels(reference, fetcher, start, end)


def _index_levels(reference: BenchmarkRef,
                  fetcher: DataFetcher,
                  index_store: DocumentStore,
                  start: str,
                  end: str) -> pd.Series:
    """Calculate a stored index and return its level series.

    Calculating a second index is real work, which is why a benchmark can only
    be requested inside a job rather than on a synchronous read.
    """
    document = index_store.read(reference.id)
    if document is None:
        raise DataNotFoundError(f"benchmark index '{reference.id}'",
                                source="DocumentStore")

    definition = build_index_definition(IndexDocument.model_validate(document))
    logger.info(f"Calculating benchmark index '{reference.id}' from {start} to {end}.")

    result = IndexCalculator(definition, fetcher).run(start_date=start, end_date=end)

    if result.index_levels.empty:
        raise DataNotFoundError(
            f"benchmark index '{reference.id}' produced no levels between "
            f"{start} and {end}",
            source="IndexCalculator")

    return result.index_levels


def _identifier_levels(reference: BenchmarkRef,
                       fetcher: DataFetcher,
                       start: str,
                       end: str) -> pd.Series:
    """Read a market-data series as the benchmark's levels."""
    frame = fetcher.fetch_market_data(reference.id, start, end,
                                      [reference.price_column])

    if frame.empty:
        raise DataNotFoundError(
            f"benchmark identifier '{reference.id}' between {start} and {end}",
            source="MarketData")

    if reference.price_column not in frame.columns:
        raise DataNotFoundError(
            f"column '{reference.price_column}' for benchmark identifier "
            f"'{reference.id}'",
            source="MarketData")

    levels = frame[reference.price_column].dropna()

    if levels.empty:
        raise DataNotFoundError(
            f"benchmark identifier '{reference.id}' has no values in column "
            f"'{reference.price_column}' between {start} and {end}",
            source="MarketData")

    return levels
