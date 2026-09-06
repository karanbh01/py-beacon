# src/beacon/server/benchmarks.py
"""
Resolving a benchmark reference to a level series.

A benchmark arrives as a reference, not data: either the id of a stored index
definition — which has to be calculated before it can be compared against — or
a market-data identifier whose price series is the benchmark directly.

A stored index of *either* face qualifies (BN-169): a benchmark needs only a
level series to compare against, and an optimised index has one exactly as a
rule-driven one does. It is calculated through whichever path its definition
requires, which is the only difference the two faces make here.
"""
import logging

import pandas as pd

from ..data.fetcher import DataFetcher
from ..exceptions import DataNotFoundError
from ..index.calculation import IndexCalculator
from ..index.derived import (
    AnyIndexDefinition,
    OptimisedIndexDefinition,
    calculate_derived_index,
)
from ..index.result import IndexResult
from .definitions import build_definition
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
    """Calculate a stored index of either face and return its level series.

    Calculating a second index is real work, which is why a benchmark can only
    be requested inside a job rather than on a synchronous read.
    """
    document = index_store.read(reference.id)
    if document is None:
        raise DataNotFoundError(f"benchmark index '{reference.id}'",
                                source="DocumentStore")

    definition = build_definition(IndexDocument.model_validate(document),
                                  index_store)
    logger.info(f"Calculating benchmark index '{reference.id}' from {start} to {end}.")

    result = _calculated(definition, fetcher, start, end)

    if result.index_levels.empty:
        raise DataNotFoundError(
            f"benchmark index '{reference.id}' produced no levels between "
            f"{start} and {end}",
            source="IndexCalculator")

    return result.index_levels


def _calculated(definition: AnyIndexDefinition,
                fetcher: DataFetcher,
                start: str,
                end: str) -> IndexResult:
    """Run whichever calculation the definition's face requires.

    The sibling design (BN-167) is why this is a dispatch rather than one
    call: `IndexCalculator` cannot receive a derivation — deliberately, so it
    can never be handed one by accident — and a derivation solves its source's
    weights rather than applying rules of its own.
    """
    if isinstance(definition, OptimisedIndexDefinition):
        return calculate_derived_index(definition, fetcher,
                                       start_date=start, end_date=end)

    return IndexCalculator(definition, fetcher).run(start_date=start,
                                                    end_date=end)


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
