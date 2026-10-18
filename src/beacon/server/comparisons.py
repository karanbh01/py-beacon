# src/beacon/server/comparisons.py
"""
A run against its comparators, on the wire: its index and its benchmark, each
with the comparator's own figures over the run's window and the run's
figures relative to it.

The run is given by its daily returns, the ones its metrics use, so the
comparison with its own index agrees with its tracking error and tracking
difference.
"""
# BN-323.
from dataclasses import dataclass

import pandas as pd

from ..analysis.comparison import Comparison, compare
from ..assumptions import DEFAULT_PERIODS_PER_YEAR
from ..backtest.result import BacktestResult
from ..index.result import IndexResult
from .schemas import (
    BenchmarkRef,
    ComparatorFigures,
    ComparisonPayload,
    ComparisonsPayload,
    RelativeFigures,
    SeriesPayload,
)

INDEX_KIND = "index"


@dataclass(frozen=True)
class Comparators:
    """What a run is compared against: its index, by id, and an external
    benchmark with its resolved levels."""
    index_id: str
    benchmark: BenchmarkRef | None = None
    benchmark_levels: pd.Series | None = None


def comparison_payloads(result: BacktestResult,
                        index_result: IndexResult,
                        comparators: Comparators) -> ComparisonsPayload:
    """The run against its index and its benchmark."""
    returns = result.get_returns()
    assumptions = result.modelling_assumptions
    periods = (assumptions.periods_per_year if assumptions is not None else None)
    risk_free = (assumptions.risk_free_rate if assumptions is not None else None)
    periods_per_year = periods or DEFAULT_PERIODS_PER_YEAR

    def against(levels: pd.Series | None,
                identifier: str,
                kind: str) -> ComparisonPayload | None:
        comparison = (compare(returns, levels, periods_per_year, risk_free or 0.0)
                      if levels is not None else None)

        return _payload(comparison, identifier, kind) if comparison is not None else None

    benchmark = comparators.benchmark

    return ComparisonsPayload(
        index=against(index_result.index_levels, comparators.index_id, INDEX_KIND),
        benchmark=(against(comparators.benchmark_levels, benchmark.id, benchmark.kind)
                   if benchmark is not None else None))


def _payload(comparison: Comparison,
             identifier: str,
             kind: str) -> ComparisonPayload:
    """One comparison on the wire."""
    return ComparisonPayload(id=identifier,
                             kind=kind,
                             start=str(comparison.start.date()),
                             end=str(comparison.end.date()),
                             observations=comparison.observations,
                             level=SeriesPayload.from_series(comparison.level),
                             own=ComparatorFigures.model_validate(comparison.own),
                             relative=RelativeFigures.model_validate(comparison.relative))
