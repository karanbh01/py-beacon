# src/beacon/backtest/books.py
"""
A book: one series of levels and weights, compared the same way whatever it
is. A backtest's result holds one for the portfolio's target index, one for
an optimised index when there is one, and one for its benchmark.
"""
# Moved out of result.py when it was split (BN-268).
from dataclasses import dataclass

import pandas as pd

from ..index.result import IndexResult


class Book:
    """One comparator's daily record: levels, weights, returns.

    The uniform surface every comparator answers through, so
    `result.index.target.weights` and `result.benchmark.levels` are the
    same spelling on every book. A book built from an `IndexResult` keeps it as
    `source`, because the snapshots it holds (what each rebalance *decided*)
    are a different fact from the daily panel (what happened between).

    A benchmark supplied as a bare level series has no weights; its
    `weights` frame is empty rather than invented.
    """

    def __init__(self,
                 levels: pd.Series,
                 weights: pd.DataFrame | None = None,
                 source: IndexResult | None = None):
        self.levels = levels
        self.weights = weights if weights is not None else pd.DataFrame()
        self.source = source

    @classmethod
    def from_index(cls,
                   result: IndexResult) -> "Book":
        """A book over an index result.

        The result's daily weights panel becomes the wide weights frame
        (dates by identifiers). A result with no daily weights panel yields
        an empty frame rather than a derived one: weights are recorded, not
        reconstructed.
        """
        # The daily panel is BN-153; results produced before it existed have
        # none.
        weights = pd.DataFrame()

        if not result.daily_weights.empty:
            weights = result.daily_weights.pivot_table(index="DATE",
                                                       columns="IDENTIFIER",
                                                       values="WEIGHT",
                                                       observed=True)

        return cls(levels=result.index_levels, weights=weights, source=result)

    @classmethod
    def from_levels(cls,
                    levels: pd.Series) -> "Book":
        """A book over a bare level series, such as a benchmark from raw data."""
        return cls(levels=pd.Series(levels).astype(float))

    @property
    def returns(self) -> pd.Series:
        """Daily returns of the levels; empty when the book is."""
        if self.levels.empty:
            return pd.Series(dtype=float)

        return self.levels.pct_change().dropna()

    def __repr__(self) -> str:
        return (f"Book(dates={len(self.levels)}, "
                f"weighted={not self.weights.empty})")


# BN-164 replaced the flat `index` / `target_index` pair, which were two names
# for one concept ("the calculated index this run aims at") chosen by mode.
# `target` is always filled because the engine has taken an IndexResult as its
# only schedule source since BN-165; `optimised` is BN-167's derived index.
@dataclass
class IndexBooks:
    """The run's calculated indices, one home each.

    The **container is always present**; its books are what can be None.
    That distinction reads as safe and is not: a guard written against
    `result.index` rather than `result.index.target` never fires.

    Attributes:
        target: The index being aimed at, pre-optimisation. Always filled
            on an engine-produced result, since the engine's schedule is
            always an IndexResult. None only on a result built by hand from
            an empty container.
        optimised: The solved index's own calculation, filled when the run
            tracked a derived (optimised) index. None on plain passive runs.
    """
    target: Book | None = None
    optimised: Book | None = None

    @property
    def tracked(self) -> Book | None:
        """The book the engine traded toward: optimised when solved, else target."""
        return self.optimised if self.optimised is not None else self.target


def index_books(index_result: IndexResult,
                target_index: IndexResult | None) -> IndexBooks:
    """A run's calculated indices, one book each.

    Given *index_result* alone, the run traded a plain calculation: it is the
    `target` book. Given *target_index* as well, the schedule the engine
    traded is an optimised calculation and *target_index* is the parent it was
    solved from, so they are the `target` and `optimised` books.
    """
    if target_index is not None:
        return IndexBooks(target=Book.from_index(target_index),
                          optimised=Book.from_index(index_result))

    return IndexBooks(target=Book.from_index(index_result))


def benchmark_book(benchmark: IndexResult | pd.Series | None) -> Book | None:
    """The benchmark of record as a book, whichever form it was given in."""
    if benchmark is None:
        return None

    if isinstance(benchmark, pd.Series):
        return Book.from_levels(benchmark)

    return Book.from_index(benchmark)
