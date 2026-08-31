# src/beacon/backtest/result.py
"""
BacktestResult — the record of a run, holding books rather than fields.

The result is an **orchestrator**: it holds the Portfolio (kept whole, not
flattened into series) plus the run-level facts — the index tracked, the
target index, the benchmark of record, unfilled orders — and its methods
answer questions by comparing books:

    result.portfolio.nav          what the money did
    result.index.levels           what the tracked index did
    result.index.weights          daily, including mid-period deletions
    result.benchmark.levels       the benchmark of record, when one was given
    result.against(other)         any comparator, after the fact

Parallel structure is the point (BN-154): three-plus books, same question,
same spelling. The old flat fields — `portfolio_nav`, `cash_history`,
`actual_weight_history`, the `portfolio_id` alias — are gone; each fact now
has one home.

## The benchmark of record versus a question asked later

`benchmark=` given to the engine is a fact about the run: stored here,
serialised, reproducible, so every reader of this result quotes excess
return against the same comparator. `against(other)` is a question asked
afterwards — it computes and returns, and **never mutates the stored
record**, because otherwise the benchmark of record would be whatever
somebody last idly compared against.

## Day zero and the trading NAV

The portfolio's NAV opens with initial capital on the eve of the first
trading day (decision 11). Metrics, the fund's fee accrual and the current
wire format all derive from :attr:`trading_nav` — the same series with that
opening row dropped — so every number computed before the redesign is
computed identically after it.
"""
from dataclasses import dataclass, field
from typing import Union

import numpy as np
import pandas as pd

from ..analysis.relative import RelativeMetrics, relative_metrics
from ..data.fetcher import DataFetcher
from ..index.result import IndexResult
from ..plot.base import PlotAccessor
from ..portfolio.base import Portfolio
from .asset_view import BacktestAssetView


@dataclass(frozen=True)
class UnfilledOrder:
    """A buy the simulation could not execute in full.

    Recorded on the result rather than only logged: a partially filled
    rebalance leaves the portfolio off its target weights, and a caller
    comparing tracking error against expectations needs to know that happened
    rather than reading it as a modelling result.

    Attributes:
        date: The rebalance date.
        asset_id: Asset that could not be fully bought.
        requested_quantity: Quantity the rebalance asked for.
        filled_quantity: Quantity actually bought; 0.0 when nothing was.
        price: Execution price used.
        shortfall_value: Notional value that went unfilled, at *price*.
    """
    date: pd.Timestamp
    asset_id: str
    requested_quantity: float
    filled_quantity: float
    price: float
    shortfall_value: float


class Book:
    """One comparator's daily record: levels, weights, returns.

    The uniform surface every comparator answers through, so
    `result.index.weights` and `result.benchmark.levels` are the same
    spelling on every book. A book built from an `IndexResult` keeps it as
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

        The daily panel (BN-153) becomes the wide weights frame; an index
        produced before the panel existed yields an empty frame rather than
        a derived one — weights are recorded, not reconstructed.
        """
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
        """A book over a bare level series — a benchmark from raw data."""
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


# What `against()` accepts: anything carrying a daily level series.
Comparable = Union["BacktestResult", Book, IndexResult, pd.Series]


@dataclass
class BacktestResult:
    """The record of one backtest run.

    Args:
        portfolio: The books — positions, weights, cash, NAV, transactions —
            kept whole and frozen by the engine on completion.
        index: The index the run tracked, when it tracked one. A run driven
            by a raw weight schedule has none.
        target_index: The post-selection, pre-optimisation index, populated
            on optimisation runs so optimised-versus-unoptimised is a
            first-class comparison.
        benchmark: The benchmark of record, when one was given to the engine.
        unfilled: Buys the simulation could not execute in full. Empty for a
            run where every rebalance leg filled, so a non-empty list is
            itself the signal that the portfolio drifted off target for a
            reason other than price movement.
    """

    #: Charts for this result. A descriptor that resolves on first
    #: access, so matplotlib is imported only when something is drawn.
    plot = PlotAccessor("BacktestPlots")
    portfolio: Portfolio
    index: Book | None = None
    target_index: Book | None = None
    benchmark: Book | None = None
    unfilled: list[UnfilledOrder] = field(default_factory=list)
    _data_fetcher: DataFetcher | None = field(default=None, repr=False,
                                              compare=False)

    @property
    def trading_nav(self) -> pd.Series:
        """NAV over the simulated days, with the day-zero row excluded.

        The portfolio's own `nav` opens with initial capital on the eve of
        the first trading day (decision 11) — the record of what the run
        started with. Every *metric* derives from this series instead, which
        matches the NAV the engine produced before the redesign exactly: the
        eve row is a starting fact, not a day the simulation traded.
        """
        nav = self.portfolio.nav

        if (self.portfolio.inception is not None and not nav.empty
                and nav.index[0] == self.portfolio.inception):
            return nav.iloc[1:]

        return nav

    @property
    def total_unfilled_value(self) -> float:
        """Total notional that went unfilled across the run."""
        return float(sum(order.shortfall_value for order in self.unfilled))

    def with_data(self,
                  data_fetcher: DataFetcher) -> 'BacktestResult':
        """Bind a DataFetcher for asset-level queries. Returns self for chaining."""
        self._data_fetcher = data_fetcher
        return self

    def asset(self,
              asset_id: str) -> BacktestAssetView:
        """Return a BacktestAssetView for an asset the run ever held.

        Args:
            asset_id: Identifier of the asset.

        Returns:
            BacktestAssetView

        Raises:
            RuntimeError: If no DataFetcher has been bound via
                :meth:`with_data`.
            KeyError: If the run's books never held *asset_id*. Membership is
                judged from the positions panel — the record of holdings —
                rather than from a weight column, so a position too small to
                round to a visible weight still counts as held.
        """
        if self._data_fetcher is None:
            raise RuntimeError(
                "No DataFetcher bound. Call .with_data(fetcher) first."
            )

        positions = self.portfolio.positions

        if positions.empty or asset_id not in set(positions["ASSET_ID"]):
            raise KeyError(
                f"Asset '{asset_id}' does not appear in this backtest's books."
            )

        return BacktestAssetView(asset_id=asset_id,
                                 data_fetcher=self._data_fetcher,
                                 portfolio=self.portfolio,
                                 index_book=self.index)

    def against(self,
                other: Comparable) -> RelativeMetrics:
        """Compare this run's NAV against any comparator, after the fact.

        The exploratory half of decision 13: the run-time benchmark is a fact
        about the run, this is a question asked later — so it computes and
        returns, and **stores nothing**. Ask against ten comparators and the
        result is byte-for-byte what it was.

        Args:
            other: Another result, a book, an index result, or a bare level
                series.

        Returns:
            RelativeMetrics: Excess return, tracking error, beta and
            correlation over the common window, as `analysis.relative`
            computes them.
        """
        return relative_metrics(self.trading_nav, _levels_of(other))

    def get_returns(self) -> pd.Series:
        """Derive a return series from portfolio NAV.

        Returns:
            pd.Series: Percentage returns (first entry is dropped).
        """
        nav = self.trading_nav

        if nav.empty:
            return pd.Series(dtype=float)

        return nav.pct_change().dropna()

    def get_tracking_error(self) -> float | None:
        """Calculate annualised tracking error against the tracked index.

        Tracking error is the annualised standard deviation of the
        difference between portfolio returns and index returns.

        Returns:
            float or None: Annualised tracking error, or None if the run
            tracked no index.
        """
        if self.index is None:
            return None

        port_returns = self.get_returns()
        index_returns = self.index.returns

        # Align on common dates
        aligned = pd.DataFrame({
            "port": port_returns,
            "index": index_returns,
        }).dropna()

        if aligned.empty:
            return None

        active_returns = aligned["port"] - aligned["index"]
        return float(active_returns.std() * np.sqrt(252))

    def get_tracking_difference(self) -> float | None:
        """Calculate cumulative tracking difference against the tracked index.

        Tracking difference is the difference between the cumulative
        portfolio return and the cumulative index return over the
        full backtest period.

        Returns:
            float or None: Tracking difference, or None if the run tracked
            no index.
        """
        if self.index is None:
            return None

        port_returns = self.get_returns()
        index_returns = self.index.returns

        if port_returns.empty or index_returns.empty:
            return None

        port_cumulative = (1 + port_returns).prod() - 1
        index_cumulative = (1 + index_returns).prod() - 1
        return float(port_cumulative - index_cumulative)

    def summary(self) -> dict[str, float | None]:
        """Calculate key performance metrics for the backtest.

        Returns:
            dict: Dictionary containing: total_return, annualised_return,
            volatility, sharpe_ratio, max_drawdown, and optionally
            tracking_error and tracking_difference.
        """
        returns = self.get_returns()
        n_periods = len(returns)
        nav = self.trading_nav
        initial = self.portfolio.initial_capital

        # Total return
        total_return = (0.0 if nav.empty or initial == 0
                        else float(nav.iloc[-1] / initial - 1))

        # Annualised return
        if n_periods > 0:
            years = n_periods / 252.0
            annualised_return = float((1 + total_return) ** (1 / years) - 1) if years > 0 else 0.0
        else:
            annualised_return = 0.0

        # Volatility (annualised)
        volatility = float(returns.std() * np.sqrt(252)) if n_periods > 1 else 0.0

        # Sharpe ratio (assumes risk-free rate = 0)
        sharpe_ratio = float(annualised_return / volatility) if volatility > 0 else 0.0

        # Max drawdown
        if not nav.empty:
            cumulative_max = nav.cummax()
            drawdown = (nav - cumulative_max) / cumulative_max
            max_drawdown = float(drawdown.min())
        else:
            max_drawdown = 0.0

        result: dict[str, float | None] = {
            "total_return": total_return,
            "annualised_return": annualised_return,
            "volatility": volatility,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": max_drawdown,
        }

        # Tracking metrics (only if the run tracked an index)
        te = self.get_tracking_error()
        td = self.get_tracking_difference()
        if te is not None:
            result["tracking_error"] = te
        if td is not None:
            result["tracking_difference"] = td

        return result

    def __repr__(self) -> str:
        n_dates = len(self.trading_nav)
        n_txns = len(self.portfolio.transactions)
        bound = self._data_fetcher is not None
        return (
            f"BacktestResult(portfolio='{self.portfolio.portfolio_id}', "
            f"dates={n_dates}, transactions={n_txns}, "
            f"index={self.index is not None}, "
            f"benchmark={self.benchmark is not None}, data_bound={bound})"
        )


def _levels_of(other: Comparable) -> pd.Series:
    """The level series a comparator carries, whichever kind it is."""
    if isinstance(other, BacktestResult):
        return other.trading_nav

    if isinstance(other, Book):
        return other.levels

    if isinstance(other, IndexResult):
        return other.index_levels

    if isinstance(other, pd.Series):
        return other.astype(float)

    raise TypeError(
        f"Cannot compare against {type(other).__name__}; expected a "
        f"BacktestResult, Book, IndexResult or level series.")
