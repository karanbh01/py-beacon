# src/beacon/backtest/result.py
"""
BacktestResult: the record of a run, holding books rather than fields.

The result is an **orchestrator**: it holds the Portfolio (kept whole, not
flattened into series) plus the run-level facts (the index tracked, the
target index, the benchmark of record, unfilled orders) and its methods
answer questions by comparing books:

    result.portfolio.nav          what the money did
    result.index.target.levels    what the index being aimed at did
    result.index.optimised        the solved index's own calculation, when one exists
    result.index.tracked          the book the engine actually traded toward
    result.benchmark.levels       the benchmark of record, when one was given
    result.against(other)         any comparator, after the fact

Parallel structure is the point: several books, same question, same
spelling. Each fact has one home.

## The benchmark of record versus a question asked later

`benchmark=` given to the engine is a fact about the run: stored here,
serialised, reproducible, so every reader of this result quotes excess
return against the same comparator. `against(other)` is a question asked
afterwards: it computes and returns, and **never mutates the stored
record**, because otherwise the benchmark of record would be whatever
somebody last idly compared against.

## Day zero and the trading NAV

The portfolio's NAV opens with initial capital on the eve of the first
trading day. :attr:`trading_nav` is the same series with that opening row
dropped, net of any management fee the fund owes, and the wire format reads
it. The return metrics start from the initial capital instead: the first
day's return runs from the capital to the first close, so the cost of buying
in counts.

## Flows and units

With flows the NAV is the fund's size, which money arriving and leaving moves
as much as the market does. Performance is then measured per unit:
:attr:`nav_per_unit` is the NAV over the units outstanding, every return
metric is computed from it (time-weighted), and the summary adds the
money-weighted return, which does count when the money arrived. Without flows
the units never change and the metrics are computed exactly as before.
"""
# BN-154 removed the old flat fields (`portfolio_nav`, `cash_history`,
# `actual_weight_history`, the `portfolio_id` alias) in favour of books.
# The day-zero row is decision 11 of the backtester redesign; deriving metrics
# from `trading_nav` keeps every number computed before the redesign
# identical after it.
from dataclasses import dataclass, field
from typing import Union

import pandas as pd

from ..analysis.relative import RelativeMetrics, relative_metrics
from ..assumptions import ModellingAssumptions
from ..data.fetcher import DataFetcher
from ..index.result import IndexResult, PriceGap
from ..plot.base import PlotAccessor
from ..portfolio.base import Portfolio
from .asset_view import BacktestAssetView
from .books import Book, IndexBooks
from .flows import FlowRecord
from .implementation import RebalanceStep
from .metrics import MetricsMixin

# Why an order went unfilled (BN-252, BN-266).
CASH_SHORT = "cash"
NO_PRICE = "no price"
EXECUTION_LIMIT = "execution limit"


@dataclass(frozen=True)
class UnfilledOrder:
    """An order the simulation could not execute in full.

    Recorded on the result rather than only logged: a partially filled
    rebalance leaves the portfolio off its target weights, and a caller
    comparing tracking error against expectations needs to know that happened
    rather than reading it as a modelling result.

    Attributes:
        date: The rebalance date.
        asset_id: Asset that could not be fully traded.
        requested_quantity: Quantity the rebalance asked for; 0.0 when the
            name could not be priced to size an order.
        filled_quantity: Quantity actually traded; 0.0 when nothing was.
        price: Execution price used; NaN when there was none.
        shortfall_value: Notional value that went unfilled; for a name with
            no price, the value the rebalance aimed to hold. NaN for an order
            an execution limit left unfinished.
        reason: Why: ``"cash"`` (not enough to buy it all), ``"no price"``
            (the name could not be priced on the day) or
            ``"execution limit"`` (still working when the next rebalance
            replaced it or the run ended).
    """
    date: pd.Timestamp
    asset_id: str
    requested_quantity: float
    filled_quantity: float
    price: float
    shortfall_value: float
    reason: str = CASH_SHORT


# RebalancePricing is BN-183. PriceGap, also BN-183, moved to the index
# layer with BN-251, when the index started recording gaps too.
@dataclass(frozen=True)
class RebalancePricing:
    """What one rebalance priced from.

    A rebalance scheduled on a day the market was shut still trades: it prices
    from the session in force through the closure. Recording both dates lets a
    reader tell that case apart from an ordinary rebalance. `date` and
    `priced_from` are equal for the ordinary rebalance, which is what makes an
    unequal pair worth reading.

    Attributes:
        date: The rebalance date from the weight schedule.
        priced_from: The session its prices were read from.
    """
    date: pd.Timestamp
    priced_from: pd.Timestamp


# What `against()` accepts: anything carrying a daily level series.
Comparable = Union["BacktestResult", Book, IndexResult, pd.Series]


@dataclass
class BacktestResult(MetricsMixin):
    """The record of one backtest run.

    Args:
        portfolio: The books (positions, weights, cash, NAV, transactions),
            kept whole and frozen by the engine on completion.
        index: The run's calculated indices, as an :class:`IndexBooks`
            container: always present, its books None when the run
            calculated none. `index.target` is the index aimed at,
            `index.optimised` the solved calculation when one exists, and
            `index.tracked` the book the engine traded toward.
        benchmark: The benchmark of record, when one was given to the engine.
        unfilled: Buys the simulation could not execute in full. Empty for a
            run where every rebalance leg filled, so a non-empty list is
            itself the signal that the portfolio drifted off target for a
            reason other than price movement.
        price_gaps: Days a name had no bar on a session its calendar says was
            open, and was therefore marked at a carried-forward price.
            Empty for a run with complete data. A market holiday is not a
            gap, since nothing is missing on a day nothing traded.
        rebalance_pricing: What each rebalance priced from, in date order.
            `date` and `priced_from` differ only where the schedule landed on
            a day the market was shut, so the run can state which session its
            trades were struck at rather than leaving it inferable.
        rebalance_steps: What the screening and redistribution stages did at
            each rebalance: the target, the names removed and by what, and
            the weights traded to.
        currency: The book's currency, which the NAV, costs and holdings'
            values are in.
        modelling_assumptions: What the run assumed, every field resolved.
            None for a result built by hand, which the summary then measures
            with a zero risk-free rate over 252 periods a year.
        flows: Each day's flow as dealt: the amount, and the units created or
            cancelled at that day's NAV per unit. Empty without flows.
        units: Units outstanding at the end of each day, from the eve of the
            first trading day.
        fees_payable: The management fee owed and not yet paid at the end of
            each day, which the NAV is net of.
        launch_price: NAV per unit at launch.
        market: For an ETF, each day's quote on the exchange: NAV per share,
            premium, spread, market price, bid and ask, and the arbitrage
            band. Empty for any other vehicle.
    """

    #: Charts for this result. A descriptor that resolves on first
    #: access, so matplotlib is imported only when something is drawn.
    plot = PlotAccessor("BacktestPlots")
    portfolio: Portfolio
    index: IndexBooks = field(default_factory=IndexBooks)
    benchmark: Book | None = None
    unfilled: list[UnfilledOrder] = field(default_factory=list)
    price_gaps: list[PriceGap] = field(default_factory=list)
    rebalance_pricing: list[RebalancePricing] = field(default_factory=list)
    rebalance_steps: list[RebalanceStep] = field(default_factory=list)
    currency: str = "USD"
    modelling_assumptions: ModellingAssumptions | None = None
    flows: list[FlowRecord] = field(default_factory=list)
    units: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    fees_payable: pd.Series = field(
        default_factory=lambda: pd.Series(dtype=float))
    launch_price: float = 1.0
    market: pd.DataFrame = field(default_factory=pd.DataFrame)
    _data_fetcher: DataFetcher | None = field(default=None, repr=False,
                                              compare=False)

    @property
    def trading_nav(self) -> pd.Series:
        """NAV over the simulated days, with the day-zero row excluded.

        The portfolio's own `nav` opens with initial capital on the eve of
        the first trading day, the record of what the run started with.
        Every *metric* derives from this series instead: the eve row is a
        starting fact, not a day the simulation traded. Net of the management
        fee owed and not yet paid, when the run had a vehicle charging one.
        """
        nav = self._net_nav()

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
                judged from the positions panel (the record of holdings)
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
                                 index_book=self.index.tracked)

    def against(self,
                other: Comparable) -> RelativeMetrics:
        """Compare this run's NAV against any comparator, after the fact.

        The run-time benchmark is a fact about the run; this is a question
        asked later, so it computes and returns, and **stores nothing**. Ask
        against ten comparators and the result is byte-for-byte what it was.

        Args:
            other: Another result, a book, an index result, or a bare level
                series.

        Returns:
            RelativeMetrics: Excess return, tracking error, beta and
            correlation over the common window, as `analysis.relative`
            computes them.
        """
        return relative_metrics(self.performance_levels(), _levels_of(other))

    def __repr__(self) -> str:
        n_dates = len(self.trading_nav)
        n_txns = len(self.portfolio.transactions)
        bound = self._data_fetcher is not None
        return (
            f"BacktestResult(portfolio='{self.portfolio.portfolio_id}', "
            f"dates={n_dates}, transactions={n_txns}, "
            f"index={self.index.tracked is not None}, "
            f"benchmark={self.benchmark is not None}, data_bound={bound})"
        )


def _levels_of(other: Comparable) -> pd.Series:
    """The level series a comparator carries, whichever kind it is."""
    if isinstance(other, BacktestResult):
        return other.performance_levels()

    if isinstance(other, Book):
        return other.levels

    if isinstance(other, IndexResult):
        return other.index_levels

    if isinstance(other, pd.Series):
        return other.astype(float)

    raise TypeError(
        f"Cannot compare against {type(other).__name__}; expected a "
        f"BacktestResult, Book, IndexResult or level series.")
