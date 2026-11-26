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
import warnings
from collections.abc import Callable
from typing import Any, Union

import pandas as pd

from ..analysis.relative import RelativeMetrics, relative_metrics
from ..assumptions import ModellingAssumptions
from ..data.fetcher import DataFetcher
from ..index.result import IndexResult, PriceGap
from ..plot.base import PlotAccessor
from ..portfolio.base import Portfolio
from ..strategy.active import ActiveStep
from ..strategy.tracking import ReplicationStep
from .analytics_access import AnalyticsMixin
from .asset_view import BacktestAssetView
from .books import Book, IndexBooks
from .cash_band import band_events_frame
from .etf import quotes_frame
from .flows import FlowRecord
from .fund_record import fund_record
from .futures import FuturesRoll, FuturesTrade, futures_days_frame
from .implementation import RebalanceStep
from .metrics import MetricsMixin
from .share_classes import ShareClassBook
from .swap import SwapReset
from .trading_record import (
    RebalancePricing,
    TradingRecord,
    UnfilledOrder,
)

# What `against()` accepts: anything carrying a daily level series.
Comparable = Union["BacktestResult", Book, IndexResult, pd.Series]


class BacktestResult(MetricsMixin, AnalyticsMixin):
    """The record of one backtest run.

    Attributes:
        portfolio: The books (positions, weights, cash, NAV, transactions),
            kept whole and frozen by the engine on completion.
        index: The run's calculated indices, as an :class:`IndexBooks`
            container: always present, its books None when the run
            calculated none. `index.target` is the index aimed at,
            `index.optimised` the solved calculation when one exists, and
            `index.tracked` the book the engine traded toward.
        benchmark: The benchmark of record, when one was given to the engine.
        fund: The vehicle's recorded data (`beacon.backtest.fund_record`):
            its NAV, units, fees, flows, dividends and share classes, and
            its ETF quotes, swap, futures, cash band and hedging, each None
            when the run's vehicle lacks it.
        trading: What the engine did at each rebalance
            (`beacon.backtest.trading_record`): its steps, pricing, unfilled
            orders and price gaps, and a replication's or an active
            strategy's choices.
        assumptions: What the run assumed, every field resolved. None for a
            result built by hand, which the summary then measures with a
            zero risk-free rate over 252 periods a year.
        analytics: Every calculation on the run
            (`beacon.analysis.analytics.Analytics`), and the analysis it was
            asked for.

    The engine builds it from what it kept; built by hand, every keyword
    but *portfolio* may be left out. The flat names the fund's and the
    engine's data had before 0.11.0 (``nav_per_unit``, ``rebalance_steps``,
    ...) still read them, warning, until 0.12.0.
    """
    # BN-459/BN-460 (Karan, 2026-11-24): the data under fund and trading; a
    # class rather than a dataclass so the flat names can warn.

    #: Charts for this result. A descriptor that resolves on first
    #: access, so matplotlib is imported only when something is drawn.
    plot = PlotAccessor("BacktestPlots")

    def __init__(self,
                 portfolio: Portfolio,
                 index: IndexBooks | None = None,
                 benchmark: Book | None = None,
                 unfilled: list[UnfilledOrder] | None = None,
                 price_gaps: list[PriceGap] | None = None,
                 rebalance_pricing: list[RebalancePricing] | None = None,
                 rebalance_steps: list[RebalanceStep] | None = None,
                 currency: str = "USD",
                 modelling_assumptions: ModellingAssumptions | None = None,
                 flows: list[FlowRecord] | None = None,
                 units: pd.Series | None = None,
                 fees_payable: pd.Series | None = None,
                 launch_price: float = 1.0,
                 market: pd.DataFrame | None = None,
                 replication: list[ReplicationStep] | None = None,
                 active: list[ActiveStep] | None = None,
                 swap_value: pd.Series | None = None,
                 swap_resets: list[SwapReset] | None = None,
                 futures_notional: pd.Series | None = None,
                 futures_rolls: list[FuturesRoll] | None = None,
                 cash_band_events: pd.DataFrame | None = None,
                 futures_trades: list[FuturesTrade] | None = None,
                 futures_days: pd.DataFrame | None = None,
                 hedge_value: pd.Series | None = None,
                 dividends_receivable: pd.Series | None = None,
                 dividends_entitled: pd.DataFrame | None = None,
                 share_classes: dict[str, ShareClassBook] | None = None,
                 analysis_request: Any = None,
                 analysis_timings: dict[str, float] | None = None,
                 analysis_block: dict[str, Any] | None = None):
        self.portfolio = portfolio
        self.index = index if index is not None else IndexBooks()
        self.benchmark = benchmark
        self.assumptions = modelling_assumptions
        self.fund = fund_record(
            portfolio, currency, launch_price, _series(units), list(flows or []),
            _series(fees_payable), _series(dividends_receivable),
            (dividends_entitled if dividends_entitled is not None
             else pd.DataFrame(columns=["GROSS", "WITHHELD"], dtype=float)),
            dict(share_classes or {}), _frame(market), _series(swap_value),
            list(swap_resets or []), _series(futures_notional), list(futures_trades or []),
            list(futures_rolls or []), _frame(futures_days), cash_band_events,
            _series(hedge_value))
        self.trading = TradingRecord(portfolio, list(rebalance_steps or []),
                                     list(rebalance_pricing or []), list(unfilled or []),
                                     list(price_gaps or []), list(replication or []) or None,
                                     list(active or []) or None)
        # BN-397/BN-398: what the run was asked to analyse, its analytics
        # block, and how long each part took.
        self._analysis_request = analysis_request
        self._analysis_timings = dict(analysis_timings or {})
        self._analysis_block = analysis_block
        self._data_fetcher: DataFetcher | None = None

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
        return relative_metrics(self._performance_levels(), _levels_of(other))

    def __repr__(self) -> str:
        n_dates = len(self.fund.trading_nav)
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
        return other._performance_levels()

    if isinstance(other, Book):
        return other.levels

    if isinstance(other, IndexResult):
        return other.index_levels

    if isinstance(other, pd.Series):
        return other.astype(float)

    raise TypeError(
        f"Cannot compare against {type(other).__name__}; expected a "
        f"BacktestResult, Book, IndexResult or level series.")


def _series(found: pd.Series | None) -> pd.Series:
    return found if found is not None else pd.Series(dtype=float)


def _frame(found: pd.DataFrame | None) -> pd.DataFrame:
    return found if found is not None else pd.DataFrame()


def _empty_series() -> pd.Series:
    return pd.Series(dtype=float)


# BN-459/BN-460: each flat name before 0.11.0, where it moved, and how to read
# it there, as it read before (empty where the vehicle lacks the part).
MOVED: dict[str, tuple[str, Callable[[BacktestResult], Any]]] = {
    "nav_per_unit": ("fund.nav_per_unit", lambda result: result.fund.nav_per_unit),
    "trading_nav": ("fund.trading_nav", lambda result: result.fund.trading_nav),
    "aum": ("fund.aum", lambda result: result.fund.aum),
    "units": ("fund.units", lambda result: result.fund.units),
    "units_outstanding": ("fund.units_outstanding",
                          lambda result: result.fund.units_outstanding),
    "launch_price": ("fund.launch_price", lambda result: result.fund.launch_price),
    "currency": ("fund.currency", lambda result: result.fund.currency),
    "fees_payable": ("fund.fees.payable", lambda result: result.fund.fees.payable),
    "fees_paid": ("fund.fees.paid", lambda result: result.fund.fees.paid),
    "flows": ("fund.flows", lambda result: result.fund.flows),
    "dividends_receivable": ("fund.dividends.receivable",
                             lambda result: result.fund.dividends.receivable),
    "dividends_entitled": ("fund.dividends.entitled",
                           lambda result: result.fund.dividends.entitled),
    "share_classes": ("fund.share_classes", lambda result: result.fund.share_classes),
    "market": ("fund.etf.market", lambda result: (
        result.fund.etf.market if result.fund.etf is not None else quotes_frame([]))),
    "swap_value": ("fund.swap.value", lambda result: (
        result.fund.swap.value if result.fund.swap is not None else _empty_series())),
    "swap_resets": ("fund.swap.resets", lambda result: (
        result.fund.swap.resets if result.fund.swap is not None else [])),
    "counterparty_exposure": ("fund.swap.counterparty_exposure", lambda result: (
        result.fund.swap.counterparty_exposure if result.fund.swap is not None
        else _empty_series())),
    "futures_notional": ("fund.futures.notional", lambda result: (
        result.fund.futures.notional if result.fund.futures is not None
        else _empty_series())),
    "futures_exposure": ("fund.futures.exposure", lambda result: (
        result.fund.futures.exposure if result.fund.futures is not None
        else _empty_series())),
    "futures_trades": ("fund.futures.trades", lambda result: (
        result.fund.futures.trades if result.fund.futures is not None else [])),
    "futures_rolls": ("fund.futures.rolls", lambda result: (
        result.fund.futures.rolls if result.fund.futures is not None else [])),
    "futures_days": ("fund.futures.days", lambda result: (
        result.fund.futures.days if result.fund.futures is not None
        else futures_days_frame({}))),
    "cash_band_events": ("fund.cash_band.events", lambda result: (
        result.fund.cash_band.events if result.fund.cash_band is not None
        else band_events_frame([]))),
    "hedge_value": ("fund.hedging.value", lambda result: (
        result.fund.hedging.value if result.fund.hedging is not None else _empty_series())),
    "rebalance_steps": ("trading.steps", lambda result: result.trading.steps),
    "rebalance_pricing": ("trading.pricing", lambda result: result.trading.pricing),
    "unfilled": ("trading.unfilled", lambda result: result.trading.unfilled),
    "total_unfilled_value": ("trading.unfilled_value",
                             lambda result: result.trading.unfilled_value),
    "price_gaps": ("trading.price_gaps", lambda result: result.trading.price_gaps),
    "total_costs": ("trading.total_costs", lambda result: result.trading.total_costs),
    "replication": ("trading.replication", lambda result: result.trading.replication or []),
    "active": ("trading.active", lambda result: result.trading.active or []),
    # BN-461.
    "modelling_assumptions": ("assumptions", lambda result: result.assumptions),
    "analysis_request": ("analytics.request", lambda result: result._analysis_request),
    "analysis_timings": ("analytics.timings", lambda result: result._analysis_timings),
    "analysis_block": ("analytics.to_dict()", lambda result: result._analysis_block),
    "run_data": ("analytics.run_data", lambda result: result._run_data),
    "portfolio_analytics": ("analytics.portfolio",
                            lambda result: result._portfolio_analytics),
    "risk": ("analytics.portfolio.risk", lambda result: result._risk),
    "attribution": ("analytics.portfolio.attribution", lambda result: result._attribution),
}

# BN-461: each method before 0.11.0, where it moved, and the method that
# still gives what it gave.
MOVED_METHODS: dict[str, tuple[str, str]] = {
    "get_returns": ("analytics.portfolio.performance.returns()", "_returns"),
    "get_annual_returns": ("analytics.portfolio.performance.total_return(by=\"year\")",
                           "_annual_returns"),
    "get_tracking_error": ("analytics.portfolio.relative.tracking_error()", "_tracking_error"),
    "get_tracking_difference": ("analytics.portfolio.relative.tracking_difference()",
                                "_tracking_difference"),
    "information_ratio": ("analytics.portfolio.relative.information_ratio()",
                          "_information_ratio"),
    "money_weighted_return": ("analytics.portfolio.performance.money_weighted_return()",
                              "_money_weighted_return"),
    "active_share": ("analytics.portfolio.relative.active_share()", "_active_share"),
    "active_attribution": ("analytics.portfolio.attribution.active_attribution()",
                           "_active_attribution"),
    "market_return": ("analytics.portfolio.vehicle.market_return()", "_market_return"),
    "market_summary": ("analytics.portfolio.vehicle.market_summary()", "_market_summary"),
    "performance_levels": ("fund.nav_per_unit, or fund.trading_nav without flows",
                           "_performance_levels"),
}


def _moved(name: str,
           path: str,
           read: Callable[[BacktestResult], Any]) -> property:
    """A flat name that reads its new home, warning."""
    def get(self: BacktestResult) -> Any:
        warnings.warn(f"BacktestResult.{name} is deprecated and will be removed in 0.12.0; "
                      f"use result.{path}.", DeprecationWarning, stacklevel=2)

        return read(self)

    return property(get, doc=f"Deprecated: `result.{path}`.")


def _moved_method(name: str,
                  path: str,
                  private: str) -> Callable[[BacktestResult], Any]:
    """A method before 0.11.0 that gives what it gave, warning."""
    def method(self: BacktestResult) -> Any:
        warnings.warn(f"BacktestResult.{name}() is deprecated and will be removed in 0.12.0; "
                      f"use result.{path}.", DeprecationWarning, stacklevel=2)

        return getattr(self, private)()

    method.__name__ = name
    method.__doc__ = f"Deprecated: `result.{path}`."

    return method


for _name, (_path, _read) in MOVED.items():
    setattr(BacktestResult, _name, _moved(_name, _path, _read))

for _name, (_path, _private) in MOVED_METHODS.items():
    setattr(BacktestResult, _name, _moved_method(_name, _path, _private))


__all__ = ["MOVED", "MOVED_METHODS", "BacktestResult", "Book", "Comparable", "IndexBooks",
           "PriceGap", "RebalancePricing", "UnfilledOrder"]
