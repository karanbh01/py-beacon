# beacon/backtest/engine.py
"""
BacktestEngine — simulates portfolio execution against a target weight schedule.
"""
import pandas as pd
import logging
from typing import Dict, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ..data.fetcher import DataFetcher
    from ..index.result import IndexResult

from ..portfolio.base import Portfolio
from .result import BacktestResult

logger = logging.getLogger(__name__)


class BacktestEngine:
    """Simulates portfolio execution against a target weight schedule.

    The engine consumes target weights from an ``IndexResult`` or a
    custom weight dictionary, and simulates trading over a date range
    using prices from a ``DataFetcher``.

    Parameters
    ----------
    start_date : str
        The start date of the backtest (YYYY-MM-DD).
    end_date : str
        The end date of the backtest (YYYY-MM-DD).
    initial_capital : float
        The starting capital for the backtest.
    data_provider : DataFetcher
        Data source for market prices.
    target_index_result : IndexResult, optional
        An IndexResult whose weight_snapshots provide the rebalance
        schedule and target weights. Mutually exclusive with
        *target_weights*.
    target_weights : dict, optional
        Custom weight schedule as a mapping of
        ``pd.Timestamp -> Dict[str, float]``. Mutually exclusive with
        *target_index_result*.
    price_column : str
        Column name to read from market data. Defaults to ``"CLOSE"``.
    """

    def __init__(self,
                 start_date: str,
                 end_date: str,
                 initial_capital: float,
                 data_provider: 'DataFetcher',
                 target_index_result: Optional['IndexResult'] = None,
                 target_weights: Optional[Dict[pd.Timestamp, Dict[str, float]]] = None,
                 price_column: str = "CLOSE"):
        if target_index_result is not None and target_weights is not None:
            raise ValueError(
                "Provide either target_index_result or target_weights, not both."
            )
        if target_index_result is None and target_weights is None:
            raise ValueError(
                "One of target_index_result or target_weights must be provided."
            )

        self.start_date: pd.Timestamp = pd.Timestamp(start_date)
        self.end_date: pd.Timestamp = pd.Timestamp(end_date)
        self.initial_capital: float = initial_capital
        self.data_provider: 'DataFetcher' = data_provider
        self.target_index_result: Optional['IndexResult'] = target_index_result
        self.price_column: str = price_column

        # Normalise weight schedule to a dict
        if target_weights is not None:
            self._weight_schedule: Dict[pd.Timestamp, Dict[str, float]] = target_weights
        else:
            self._weight_schedule = target_index_result.weight_snapshots

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _fetch_price(self, asset_id: str, date: pd.Timestamp) -> Optional[float]:
        """Fetch a single closing price for *asset_id* on *date*."""
        date_str = date.strftime("%Y-%m-%d")
        try:
            df = self.data_provider.fetch_market_data(asset_id, date_str, date_str)
            if not df.empty and self.price_column in df.columns:
                val = df[self.price_column].iloc[0]
                if pd.notna(val):
                    return float(val)
        except Exception as e:
            logger.error(f"Error fetching price for {asset_id} on {date_str}: {e}")
        return None

    def _update_portfolio_prices(self, portfolio: Portfolio, date: pd.Timestamp) -> None:
        """Fetch prices for all holdings and push into the portfolio."""
        prices: Dict[str, float] = {}
        for asset_id in portfolio.holdings:
            price = self._fetch_price(asset_id, date)
            if price is not None:
                prices[asset_id] = price
        portfolio.update_prices(prices)

    def _get_target_weights_for_date(self, date: pd.Timestamp) -> Optional[Dict[str, float]]:
        """Return target weights if *date* is a rebalance date, else None."""
        return self._weight_schedule.get(date)

    def _rebalance(self, portfolio: Portfolio, target_weights: Dict[str, float],
                   date: pd.Timestamp) -> None:
        """Adjust *portfolio* to match *target_weights*. Sells first, then buys."""
        current_value = portfolio.get_total_value()
        if current_value <= 0:
            logger.warning(f"[{date}] Portfolio value is {current_value:.2f}. Skipping rebalance.")
            return

        logger.info(f"[{date}] Rebalancing to target weights: {target_weights}")

        # --- Phase 1: Sells (assets not in target, or overweight) ---
        for asset_id, holding in list(portfolio.holdings.items()):
            price = self._fetch_price(asset_id, date)
            if price is None:
                logger.warning(f"[{date}] No price for {asset_id}. Skipping sell.")
                continue

            target_w = target_weights.get(asset_id, 0.0)
            if target_w == 0:
                # Fully exit
                portfolio.execute_sell(asset_id, holding.quantity, price, date=date)
                logger.debug(f"[{date}] Sold all {holding.quantity:.4f} of {asset_id}")
            else:
                target_value = current_value * target_w
                current_asset_value = holding.quantity * price
                if current_asset_value > target_value + 1e-6:
                    excess_value = current_asset_value - target_value
                    qty_to_sell = excess_value / price
                    if qty_to_sell > 1e-9:
                        portfolio.execute_sell(asset_id, qty_to_sell, price, date=date)
                        logger.debug(f"[{date}] Trimmed {qty_to_sell:.4f} of {asset_id}")

        # --- Phase 2: Buys (new or underweight) ---
        for asset_id, target_w in target_weights.items():
            if target_w <= 0:
                continue

            price = self._fetch_price(asset_id, date)
            if price is None or price <= 0:
                logger.warning(f"[{date}] No valid price for {asset_id}. Skipping buy.")
                continue

            target_value = current_value * target_w
            current_holding_value = 0.0
            if asset_id in portfolio.holdings:
                current_holding_value = portfolio.holdings[asset_id].quantity * price

            deficit = target_value - current_holding_value
            if deficit > 1e-6:
                qty_to_buy = deficit / price
                if portfolio.cash_balance >= deficit:
                    portfolio.execute_buy(asset_id, qty_to_buy, price, date=date)
                    logger.debug(f"[{date}] Bought {qty_to_buy:.4f} of {asset_id}")
                else:
                    logger.warning(
                        f"[{date}] Insufficient cash for {asset_id}. "
                        f"Need {deficit:.2f}, have {portfolio.cash_balance:.2f}"
                    )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    #todo: vectorise run for efficiency, currently iterative and may be slow
    def run(self) -> BacktestResult:
        """Execute the backtest and return a :class:`BacktestResult`.

        Returns
        -------
        BacktestResult
        """
        logger.info(
            f"Starting backtest from {self.start_date.date()} to "
            f"{self.end_date.date()} with capital {self.initial_capital:.2f}"
        )

        portfolio = Portfolio(portfolio_id="backtest_portfolio",
                              initial_cash=self.initial_capital)

        trading_days = pd.bdate_range(start=self.start_date, end=self.end_date, freq="B")
        if trading_days.empty:
            logger.warning("No trading days in the specified date range.")
            return self._build_empty_result(portfolio)

        nav_records: Dict[pd.Timestamp, float] = {}
        cash_records: Dict[pd.Timestamp, float] = {}
        weight_records = []

        for idx, date in enumerate(trading_days):
            # 1. Update prices for existing holdings
            self._update_portfolio_prices(portfolio, date)

            # 2. Check for rebalance
            target_w = self._get_target_weights_for_date(date)
            if target_w is not None:
                self._rebalance(portfolio, target_w, date)
                # Re-price after rebalance
                self._update_portfolio_prices(portfolio, date)

            # 3. Record end-of-day state
            nav = portfolio.get_total_value()
            nav_records[date] = nav
            cash_records[date] = portfolio.cash_balance

            daily_weights: Dict[str, float] = {}
            for asset_id, w in portfolio.get_weights().items():
                daily_weights[f"{asset_id}_weight"] = w
            weight_records.append({"Date": date, **daily_weights})

            # Progress logging
            n = len(trading_days)
            if n > 10 and idx % (n // 10) == 0:
                logger.info(
                    f"Backtest progress: {(idx + 1) / n * 100:.0f}% "
                    f"({date.date()}, NAV={nav:.2f})"
                )

        logger.info(f"Backtest finished. Final NAV: {nav_records[trading_days[-1]]:.2f}")

        portfolio_nav = pd.Series(nav_records, dtype=float)
        portfolio_nav.index.name = "Date"
        cash_history = pd.Series(cash_records, dtype=float)
        cash_history.index.name = "Date"
        weight_df = pd.DataFrame(weight_records)
        if not weight_df.empty:
            weight_df.set_index("Date", inplace=True)

        return BacktestResult(
            portfolio_id=portfolio.portfolio_id,
            initial_capital=self.initial_capital,
            portfolio_nav=portfolio_nav,
            cash_history=cash_history,
            transactions=list(portfolio.transactions),
            actual_weight_history=weight_df,
            target_index_result=self.target_index_result,
        )

    def _build_empty_result(self, portfolio: Portfolio) -> BacktestResult:
        """Build a BacktestResult with no data."""
        return BacktestResult(
            portfolio_id=portfolio.portfolio_id,
            initial_capital=self.initial_capital,
            portfolio_nav=pd.Series(dtype=float),
            cash_history=pd.Series(dtype=float),
            transactions=[],
            actual_weight_history=pd.DataFrame(),
            target_index_result=self.target_index_result,
        )
